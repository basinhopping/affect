"""The smoke test: run before launching a sweep, on one GPU, a few minutes.

1. Expands the sweep file exactly as enqueue would (nothing is queued).
2. Every distinct configuration of the sweep: builds its features, model, gender head and
   projection head, takes one training step on four utterances with the same code a real job runs
   (train.loop.Step, including speed perturbation, mixup and the contrastive loss), and predicts two
   utterances the way the job will (whole or in segments), so a setting that crashes fails here,
   not overnight.
3. The sweep's first job as a real job, cut to 2 epochs of 5 batches, written like any run
   folder under artifacts/runs/smoke_<utc>/; the same for the first job that uses each option a
   real job sets up for itself: global and speaker normalization (statistics), speed
   perturbation, mixup, each gender head, the contrastive loss, segment evaluation, each
   normalization layer, and each model.
4. Times the base configuration at full size: training steps and one validation pass, to
   estimate minutes per job.
Ends with SMOKE OK or SMOKE FAILED.
"""
import copy
import datetime
import time
import traceback

import numpy as np

from affect.registry import FEATURES, MODELS
from affect.runs.spec import config_hash, input_width, plain, run_id, validate_job
from affect.sweep.design import expand
from affect.sweep.enqueue import load_sweep_file
from affect.train.data import (GENDERS, VA_SCALES, gender_index, job_source, prepare, rows_per_speaker, speaker_index,
                               training_waves, va_ratings)
from affect.train.loop import (Normalizer, Step, augmentation_rng, make_aux_head, make_contrastive_head,
                               make_normalizer, make_va_head, predict, segment_samples, trainable_parameters)

# Options a real job sets up for itself, each tried in one short real job (step 3): name, test.
COVERED = (
    ("global normalization", lambda s: s["features"]["normalization"] == "global"),
    ("speaker normalization", lambda s: s["features"]["normalization"] == "speaker"),
    ("speed perturbation", lambda s: tuple(s["augment"].get("speed_factors", [1.0])) != (1.0,)),
    ("mixup", lambda s: s["augment"].get("mixup_alpha", 0.0) > 0),
    ("multitask gender head", lambda s: s["train"].get("aux_gender", "none") == "multitask"),
    ("adversarial gender head", lambda s: s["train"].get("aux_gender", "none") == "adversarial"),
    ("contrastive loss", lambda s: s["train"].get("contrastive", "none") != "none"),
    ("segment evaluation", lambda s: s["train"].get("eval_segment_s") is not None),
    ("subspectral norm", lambda s: s["model"].get("norm", "batch") == "subspectral"),
    ("group norm", lambda s: s["model"].get("norm", "batch") == "group"),
    ("no norm", lambda s: s["model"].get("norm", "batch") == "none"),
    ("model cnn_transformer", lambda s: s["model"]["name"] == "cnn_transformer"),
    ("model cnn_lif", lambda s: s["model"]["name"] == "cnn_lif"),
    ("model snn, frames", lambda s: s["model"]["name"] == "snn" and s["model"]["encoding"] == "frames"),
    ("model snn, repeat", lambda s: s["model"]["name"] == "snn" and s["model"]["encoding"] == "repeat"),
    # added 2026-10-24
    ("valence and activation head", lambda s: s["train"].get("aux_va", "none") != "none"),
    ("external test", lambda s: s["train"].get("external_test", "none") != "none"),
    ("model cnn_rnn", lambda s: s["model"]["name"] == "cnn_rnn"),
    # added 2026-10-26: heads on a pretrained encoder's frames
    ("model frames_pool", lambda s: s["model"]["name"] == "frames_pool"),
    ("model frames_transformer", lambda s: s["model"]["name"] == "frames_transformer"),
    ("model frames_lif", lambda s: s["model"]["name"] == "frames_lif"),
    ("model frames_rnn", lambda s: s["model"]["name"] == "frames_rnn"),
)


def _parts(job, prepared, device, stats_rows):
    """Features, normalizer, model, gender head, projection head, valence and activation head and
    optimizer, built as train_job builds them."""
    import torch

    features = FEATURES.get(job.features_name).target(job.features, prepared.sample_rate, device)
    normalizer = make_normalizer(job.features.normalization, features, prepared, stats_rows, 4, device,
                                 job_source(job, prepared)[0])
    model = MODELS.get(job.model_name).target(job.model, input_width(job), len(prepared.class_names)).to(device)
    aux_head = make_aux_head(job.train, model, device)
    contrastive_head = make_contrastive_head(job.train, model, device)
    va_head = make_va_head(job.train, model, device)
    optimizer = torch.optim.AdamW(trainable_parameters(model, aux_head, contrastive_head, va_head), lr=job.train.lr,
                                  weight_decay=job.train.weight_decay)
    return features, normalizer, model, aux_head, contrastive_head, va_head, optimizer


def _targets(job, prepared, rows):
    n_classes = len(prepared.class_names)
    targets = np.eye(n_classes, dtype=np.float32)[prepared.labels[rows]]
    genders = None
    if job.train.aux_gender != "none":
        genders = np.eye(len(GENDERS), dtype=np.float32)[gender_index(prepared.utterances)[rows]]
    return targets, genders


def _va_targets(job, prepared, rows):
    """The rows' valence and activation ratings standardized with every utterance's means and SDs (a real job uses
    its training rows'), or None without the head."""
    if job.train.aux_va == "none":
        return None
    ratings = va_ratings(prepared.utterances)
    return ((ratings[rows] - ratings.mean(axis=0)) / ratings.std(axis=0)).astype(np.float32)


def _one_step(job_spec: dict, prepared, device) -> str:
    import torch

    job = validate_job(job_spec["settings"])
    rng = np.random.default_rng(0)
    augment_rng = augmentation_rng(0)
    generator = torch.Generator().manual_seed(0)
    rows = np.arange(4)
    if job.train.contrastive != "none" and len(set(prepared.labels[rows].tolist())) == 4:
        # four different labels would leave the contrastive loss no positive pair: the fourth row becomes the next
        # utterance with the first one's label
        rows[3] = int(np.flatnonzero(prepared.labels == prepared.labels[0])[1])
    # The same constructor as a real job; speaker statistics from 3 utterances of every speaker.
    if job.features.normalization == "speaker":
        stats_rows = rows_per_speaker(speaker_index(prepared.utterances)[1], 3)
    else:
        stats_rows = np.arange(8)
    features, normalizer, model, aux_head, contrastive_head, va_head, optimizer = _parts(job, prepared, device,
                                                                                         stats_rows)
    source, rate = job_source(job, prepared)
    crop_len = int(round(job.train.crop_s * rate))
    waves, lengths = training_waves(source, rows, crop_len, job.augment.speed_factors, rng, augment_rng)
    model.train()
    for head in (aux_head, contrastive_head, va_head):
        if head is not None:
            head.train()
    targets, genders = _targets(job, prepared, rows)
    step = Step(job, model, aux_head, features, normalizer, None, device, contrastive_head, va_head)
    loss, _, gender_loss, contrastive_loss = step.loss(
        waves, lengths, rows, targets, genders, generator, augment_rng, prepared.labels[rows],
        _va_targets(job, prepared, rows))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    predicted = predict(model, features, normalizer, source, np.arange(4, 6), 2, device, job.train.precision,
                        segment_samples(job.train, rate), aux_head, va_head)
    n_classes = len(prepared.class_names)
    bad = predicted.probs.shape != (2, n_classes) or not np.isfinite(predicted.probs).all() or not torch.isfinite(loss)
    if aux_head is not None:
        bad = bad or predicted.aux_probs.shape != (2, len(GENDERS)) or not np.isfinite(predicted.aux_probs).all()
    if va_head is not None:
        bad = bad or predicted.va.shape != (2, len(VA_SCALES)) or not np.isfinite(predicted.va).all()
    if bad:
        raise FloatingPointError(f"bad output: loss {loss.item()}, probabilities {predicted.probs}, "
                                 f"gender {predicted.aux_probs}, valence and activation {predicted.va}")
    detail = f"loss {loss.item():.3f}, {sum(p.numel() for p in model.parameters())} parameters"
    if gender_loss is not None:
        detail += f", gender loss {gender_loss.item():.3f}"
    if contrastive_loss is not None:
        detail += f", contrastive loss {contrastive_loss.item():.3f}"
    if step.last_va_loss is not None:
        detail += f", valence and activation loss {step.last_va_loss.item():.3f}"
    return detail


def _timing(job_spec: dict, prepared, device, steps: int) -> dict:
    """Training steps at full batch size, with the job's augmentations, and one validation pass
    the way the job evaluates. Normalisation is per utterance here (no statistics pass)."""
    import torch

    job = validate_job(job_spec["settings"])
    rng = np.random.default_rng(0)
    augment_rng = augmentation_rng(0)
    generator = torch.Generator().manual_seed(0)
    features = FEATURES.get(job.features_name).target(job.features, prepared.sample_rate, device)
    n_classes = len(prepared.class_names)
    model = MODELS.get(job.model_name).target(job.model, input_width(job), n_classes).to(device)
    aux_head = make_aux_head(job.train, model, device)
    contrastive_head = make_contrastive_head(job.train, model, device)
    va_head = make_va_head(job.train, model, device)
    optimizer = torch.optim.AdamW(trainable_parameters(model, aux_head, contrastive_head, va_head), lr=job.train.lr,
                                  weight_decay=job.train.weight_decay)
    normalizer = Normalizer("utterance") if job.features.normalization != "none" else Normalizer("none")
    step = Step(job, model, aux_head, features, normalizer, None, device, contrastive_head, va_head)
    source, rate = job_source(job, prepared)
    crop_len = int(round(job.train.crop_s * rate))
    batch = job.train.batch_size
    model.train()
    for head in (aux_head, contrastive_head, va_head):
        if head is not None:
            head.train()
    for k in range(steps + 5):
        if k == 5:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
        rows = rng.choice(len(prepared.utterances), size=batch, replace=False)
        waves, lengths = training_waves(source, rows, crop_len, job.augment.speed_factors, rng, augment_rng)
        targets, genders = _targets(job, prepared, rows)
        loss, _, _, _ = step.loss(waves, lengths, rows, targets, genders, generator, augment_rng, prepared.labels[rows],
                                  _va_targets(job, prepared, rows))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = (time.perf_counter() - start) / steps
    val_rows = np.arange(min(1100, len(prepared.utterances)))
    predicted = predict(model, features, normalizer, source, val_rows, job.train.eval_batch_size, device,
                        job.train.precision, segment_samples(job.train, rate), aux_head, va_head)
    return {"step_seconds": step_seconds, "eval_seconds_1100_utterances": predicted.seconds,
            "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20}


def smoke(sweep_file, device) -> bool:
    from affect.sweep.worker import execute

    ok = True
    sweep = load_sweep_file(sweep_file)
    jobs, summary = expand(sweep)
    print(f"sweep {summary['sweep_id']}: {summary['configs']} configurations, {summary['jobs']} jobs")
    for stage in summary["stages"]:
        print(f"  stage {stage['name']}: {stage['configs']} configurations x {len(stage['seeds'])} seeds, "
              f"priority {stage['priority']}, invalid random draws skipped {stage['invalid_random_draws_skipped']}")
    start = time.perf_counter()
    job0 = jobs[0]
    base = validate_job(job0["settings"])
    prepared = prepare(base.data_name, base.data)
    print(f"data: {len(prepared.utterances)} utterances loaded in {time.perf_counter() - start:.1f}s, "
          f"classes {prepared.class_names}")

    distinct = {}
    for job in jobs:
        distinct.setdefault(job["config_hash"], job)
    print(f"one training step for each of {len(distinct)} configurations:")
    for n, job in enumerate(distinct.values(), 1):
        try:
            detail = _one_step(job, prepared, device)
            if n <= 3 or n == len(distinct) or n % 25 == 0:
                print(f"  {n:4d} ok  {job['stage']}: {job['label']} ({detail})")
        except Exception:
            ok = False
            print(f"  {n:4d} FAILED  {job['stage']}: {job['label']}")
            traceback.print_exc()

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # A short real job for the first job of the sweep, and for the first job using each option
    # that a real job sets up for itself (COVERED).
    short_jobs = [("first job", job0)]
    for what, uses in COVERED:
        found = next((j for j in jobs if uses(j["settings"])), None)
        if found is not None and all(found is not j for _, j in short_jobs):
            short_jobs.append((f"first job with {what}", found))
    for what, job in short_jobs:
        settings = copy.deepcopy(job["settings"])
        settings["train"].update({"epochs": 2, "warmup_epochs": 1, "max_batches_per_epoch": 5})
        short = validate_job(settings)
        spec = dict(job, sweep_id=f"smoke_{stamp}", stage="smoke", label=f"{what}, 2 epochs of 5 batches",
                    settings=plain(short), config_hash=config_hash(short), run_id=run_id(short, job["fold"], job["seed"]))
        print(f"a short real job ({what}) into artifacts/runs/smoke_{stamp}/{spec['run_id']}:")
        try:
            metrics = execute(spec, device, lambda m: print("  " + m))
            print(f"  ok: {metrics['epochs_run']} epochs, validation UA {metrics['best_val_ua']:.4f} (meaningless at this length)")
        except Exception:
            ok = False
            print("  FAILED")
            traceback.print_exc()

    try:
        t = _timing(job0, prepared, device, 30)
        base_train = base.train
        n_train = int(len(prepared.utterances) * 3 / 5)
        steps = n_train // base_train.batch_size
        epoch_seconds = steps * t["step_seconds"] + t["eval_seconds_1100_utterances"]
        print(f"timing, base configuration on this GPU alone: {1000 * t['step_seconds']:.1f} ms per training step "
              f"(batch {base_train.batch_size}, crop {base_train.crop_s}s), {t['eval_seconds_1100_utterances']:.1f}s "
              f"per validation pass, peak GPU memory {t['peak_gpu_mib']:.0f} MiB")
        print(f"  about {epoch_seconds:.0f}s per epoch; a job stopping after 30 epochs takes about "
              f"{30 * epoch_seconds / 60:.1f} min, one running all {base_train.epochs} epochs "
              f"{base_train.epochs * epoch_seconds / 60:.1f} min")
    except Exception:
        ok = False
        print("timing FAILED")
        traceback.print_exc()
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return ok
