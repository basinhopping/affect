"""One job: train on the fold's training sessions, keep the epoch with the best validation UA,
then predict validation and test with that epoch's weights (plan section 2, 6.5).

Selection uses validation only. Test predictions are written into the run folder, and the
status report never reads them (plan 3.5).

Added 2026-09-26, each off in every configuration written before then, which runs exactly as it
did (the same random draws in the same order): speed perturbation and mixup (job.augment), a
gender head, multitask or adversarial (job.train.aux_gender), and evaluation on averaged
segments (job.train.eval_segment_s). Speed perturbation and mixup draw from their own random
stream (augmentation_rng), so the crops, the epoch order and the masks keep the streams they
always had.
"""
import csv
import dataclasses
import io
import math
import time

import numpy as np

from affect.augment import mixup
from affect.augment.speed import cached_bytes
from affect.augment.specaugment import spec_augment
from affect.data.splits import assign
from affect.eval.metrics import report, report_by_group
from affect.registry import FEATURES, MODELS
from affect.runs.provenance import collect, utc_now
from affect.runs.spec import check_job, config_hash, validate_job
from affect.train.data import (GENDERS, class_weights, eval_batches, gender_index, prepare, segment_batches,
                               speaker_index, train_order, training_waves)


def augmentation_rng(seed: int) -> np.random.Generator:
    """The random stream of speed perturbation and mixup: the job seed plus a fixed second word,
    so it is separate from the crop and epoch-order stream (default_rng(seed))."""
    return np.random.default_rng([seed, 20260927])


def lr_factor(step: int, warmup_steps: int, total_steps: int, schedule: str) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class Normalizer:
    """utterance: per band, mean 0 and SD 1 over the utterance's real frames.
    global: per band, with training-set statistics. speaker: per band, with the statistics of the
    row's speaker (mean and std are [speakers, bands], speaker_of maps a row to its speaker).
    none: unchanged. After normalising, padded frames are set to 0.
    Called with the batch, its real frame counts, and the batch's rows (used by speaker only)."""

    def __init__(self, mode: str, mean=None, std=None, speaker_of=None):
        if mode == "speaker" and speaker_of is None:
            raise ValueError("speaker normalization needs the row-to-speaker map")
        self.mode, self.mean, self.std, self.speaker_of = mode, mean, std, speaker_of

    def __call__(self, x, valid_frames, rows=None):
        import torch

        if self.mode == "none":
            return x
        frames = x.shape[-1]
        mask = (torch.arange(frames, device=x.device)[None, :] < valid_frames[:, None]).to(x.dtype)[:, None, :]
        if self.mode == "utterance":
            count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean = (x * mask).sum(dim=-1, keepdim=True) / count
            var = (((x - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / count
            x = (x - mean) / torch.sqrt(var + 1e-5)
        elif self.mode == "global":
            x = (x - self.mean[None, :, None]) / self.std[None, :, None]
        elif self.mode == "speaker":
            if rows is None or len(rows) != x.shape[0]:
                raise ValueError("speaker normalization needs the rows of the batch")
            idx = torch.as_tensor(self.speaker_of[np.asarray(rows)], device=x.device)
            x = (x - self.mean[idx][:, :, None]) / self.std[idx][:, :, None]
        else:
            raise ValueError(self.mode)
        return x * mask


def global_stats(features, audio, rows, batch_size, device):
    """Per-band mean and SD of the log-mel over every real frame of the given rows."""
    import torch

    total = sq = count = None
    for _, waves, lengths in eval_batches(audio, rows, batch_size):
        x = features(torch.from_numpy(waves).to(device)).double()
        frames = features.frames(torch.from_numpy(lengths).to(device)).clamp_max(x.shape[-1])
        mask = (torch.arange(x.shape[-1], device=device)[None, :] < frames[:, None]).double()[:, None, :]
        s, s2, n = (x * mask).sum(dim=(0, 2)), (x * x * mask).sum(dim=(0, 2)), mask.sum()
        total, sq, count = (s, s2, n) if total is None else (total + s, sq + s2, count + n)
    mean = total / count
    std = torch.sqrt((sq / count - mean * mean).clamp_min(1e-10))
    return mean.float(), std.float()


def speaker_stats(features, audio, rows, speaker_of, n_speakers, batch_size, device):
    """Per speaker and band, mean and SD of the log-mel over every real frame of the given rows.
    Returns two [n_speakers, n_mels] tensors. Every speaker must have at least one row."""
    import torch

    total = torch.zeros(n_speakers, features.n_mels, dtype=torch.float64, device=device)
    sq = torch.zeros_like(total)
    count = torch.zeros(n_speakers, dtype=torch.float64, device=device)
    for positions, waves, lengths in eval_batches(audio, rows, batch_size):
        x = features(torch.from_numpy(waves).to(device)).double()
        frames = features.frames(torch.from_numpy(lengths).to(device)).clamp_max(x.shape[-1])
        mask = (torch.arange(x.shape[-1], device=device)[None, :] < frames[:, None]).double()[:, None, :]
        spk = torch.as_tensor(speaker_of[rows[positions]], device=device)
        total.index_add_(0, spk, (x * mask).sum(dim=2))
        sq.index_add_(0, spk, (x * x * mask).sum(dim=2))
        count.index_add_(0, spk, mask.sum(dim=(1, 2)))
    if bool((count == 0).any()):
        raise ValueError(f"speakers {torch.nonzero(count == 0).flatten().tolist()} have no frames for statistics")
    mean = total / count[:, None]
    std = torch.sqrt((sq / count[:, None] - mean * mean).clamp_min(1e-10))
    return mean.float(), std.float()


def make_normalizer(mode: str, features, prepared, stats_rows, batch_size, device) -> Normalizer:
    """The normalizer a job uses. global: statistics over stats_rows (the training rows).
    speaker: per-speaker statistics over stats_rows, which must include every speaker the job sees;
    train_job passes every row, so each speaker is normalized with its own audio (labels unused)."""
    if mode == "global":
        mean, std = global_stats(features, prepared.audio, stats_rows, batch_size, device)
        return Normalizer("global", mean, std)
    if mode == "speaker":
        names, speaker_of = speaker_index(prepared.utterances)
        mean, std = speaker_stats(features, prepared.audio, stats_rows, speaker_of, len(names), batch_size, device)
        return Normalizer("speaker", mean, std, speaker_of)
    return Normalizer(mode)


def soft_cross_entropy(logits, targets, weights):
    import torch

    per_class = -targets * torch.log_softmax(logits.float(), dim=-1)
    if weights is not None:
        per_class = per_class * weights[None, :]
    return per_class.sum(dim=-1).mean()


def make_aux_head(train_settings, model, device):
    """The gender head a job's settings ask for, on the device, or None."""
    if train_settings.aux_gender == "none":
        return None
    from affect.models.heads import build_aux_head

    return build_aux_head(train_settings.aux_gender, model.embedding_dim, len(GENDERS), model.head_dropout).to(device)


def trainable_parameters(model, aux_head) -> list:
    return list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])


def segment_samples(train_settings, sample_rate: int):
    """(segment, hop) in samples for segment evaluation, or None for whole utterances."""
    if train_settings.eval_segment_s is None:
        return None
    return (int(round(train_settings.eval_segment_s * sample_rate)),
            int(round(train_settings.eval_segment_hop_s * sample_rate)))


class Step:
    """The loss of one training batch: features, normalisation, SpecAugment masks, mixup, the
    model, and the gender head when there is one. train_job and the smoke test both use it, so the
    smoke test runs the code a real job runs. Returns (loss to minimise, emotion loss, gender loss
    or None)."""

    def __init__(self, job, model, aux_head, features, normalizer, weights, device):
        self.augment, self.train = job.augment, job.train
        self.model, self.aux_head, self.features, self.normalizer = model, aux_head, features, normalizer
        self.weights, self.device = weights, device

    def loss(self, waves, lengths, rows, targets, gender_targets, mask_generator, augment_rng):
        import torch

        device = self.device
        x = self.features(torch.from_numpy(waves).to(device, non_blocking=True))
        frames = self.features.frames(torch.from_numpy(lengths).to(device)).clamp_max(x.shape[-1])
        x = spec_augment(self.normalizer(x, frames, rows), frames, self.augment, mask_generator)
        t = torch.from_numpy(targets).to(device)
        g = torch.from_numpy(gender_targets).to(device) if self.aux_head is not None else None
        if self.augment.mixup_alpha > 0:
            weights, partner = mixup.draw(self.augment.mixup_alpha, x.shape[0], augment_rng)
            x, frames, mixed = mixup.mix(x, frames, [t] if g is None else [t, g], weights, partner)
            t = mixed[0]
            g = mixed[1] if g is not None else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.train.precision == "bf16"):
            if self.aux_head is None:
                logits = self.model(x, frames)
            else:
                embedding = self.model.embed(x, frames)
                logits = self.model.classify(embedding)
                aux_logits = self.aux_head(embedding)
        emotion = soft_cross_entropy(logits, t, self.weights)
        # A spiking model's activity penalty (affect.models.lif, added 2026-09-29); other models
        # have none, and their loss is exactly what it was.
        penalty = self.model.activity_penalty() if hasattr(self.model, "activity_penalty") else None
        total = emotion if penalty is None else emotion + penalty
        if self.aux_head is None:
            return total, emotion, None
        gender = soft_cross_entropy(aux_logits, g, None)
        return total + self.train.aux_gender_weight * gender, emotion, gender


@dataclasses.dataclass
class Predictions:
    probs: np.ndarray               # [rows, classes] class probabilities
    aux_probs: np.ndarray | None    # [rows, 2] gender probabilities (female, male) when a gender head was given
    seconds: float


def predict(model, features, normalizer, audio, rows, batch_size, device, precision, segment=None,
            aux_head=None) -> Predictions:
    """Class probabilities for the given rows: from each whole utterance (segment None), or the
    mean of the probabilities of its segments (segment = (length, hop) in samples)."""
    import torch

    model.eval()
    if aux_head is not None:
        aux_head.eval()
    sums = aux_sums = None
    counts = np.zeros(rows.shape[0], dtype=np.float64)
    batches = (eval_batches(audio, rows, batch_size) if segment is None
               else segment_batches(audio, rows, batch_size, segment[0], segment[1]))
    start = time.perf_counter()
    with torch.no_grad():
        for positions, waves, lengths in batches:
            x = features(torch.from_numpy(waves).to(device))
            frames = features.frames(torch.from_numpy(lengths).to(device)).clamp_max(x.shape[-1])
            x = normalizer(x, frames, rows[positions])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                if aux_head is None:
                    logits = model(x, frames)
                else:
                    embedding = model.embed(x, frames)
                    logits = model.classify(embedding)
                    aux_logits = aux_head(embedding)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            if sums is None:
                sums = np.zeros((rows.shape[0], probs.shape[1]), dtype=np.float64)
            if aux_head is not None:
                aux_probs = torch.softmax(aux_logits.float(), dim=-1).cpu().numpy()
                if aux_sums is None:
                    aux_sums = np.zeros((rows.shape[0], aux_probs.shape[1]), dtype=np.float64)
            if segment is None:
                sums[positions] = probs
                if aux_head is not None:
                    aux_sums[positions] = aux_probs
            else:
                np.add.at(sums, positions, probs)
                if aux_head is not None:
                    np.add.at(aux_sums, positions, aux_probs)
            np.add.at(counts, positions, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if segment is not None:
        sums /= counts[:, None]
        if aux_sums is not None:
            aux_sums /= counts[:, None]
    elif not (counts == 1).all():
        raise RuntimeError("whole-utterance evaluation must see every row exactly once")
    return Predictions(sums, aux_sums, seconds)


def predictions_csv(prepared, rows, probs) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utt_id", "speaker", "kind", "label", "pred"] + [f"prob_{c}" for c in prepared.class_names])
    for k, row in enumerate(rows):
        u = prepared.utterances[row]
        writer.writerow([u.utt_id, u.speaker, u.kind, u.label, prepared.class_names[int(probs[k].argmax())]]
                        + [f"{p:.6f}" for p in probs[k]])
    return buffer.getvalue()


def evaluate(prepared, rows, probs) -> dict:
    y_true = prepared.labels[rows]
    y_pred = probs.argmax(axis=1)
    out = report(y_true, y_pred, prepared.class_names)
    out["by_kind"] = report_by_group(y_true, y_pred, prepared.kinds[rows], prepared.class_names)
    out["mean_nll"] = float(-np.mean(np.log(np.clip(probs[np.arange(rows.shape[0]), y_true], 1e-12, None))))
    return out


def gender_report(true_gender: np.ndarray, aux_probs: np.ndarray) -> dict:
    """How well the gender head predicts the speaker's gender: accuracy, recall per gender present
    (a split may hold one gender only), and the mean of those recalls."""
    predicted = aux_probs.argmax(axis=1)
    out = {"accuracy": float(np.mean(predicted == true_gender)), "n": int(true_gender.shape[0])}
    recalls = []
    for k, name in enumerate(GENDERS):
        members = true_gender == k
        out[f"n_{name}"] = int(members.sum())
        out[f"recall_{name}"] = float(np.mean(predicted[members] == k)) if members.any() else None
        if members.any():
            recalls.append(out[f"recall_{name}"])
    out["ua"] = float(np.mean(recalls))
    return out


def train_job(spec: dict, device, log) -> dict:
    """Run one job. Returns every file of the run folder's content, keyed by file name."""
    import torch

    started = utc_now()
    job = validate_job(spec["settings"])
    prepared = prepare(job.data_name, job.data)
    check_job(job, prepared.sample_rate)
    if config_hash(job) != spec["config_hash"]:
        raise ValueError("the settings do not hash to the job's config_hash; the job file was edited")
    fold = next((f for f in prepared.folds if f.name == spec["fold"]), None)
    if fold is None:
        raise ValueError(f"no fold {spec['fold']!r}; folds are {[f.name for f in prepared.folds]}")
    roles = assign(prepared.utterances, fold)
    train_rows = np.array([prepared.position[u] for u in roles["train"]], dtype=np.int64)
    val_rows = np.array([prepared.position[u] for u in roles["val"]], dtype=np.int64)
    test_rows = np.array([prepared.position[u] for u in roles["test"]], dtype=np.int64)
    ts, seed = job.train, int(spec["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mask_generator = torch.Generator().manual_seed(seed)
    augment_rng = augmentation_rng(seed)
    n_classes = len(prepared.class_names)

    features = FEATURES.get(job.features_name).target(job.features, prepared.sample_rate, device)
    targets = (prepared.soft if ts.labels == "soft" else np.eye(n_classes, dtype=np.float32)[prepared.labels])[train_rows]
    train_labels = prepared.labels[train_rows]
    if ts.control == "shuffle_train_labels":
        permutation = np.random.default_rng(seed + 1_000_003).permutation(train_rows.shape[0])
        targets, train_labels = targets[permutation], train_labels[permutation]
    if ts.label_smoothing > 0:
        targets = (1.0 - ts.label_smoothing) * targets + ts.label_smoothing / n_classes
    targets = targets.astype(np.float32)
    genders = gender_targets = None
    if ts.aux_gender != "none":
        genders = gender_index(prepared.utterances)
        gender_targets = np.eye(len(GENDERS), dtype=np.float32)[genders[train_rows]]   # never shuffled
    weights = None
    if ts.class_balance == "weighted_loss":
        weights = torch.from_numpy(class_weights(train_labels, n_classes)).to(device)
    stats_rows = np.arange(len(prepared.utterances)) if job.features.normalization == "speaker" else train_rows
    normalizer = make_normalizer(job.features.normalization, features, prepared, stats_rows, ts.eval_batch_size, device)

    model = MODELS.get(job.model_name).target(job.model, job.features.n_mels, n_classes).to(device)
    aux_head = make_aux_head(ts, model, device)
    n_parameters = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(trainable_parameters(model, aux_head), lr=ts.lr, weight_decay=ts.weight_decay)
    step = Step(job, model, aux_head, features, normalizer, weights, device)
    segment = segment_samples(ts, prepared.sample_rate)
    steps_per_epoch = train_rows.shape[0] // ts.batch_size
    if ts.max_batches_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, ts.max_batches_per_epoch)
    total_steps = ts.epochs * steps_per_epoch
    warmup_steps = ts.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step_number: lr_factor(step_number, warmup_steps, total_steps, ts.schedule))
    crop_len = int(round(ts.crop_s * prepared.sample_rate))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    extras = []
    if tuple(job.augment.speed_factors) != (1.0,):
        extras.append(f"speeds {list(job.augment.speed_factors)}")
    if job.augment.mixup_alpha > 0:
        extras.append(f"mixup alpha {job.augment.mixup_alpha}")
    if aux_head is not None:
        extras.append(f"{ts.aux_gender} gender head, weight {ts.aux_gender_weight}")
    if segment is not None:
        extras.append(f"evaluation on {ts.eval_segment_s}s segments every {ts.eval_segment_hop_s}s")
    log(f"run {spec['run_id']}: fold {fold.name} (val {fold.val_group}), seed {seed}, train {train_rows.shape[0]}, "
        f"val {val_rows.shape[0]}, test {test_rows.shape[0]}, {n_parameters} parameters, {steps_per_epoch} steps/epoch"
        + "".join(f", {e}" for e in extras))
    curves, best_ua, best_epoch, best_state, best_aux_state, stopped_early = [], -1.0, 0, None, None, False
    best_train_report = None
    train_start = time.perf_counter()
    for epoch in range(1, ts.epochs + 1):
        model.train()
        if aux_head is not None:
            aux_head.train()
        epoch_start = time.perf_counter()
        order = train_order(train_labels, ts.class_balance, rng)
        loss_sum = torch.zeros((), device=device)
        emotion_sum = torch.zeros((), device=device)
        gender_sum = torch.zeros((), device=device)
        for b in range(steps_per_epoch):
            picked = order[b * ts.batch_size:(b + 1) * ts.batch_size]
            rows = train_rows[picked]
            waves, lengths = training_waves(prepared.audio, rows, crop_len, job.augment.speed_factors, rng, augment_rng)
            loss, emotion_loss, gender_loss = step.loss(
                waves, lengths, rows, targets[picked], gender_targets[picked] if gender_targets is not None else None,
                mask_generator, augment_rng)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum = loss_sum + loss.detach()
            if gender_loss is not None:
                emotion_sum = emotion_sum + emotion_loss.detach()
                gender_sum = gender_sum + gender_loss.detach()
        train_loss = float(loss_sum.item()) / max(1, steps_per_epoch)
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"training loss is {train_loss} in epoch {epoch}")
        train_seconds = time.perf_counter() - epoch_start
        predicted = predict(model, features, normalizer, prepared.audio, val_rows, ts.eval_batch_size, device,
                            ts.precision, segment)
        val = evaluate(prepared, val_rows, predicted.probs)
        train_report = train_pass = None
        if ts.eval_train == "every_epoch":
            # the training utterances predicted like the validation ones (eval mode, no augmentation, true
            # labels), with the weights just validated; predict changes no weight, statistic or random state
            train_pass = predict(model, features, normalizer, prepared.audio, train_rows, ts.eval_batch_size,
                                 device, ts.precision, segment)
            train_report = evaluate(prepared, train_rows, train_pass.probs)
        curve = {"epoch": epoch, "train_loss": train_loss, "val_ua": val["ua"], "val_wa": val["wa"],
                 "val_macro_f1": val["macro_f1"], "val_mean_nll": val["mean_nll"],
                 "lr": optimizer.param_groups[0]["lr"], "train_seconds": round(train_seconds, 3),
                 "eval_seconds": round(predicted.seconds, 3)}
        if aux_head is not None:
            curve["train_loss_emotion"] = float(emotion_sum.item()) / max(1, steps_per_epoch)
            curve["train_loss_gender"] = float(gender_sum.item()) / max(1, steps_per_epoch)
        if train_report is not None:
            curve.update({"train_ua": train_report["ua"], "train_wa": train_report["wa"],
                          "train_macro_f1": train_report["macro_f1"], "train_mean_nll": train_report["mean_nll"],
                          "train_eval_seconds": round(train_pass.seconds, 3)})
        curves.append(curve)
        improved = val["ua"] > best_ua
        if improved:
            best_ua, best_epoch = val["ua"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_train_report = train_report
            if aux_head is not None:
                best_aux_state = {k: v.detach().clone() for k, v in aux_head.state_dict().items()}
        train_note = f" train UA {train_report['ua']:.4f}" if train_report is not None else ""
        log(f"  epoch {epoch:3d} loss {train_loss:.4f}{train_note} val UA {val['ua']:.4f} WA {val['wa']:.4f} "
            f"({train_seconds:.1f}s + {predicted.seconds:.1f}s){' best' if improved else ''}")
        if epoch - best_epoch >= ts.early_stopping_patience:
            stopped_early = True
            break
    train_total = time.perf_counter() - train_start

    model.load_state_dict(best_state)
    if aux_head is not None:
        aux_head.load_state_dict(best_aux_state)
    spiking = hasattr(model, "spike_report")          # the LIF models record spikes on this pass (2026-09-29)
    if spiking:
        model.record(True)
    val_pred = predict(model, features, normalizer, prepared.audio, val_rows, ts.eval_batch_size, device, ts.precision,
                       segment, aux_head)
    spikes = None
    if spiking:
        model.record(False)
        spikes = model.spike_report()
        rates = ", ".join(f"{k} {v:.3f}" for k, v in spikes["firing_rate"].items() if v is not None)
        log(f"  spikes on validation: {rates}" + (f"; SILENT {spikes['silent_layers']}" if spikes["silent_layers"] else ""))
    test_pred = predict(model, features, normalizer, prepared.audio, test_rows, ts.eval_batch_size, device,
                        ts.precision, segment, aux_head)
    val_final = evaluate(prepared, val_rows, val_pred.probs)
    if abs(val_final["ua"] - best_ua) > 1e-9:
        log(f"  note: best-epoch weights reloaded give val UA {val_final['ua']:.6f}, recorded {best_ua:.6f}")
    peak_mib = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
    provenance = collect("affect.train.loop")
    provenance.update({
        "started_utc": started, "finished_utc": utc_now(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "data_manifest_sha256": prepared.manifest_sha256, "folds_hash": prepared.folds_hash,
    })
    metrics_val = {"best_epoch": best_epoch, "best_val_ua": best_ua, "epochs_run": len(curves),
                   "stopped_early": stopped_early, "curves": curves, "final": val_final}
    if ts.eval_train == "every_epoch":
        # the training utterances' full report (per-class recall, confusion matrix) at the chosen epoch
        metrics_val["train_final"] = best_train_report
    metrics_test = {"final": evaluate(prepared, test_rows, test_pred.probs)}
    efficiency = {"n_parameters": n_parameters, "train_seconds": round(train_total, 2),
                  "seconds_per_epoch": round(train_total / len(curves), 3), "epochs_run": len(curves),
                  "test_seconds": round(test_pred.seconds, 3),
                  "ms_per_test_utterance": round(1000 * test_pred.seconds / test_rows.shape[0], 3),
                  "peak_gpu_mib": round(peak_mib, 1) if peak_mib is not None else None}
    if aux_head is not None:
        metrics_val["aux_gender"] = gender_report(genders[val_rows], val_pred.aux_probs)
        metrics_test["aux_gender"] = gender_report(genders[test_rows], test_pred.aux_probs)
        efficiency["aux_parameters"] = int(sum(p.numel() for p in aux_head.parameters()))
    if tuple(job.augment.speed_factors) != (1.0,):
        efficiency["speed_cache_mib_in_process"] = round(cached_bytes() / 2 ** 20, 1)
    if spikes is not None:
        efficiency["spikes_validation"] = spikes
    return {
        "metrics_val.json": metrics_val,
        "predictions_val.csv": predictions_csv(prepared, val_rows, val_pred.probs),
        "metrics_test.json": metrics_test,
        "predictions_test.csv": predictions_csv(prepared, test_rows, test_pred.probs),
        "efficiency.json": efficiency,
        "provenance.json": provenance,
    }
