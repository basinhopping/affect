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

Added 2026-10-15, off in every configuration written before then, which runs exactly as it did: a
supervised contrastive loss through a projection head (job.train.contrastive). The head is built
after the model and the gender head, so a job without it makes the same random draws as before.

Added 2026-10-20, off in every configuration written before then: an external test (job.train.external_test).
After the test set, the chosen epoch's model scores a corpus it never trains or chooses on (EmoFilm), with the
job's own normalization; it draws no random number and changes nothing the job measured before.

Added 2026-10-24, off in every configuration written before then, which runs exactly as it did: a valence and
activation head (job.train.aux_va), built after every other head, so a job without it makes the same random draws as
before; and the external test of a model with more classes than the corpus (four_merged on EmoFilm's three).
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
from affect.runs.spec import check_job, config_hash, input_width, validate_job
from affect.train.data import (EXTERNAL_LABEL_SETS, GENDERS, VA_SCALES, class_weights, eval_batches, gender_index,
                               job_source, prepare, prepare_external, segment_batches, speaker_index, train_order,
                               training_waves, va_ratings)


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


def make_normalizer(mode: str, features, prepared, stats_rows, batch_size, device, audio=None) -> Normalizer:
    """The normalizer a job uses. global: statistics over stats_rows (the training rows).
    speaker: per-speaker statistics over stats_rows, which must include every speaker the job sees;
    train_job passes every row, so each speaker is normalized with its own audio (labels unused).
    audio: what the features read (train.data.job_source; added 2026-10-26), prepared.audio when None."""
    if mode == "global":
        mean, std = global_stats(features, prepared.audio if audio is None else audio, stats_rows, batch_size, device)
        return Normalizer("global", mean, std)
    if mode == "speaker":
        names, speaker_of = speaker_index(prepared.utterances)
        mean, std = speaker_stats(features, prepared.audio if audio is None else audio, stats_rows, speaker_of,
                                  len(names), batch_size, device)
        return Normalizer("speaker", mean, std, speaker_of)
    return Normalizer(mode)


def soft_cross_entropy(logits, targets, weights):
    import torch

    per_class = -targets * torch.log_softmax(logits.float(), dim=-1)
    if weights is not None:
        per_class = per_class * weights[None, :]
    return per_class.sum(dim=-1).mean()


def supervised_contrastive_loss(projections, labels, temperature: float):
    """The supervised contrastive loss of one batch with one view of each utterance (Khosla et al.
    2020, equation 2, L_out; added 2026-10-15). projections: [n, d], n >= 2, any scale: each row is
    L2-normalised here, in float32 (float64 stays float64). labels: [n] class indices. For anchor i,
    with A(i) every row but i and P(i) the rows of A(i) with i's label:
        loss_i = -(1 / |P(i)|) sum over p in P(i) of log( exp(z_i.z_p / t) / sum over a in A(i) of exp(z_i.z_a / t) )
    Anchors with an empty P(i) are left out and the loss is the mean of loss_i over the others; it is 0,
    with gradients of 0, when every anchor's P(i) is empty. Equation 2 as written: the reference code of
    the paper also multiplies the loss by t / 0.07, which this does not. Computed without autocast, so the
    similarities are float32 even when the caller is inside an autocast region."""
    import torch

    with torch.autocast(device_type=projections.device.type, enabled=False):
        z = projections if projections.dtype == torch.float64 else projections.float()
        z = torch.nn.functional.normalize(z, dim=1)
        labels = labels.to(z.device)
        n = z.shape[0]
        own = torch.eye(n, dtype=torch.bool, device=z.device)
        logits = (z @ z.T / temperature).masked_fill(own, float("-inf"))   # i itself is not in A(i)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive = (labels[:, None] == labels[None, :]) & ~own
        n_positive = positive.sum(dim=1)
        per_anchor = log_prob.masked_fill(~positive, 0.0).sum(dim=1) / n_positive.clamp_min(1)
        n_anchors = (n_positive > 0).sum()
        return -per_anchor.sum() / n_anchors.clamp_min(1)


def concordance(pred, target):
    """Lin's (1989) concordance correlation coefficient of each column: 2 cov / (var_pred + var_target + (mean_pred -
    mean_target)^2), with population (1 / n) moments. pred and target: [n, k] float tensors; returns [k]. A column
    whose denominator is 0 (both constant and equal) gives 0. Unchanged when the same affine map is applied to both."""
    pm, tm = pred.mean(dim=0), target.mean(dim=0)
    pc, tc = pred - pm, target - tm
    covariance = (pc * tc).mean(dim=0)
    denominator = (pc * pc).mean(dim=0) + (tc * tc).mean(dim=0) + (pm - tm) ** 2
    return 2.0 * covariance / denominator.clamp_min(1e-12)


def concordance_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """concordance in float64 numpy, for the reports: pred and target [n, k]; returns [k]."""
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    pm, tm = pred.mean(axis=0), target.mean(axis=0)
    pc, tc = pred - pm, target - tm
    denominator = (pc * pc).mean(axis=0) + (tc * tc).mean(axis=0) + (pm - tm) ** 2
    return 2.0 * (pc * tc).mean(axis=0) / np.maximum(denominator, 1e-12)


def va_loss(pred, target, kind: str):
    """The valence and activation loss of one batch (train.aux_va_loss, added 2026-10-24), in float32 outside autocast:
    ccc, 1 minus the mean over the columns of their concordance correlation coefficients; mse, the mean squared error.
    pred and target: [n, 2], target standardized with the fold's training statistics."""
    import torch

    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        if kind == "ccc":
            return 1.0 - concordance(p, t).mean()
        if kind == "mse":
            return ((p - t) ** 2).mean()
    raise ValueError(f"unknown valence and activation loss {kind!r}")


def make_aux_head(train_settings, model, device):
    """The gender head a job's settings ask for, on the device, or None."""
    if train_settings.aux_gender == "none":
        return None
    from affect.models.heads import build_aux_head

    return build_aux_head(train_settings.aux_gender, model.embedding_dim, len(GENDERS), model.head_dropout).to(device)


def make_va_head(train_settings, model, device):
    """The valence and activation head a job's settings ask for (train.aux_va, added 2026-10-24), on the device, or
    None: dropout and one linear layer from the model's pooled summary to two outputs (VA_SCALES), the gender head's
    multitask form. Built after every other head, so a job without it makes the random draws it always did."""
    if train_settings.aux_va == "none":
        return None
    from affect.models.heads import build_aux_head

    return build_aux_head("multitask", model.embedding_dim, len(VA_SCALES), model.head_dropout).to(device)


def make_contrastive_head(train_settings, model, device):
    """The projection head of the contrastive loss a job's settings ask for, on the device, or None."""
    if train_settings.contrastive == "none":
        return None
    from affect.models.heads import build_projection_head

    return build_projection_head(model.embedding_dim, train_settings.contrastive_dim).to(device)


def trainable_parameters(model, aux_head, contrastive_head=None, va_head=None) -> list:
    """The model's parameters, then the gender head's, the projection head's and the valence and activation head's
    when there are any, so a job without extra heads gives the optimizer the list it always did."""
    out = list(model.parameters())
    for head in (aux_head, contrastive_head, va_head):
        if head is not None:
            out += list(head.parameters())
    return out


def segment_samples(train_settings, sample_rate: int):
    """(segment, hop) in samples for segment evaluation, or None for whole utterances."""
    if train_settings.eval_segment_s is None:
        return None
    return (int(round(train_settings.eval_segment_s * sample_rate)),
            int(round(train_settings.eval_segment_hop_s * sample_rate)))


class Step:
    """The loss of one training batch: features, normalisation, SpecAugment masks, mixup, the
    model, the gender head when there is one, the contrastive loss through its projection head
    when there is one, and the valence and activation head when there is one (added 2026-10-24).
    train_job and the smoke test both use it, so the smoke test runs the code a real job runs.
    Returns (loss to minimise, emotion loss, gender loss or None, contrastive loss or None); the
    contrastive loss needs the batch's class labels (labels, [n] class indices), the valence and
    activation head the batch's standardized ratings (va_targets, [n, 2]), and its loss is kept in
    last_va_loss (None without the head), so the four values returned stay what they always were."""

    def __init__(self, job, model, aux_head, features, normalizer, weights, device, contrastive_head=None,
                 va_head=None):
        self.augment, self.train = job.augment, job.train
        self.model, self.aux_head, self.features, self.normalizer = model, aux_head, features, normalizer
        self.weights, self.device = weights, device
        self.contrastive_head = contrastive_head
        self.va_head = va_head
        self.last_va_loss = None
        if (contrastive_head is not None) != (job.train.contrastive != "none"):
            raise ValueError(f"train.contrastive is {job.train.contrastive!r}, but the projection head is "
                             f"{'missing' if contrastive_head is None else 'given'}")
        if contrastive_head is not None and job.augment.mixup_alpha > 0:
            raise ValueError("a contrastive loss needs mixup_alpha 0 (mixup blends the labels it pairs by)")
        if (va_head is not None) != (job.train.aux_va != "none"):
            raise ValueError(f"train.aux_va is {job.train.aux_va!r}, but the valence and activation head is "
                             f"{'missing' if va_head is None else 'given'}")

    def loss(self, waves, lengths, rows, targets, gender_targets, mask_generator, augment_rng, labels=None,
             va_targets=None):
        import torch

        device = self.device
        x = self.features(torch.from_numpy(waves).to(device, non_blocking=True))
        frames = self.features.frames(torch.from_numpy(lengths).to(device)).clamp_max(x.shape[-1])
        x = spec_augment(self.normalizer(x, frames, rows), frames, self.augment, mask_generator)
        t = torch.from_numpy(targets).to(device)
        g = torch.from_numpy(gender_targets).to(device) if self.aux_head is not None else None
        y = None
        if self.contrastive_head is not None:
            if labels is None or len(labels) != targets.shape[0]:
                raise ValueError("the contrastive loss needs one class label per utterance of the batch")
            y = torch.as_tensor(np.asarray(labels), device=device)
        v = None
        if self.va_head is not None:
            if va_targets is None or va_targets.shape != (targets.shape[0], len(VA_SCALES)):
                raise ValueError("the valence and activation head needs the batch's two standardized ratings per "
                                 "utterance")
            v = torch.from_numpy(np.ascontiguousarray(va_targets, dtype=np.float32)).to(device)
        if self.augment.mixup_alpha > 0:
            weights, partner = mixup.draw(self.augment.mixup_alpha, x.shape[0], augment_rng)
            to_mix = [t] + ([g] if g is not None else []) + ([v] if v is not None else [])
            x, frames, mixed = mixup.mix(x, frames, to_mix, weights, partner)
            t = mixed[0]
            g = mixed[1] if g is not None else None
            v = mixed[-1] if v is not None else None
        heads = self.aux_head is not None or self.contrastive_head is not None or self.va_head is not None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.train.precision == "bf16"):
            if not heads:
                logits = self.model(x, frames)
            else:
                embedding = self.model.embed(x, frames)
                logits = self.model.classify(embedding)
                aux_logits = self.aux_head(embedding) if self.aux_head is not None else None
                projections = self.contrastive_head(embedding) if self.contrastive_head is not None else None
                va_out = self.va_head(embedding) if self.va_head is not None else None
        emotion = soft_cross_entropy(logits, t, self.weights)
        # A spiking model's activity penalty (affect.models.lif, added 2026-09-29); other models
        # have none, and their loss is exactly what it was.
        penalty = self.model.activity_penalty() if hasattr(self.model, "activity_penalty") else None
        total = emotion if penalty is None else emotion + penalty
        gender = contrastive = None
        if self.aux_head is not None:
            gender = soft_cross_entropy(aux_logits, g, None)
            total = total + self.train.aux_gender_weight * gender
        if self.contrastive_head is not None:
            contrastive = supervised_contrastive_loss(projections, y, self.train.contrastive_temperature)
            total = total + self.train.contrastive_weight * contrastive
        self.last_va_loss = None
        if self.va_head is not None:
            self.last_va_loss = va_loss(va_out, v, self.train.aux_va_loss)
            total = total + self.train.aux_va_weight * self.last_va_loss
        return total, emotion, gender, contrastive


@dataclasses.dataclass
class Predictions:
    probs: np.ndarray               # [rows, classes] class probabilities
    aux_probs: np.ndarray | None    # [rows, 2] gender probabilities (female, male) when a gender head was given
    seconds: float
    va: np.ndarray | None = None    # [rows, 2] the valence and activation head's outputs (standardized), when given


def predict(model, features, normalizer, audio, rows, batch_size, device, precision, segment=None,
            aux_head=None, va_head=None) -> Predictions:
    """Class probabilities for the given rows: from each whole utterance (segment None), or the
    mean of the probabilities of its segments (segment = (length, hop) in samples). With a valence
    and activation head (added 2026-10-24), also its outputs, averaged over the segments the same way;
    the class probabilities are the same with or without it."""
    import torch

    model.eval()
    if aux_head is not None:
        aux_head.eval()
    if va_head is not None:
        va_head.eval()
    sums = aux_sums = va_sums = None
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
                if aux_head is None and va_head is None:
                    logits = model(x, frames)
                else:
                    embedding = model.embed(x, frames)
                    logits = model.classify(embedding)
                    aux_logits = aux_head(embedding) if aux_head is not None else None
                    va_out = va_head(embedding) if va_head is not None else None
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            if sums is None:
                sums = np.zeros((rows.shape[0], probs.shape[1]), dtype=np.float64)
            if aux_head is not None:
                aux_probs = torch.softmax(aux_logits.float(), dim=-1).cpu().numpy()
                if aux_sums is None:
                    aux_sums = np.zeros((rows.shape[0], aux_probs.shape[1]), dtype=np.float64)
            if va_head is not None:
                va_values = va_out.float().cpu().numpy()
                if va_sums is None:
                    va_sums = np.zeros((rows.shape[0], va_values.shape[1]), dtype=np.float64)
            if segment is None:
                sums[positions] = probs
                if aux_head is not None:
                    aux_sums[positions] = aux_probs
                if va_head is not None:
                    va_sums[positions] = va_values
            else:
                np.add.at(sums, positions, probs)
                if aux_head is not None:
                    np.add.at(aux_sums, positions, aux_probs)
                if va_head is not None:
                    np.add.at(va_sums, positions, va_values)
            np.add.at(counts, positions, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if segment is not None:
        sums /= counts[:, None]
        if aux_sums is not None:
            aux_sums /= counts[:, None]
        if va_sums is not None:
            va_sums /= counts[:, None]
    elif not (counts == 1).all():
        raise RuntimeError("whole-utterance evaluation must see every row exactly once")
    return Predictions(sums, aux_sums, seconds, va_sums)


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


def va_report(z_true: np.ndarray, z_pred: np.ndarray, scale: tuple) -> dict:
    """How well the valence and activation head predicts the ratings (train.aux_va, added 2026-10-24): per dimension
    the concordance correlation coefficient (the same on the standardized and on the rating scale), Pearson's r, and
    the RMSE in rating points; and the mean of the two coefficients. z_true, z_pred: [rows, 2], standardized with
    scale = (the training utterances' means, their SDs)."""
    mean, sd = scale
    ccc = concordance_numpy(z_pred, z_true)
    out = {"n": int(z_true.shape[0]), "scales": list(VA_SCALES), "train_mean": [float(m) for m in mean],
           "train_sd": [float(s) for s in sd]}
    for j, name in enumerate(VA_SCALES):
        out[f"ccc_{name}"] = float(ccc[j])
        a, b = z_true[:, j].astype(np.float64), z_pred[:, j].astype(np.float64)
        out[f"pearson_{name}"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(((b - a) * sd[j]) ** 2)))
    out["mean_ccc"] = float(np.mean(ccc))
    return out


def predictions_va_csv(prepared, rows, z_true, z_pred, scale) -> str:
    """One row per utterance: its class label, and each dimension's rating and the head's prediction, both on the
    rating scale (the standardized values times the training SD plus the training mean)."""
    mean, sd = scale
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utt_id", "speaker", "label"] + [f"true_{s}" for s in VA_SCALES]
                    + [f"pred_{s}" for s in VA_SCALES])
    dims = range(len(VA_SCALES))
    for k, row in enumerate(rows):
        u = prepared.utterances[row]
        writer.writerow([u.utt_id, u.speaker, u.label] + [f"{z_true[k, j] * sd[j] + mean[j]:.4f}" for j in dims]
                        + [f"{z_pred[k, j] * sd[j] + mean[j]:.6f}" for j in dims])
    return buffer.getvalue()


def strict_report(y_true: np.ndarray, y_model: np.ndarray, corpus_classes: tuple, model_classes: tuple) -> dict:
    """A model with more classes than the corpus, scored over all of its classes (added 2026-10-24): a clip predicted
    as a class the corpus does not have (four_merged's neu on EmoFilm) is an error. y_true: corpus class indices;
    y_model: indices of the model's classes. Recall per corpus class, its mean (UA) and the share of right answers
    (WA), and for each corpus class the share of its clips given to each of the model's classes. A corpus class with
    no clip (in a group) gets recall and shares None, and the UA is then None, as in eval.metrics.report_by_group."""
    cm = np.zeros((len(corpus_classes), len(model_classes)), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_model)), 1)
    support = cm.sum(axis=1)
    hits = np.array([cm[i, model_classes.index(c)] for i, c in enumerate(corpus_classes)])
    recall = [float(hits[i] / support[i]) if support[i] else None for i in range(len(corpus_classes))]
    defined = all(r is not None for r in recall)
    return {"model_class_names": list(model_classes), "n": int(cm.sum()),
            "support": dict(zip(corpus_classes, support.tolist())),
            "ua": float(np.mean(recall)) if defined else None,
            "wa": float(hits.sum() / cm.sum()), "recall": dict(zip(corpus_classes, recall)),
            "share_given": {c: (dict(zip(model_classes, (cm[i] / support[i]).tolist())) if support[i] else None)
                            for i, c in enumerate(corpus_classes)},
            "share_given_all": dict(zip(model_classes, (cm.sum(axis=0) / cm.sum()).tolist())),
            "confusion_true_by_predicted": cm.tolist()}


def external_files(external, model, features, normalizer, ts, segment, device, model_classes=None) -> tuple:
    """The chosen epoch's model on a test corpus (train.external_test, added 2026-10-20), predicted like the test
    set (whole utterances or the job's segments): (metrics, predictions CSV text, seconds). The normalizer is the
    job's own; runs.spec.check_job allows an external test only with one that needs no IEMOCAP speaker.
    model_classes (added 2026-10-24): the model's classes when they are more than the corpus's (four_merged on
    EmoFilm). Each clip is then given the most probable of the corpus's classes (the model's other classes left out,
    the rest renormalised), which every metric below scores; metrics["strict"] scores the model's own choice over all
    of its classes (strict_report), and the CSV holds both choices and the probabilities of all of its classes. None,
    or the corpus's own classes: the files of 2026-10-20, unchanged."""
    if model_classes is not None and tuple(model_classes) != tuple(external.class_names):
        return _external_files_more_classes(external, model, features, normalizer, ts, segment, device,
                                            tuple(model_classes))
    rows = np.arange(len(external.utterances), dtype=np.int64)
    pred = predict(model, features, normalizer, external.audio, rows, ts.eval_batch_size, device, ts.precision,
                   segment)
    y_true, y_pred = external.labels, pred.probs.argmax(axis=1)
    metrics = report(y_true, y_pred, external.class_names)
    for field in ("language", "gender"):
        groups = np.array([u.extra[field] for u in external.utterances])
        metrics[f"by_{field}"] = report_by_group(y_true, y_pred, groups, external.class_names)
    metrics["mean_nll"] = float(-np.mean(np.log(np.clip(pred.probs[rows, y_true], 1e-12, None))))
    metrics.update({"dataset": external.dataset, "manifest_sha256": external.manifest_sha256,
                    "seconds": round(pred.seconds, 3)})
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utt_id", "speaker", "film", "language", "gender", "label", "pred"]
                    + [f"prob_{c}" for c in external.class_names])
    for k, u in enumerate(external.utterances):
        writer.writerow([u.utt_id, u.speaker, u.extra["film"], u.extra["language"], u.extra["gender"], u.label,
                         external.class_names[int(pred.probs[k].argmax())]] + [f"{p:.6f}" for p in pred.probs[k]])
    return metrics, buffer.getvalue(), pred.seconds


def _external_files_more_classes(external, model, features, normalizer, ts, segment, device, model_classes) -> tuple:
    """external_files for a model whose classes include the corpus's and more (added 2026-10-24)."""
    missing = [c for c in external.class_names if c not in model_classes]
    if missing:
        raise ValueError(f"{external.dataset}: classes {missing} are not among the model's {list(model_classes)}")
    rows = np.arange(len(external.utterances), dtype=np.int64)
    pred = predict(model, features, normalizer, external.audio, rows, ts.eval_batch_size, device, ts.precision,
                   segment)
    columns = [model_classes.index(c) for c in external.class_names]
    shared = pred.probs[:, columns]
    shared = shared / shared.sum(axis=1, keepdims=True)
    y_true, y_pred, y_model = external.labels, shared.argmax(axis=1), pred.probs.argmax(axis=1)
    metrics = report(y_true, y_pred, external.class_names)
    for field in ("language", "gender"):
        groups = np.array([u.extra[field] for u in external.utterances])
        metrics[f"by_{field}"] = report_by_group(y_true, y_pred, groups, external.class_names)
    metrics["mean_nll"] = float(-np.mean(np.log(np.clip(shared[rows, y_true], 1e-12, None))))
    metrics["scoring"] = ("each clip is given the most probable of the corpus's classes; the model's other classes "
                          f"({', '.join(c for c in model_classes if c not in external.class_names)}) are left out and "
                          "the rest renormalised. strict: the model's own choice over all of its classes, any other "
                          "class an error")
    metrics["strict"] = strict_report(y_true, y_model, external.class_names, model_classes)
    for field in ("language", "gender"):
        groups = np.array([u.extra[field] for u in external.utterances])
        metrics["strict"][f"by_{field}"] = {
            str(g): strict_report(y_true[groups == g], y_model[groups == g], external.class_names, model_classes)
            for g in sorted(set(groups.tolist()))}
    metrics.update({"dataset": external.dataset, "manifest_sha256": external.manifest_sha256,
                    "seconds": round(pred.seconds, 3)})
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utt_id", "speaker", "film", "language", "gender", "label", "pred", "pred_all"]
                    + [f"prob_{c}" for c in model_classes])
    for k, u in enumerate(external.utterances):
        writer.writerow([u.utt_id, u.speaker, u.extra["film"], u.extra["language"], u.extra["gender"], u.label,
                         external.class_names[int(y_pred[k])], model_classes[int(y_model[k])]]
                        + [f"{p:.6f}" for p in pred.probs[k]])
    return metrics, buffer.getvalue(), pred.seconds


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
    # what the features read and its rate per second: the samples at the sample rate for log-mel, a pretrained
    # encoder's frames at its frame rate (added 2026-10-26); log-mel jobs read exactly what they always did
    source, rate = job_source(job, prepared)
    targets = (prepared.soft if ts.labels == "soft" else np.eye(n_classes, dtype=np.float32)[prepared.labels])[train_rows]
    train_labels = prepared.labels[train_rows]
    permutation = None
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
    va_z = va_targets = va_scale = None
    if ts.aux_va != "none":
        # the ratings of every row, standardized with the training rows' means and SDs (added 2026-10-24); the
        # leak-test control shuffles them with the labels, since they carry the emotion too
        ratings = va_ratings(prepared.utterances)
        va_scale = (ratings[train_rows].mean(axis=0), ratings[train_rows].std(axis=0))
        if not (va_scale[1] > 0).all():
            raise ValueError(f"the training ratings have SDs {va_scale[1].tolist()}; a constant cannot be learned")
        va_z = ((ratings - va_scale[0]) / va_scale[1]).astype(np.float32)
        va_targets = va_z[train_rows] if permutation is None else va_z[train_rows][permutation]
    weights = None
    if ts.class_balance == "weighted_loss":
        weights = torch.from_numpy(class_weights(train_labels, n_classes)).to(device)
    stats_rows = np.arange(len(prepared.utterances)) if job.features.normalization == "speaker" else train_rows
    normalizer = make_normalizer(job.features.normalization, features, prepared, stats_rows, ts.eval_batch_size, device,
                                 source)

    model = MODELS.get(job.model_name).target(job.model, input_width(job), n_classes).to(device)
    aux_head = make_aux_head(ts, model, device)
    contrastive_head = make_contrastive_head(ts, model, device)     # after the others: their draws are unchanged
    va_head = make_va_head(ts, model, device)                       # after these, for the same reason (2026-10-24)
    n_parameters = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(trainable_parameters(model, aux_head, contrastive_head, va_head), lr=ts.lr,
                                  weight_decay=ts.weight_decay)
    step = Step(job, model, aux_head, features, normalizer, weights, device, contrastive_head, va_head)
    segment = segment_samples(ts, rate)
    steps_per_epoch = train_rows.shape[0] // ts.batch_size
    if ts.max_batches_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, ts.max_batches_per_epoch)
    total_steps = ts.epochs * steps_per_epoch
    warmup_steps = ts.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step_number: lr_factor(step_number, warmup_steps, total_steps, ts.schedule))
    crop_len = int(round(ts.crop_s * rate))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    extras = []
    if tuple(job.augment.speed_factors) != (1.0,):
        extras.append(f"speeds {list(job.augment.speed_factors)}")
    if job.augment.mixup_alpha > 0:
        extras.append(f"mixup alpha {job.augment.mixup_alpha}")
    if aux_head is not None:
        extras.append(f"{ts.aux_gender} gender head, weight {ts.aux_gender_weight}")
    if contrastive_head is not None:
        extras.append(f"supervised contrastive loss, weight {ts.contrastive_weight}, temperature "
                      f"{ts.contrastive_temperature}, projection to {ts.contrastive_dim}")
    if va_head is not None:
        extras.append(f"valence and activation head, {ts.aux_va_loss} loss, weight {ts.aux_va_weight}")
    if segment is not None:
        extras.append(f"evaluation on {ts.eval_segment_s}s segments every {ts.eval_segment_hop_s}s")
    log(f"run {spec['run_id']}: fold {fold.name} (val {fold.val_group}), seed {seed}, train {train_rows.shape[0]}, "
        f"val {val_rows.shape[0]}, test {test_rows.shape[0]}, {n_parameters} parameters, {steps_per_epoch} steps/epoch"
        + "".join(f", {e}" for e in extras))
    curves, best_ua, best_epoch, best_state, best_aux_state, stopped_early = [], -1.0, 0, None, None, False
    best_train_report = best_va_state = None
    train_start = time.perf_counter()
    for epoch in range(1, ts.epochs + 1):
        model.train()
        if aux_head is not None:
            aux_head.train()
        if contrastive_head is not None:
            contrastive_head.train()
        if va_head is not None:
            va_head.train()
        epoch_start = time.perf_counter()
        order = train_order(train_labels, ts.class_balance, rng)
        loss_sum = torch.zeros((), device=device)
        emotion_sum = torch.zeros((), device=device)
        gender_sum = torch.zeros((), device=device)
        contrastive_sum = torch.zeros((), device=device)
        va_sum = torch.zeros((), device=device)
        for b in range(steps_per_epoch):
            picked = order[b * ts.batch_size:(b + 1) * ts.batch_size]
            rows = train_rows[picked]
            waves, lengths = training_waves(source, rows, crop_len, job.augment.speed_factors, rng, augment_rng)
            # train_labels[picked]: the batch's labels as training sees them (shuffled under the leak-test
            # control, like the targets); only the contrastive loss reads them
            loss, emotion_loss, gender_loss, contrastive_loss = step.loss(
                waves, lengths, rows, targets[picked], gender_targets[picked] if gender_targets is not None else None,
                mask_generator, augment_rng, train_labels[picked],
                va_targets[picked] if va_targets is not None else None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum = loss_sum + loss.detach()
            if gender_loss is not None or contrastive_loss is not None or step.last_va_loss is not None:
                emotion_sum = emotion_sum + emotion_loss.detach()
            if gender_loss is not None:
                gender_sum = gender_sum + gender_loss.detach()
            if contrastive_loss is not None:
                contrastive_sum = contrastive_sum + contrastive_loss.detach()
            if step.last_va_loss is not None:
                va_sum = va_sum + step.last_va_loss.detach()
        train_loss = float(loss_sum.item()) / max(1, steps_per_epoch)
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"training loss is {train_loss} in epoch {epoch}")
        train_seconds = time.perf_counter() - epoch_start
        predicted = predict(model, features, normalizer, source, val_rows, ts.eval_batch_size, device,
                            ts.precision, segment, None, va_head)
        val = evaluate(prepared, val_rows, predicted.probs)
        train_report = train_pass = None
        if ts.eval_train == "every_epoch":
            # the training utterances predicted like the validation ones (eval mode, no augmentation, true
            # labels), with the weights just validated; predict changes no weight, statistic or random state
            train_pass = predict(model, features, normalizer, source, train_rows, ts.eval_batch_size,
                                 device, ts.precision, segment)
            train_report = evaluate(prepared, train_rows, train_pass.probs)
        curve = {"epoch": epoch, "train_loss": train_loss, "val_ua": val["ua"], "val_wa": val["wa"],
                 "val_macro_f1": val["macro_f1"], "val_mean_nll": val["mean_nll"],
                 "lr": optimizer.param_groups[0]["lr"], "train_seconds": round(train_seconds, 3),
                 "eval_seconds": round(predicted.seconds, 3)}
        if aux_head is not None or contrastive_head is not None or va_head is not None:
            curve["train_loss_emotion"] = float(emotion_sum.item()) / max(1, steps_per_epoch)
        if aux_head is not None:
            curve["train_loss_gender"] = float(gender_sum.item()) / max(1, steps_per_epoch)
        if contrastive_head is not None:
            curve["train_loss_contrastive"] = float(contrastive_sum.item()) / max(1, steps_per_epoch)
        if va_head is not None:
            curve["train_loss_va"] = float(va_sum.item()) / max(1, steps_per_epoch)
            epoch_va = va_report(va_z[val_rows], predicted.va, va_scale)
            curve.update({f"val_ccc_{s}": epoch_va[f"ccc_{s}"] for s in VA_SCALES})
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
            if va_head is not None:
                best_va_state = {k: v.detach().clone() for k, v in va_head.state_dict().items()}
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
    if va_head is not None:
        va_head.load_state_dict(best_va_state)
    spiking = hasattr(model, "spike_report")          # the LIF models record spikes on this pass (2026-09-29)
    if spiking:
        model.record(True)
    val_pred = predict(model, features, normalizer, source, val_rows, ts.eval_batch_size, device, ts.precision,
                       segment, aux_head, va_head)
    spikes = None
    if spiking:
        model.record(False)
        spikes = model.spike_report()
        rates = ", ".join(f"{k} {v:.3f}" for k, v in spikes["firing_rate"].items() if v is not None)
        log(f"  spikes on validation: {rates}" + (f"; SILENT {spikes['silent_layers']}" if spikes["silent_layers"] else ""))
    test_pred = predict(model, features, normalizer, source, test_rows, ts.eval_batch_size, device,
                        ts.precision, segment, aux_head, va_head)
    val_final = evaluate(prepared, val_rows, val_pred.probs)
    if abs(val_final["ua"] - best_ua) > 1e-9:
        log(f"  note: best-epoch weights reloaded give val UA {val_final['ua']:.6f}, recorded {best_ua:.6f}")
    peak_mib = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
    external = None
    if ts.external_test != "none":
        # after every IEMOCAP number and the memory peak, so none of them changes (added 2026-10-20). The corpus is
        # read in its own label set (EXTERNAL_LABEL_SETS): for a three-class job the job's own, as before; a job
        # with more classes (four_merged, added 2026-10-24) is scored on the corpus's classes (external_files)
        ext = prepare_external(ts.external_test, EXTERNAL_LABEL_SETS[ts.external_test])
        if not set(ext.class_names) <= set(prepared.class_names) or ext.sample_rate != prepared.sample_rate:
            raise ValueError(f"{ext.dataset}: classes {ext.class_names} at {ext.sample_rate} Hz, the job's "
                             f"{prepared.class_names} at {prepared.sample_rate} Hz")
        more = None if ext.class_names == prepared.class_names else prepared.class_names
        external = (ext, *external_files(ext, model, features, normalizer, ts, segment, device, more))
        # the log is read with the validation files, so it names the external test without its result
        log(f"  {ext.dataset}: {len(ext.utterances)} utterances scored (metrics_{ext.dataset}.json), never trained "
            f"or chosen on")
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
    if contrastive_head is not None:
        efficiency["contrastive_parameters"] = int(sum(p.numel() for p in contrastive_head.parameters()))
    if va_head is not None:
        metrics_val["aux_va"] = va_report(va_z[val_rows], val_pred.va, va_scale)
        metrics_test["aux_va"] = va_report(va_z[test_rows], test_pred.va, va_scale)
        efficiency["va_parameters"] = int(sum(p.numel() for p in va_head.parameters()))
    if tuple(job.augment.speed_factors) != (1.0,):
        efficiency["speed_cache_mib_in_process"] = round(cached_bytes() / 2 ** 20, 1)
    if spikes is not None:
        efficiency["spikes_validation"] = spikes
    files = {
        "metrics_val.json": metrics_val,
        "predictions_val.csv": predictions_csv(prepared, val_rows, val_pred.probs),
        "metrics_test.json": metrics_test,
        "predictions_test.csv": predictions_csv(prepared, test_rows, test_pred.probs),
        "efficiency.json": efficiency,
        "provenance.json": provenance,
    }
    if va_head is not None:
        files["predictions_va_val.csv"] = predictions_va_csv(prepared, val_rows, va_z[val_rows], val_pred.va, va_scale)
        files["predictions_va_test.csv"] = predictions_va_csv(prepared, test_rows, va_z[test_rows], test_pred.va,
                                                              va_scale)
    if external is not None:
        ext, ext_metrics, ext_csv, ext_seconds = external
        provenance["external_test"] = {"dataset": ext.dataset, "manifest_sha256": ext.manifest_sha256,
                                       "n_utterances": len(ext.utterances)}
        efficiency[f"{ext.dataset}_seconds"] = round(ext_seconds, 3)
        files[f"metrics_{ext.dataset}.json"] = ext_metrics
        files[f"predictions_{ext.dataset}.csv"] = ext_csv
    return files
