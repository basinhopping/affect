"""Training settings: one dataclass, every field required (plan 5.3, optimisation).

aux_gender, aux_gender_weight, eval_segment_s and eval_segment_hop_s were added on 2026-09-26.
Settings written before then lack them, which means no gender head and whole-utterance
evaluation, and they keep their configuration hashes (affect.settings, ADDED_FIELDS).
eval_train was added on 2026-10-03 (Dr. Luitel asked for the training accuracy, to see over- or
underfitting); settings without it mean "none", what the loop did before, with the same hashes.
"""
import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class TrainSettings:
    epochs: int                    # at most this many passes over the training sessions
    batch_size: int                # training batch size; the last incomplete batch is dropped
    lr: float                      # peak learning rate (AdamW)
    weight_decay: float            # AdamW weight decay
    warmup_epochs: int             # linear warmup from 0 to lr
    schedule: Literal["cosine", "constant"]   # after warmup: cosine decay to 0 at `epochs`, or constant
    early_stopping_patience: int   # stop after this many epochs without a better validation UA
    crop_s: float                  # training crop in seconds; shorter utterances are zero-padded and masked
    class_balance: Literal["none", "weighted_loss", "balanced_sampler"]
    labels: Literal["hard", "soft"]  # soft: each evaluator's votes over the four classes
    label_smoothing: float         # mixes the target with a uniform distribution by this weight
    precision: Literal["bf16", "fp32"]
    eval_batch_size: int           # whole utterances per batch when evaluating
    control: Literal["none", "shuffle_train_labels"]  # shuffle_train_labels: the leak test, expect chance
    max_batches_per_epoch: int | None  # None in every real run; small only in the smoke test
    aux_gender: Literal["none", "multitask", "adversarial"]
    # a second output predicts the speaker's gender from the pooled features. multitask: the shared
    # layers learn gender too (Latif 2022, Li 2019). adversarial: a gradient reversal between the
    # shared layers and that output makes the shared layers unlearn gender while the output tries to
    # predict it (Ganin and Lempitsky 2015). none: no second output. Emotion alone picks the epoch.
    aux_gender_weight: float       # weight of the gender loss added to the emotion loss; 0 with none
    eval_segment_s: float | None
    # None: validation and test see each utterance whole. A length in seconds: each utterance is cut
    # into segments of this length (one segment if it is shorter), and the class probabilities of its
    # segments are averaged (Satt 2017; Zhu 2022). eval_batch_size then counts segments.
    eval_segment_hop_s: float | None  # seconds between segment starts; None with whole utterances
    eval_train: Literal["none", "every_epoch"]
    # every_epoch: after each epoch's validation pass, the training utterances are predicted the same
    # way (eval mode, whole utterances or the same segments, no augmentation, true labels), and the
    # epoch's training UA, WA, macro-F1 and mean NLL are recorded beside the validation ones, with the
    # full report of the chosen epoch. Only measured: the weights, the epoch chosen and every validation
    # and test number are the same as with none. none: no training pass (the loop before 2026-10-03).

    ADDED_FIELDS = (("2026-09-26", {"aux_gender": "none", "aux_gender_weight": 0.0, "eval_segment_s": None,
                                    "eval_segment_hop_s": None}),
                    ("2026-10-03", {"eval_train": "none"}))


def check(settings: TrainSettings) -> None:
    if settings.epochs < 1 or settings.batch_size < 2 or settings.eval_batch_size < 1:
        raise ValueError("train: epochs >= 1, batch_size >= 2 and eval_batch_size >= 1 are required")
    if settings.lr <= 0 or settings.weight_decay < 0:
        raise ValueError("train: lr must be positive and weight_decay not negative")
    if not 0 <= settings.warmup_epochs < settings.epochs:
        raise ValueError("train: warmup_epochs must be at least 0 and below epochs")
    if settings.early_stopping_patience < 1:
        raise ValueError("train: early_stopping_patience must be at least 1")
    if settings.crop_s <= 0:
        raise ValueError("train: crop_s must be positive")
    if not 0.0 <= settings.label_smoothing < 1.0:
        raise ValueError("train: label_smoothing must be in [0, 1)")
    if settings.max_batches_per_epoch is not None and settings.max_batches_per_epoch < 1:
        raise ValueError("train: max_batches_per_epoch must be None or positive")
    if settings.aux_gender == "none":
        if settings.aux_gender_weight != 0:
            raise ValueError("train: aux_gender_weight must be 0 when aux_gender is none")
    elif not settings.aux_gender_weight > 0:
        raise ValueError(f"train: aux_gender {settings.aux_gender!r} needs aux_gender_weight above 0")
    if (settings.eval_segment_s is None) != (settings.eval_segment_hop_s is None):
        raise ValueError("train: eval_segment_s and eval_segment_hop_s are both None (whole utterances) or both set")
    if settings.eval_segment_s is not None:
        if not settings.eval_segment_s > 0:
            raise ValueError("train: eval_segment_s must be positive")
        if not 0 < settings.eval_segment_hop_s <= settings.eval_segment_s:
            raise ValueError("train: eval_segment_hop_s must be above 0 and at most eval_segment_s")
