"""A job's settings: five sections, each validated against its declaration (plan 6.3).

    data      {"name": <registered dataset>, ...its settings}
    features  {"name": <registered feature>, ...its settings}
    augment   the augmentation settings
    model     {"name": <registered model>, ...its settings}
    train     the training settings

A missing or unknown section raises, and so does anything the sections' own validation refuses.
`check_job` then applies the checks that need several sections at once (a window longer than
n_fft, pooling that leaves no mel bands, a contrastive loss with mixup), so a bad value fails when
the sweep is enqueued, not at 3 a.m. The validated settings are hashed; the hash, the fold and the
seed name the run.

`plain` gives every setting, including fields added after runs existed (settings.ADDED_FIELDS);
`canonical`, which is what gets hashed, leaves an added field out while it holds its value from
before it existed, so a configuration that does what an older one did keeps the older hash.
"""
import dataclasses

from affect.augment.settings import AugmentSettings
from affect.augment.settings import check as check_augment
from affect.hashing import sha256_json
from affect.registry import DATASETS, FEATURES, MODELS
from affect.settings import SettingsError, canonical_plain, to_plain, validate
from affect.train.settings import TrainSettings
from affect.train.settings import check as check_train

SECTIONS = ("data", "features", "augment", "model", "train")


@dataclasses.dataclass(frozen=True)
class Job:
    data_name: str
    data: object
    features_name: str
    features: object
    augment: AugmentSettings
    model_name: str
    model: object
    train: TrainSettings


def validate_job(raw) -> Job:
    if type(raw) is not dict:
        raise SettingsError(f"job settings must be a mapping, got {type(raw).__name__}")
    missing = [s for s in SECTIONS if s not in raw]
    unknown = sorted(k for k in raw if k not in SECTIONS)
    if missing or unknown:
        raise SettingsError(f"job settings: missing sections {missing}, unknown sections {unknown}")
    data_entry, data = DATASETS.validate_section(raw["data"], "data")
    feature_entry, features = FEATURES.validate_section(raw["features"], "features")
    augment = validate(AugmentSettings, raw["augment"], "augment")
    model_entry, model = MODELS.validate_section(raw["model"], "model")
    train = validate(TrainSettings, raw["train"], "train")
    return Job(data_entry.name, data, feature_entry.name, features, augment, model_entry.name, model, train)


def plain(job: Job) -> dict:
    return {
        "data": {"name": job.data_name, **to_plain(job.data)},
        "features": {"name": job.features_name, **to_plain(job.features)},
        "augment": to_plain(job.augment),
        "model": {"name": job.model_name, **to_plain(job.model)},
        "train": to_plain(job.train),
    }


def canonical(job: Job) -> dict:
    """plain(job) without the added fields that hold their earlier values; this is hashed."""
    return {
        "data": {"name": job.data_name, **canonical_plain(job.data)},
        "features": {"name": job.features_name, **canonical_plain(job.features)},
        "augment": canonical_plain(job.augment),
        "model": {"name": job.model_name, **canonical_plain(job.model)},
        "train": canonical_plain(job.train),
    }


def config_hash(job: Job) -> str:
    return sha256_json(canonical(job))


def run_id(job: Job, fold: str, seed: int) -> str:
    return f"{config_hash(job)[:12]}_{fold}_seed{seed}"


def control_of(settings: dict) -> str:
    """The leak-test control recorded in a run's settings: train.control for the neural models,
    protocol.control for the classical baselines (affect.classical)."""
    return settings["train"]["control"] if "train" in settings else settings["protocol"]["control"]


# Models that read a log-mel image (a convolutional front end), and models that read a pretrained encoder's frames
# (affect.models.frames, added 2026-10-26); each needs its own kind of features.
LOGMEL_MODELS = ("cnn", "cnn_transformer", "cnn_lif", "snn", "cnn_rnn")
FRAME_MODELS = ("frames_pool", "frames_transformer", "frames_lif", "frames_rnn")


def input_width(job: Job) -> int:
    """The width of what the model reads at each step: the mel bands of a log-mel job, the encoder's hidden size of a
    frames job (added 2026-10-26)."""
    if job.features_name == "pretrained_frames":
        from affect.features import frames

        return frames.width(job.features)
    return job.features.n_mels


def check_job(job: Job, sample_rate: int) -> None:
    """Checks across sections. Needs no torch, so it runs at enqueue time."""
    from affect.features import frames, logmel
    from affect.models import cnn, cnn_lif, cnn_rnn, cnn_transformer, snn
    from affect.models import frames as frame_models

    if job.features_name == "logmel":
        logmel.check(job.features, sample_rate)
        if job.model_name not in LOGMEL_MODELS:
            raise ValueError(f"model {job.model_name!r} reads a pretrained encoder's frames, not log-mel features")
    elif job.features_name == "pretrained_frames":
        # added 2026-10-26: the frames are the encoder's, so nothing that changes the audio, and no corpus without
        # extracted frames
        frames.check(job.features)
        if job.model_name not in FRAME_MODELS:
            raise ValueError(f"model {job.model_name!r} reads a log-mel image, not a pretrained encoder's frames")
        if tuple(job.augment.speed_factors) != (1.0,):
            raise ValueError("features pretrained_frames: speed perturbation changes the audio, not the frames")
        if job.train.external_test != "none":
            raise ValueError("features pretrained_frames: the external test needs the corpus's frames, not extracted")
    else:
        raise ValueError(f"no check for features {job.features_name!r}")
    n_mels = input_width(job)
    model_checks = {"cnn": cnn.check, "cnn_transformer": cnn_transformer.check, "cnn_lif": cnn_lif.check,
                    "snn": snn.check, "cnn_rnn": cnn_rnn.check, **{m: frame_models.check for m in FRAME_MODELS}}
    if job.model_name not in model_checks:
        raise ValueError(f"no check for model {job.model_name!r}")
    model_checks[job.model_name](job.model, n_mels)
    check_augment(job.augment, n_mels)
    check_train(job.train)
    if job.train.contrastive != "none" and job.augment.mixup_alpha > 0:
        # the contrastive loss pairs utterances by label (added 2026-10-15); mixup blends two utterances
        # and their labels, so an utterance would no longer have one label to be paired by
        raise ValueError("train: a contrastive loss needs mixup_alpha 0 (mixup blends the labels it pairs by)")
    if job.train.external_test != "none":
        # added 2026-10-20: EmoFilm holds IEMOCAP's three classes only, and its speakers are not IEMOCAP's, so a
        # normalization by IEMOCAP speaker statistics has nothing to apply to its clips. Since 2026-10-24 a label set
        # with more classes than those three (four_merged) is allowed too: its model is scored on the three.
        from affect.data.emofilm import CLASS_ORDER as EXTERNAL_CLASSES
        from affect.data.iemocap import CLASS_ORDER
        from affect.train.data import EXTERNAL_LABEL_SETS

        needed = EXTERNAL_CLASSES[EXTERNAL_LABEL_SETS[job.train.external_test]]
        label_set = getattr(job.data, "label_set", None)
        if job.data_name != "iemocap" or not set(needed) <= set(CLASS_ORDER.get(label_set, ())):
            raise ValueError(f"train: external_test {job.train.external_test} needs an IEMOCAP label set that holds "
                             f"its classes {list(needed)} (three_no_neutral or four_merged)")
        if job.features.normalization == "speaker":
            raise ValueError("train: external_test emofilm needs a normalization other than speaker")
    if job.train.aux_va != "none" and job.data_name != "iemocap":
        # added 2026-10-24: the valence and activation ratings are IEMOCAP's (extra "v" and "a")
        raise ValueError("train: aux_va needs IEMOCAP, whose utterances carry valence and activation ratings")
