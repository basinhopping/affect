"""What a job needs from its dataset, loaded once per worker process and reused by its jobs.

All audio of the chosen utterances is held in memory as 16-bit samples (the four IEMOCAP classes:
about 0.8 GB), so no job reads audio from disk after the first. Features are computed on the GPU
per batch (affect.features).
"""
import dataclasses
import wave

import numpy as np

from affect import paths
from affect.augment.speed import at_speed, draw_speeds
from affect.data.splits import folds_hash
from affect.hashing import sha256_file, sha256_json
from affect.registry import DATASETS
from affect.settings import to_plain


@dataclasses.dataclass
class Prepared:
    dataset: str
    utterances: list
    position: dict          # utt_id -> row
    audio: list             # int16 arrays, one per utterance, in row order
    labels: np.ndarray      # class index per row
    soft: np.ndarray        # [rows, classes] vote distribution per row
    kinds: np.ndarray
    class_names: tuple
    folds: tuple
    folds_hash: str
    manifest_sha256: str
    sample_rate: int


_PREPARED: dict = {}
_AUDIO: dict = {}           # shared by every Prepared with the same utterances (e.g. different splits)


def read_int16(path, n_frames: int) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()) != (1, 2, "NONE"):
            raise ValueError(f"{path}: not mono 16-bit PCM")
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").copy()
    if samples.shape[0] != n_frames:
        raise ValueError(f"{path}: {samples.shape[0]} frames, the table says {n_frames}")
    return samples


def _audio(root, utterances: list) -> list:
    """The audio of these utterances, read once per process however many data settings use it.

    Splits of the same label set and subset hold the same utterances, so a worker that runs jobs of
    several split schemes keeps one copy of the audio in memory, not one per scheme.
    """
    key = (str(root), tuple((u.utt_id, u.audio_path, u.n_frames) for u in utterances))
    if key not in _AUDIO:
        _AUDIO[key] = [read_int16(root / u.audio_path, u.n_frames) for u in utterances]
    return _AUDIO[key]


def prepare(data_name: str, data_settings) -> Prepared:
    key = (data_name, sha256_json(to_plain(data_settings)))
    if key in _PREPARED:
        return _PREPARED[key]
    ds = DATASETS.get(data_name).target()
    utterances = ds.utterances(data_settings)
    class_names = tuple(ds.class_order(data_settings))
    root = paths.assembled_dir(data_name)
    rates = {u.sample_rate for u in utterances}
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates {rates}")
    audio = _audio(root, utterances)
    labels = np.array([class_names.index(u.label) for u in utterances], dtype=np.int64)
    soft = np.asarray(ds.soft_targets(data_settings, utterances), dtype=np.float32)
    if soft.shape != (len(utterances), len(class_names)) or not np.allclose(soft.sum(axis=1), 1.0):
        raise ValueError("soft targets must be one distribution over the classes per utterance")
    folds = ds.folds(data_settings, utterances)
    _, manifest_sha = sha256_file(root / "MANIFEST.csv")
    prepared = Prepared(
        dataset=data_name, utterances=utterances, position={u.utt_id: i for i, u in enumerate(utterances)},
        audio=audio, labels=labels, soft=soft, kinds=np.array([u.kind for u in utterances]),
        class_names=class_names, folds=folds, folds_hash=folds_hash(utterances, folds),
        manifest_sha256=manifest_sha, sample_rate=rates.pop(),
    )
    _PREPARED[key] = prepared
    return prepared


@dataclasses.dataclass
class External:
    """A test corpus for train.external_test (added 2026-10-20): its utterances, audio and labels, and no folds."""
    dataset: str
    utterances: list
    audio: list
    labels: np.ndarray
    class_names: tuple
    manifest_sha256: str
    sample_rate: int


_EXTERNAL: dict = {}
# The label set each test corpus is read in (train.external_test): EmoFilm has only three_no_neutral, whose classes
# a model of IEMOCAP's three_no_neutral or of a set holding those classes (four_merged, added 2026-10-24) predicts.
EXTERNAL_LABEL_SETS = {"emofilm": "three_no_neutral"}


def prepare_external(data_name: str, label_set: str) -> External:
    """Every utterance of a test corpus (EmoFilm: all three languages) in the given label set, read once per
    process. Its dataset has no fold scheme or vote distributions, so prepare() is not used."""
    from affect.settings import validate

    key = (data_name, label_set)
    if key in _EXTERNAL:
        return _EXTERNAL[key]
    entry = DATASETS.get(data_name)
    settings = validate(entry.settings_class, {"label_set": label_set, "subset": "all"}, "external test")
    ds = entry.target()
    utterances = ds.utterances(settings)
    class_names = tuple(ds.class_order(settings))
    rates = {u.sample_rate for u in utterances}
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates {rates}")
    root = paths.assembled_dir(data_name)
    _, manifest_sha = sha256_file(root / "MANIFEST.csv")
    external = External(dataset=data_name, utterances=utterances, audio=_audio(root, utterances),
                        labels=np.array([class_names.index(u.label) for u in utterances], dtype=np.int64),
                        class_names=class_names, manifest_sha256=manifest_sha, sample_rate=rates.pop())
    _EXTERNAL[key] = external
    return external


def job_source(job, prepared) -> tuple:
    """(what a job's features read, one array per row of prepared; its rate per second): the int16 samples and the
    sample rate for a log-mel job, the pretrained encoder's frames and its frame rate for a frames job (added
    2026-10-26; affect.features.frames). crop_s and the evaluation segments are converted with that rate."""
    if job.features_name == "pretrained_frames":
        from affect.features import frames

        return frames.source(prepared.dataset, job.features, prepared.utterances), frames.frame_rate(job.features)
    return prepared.audio, prepared.sample_rate


def speaker_index(utterances: list) -> tuple:
    """(speaker names in sorted order, the speaker's position in that order for every row)."""
    names = tuple(sorted({u.speaker for u in utterances}))
    lookup = {name: i for i, name in enumerate(names)}
    return names, np.array([lookup[u.speaker] for u in utterances], dtype=np.int64)


def rows_per_speaker(speaker_of: np.ndarray, k: int) -> np.ndarray:
    """The first k rows of every speaker, so that a small sample still covers all speakers."""
    return np.concatenate([np.flatnonzero(speaker_of == s)[:k] for s in range(int(speaker_of.max()) + 1)])


def train_order(labels: np.ndarray, class_balance: str, rng: np.random.Generator) -> np.ndarray:
    """Positions within the training set for one epoch."""
    n = labels.shape[0]
    if class_balance == "balanced_sampler":
        counts = np.bincount(labels)
        weights = 1.0 / counts[labels]
        return rng.choice(n, size=n, replace=True, p=weights / weights.sum())
    return rng.permutation(n)


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """n / (classes x count) per class, from the training labels."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"a class has no training examples: counts {counts.tolist()}")
    return (labels.shape[0] / (n_classes * counts)).astype(np.float32)


def _scale(arrays: list, waves: np.ndarray) -> np.ndarray:
    """int16 samples to [-1, 1); anything else (a pretrained encoder's frames, added 2026-10-26) as it is."""
    if np.issubdtype(arrays[0].dtype, np.integer):
        waves /= 32768.0
    return waves


def crop_arrays(arrays: list, crop_len: int, rng: np.random.Generator) -> tuple:
    """A random crop of crop_len samples from each int16 array, zero-padded when shorter.
    Returns (waves float32 [batch, crop_len] in [-1, 1), real sample counts [batch]).
    Draws one start per array longer than crop_len, in order, and nothing else. Since 2026-10-26 an array may
    also be a pretrained encoder's frames [frames, width] (float16): crop_len then counts frames, the crops are
    float32 [batch, crop_len, width] and keep their values."""
    waves = np.zeros((len(arrays), crop_len) + tuple(arrays[0].shape[1:]), dtype=np.float32)
    valid = np.empty(len(arrays), dtype=np.int64)
    for j, a in enumerate(arrays):
        if a.shape[0] > crop_len:
            start = int(rng.integers(0, a.shape[0] - crop_len + 1))
            waves[j] = a[start:start + crop_len]
            valid[j] = crop_len
        else:
            waves[j, :a.shape[0]] = a
            valid[j] = a.shape[0]
    return _scale(arrays, waves), valid


def crop_batch(audio: list, rows: np.ndarray, crop_len: int, rng: np.random.Generator) -> tuple:
    """A random crop of crop_len samples from each row, zero-padded when shorter (crop_arrays)."""
    return crop_arrays([audio[row] for row in rows], crop_len, rng)


def training_waves(audio: list, rows: np.ndarray, crop_len: int, speed_factors: tuple,
                   rng: np.random.Generator, augment_rng: np.random.Generator) -> tuple:
    """The crops of one training batch. Without speed perturbation (speed_factors [1.0]) this is
    crop_batch exactly, drawing nothing from augment_rng. Otherwise every row first gets a speed
    from augment_rng (affect.augment.speed) and is cropped from its resampled waveform."""
    if tuple(speed_factors) == (1.0,):
        return crop_batch(audio, rows, crop_len, rng)
    speeds = draw_speeds(speed_factors, rows.shape[0], augment_rng)
    return crop_arrays([at_speed(audio, row, speed) for row, speed in zip(rows, speeds)], crop_len, rng)


def eval_batches(audio: list, rows: np.ndarray, batch_size: int):
    """Whole utterances, longest first, padded to the longest in each batch.
    Yields (positions within `rows`, waves float32 [batch, longest], real sample counts)."""
    order = sorted(range(rows.shape[0]), key=lambda k: -audio[rows[k]].shape[0])
    for start in range(0, len(order), batch_size):
        positions = np.array(order[start:start + batch_size], dtype=np.int64)
        lengths = np.array([audio[rows[k]].shape[0] for k in positions], dtype=np.int64)
        waves = np.zeros((positions.shape[0], int(lengths.max())) + tuple(audio[rows[0]].shape[1:]), dtype=np.float32)
        for j, k in enumerate(positions):
            waves[j, :lengths[j]] = audio[rows[k]]
        yield positions, _scale(audio, waves), lengths


def segment_starts(n_samples: int, segment: int, hop: int) -> list:
    """Starts of the segments of one utterance: every hop samples while a whole segment fits,
    plus one segment ending at the last sample if the regular ones stop short of it. An utterance
    of at most `segment` samples is a single segment starting at 0."""
    if segment < 1 or hop < 1:
        raise ValueError("segment and hop must be at least one sample")
    if n_samples <= segment:
        return [0]
    starts = list(range(0, n_samples - segment + 1, hop))
    if starts[-1] + segment < n_samples:
        starts.append(n_samples - segment)
    return starts


def segment_batches(audio: list, rows: np.ndarray, batch_size: int, segment: int, hop: int):
    """Every segment of every utterance (segment_starts), longest first, batch_size segments per
    batch, padded to the longest in the batch. Yields (the position within `rows` of each
    segment's utterance, waves float32 [batch, longest], real sample counts)."""
    items = []
    for k in range(rows.shape[0]):
        n = audio[rows[k]].shape[0]
        for start in segment_starts(n, segment, hop):
            items.append((k, start, min(segment, n)))
    items.sort(key=lambda item: -item[2])
    for first in range(0, len(items), batch_size):
        chunk = items[first:first + batch_size]
        positions = np.array([k for k, _, _ in chunk], dtype=np.int64)
        lengths = np.array([length for _, _, length in chunk], dtype=np.int64)
        waves = np.zeros((len(chunk), int(lengths.max())) + tuple(audio[rows[0]].shape[1:]), dtype=np.float32)
        for j, (k, start, length) in enumerate(chunk):
            waves[j, :length] = audio[rows[k]][start:start + length]
        yield positions, _scale(audio, waves), lengths


GENDERS = ("F", "M")
VA_SCALES = ("v", "a")      # valence and activation (arousal), the order of the valence and activation head's outputs


def va_ratings(utterances: list) -> np.ndarray:
    """[rows, 2] float64: each utterance's valence and activation rating, extra "v" and "a" (IEMOCAP's summary files:
    the means over the evaluators on its 1 to 5 scales; train.aux_va, added 2026-10-24). Raises if a rating is missing,
    is not a finite number, or lies outside 0 to 6 (the check of affect.classical.va)."""
    out = np.empty((len(utterances), len(VA_SCALES)), dtype=np.float64)
    for i, u in enumerate(utterances):
        for j, scale in enumerate(VA_SCALES):
            value = u.extra.get(scale)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) \
                    or not 0 <= value <= 6:
                raise ValueError(f"{u.utt_id}: rating {scale!r} is {value!r}, not a number from 0 to 6")
            out[i, j] = float(value)
    return out


def gender_index(utterances: list) -> np.ndarray:
    """0 for a female and 1 for a male speaker, per row, from extra["speaker_gender"] (IEMOCAP).
    Raises if any utterance has no such field."""
    out = np.empty(len(utterances), dtype=np.int64)
    for i, u in enumerate(utterances):
        gender = u.extra.get("speaker_gender")
        if gender not in GENDERS:
            raise ValueError(f"{u.utt_id}: speaker gender {gender!r} is not one of {GENDERS}")
        out[i] = GENDERS.index(gender)
    return out
