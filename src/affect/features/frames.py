"""A pretrained encoder's frames as the input of a trained head (feature "pretrained_frames", added 2026-10-26).

Dr. Luitel, 2026-10-21: "try my own lif/transformer on top of the pretrained, well lets do a sweep and see what is
best". The encoder (WavLM Large, or another of affect.classical.settings.PRETRAINED_MODELS) stays frozen: one of its
hidden layers is extracted once for every utterance, every frame kept (50 frames per second; affect.classical.
pretrained.extract_frames), into artifacts/features/<dataset>/<model>_l<layer>_frames/:

    frames.npy    float16 [frames of every utterance, width], the utterances one after another
    offsets.npy   int64 [utterances + 1]: utterance i holds the rows offsets[i] to offsets[i + 1]
    utt_ids.txt   the utterances in that order
    record.json   the encoder, its weights' SHA-256, the layer, the counts; provenance.json

A job with these features reads each utterance's frames where a log-mel job reads its samples
(affect.train.data.job_source): the training crops, the validation and test passes and the normalization statistics
all take frames, and crop_s and the evaluation segments count seconds at the encoder's frame rate. The feature
callable turns a batch of frames [batch, frames, width] into [batch, width, frames], the layout of a log-mel batch,
so the normalization (per dimension), SpecAugment's time masks and the heads see what they always see; its width
plays the part of the mel bands. The heads are affect.models.frames. Speed perturbation needs the audio, so
runs.spec.check_job refuses it with these features. The external test (refused until 2026-11-05, the films of the
heads) reads the test corpus's frames of the same encoder and layer, extracted like IEMOCAP's into
artifacts/features/<corpus>/ (extract-frames --dataset emofilm; affect.train.data.external_source), and so does the
warm start on its films. Whisper large-v3's frames (added 2026-11-01) are the
encoder's outputs that cover each utterance's own samples (affect.classical.pretrained.load_whisper_frame_encoder);
AST's (added 2026-11-02) its patch tokens of each time column averaged over the frequency rows, 10 a second
(load_ast_frame_encoder); w2v-BERT 2.0's (added 2026-11-02) its outputs for the filterbank frames its feature extractor
marks real (load_frame_encoder).
"""
import dataclasses
import json
from typing import Literal

import numpy as np

from affect import paths
from affect.classical.settings import PRETRAINED_MODELS, SPECTROGRAM_MODELS
from affect.registry import register_feature

# the waveform encoders, and (added 2026-11-01) Whisper large-v3's encoder, which reads a log-mel spectrogram: the
# heads on its frames are the spectrogram version of the heads (Harrison, 2026-11-01: only models that read spectrograms
# are compared; the waveform encoders are the control). Since 2026-11-02 (Dr. Luitel: every pretrained spectrogram
# model, with my heads tuned on each) every encoder: AST, whose outputs are time-frequency patches, gives one frame per
# time column, its patch tokens averaged over the frequency rows (affect.classical.pretrained.load_ast_frame_encoder),
# and w2v-BERT 2.0 its Conformer's outputs. The spectrogram encoders are also probed through their layers' mean vectors
# (affect.classical.pretrained).
ENCODERS = tuple(PRETRAINED_MODELS)
# frames per second of each encoder's hidden layers: 320 samples at 16 kHz a frame (the waveform encoders' convolutional
# stride; Whisper's 10 ms mel hop times its second convolution's stride of 2; w2v-BERT 2.0's 10 ms filterbank frames
# stacked in pairs); AST's time columns are 10 Kaldi frames of 10 ms apart
FRAME_RATES = {model: (10 if model == "ast_audioset" else 50) for model in ENCODERS}
assert set(SPECTROGRAM_MODELS) <= set(ENCODERS)


@dataclasses.dataclass(frozen=True)
class FramesSettings:
    model: Literal[ENCODERS]       # the frozen encoder (affect.classical.settings.PRETRAINED_MODELS)
    layer: int                     # its hidden layer: 0 (the encoder's input projection or embedding) to the last
    normalization: Literal["utterance", "global", "speaker", "none"]
    # as the log-mel's (affect.train.loop.Normalizer), per dimension of the layer: utterance, each dimension standardized
    # over the utterance's own frames; global, with the training frames' statistics; speaker, with every frame of the
    # row's speaker (its audio, labels unused); none, the encoder's values as they are


def check(settings: FramesSettings) -> None:
    layers = PRETRAINED_MODELS[settings.model][0]
    if not 0 <= settings.layer < layers:
        raise ValueError(f"features: {settings.model} has layers 0 to {layers - 1}, not {settings.layer}")


def width(settings: FramesSettings) -> int:
    """The width of a frame: the encoder's hidden size."""
    return PRETRAINED_MODELS[settings.model][1]


def frame_rate(settings: FramesSettings) -> int:
    return FRAME_RATES[settings.model]


def feature_set(settings: FramesSettings) -> str:
    return f"{settings.model}_l{settings.layer}_frames"


def store_dir(dataset: str, settings: FramesSettings):
    return paths.artifacts_dir() / "features" / dataset / feature_set(settings)


_STORES: dict = {}


def load_store(dataset: str, settings: FramesSettings) -> tuple:
    """(frames memory-mapped float16 [total, width], offsets int64 [n + 1], {utt_id: index}), read once per process;
    the folder must be complete (record.json) and its arrays must agree with it."""
    folder = store_dir(dataset, settings)
    key = str(folder)
    if key not in _STORES:
        record_path = folder / "record.json"
        if not record_path.is_file():
            raise FileNotFoundError(f"no frames for {feature_set(settings)} in {folder} (extract-frames first)")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        frames = np.load(folder / "frames.npy", mmap_mode="r")
        offsets = np.load(folder / "offsets.npy")
        ids = (folder / "utt_ids.txt").read_text(encoding="utf-8").split()
        if (frames.dtype != np.float16 or frames.ndim != 2 or frames.shape[1] != width(settings)
                or offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != frames.shape[0]
                or (np.diff(offsets) < 1).any() or record.get("n_utterances") != len(ids)
                or record.get("total_frames") != frames.shape[0]):
            raise ValueError(f"{folder}: frames {frames.dtype} {frames.shape}, offsets {offsets.shape}, "
                             f"{len(ids)} utterances: they do not agree with record.json")
        _STORES[key] = (frames, offsets, {u: i for i, u in enumerate(ids)})
    return _STORES[key]


def source(dataset: str, settings: FramesSettings, utterances: list) -> list:
    """Each utterance's frames, float16 [frames, width] (views of the memory-mapped store), in the given order."""
    frames, offsets, index = load_store(dataset, settings)
    missing = [u.utt_id for u in utterances if u.utt_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} utterances have no frames in {feature_set(settings)}, e.g. {missing[:3]}")
    return [frames[offsets[index[u.utt_id]]:offsets[index[u.utt_id] + 1]] for u in utterances]


@register_feature("pretrained_frames", settings=FramesSettings)
class FrameFeatures:
    """Callable on a batch of frames [batch, frames, width] -> [batch, width, frames] (float32), with frames(lengths)
    the real frames of each row (the lengths themselves) and n_mels its width, as LogMel has."""

    def __init__(self, settings: FramesSettings, sample_rate: int, device):
        check(settings)
        self.settings, self.device = settings, device
        self.n_mels = width(settings)
        self.rate = frame_rate(settings)

    def frames(self, lengths):
        return lengths

    def __call__(self, x):
        if x.ndim != 3 or x.shape[2] != self.n_mels:
            raise ValueError(f"frames of shape {tuple(x.shape)}, expected [batch, frames, {self.n_mels}]")
        return x.float().transpose(1, 2)
