"""Whisper's own log-mel spectrogram as the input of the networks trained from scratch (feature "whisper_logmel", added
2026-11-10, week 33; Dr. Luitel, 2026-11-09: control the spectrogram, Whisper's log-mel for the networks, with our own
implementation and an equality test).

Whisper large-v3's feature extractor (transformers' WhisperFeatureExtractor, which made the encoder inputs of the
heads' frames: affect.classical.pretrained) turns 16 kHz audio into mel bands every 10 ms:

    a periodic Hann window of 400 samples (25 ms), a hop of 160 (10 ms), a 400-point FFT, the power spectrum;
    centred frames: 200 samples reflected before the start (torch.stft's reflect padding); Whisper pads every input with
        zeros to 30 s, so the frames at the end of an utterance see zeros after it, and so does this implementation
        (200 zeros after the row);
    n_mels filters from 0 to 8 kHz on Slaney's mel scale with Slaney's area normalisation (librosa.filters.mel's
        default, as affect.features.logmel.mel_filterbank builds it; 128 for large-v3);
    log10 of the mel power floored at 1e-10; every value below the row's largest minus 8 raised to it (an 80 dB range);
        then (x + 4) / 4.

A row of n samples has 1 + n // 160 frames, the frames whose centres fall within it (frames(n), as LogMel counts them
with its 10 ms hop). The extractor's 3,000 frames of an utterance under 30 s start with exactly these frames, from the
same windowed samples, so the values agree to float32 rounding (tests/test_whisper_logmel.py: against the numpy
transcription of the extractor here in every environment, and against transformers' extractor itself on the node).
The largest value is taken as the extractor takes it, over its 30 s: over the row's frames and the two frames after
them, whose windows may still hold the last 200 samples (every later frame holds zeros only, log10(1e-10) = -10, below
any real frame). A batch pads its shorter rows with zeros, exactly what Whisper adds after them, so a row's value does
not depend on the batch. Whisper cuts an utterance over 30 s into pieces; this feature reads it whole, so for those few
utterances the 80 dB floor comes from the whole utterance.

The networks then normalize as they do with their own log-mel (per band over the utterance's real frames, "utterance"),
so the affine (x + 4) / 4 and the base of the log do not change what they see; the window, the bands and the 80 dB
floor do. Everything else about a network's job (its crops, SpecAugment, training) stays its own: runs.spec.check_job
treats this feature as it treats logmel.
"""
import dataclasses
from typing import Literal

import numpy as np

from affect.features.logmel import mel_filterbank
from affect.registry import register_feature

# Whisper's definition, not settings: the sample rate; the window and FFT (25 ms) and the hop (10 ms) in samples; the
# top of the filters (half the sample rate); the mel power's floor before the log; the range kept below the largest
# value, in log10 units (80 dB)
WHISPER = {"sample_rate": 16000, "n_fft": 400, "hop": 160, "f_max": 8000.0, "floor": 1e-10, "range": 8.0}
SAMPLE_RATE, N_FFT, HOP = WHISPER["sample_rate"], WHISPER["n_fft"], WHISPER["hop"]
F_MAX, FLOOR, RANGE = WHISPER["f_max"], WHISPER["floor"], WHISPER["range"]
PAD = N_FFT // 2     # the centring: 200 samples before (reflected) and after (zeros)
TAIL = PAD + 2 * HOP  # zeros after a row: the centring and the two frames after it that may hold its last samples


@dataclasses.dataclass(frozen=True)
class WhisperLogMelSettings:
    n_mels: Literal[80, 128]      # Whisper's mel bands: 128 for large-v3 (80 for the earlier sizes)
    normalization: Literal["utterance", "global", "speaker", "none"]
    # as logmel's (affect.train.loop.Normalizer): utterance, each band standardized over the utterance's own frames


def check(settings: WhisperLogMelSettings, sample_rate: int) -> None:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"features: Whisper's log-mel is defined at {SAMPLE_RATE} Hz, not {sample_rate}")


def filterbank(n_mels: int) -> np.ndarray:
    """[n_mels, 201] float64: Whisper's mel filters (librosa.filters.mel(sr=16000, n_fft=400, n_mels), Slaney)."""
    return mel_filterbank(SAMPLE_RATE, N_FFT, n_mels, 0.0, F_MAX)


def frame_count(n_samples):
    """The frames of a row of n samples: those whose centres fall within it (int or integer tensor or array)."""
    return 1 + n_samples // HOP


def reference(wave: np.ndarray, n_mels: int) -> np.ndarray:
    """The same spectrogram in float64 numpy, one row: wave float [n] -> [n_mels, 1 + n // 160]. The reference that
    the torch feature and the extractor are compared with."""
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1 or wave.shape[0] <= PAD:
        raise ValueError(f"a row needs more than {PAD} samples, got shape {wave.shape}")
    padded = np.concatenate([wave[1:PAD + 1][::-1], wave, np.zeros(TAIL)])
    n_frames = frame_count(wave.shape[0])
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames + 2)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)          # periodic Hann
    spectrum = np.fft.rfft(padded[index] * window[None, :], n=N_FFT, axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2                              # [frames + 2, 201]
    mel = filterbank(n_mels) @ power.T                                           # [n_mels, frames + 2]
    log = np.log10(np.maximum(mel, FLOOR))
    log = np.maximum(log, log.max() - RANGE)
    return ((log + 4.0) / 4.0)[:, :n_frames]


@register_feature("whisper_logmel", settings=WhisperLogMelSettings)
class WhisperLogMel:
    """Callable on a batch of waveforms [batch, samples] -> [batch, n_mels, 1 + samples // 160], as LogMel is, with
    frames(n) and n_mels."""

    def __init__(self, settings: WhisperLogMelSettings, sample_rate: int, device):
        import torch

        check(settings, sample_rate)
        self.settings = settings
        self.n_mels = settings.n_mels
        self.hop = HOP
        self.window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float32, device=device)
        self.filterbank = torch.tensor(filterbank(settings.n_mels), dtype=torch.float32, device=device)

    def frames(self, n_samples):
        return frame_count(n_samples)

    def __call__(self, wave):
        import torch

        wave = wave.float()
        if wave.ndim != 2 or wave.shape[1] <= PAD:
            raise ValueError(f"a batch [rows, samples] of more than {PAD} samples expected, got {tuple(wave.shape)}")
        n_frames = int(frame_count(wave.shape[1]))
        padded = torch.cat([wave[:, 1:PAD + 1].flip(-1), wave, wave.new_zeros(wave.shape[0], TAIL)], dim=1)
        spec = torch.stft(padded, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=self.window, center=False,
                          return_complex=True)                                   # [rows, 201, frames + 2]
        power = spec.real.square() + spec.imag.square()
        mel = torch.matmul(self.filterbank, power)                              # [rows, n_mels, frames + 2]
        log = torch.clamp(mel, min=FLOOR).log10()
        top = log.amax(dim=(1, 2), keepdim=True)
        log = torch.maximum(log, top - RANGE)
        return ((log + 4.0) / 4.0)[:, :, :n_frames]
