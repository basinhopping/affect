"""Log-mel spectrogram, computed in torch on the job's device, one batch at a time (plan 4.1).

The mel filterbank is built with numpy exactly as librosa.filters.mel builds its default
(Slaney mel scale, Slaney area normalisation); a test on the node compares both, and compares the
whole log-mel with librosa.feature.melspectrogram. Framing: a Hann window of win_ms, a hop of
hop_ms, centred frames with zero padding, so an utterance of n samples has 1 + n // hop frames.
"""
import dataclasses
from typing import Literal

import numpy as np

from affect.registry import register_feature


@dataclasses.dataclass(frozen=True)
class LogMelSettings:
    win_ms: float                 # analysis window length, milliseconds
    hop_ms: float                 # step between frames, milliseconds
    n_fft: int                    # FFT size, at least the window length in samples
    n_mels: int                   # number of mel bands
    f_min: float                  # lowest band edge, Hz
    f_max: float                  # highest band edge, Hz; at most half the sample rate
    log_offset: float             # added to the mel power before the log, so silence stays finite
    normalization: Literal["utterance", "global", "speaker", "none"]
    # utterance: each mel band scaled to mean 0, SD 1 over that utterance's frames
    # global: the same, with one mean and SD per band from all training-session frames
    # speaker: the same, with one mean and SD per band for each speaker, from all of that speaker's
    #   utterances (audio only, labels unused). For validation and test speakers this uses their
    #   unlabeled audio, which is stated with any result that uses it (plan 5.3).
    # none: raw log-mel (loudness kept; loudness is an arousal cue)


def _slaney_constants() -> tuple:
    """Slaney's mel scale: linear below 1 kHz (200/3 Hz per mel), logarithmic above."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    return f_sp, min_log_hz, min_log_hz / f_sp, np.log(6.4) / 27.0


def hz_to_mel(freq):
    f_sp, min_log_hz, min_log_mel, logstep = _slaney_constants()
    freq = np.asarray(freq, dtype=np.float64)
    log_region = freq >= min_log_hz
    return np.where(log_region, min_log_mel + np.log(np.maximum(freq, 1e-12) / min_log_hz) / logstep, freq / f_sp)


def mel_to_hz(mels):
    f_sp, min_log_hz, min_log_mel, logstep = _slaney_constants()
    mels = np.asarray(mels, dtype=np.float64)
    log_region = mels >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mels - min_log_mel)), f_sp * mels)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """[n_mels, 1 + n_fft // 2] weights, as librosa.filters.mel(htk=False, norm="slaney")."""
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((n_mels, fft_freqs.shape[0]))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    return weights * enorm[:, None]


def window_samples(settings: LogMelSettings, sample_rate: int) -> tuple:
    """(window length, hop) in samples."""
    return int(round(settings.win_ms * sample_rate / 1000)), int(round(settings.hop_ms * sample_rate / 1000))


def check(settings: LogMelSettings, sample_rate: int) -> None:
    """Raise ValueError for settings that cannot work, before any job runs."""
    win, hop = window_samples(settings, sample_rate)
    if hop < 1 or win < 1:
        raise ValueError(f"features: window {win} and hop {hop} samples must be positive")
    if win > settings.n_fft:
        raise ValueError(f"features: window of {win} samples is longer than n_fft {settings.n_fft}")
    if not 0 <= settings.f_min < settings.f_max <= sample_rate / 2:
        raise ValueError(f"features: need 0 <= f_min < f_max <= {sample_rate / 2}, got {settings.f_min}, {settings.f_max}")
    if settings.n_mels < 1 or settings.log_offset <= 0:
        raise ValueError("features: n_mels must be positive and log_offset above zero")
    fb = mel_filterbank(sample_rate, settings.n_fft, settings.n_mels, settings.f_min, settings.f_max)
    empty = int((fb.max(axis=1) == 0).sum())
    if empty:
        raise ValueError(f"features: {empty} of {settings.n_mels} mel bands contain no FFT bin "
                         f"(n_fft {settings.n_fft}, f_max {settings.f_max}); use fewer bands or a larger n_fft")


@register_feature("logmel", settings=LogMelSettings)
class LogMel:
    """Callable on a batch of waveforms [batch, samples] -> log-mel [batch, n_mels, frames]."""

    def __init__(self, settings: LogMelSettings, sample_rate: int, device):
        import torch

        check(settings, sample_rate)
        self.settings = settings
        self.win, self.hop = window_samples(settings, sample_rate)
        self.n_fft = settings.n_fft
        self.n_mels = settings.n_mels
        self.window = torch.hann_window(self.win, periodic=True, dtype=torch.float32, device=device)
        fb = mel_filterbank(sample_rate, settings.n_fft, settings.n_mels, settings.f_min, settings.f_max)
        self.filterbank = torch.tensor(fb, dtype=torch.float32, device=device)

    def frames(self, n_samples):
        """Frames produced for n samples (int or integer tensor)."""
        return 1 + n_samples // self.hop

    def __call__(self, wave):
        import torch

        spec = torch.stft(wave.float(), n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window, center=True, pad_mode="constant", return_complex=True)
        power = spec.real.square() + spec.imag.square()            # [batch, freq, frames]
        mel = torch.matmul(self.filterbank, power)                  # [batch, n_mels, frames]
        return torch.log(mel + self.settings.log_offset)
