"""Speed perturbation of training waveforms (added 2026-09-26).

A speed of 1.1 plays the utterance 10% faster, which also raises its pitch, like sox "speed" as
used by Aldeneh 2017 and Latif 2019 on IEMOCAP. The waveform is resampled with a polyphase filter
(scipy.signal.resample_poly, Kaiser window, beta 5) from 16 kHz to 16 kHz / speed and then read
as 16 kHz again, so an utterance of n samples becomes ceil(n / speed) samples long.

Each worker process keeps every resampled utterance it has made (a few hundred MB per speed for
the four IEMOCAP classes), so from the second epoch on no utterance is resampled again. The
resampled audio depends only on the utterance and the speed, so keeping it changes no result.
scipy is imported only when a waveform is resampled, so the settings checks work without it.
"""
from fractions import Fraction

import numpy as np

_CACHE: dict = {}       # (id(audio list), row, speed) -> (the audio list, resampled int16 samples)


def speed_ratio(speed: float) -> tuple:
    """(up, down) with up / down = 1 / speed exactly, both at most 100. Raises ValueError when the
    speed is not such a ratio (0.9 is 10 / 9 and 1.1 is 10 / 11; 1.0 is 1 / 1)."""
    if not speed > 0:
        raise ValueError(f"speed {speed} must be positive")
    exact = Fraction(speed).limit_denominator(100)
    if abs(float(exact) - speed) > 1e-9:
        raise ValueError(f"speed {speed} is not a ratio of whole numbers up to 100 (such as 0.9 or 1.05)")
    return exact.denominator, exact.numerator


def change_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """int16 samples played at the given speed, as int16 samples (rounded, clipped to int16)."""
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError("change_speed takes one int16 waveform")
    if speed == 1.0:
        return samples
    from scipy.signal import resample_poly

    up, down = speed_ratio(speed)
    out = resample_poly(samples.astype(np.float64), up, down, window=("kaiser", 5.0))
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def at_speed(audio: list, row: int, speed: float) -> np.ndarray:
    """audio[row] at the given speed, resampled once per process and kept."""
    if speed == 1.0:
        return audio[row]
    key = (id(audio), int(row), float(speed))
    entry = _CACHE.get(key)
    if entry is None or entry[0] is not audio:
        entry = (audio, change_speed(audio[row], speed))
        _CACHE[key] = entry
    return entry[1]


def draw_speeds(speed_factors: tuple, n: int, rng: np.random.Generator) -> np.ndarray:
    """One speed per example, each factor with equal probability. With a single factor nothing is
    drawn, so a job without speed perturbation uses no random numbers here."""
    factors = np.asarray(speed_factors, dtype=np.float64)
    if factors.shape[0] == 1:
        return np.full(n, factors[0])
    return factors[rng.integers(0, factors.shape[0], size=n)]


def cached_bytes() -> int:
    """Bytes held by the resampled copies in this process."""
    return int(sum(entry[1].nbytes for entry in _CACHE.values()))
