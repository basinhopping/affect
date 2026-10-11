"""Reading wav files with the standard library.

Every IEMOCAP sentence wav is mono, 16-bit, 16 kHz, uncompressed PCM (verified in Phase 0 for
all readable files, and checked again for every file when the dataset is assembled). The
standard `wave` module reads that exactly, so no audio library is needed. Current torchaudio
no longer loads audio (plan 8.3), and this avoids depending on one.
"""
import dataclasses
import wave
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_width: int      # bytes per sample
    sample_rate: int
    n_frames: int
    compression: str       # "NONE" for PCM

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sample_rate


def read_wav_info(path: Path) -> WavInfo:
    with wave.open(str(path), "rb") as handle:
        return WavInfo(
            channels=handle.getnchannels(),
            sample_width=handle.getsampwidth(),
            sample_rate=handle.getframerate(),
            n_frames=handle.getnframes(),
            compression=handle.getcomptype(),
        )


def read_pcm16_mono(path: Path, expected_rate: int):
    """Samples of a mono 16-bit PCM wav as float32 in [-1, 1). Refuses any other format."""
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        info = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype())
        if info != (1, 2, expected_rate, "NONE"):
            raise ValueError(f"{path}: expected mono 16-bit {expected_rate} Hz PCM, got {info}")
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)
    samples = np.frombuffer(raw, dtype="<i2")
    if samples.shape[0] != n_frames:
        raise ValueError(f"{path}: header says {n_frames} frames, file holds {samples.shape[0]}")
    return samples.astype(np.float32) / 32768.0
