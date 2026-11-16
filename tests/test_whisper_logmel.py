"""Whisper's log-mel for the networks (affect.features.whisper_logmel, added 2026-11-10).

Everywhere: the mel filters equal a transcription of transformers' mel_filter_bank (Slaney scale and norm, 0 to 8 kHz,
201 FFT bins); the numpy reference equals a transcription of WhisperFeatureExtractor's own steps on a 30 s input
(zeros to 30 s, 200 reflected samples before, frames of 400 every 160 with a periodic Hann window, the power spectrum,
the filters, log10 floored at 1e-10, the last frame dropped, everything below the largest value minus 8 raised to it,
then (x + 4) / 4) on the frames that cover the utterance, for lengths that end anywhere within a hop; the frame count;
and the job checks (whisper_logmel goes with the log-mel models at 16 kHz, and a frames head refuses it).
With torch (the node): the torch feature on a zero-padded batch of rows of different lengths agrees with the reference
row by row, so a row's values do not depend on its batch. With transformers and Whisper large-v3's
preprocessor_config.json in ~/cache/huggingface (the node): the reference and the torch feature agree with the
extractor itself on stand-in audio, and on real IEMOCAP utterances when the assembled IEMOCAP is there.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from affect import catalog
from affect.features import whisper_logmel as W
from tests.support import REPO, needs_assembled, needs_module

catalog.load_all()


def hf_filters(n_mels: int) -> np.ndarray:
    """[n_mels, 201]: transformers.audio_utils.mel_filter_bank(201, n_mels, 0, 8000, 16000, norm="slaney",
    mel_scale="slaney"), written out (transposed to the filters-by-bins layout)."""
    def to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mels = 3.0 * f / 200.0
        log_region = f >= 1000.0
        return np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-12) / 1000.0) * (27.0 / np.log(6.4)), mels)

    def to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = 200.0 * m / 3.0
        log_region = m >= 15.0
        return np.where(log_region, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), f)

    fft_freqs = np.linspace(0, 16000 // 2, 201)
    filter_freqs = to_hz(np.linspace(to_mel(0.0), to_mel(8000.0), n_mels + 2))
    diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    bank = np.maximum(np.zeros(1), np.minimum(down, up))
    bank *= np.expand_dims(2.0 / (filter_freqs[2:n_mels + 2] - filter_freqs[:n_mels]), 0)
    return bank.T


def extractor_steps(wave: np.ndarray, n_mels: int) -> np.ndarray:
    """WhisperFeatureExtractor's steps on one waveform, in float64: [n_mels, 3000]."""
    x = np.zeros(480000)
    x[:wave.shape[0]] = wave
    padded = np.pad(x, 200, mode="reflect")
    n_frames = 1 + (padded.shape[0] - 400) // 160
    index = np.arange(400)[None, :] + 160 * np.arange(n_frames)[:, None]
    window = np.hanning(401)[:-1]                       # periodic Hann, as transformers' window_function
    spectrum = np.fft.rfft(padded[index] * window, axis=1)
    power = np.abs(spectrum) ** 2
    mel = hf_filters(n_mels) @ power.T
    log = np.log10(np.maximum(mel, 1e-10))[:, :-1]
    log = np.maximum(log, log.max() - 8.0)
    return (log + 4.0) / 4.0


def waves(seed: int = 0) -> list:
    """Stand-in utterances: noise shaped by a slow envelope, lengths ending anywhere within a hop, one loud ending."""
    rng = np.random.default_rng(seed)
    out = []
    for n in (8000, 16123, 24000 + 121, 24000 + 159, 32000 + 1, 40000 + 77):
        t = np.arange(n) / 16000
        env = 0.05 + 0.2 * np.abs(np.sin(2 * np.pi * 1.3 * t))
        out.append((rng.normal(0, 1, n) * env).astype(np.float32))
    # 24,159 samples (the last frame's centre is 159 before the end), nearly silent but for the last 40, which sit near
    # the centre of the frame after it: the largest value and so the 80 dB floor come from that frame
    loud = (rng.normal(0, 1, 24159) * 1e-5).astype(np.float32)
    loud[-40:] = rng.normal(0, 0.5, 40)
    out.append(loud)
    return out


class TestReference(unittest.TestCase):
    def test_filters_are_transformers(self):
        for n_mels in (80, 128):
            ours, theirs = W.filterbank(n_mels), hf_filters(n_mels)
            self.assertEqual(ours.shape, (n_mels, 201))
            self.assertLess(np.abs(ours - theirs).max(), 1e-12 * np.abs(theirs).max())

    def test_reference_is_the_extractor_on_the_utterance_frames(self):
        for wave in waves():
            n = wave.shape[0]
            ours = W.reference(wave, 128)
            self.assertEqual(ours.shape, (128, 1 + n // 160))
            self.assertEqual(int(W.frame_count(n)), 1 + n // 160)
            theirs = extractor_steps(wave.astype(np.float64), 128)[:, :1 + n // 160]
            self.assertLess(np.abs(ours - theirs).max(), 1e-9, n)
            # the 80 dB floor holds: nothing below the largest value minus 8, on the (x + 4) / 4 scale
            self.assertGreaterEqual(ours.min(), (theirs.max() * 4 - 4 - 8 + 4) / 4 - 1e-9)

    def test_job_checks(self):
        from affect.runs.spec import check_job, input_width, plain, validate_job

        sweep = json.loads((REPO / "sweeps" / "spk32_cnn.json").read_text())
        settings = plain(validate_job(sweep["base"]))
        settings["features"] = {"name": "whisper_logmel", "n_mels": 128, "normalization": "utterance"}
        job = validate_job(settings)
        check_job(job, 16000)
        self.assertEqual(input_width(job), 128)
        with self.assertRaisesRegex(ValueError, "16000"):
            check_job(job, 44100)
        too_wide = json.loads(json.dumps(settings))
        too_wide["augment"]["freq_mask_width"] = 129
        with self.assertRaises(ValueError):
            check_job(validate_job(too_wide), 16000)
        head = plain(validate_job(json.loads((REPO / "sweeps" / "heads31w_pool_so4.json").read_text())["base"]))
        head["features"] = settings["features"]
        with self.assertRaisesRegex(ValueError, "frames"):
            check_job(validate_job(head), 16000)
        with self.assertRaises(ValueError):
            validate_job({**settings, "features": {"name": "whisper_logmel", "n_mels": 64,
                                                   "normalization": "utterance"}})


@needs_module("torch")
class TestTorch(unittest.TestCase):
    def test_a_batch_gives_each_row_its_reference(self):
        import torch

        rows = waves(1)
        longest = max(w.shape[0] for w in rows)
        batch = np.zeros((len(rows), longest), dtype=np.float32)
        for i, w in enumerate(rows):
            batch[i, :w.shape[0]] = w
        feature = W.WhisperLogMel(W.WhisperLogMelSettings(n_mels=128, normalization="utterance"), 16000,
                                  torch.device("cpu"))
        out = feature(torch.from_numpy(batch)).numpy()
        self.assertEqual(out.shape, (len(rows), 128, 1 + longest // 160))
        for i, w in enumerate(rows):
            n = int(feature.frames(w.shape[0]))
            ref = W.reference(w, 128)
            self.assertLess(np.abs(out[i, :, :n] - ref).max(), 1e-3)
            self.assertLess(np.median(np.abs(out[i, :, :n] - ref)), 1e-5)
            alone = feature(torch.from_numpy(w[None, :])).numpy()[0]
            self.assertLess(np.abs(alone - out[i, :, :n]).max(), 1e-5, "a row's values do not depend on its batch")


def whisper_snapshot():
    found = sorted((Path.home() / "cache" / "huggingface").glob(
        "models--openai--whisper-large-v3/snapshots/*/preprocessor_config.json"))
    return found[-1].parent if found else None


@needs_module("torch")
@needs_module("transformers")
class TestExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = whisper_snapshot()

    def extractor(self):
        from transformers import AutoFeatureExtractor

        if self.snapshot is None:
            self.skipTest("no Whisper large-v3 preprocessor_config.json in ~/cache/huggingface")
        return AutoFeatureExtractor.from_pretrained(str(self.snapshot), local_files_only=True)

    def compare(self, extractor, wave: np.ndarray):
        import torch

        n = wave.shape[0]
        theirs = extractor(wave, sampling_rate=16000, return_tensors="np")["input_features"][0][:, :1 + n // 160]
        feature = W.WhisperLogMel(W.WhisperLogMelSettings(n_mels=128, normalization="utterance"), 16000,
                                  torch.device("cpu"))
        ours = feature(torch.from_numpy(wave[None, :].astype(np.float32))).numpy()[0]
        ref = W.reference(wave, 128)
        self.assertEqual(theirs.shape, ours.shape)
        return float(np.abs(ours - theirs).max()), float(np.abs(ref - theirs).max())

    def test_stand_in_audio(self):
        extractor = self.extractor()
        self.assertEqual((extractor.feature_size, extractor.hop_length, extractor.n_fft), (128, 160, 400))
        self.assertLess(np.abs(np.asarray(extractor.mel_filters).T - W.filterbank(128)).max(), 1e-6)
        for wave in waves(2):
            torch_diff, ref_diff = self.compare(extractor, wave)
            self.assertLess(torch_diff, 1e-3)
            self.assertLess(ref_diff, 1e-3)

    @needs_assembled("iemocap")
    def test_real_utterances(self):
        from affect.registry import DATASETS
        from affect.settings import validate
        from affect.train.data import read_int16
        from affect import paths

        extractor = self.extractor()
        entry = DATASETS.get("iemocap")
        settings = validate(entry.settings_class, {"label_set": "four_merged", "subset": "all",
                                                   "split": "leave_one_speaker_out"}, "data")
        utterances = entry.target().utterances(settings)
        picked = [u for u in utterances if u.n_frames < 470000][::250]
        worst = 0.0
        for u in picked:
            wave = read_int16(paths.assembled_dir("iemocap") / u.audio_path, u.n_frames).astype(np.float32) / 32768.0
            torch_diff, ref_diff = self.compare(extractor, wave)
            worst = max(worst, torch_diff, ref_diff)
        print(f"Whisper's log-mel against the extractor on {len(picked)} IEMOCAP utterances: largest difference "
              f"{worst:.2e}")
        self.assertGreater(len(picked), 20)
        self.assertLess(worst, 1e-3)


if __name__ == "__main__":
    unittest.main()
