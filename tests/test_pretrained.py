"""Pretrained encoders' layers as classical feature sets (affect.classical.pretrained, added 2026-10-19), without torch
or transformers: the feature-set names and the settings that accept them; extract() with a stand-in encoder on a small
dataset of real WAV files, into feature folders that affect.classical.functionals.load reads back exactly; the
refusals (folders that exist, an encoder of the wrong shape, non-finite values, a wrong sample rate); and an SVM sweep
on one layer's feature set end to end into run folders (needs scikit-learn)."""
import dataclasses
import json
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Literal

import numpy as np

from affect import catalog, paths
from affect.data.base import Utterance
from affect.data.splits import leave_one_out_next_validation
from affect.registry import DATASETS
from tests.support import needs_module

catalog.load_all()


@dataclasses.dataclass(frozen=True)
class FakeSettings:
    label_set: Literal["four"]


class FakePretrainedDataset:
    """Three groups of two speakers (named like IEMOCAP's), four classes, 24 utterances per speaker; each utterance a
    real WAV file whose samples carry its class."""
    name = "fake_pretrained"
    sample_rate = 16000

    def class_order(self, settings):
        return ("a", "b", "c", "d")

    def utterances(self, settings):
        out = []
        for g in (1, 2, 3):
            for gender in ("F", "M"):
                for i in range(24):
                    out.append(Utterance(
                        utt_id=f"G{g}_{gender}_{i:03d}", dataset="fake_pretrained", speaker=f"Ses0{g}_{gender}",
                        fold_group=f"G{g}", dialog=f"G{g}_d", kind="impro" if i % 2 else "script",
                        raw_label="abcd"[i % 4], label="abcd"[i % 4], audio_path=f"G{g}_{gender}_{i:03d}.wav",
                        n_frames=800 + 40 * i, sample_rate=self.sample_rate, transcript=None,
                        extra={"speaker_gender": gender}))
        return sorted(out, key=lambda u: u.utt_id)

    def folds(self, settings, utterances):
        return leave_one_out_next_validation(("G1", "G2", "G3"))

    def fold_names(self, settings):
        return ("test_G1", "test_G2", "test_G3")


if "fake_pretrained" not in DATASETS.names():
    DATASETS.register("fake_pretrained", settings=FakeSettings)(FakePretrainedDataset)


def stand_in_encoder(layers=13, width=768, bad=None):
    """encode(int16 samples) -> [layers, width]: a class-carrying vector (the samples' level and length) per layer."""
    def encode(samples):
        x = samples.astype(np.float64) / 32768.0
        base = np.array([x.mean(), x.std(), len(x) / 1e4])
        v = np.zeros((layers, width), dtype=np.float32)
        for k in range(layers):
            v[k, :3] = base * (k + 1)
            v[k, 3 + k] = 1.0
        if bad == "nan":
            v[0, 0] = np.nan
        return v
    info = {"layers": layers, "hidden_size": width, "model_type": "stand-in", "transformers_version": None,
            "torch_version": None, "device": "cpu", "dtype": "float32", "feature_extractor": {"sampling_rate": 16000}}
    return encode, info


class TempRepo:
    """A repository root with the fake dataset's WAV files, MANIFEST.csv and a stand-in snapshot."""
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src" / "affect").mkdir(parents=True)
        (root / "src" / "affect" / "__init__.py").write_text("# stand-in package for code_hash\n")
        assembled = root / "data" / "fake_pretrained" / "assembled"
        assembled.mkdir(parents=True)
        (assembled / "MANIFEST.csv").write_text("path,bytes,sha256\n")
        rng = np.random.default_rng(0)
        for u in FakePretrainedDataset().utterances(None):
            level = 1000 * (1 + "abcd".index(u.label))
            samples = (rng.normal(0, level, size=u.n_frames)).clip(-32768, 32767).astype("<i2")
            with wave.open(str(assembled / u.audio_path), "wb") as h:
                h.setnchannels(1)
                h.setsampwidth(2)
                h.setframerate(16000)
                h.writeframes(samples.tobytes())
        self.snapshot = root / "cache" / "models--x" / "snapshots" / "0123abcd"
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "config.json").write_text(json.dumps({"model_type": "stand-in", "num_hidden_layers": 12}))
        (self.snapshot / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 16000}))
        (self.snapshot / "pytorch_model.bin").write_bytes(b"stand-in weights")
        self.saved = paths.repo_root
        paths.repo_root = lambda: root
        self.root = root
        return self

    def __exit__(self, *exc):
        paths.repo_root = self.saved
        self.tmp.cleanup()


DATA = {"label_set": "four"}


class TestNames(unittest.TestCase):
    def test_feature_sets(self):
        from affect.classical.pretrained import REPOS, feature_sets
        from affect.classical.settings import PRETRAINED_FEATURE_SETS, PRETRAINED_MODELS, FunctionalSettings
        from affect.settings import validate

        self.assertEqual(sorted(REPOS), sorted(PRETRAINED_MODELS))
        self.assertEqual(len(PRETRAINED_FEATURE_SETS), 3 * 13 + 3 * 25)
        self.assertEqual(len(set(PRETRAINED_FEATURE_SETS)), len(PRETRAINED_FEATURE_SETS))
        self.assertEqual(feature_sets("wavlm_base_plus")[0], "wavlm_base_plus_l0")
        self.assertEqual(feature_sets("hubert_base")[-1], "hubert_base_l12")
        self.assertEqual(feature_sets("wavlm_large")[-1], "wavlm_large_l24")
        self.assertEqual(len(feature_sets("wav2vec2_large")), 25)
        with self.assertRaises(ValueError):
            feature_sets("whisper")
        for name in ("eGeMAPSv02", "wav2vec2_base_l7", "hubert_large_l24"):
            validate(FunctionalSettings, {"feature_set": name, "normalization": "speaker"}, "functionals")
        for name in ("wavlm_base_plus_l13", "wavlm_large_l25"):
            with self.assertRaises(ValueError):
                validate(FunctionalSettings, {"feature_set": name, "normalization": "speaker"}, "functionals")


class TestExtract(unittest.TestCase):
    def test_writes_and_loads(self):
        from affect.classical import functionals
        from affect.classical.pretrained import extract

        with TempRepo() as repo:
            encode, info = stand_in_encoder()
            written = extract("fake_pretrained", DATA, "wavlm_base_plus", repo.snapshot, "cpu", lambda m: None,
                              encoder=(encode, info))
            self.assertEqual([p.name for p in written], [f"wavlm_base_plus_l{k}" for k in range(13)])
            utts = FakePretrainedDataset().utterances(None)
            ids = [u.utt_id for u in utts][::-1]                          # load() follows the order asked for
            for k in (0, 6, 12):
                values, names, record = functionals.load("fake_pretrained", f"wavlm_base_plus_l{k}", ids)
                self.assertEqual(values.shape, (len(ids), 768))
                self.assertEqual(names[0], f"wavlm_base_plus_l{k}_d0")
                self.assertEqual((record["layer"], record["commit"], record["n_features"]), (k, "0123abcd", 768))
                self.assertEqual(record["weights_file"], "pytorch_model.bin")
                u = utts[-1]
                expected = encode(np.frombuffer(
                    wave.open(str(repo.root / "data/fake_pretrained/assembled" / u.audio_path)).readframes(10 ** 6),
                    dtype="<i2"))[k]
                np.testing.assert_allclose(values[0], expected, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                extract("fake_pretrained", DATA, "wavlm_base_plus", repo.snapshot, "cpu", lambda m: None,
                        encoder=(encode, info))

    def test_refusals(self):
        from affect.classical.pretrained import extract

        with TempRepo() as repo:
            encode, info = stand_in_encoder(layers=12)
            with self.assertRaises(ValueError):          # the table says 13 layers
                extract("fake_pretrained", DATA, "hubert_base", repo.snapshot, "cpu", lambda m: None,
                        encoder=(encode, info))
            encode, info = stand_in_encoder(bad="nan")
            with self.assertRaises(ValueError):
                extract("fake_pretrained", DATA, "hubert_base", repo.snapshot, "cpu", lambda m: None,
                        encoder=(encode, info))
            encode, info = stand_in_encoder()
            info = {**info, "feature_extractor": {"sampling_rate": 22050}}
            with self.assertRaises(ValueError):
                extract("fake_pretrained", DATA, "hubert_base", repo.snapshot, "cpu", lambda m: None,
                        encoder=(encode, info))
            self.assertFalse((repo.root / "artifacts" / "features").exists(), "a refusal writes nothing")
            (repo.snapshot / "pytorch_model.bin").unlink()
            with self.assertRaises(FileNotFoundError):
                extract("fake_pretrained", DATA, "hubert_base", repo.snapshot, "cpu", lambda m: None,
                        encoder=stand_in_encoder())

    def test_describe(self):
        from affect.classical.pretrained import describe

        with TempRepo() as repo:
            self.assertEqual(describe(repo.snapshot)["preprocessor_config.json"], {"sampling_rate": 16000})


@needs_module("sklearn")
class TestSweep(unittest.TestCase):
    def test_svm_on_a_layer(self):
        from affect.classical.pretrained import extract
        from affect.classical.runner import run

        with TempRepo() as repo:
            extract("fake_pretrained", DATA, "wav2vec2_base", repo.snapshot, "cpu", lambda m: None,
                    encoder=stand_in_encoder())
            sweep = {"sweep_id": "pretrained_test", "description": "test", "folds": ["test_G1", "test_G2", "test_G3"],
                     "base": {"data": {"name": "fake_pretrained", "label_set": "four"},
                              "functionals": {"feature_set": "wav2vec2_base_l6", "normalization": "speaker"},
                              "method": {"name": "svm", "kernel": "rbf", "c_values": [1.0, 10.0],
                                         "gamma_scales": [1.0], "class_weight": "balanced"},
                              "protocol": {"control": "none", "refit": "train"}},
                     "stages": [{"name": "baseline", "kind": "base", "priority": 100, "seeds": [1]},
                                {"name": "layers", "kind": "grid", "priority": 90, "seeds": [1],
                                 "factors": {"functionals.feature_set": ["wav2vec2_base_l0", "wav2vec2_base_l12"]}}]}
            path = repo.root / "pretrained_test.json"
            path.write_text(json.dumps(sweep))
            result = run(path, 2, lambda m: None)
            self.assertEqual(result["failed"], 0)
            folders = sorted((repo.root / "artifacts" / "runs" / "pretrained_test").iterdir())
            self.assertEqual(len(folders), 3 * 3)
            m = json.loads((folders[0] / "metrics_val.json").read_text())
            self.assertGreater(m["final"]["ua"], 0.5, "the stand-in features carry the class")


def stand_in_msd_encoder(layers=13, width=768):
    """The stand-in encoder with pooling mean_sd: each layer's stand-in mean vector, then a stand-in SD vector."""
    encode_mean, info = stand_in_encoder(layers, width)

    def encode(samples):
        mean = encode_mean(samples)
        return np.concatenate([mean, np.abs(mean) * 0.5 + 0.01], axis=1)
    return encode, {**info, "pooling": "mean_sd"}


class TestPooling(unittest.TestCase):
    """Added 2026-10-20: mean and SD feature sets (<model>_l<k>_msd) and the layer average (<model>_lavg)."""

    def test_names(self):
        from affect.classical.pretrained import feature_sets
        from affect.classical.settings import (PRETRAINED_AVERAGE_SETS, PRETRAINED_MSD_SETS, FunctionalSettings)
        from affect.settings import validate

        self.assertEqual(feature_sets("hubert_base", "mean_sd")[3], "hubert_base_l3_msd")
        self.assertEqual(len(PRETRAINED_MSD_SETS), 3 * 13 + 3 * 25)
        self.assertIn("wavlm_large_lavg", PRETRAINED_AVERAGE_SETS)
        for name in ("wavlm_large_l24_msd", "wav2vec2_base_lavg"):
            validate(FunctionalSettings, {"feature_set": name, "normalization": "speaker"}, "functionals")
        with self.assertRaises(ValueError):
            feature_sets("hubert_base", "max")

    def test_mean_sd_extraction(self):
        from affect.classical import functionals
        from affect.classical.pretrained import extract

        with TempRepo() as repo:
            encode, info = stand_in_msd_encoder()
            written = extract("fake_pretrained", DATA, "hubert_base", repo.snapshot, "cpu", lambda m: None,
                              encoder=(encode, info), pooling="mean_sd")
            self.assertEqual([p.name for p in written], [f"hubert_base_l{k}_msd" for k in range(13)])
            utts = FakePretrainedDataset().utterances(None)
            values, names, record = functionals.load("fake_pretrained", "hubert_base_l5_msd", [u.utt_id for u in utts])
            self.assertEqual(values.shape, (len(utts), 2 * 768))
            self.assertEqual(record["n_features"], 2 * 768)
            self.assertIn("SD", record["pooling"])
            u = utts[0]
            samples = np.frombuffer(wave.open(str(repo.root / "data/fake_pretrained/assembled" / u.audio_path))
                                    .readframes(10 ** 6), dtype="<i2")
            np.testing.assert_allclose(values[0], encode(samples)[5], rtol=0, atol=0)
            with self.assertRaises(ValueError):          # an encoder of mean vectors only, asked for mean and SD
                extract("fake_pretrained", DATA, "wav2vec2_base", repo.snapshot, "cpu", lambda m: None,
                        encoder=stand_in_encoder(), pooling="mean_sd")

    def test_layer_average(self):
        from affect.classical import functionals
        from affect.classical.pretrained import derive_layer_average, extract

        with TempRepo() as repo:
            with self.assertRaises(FileNotFoundError):
                derive_layer_average("fake_pretrained", "wavlm_base_plus", lambda m: None)
            extract("fake_pretrained", DATA, "wavlm_base_plus", repo.snapshot, "cpu", lambda m: None,
                    encoder=stand_in_encoder())
            derive_layer_average("fake_pretrained", "wavlm_base_plus", lambda m: None)
            ids = [u.utt_id for u in FakePretrainedDataset().utterances(None)]
            average, names, record = functionals.load("fake_pretrained", "wavlm_base_plus_lavg", ids)
            layers = [functionals.load("fake_pretrained", f"wavlm_base_plus_l{k}", ids)[0] for k in range(13)]
            expected = (sum(layer.astype(np.float64) for layer in layers) / 13).astype(np.float32)
            np.testing.assert_array_equal(average, expected.astype(np.float64))
            self.assertEqual((record["layer"], record["n_features"], len(record["derived_from"])), ("average", 768, 13))
            self.assertEqual(names[0], "wavlm_base_plus_lavg_d0")
            with self.assertRaises(FileExistsError):
                derive_layer_average("fake_pretrained", "wavlm_base_plus", lambda m: None)

    @needs_module("torch")
    def test_pool_layers(self):
        """The pooling load_encoder applies to the real encoder's hidden layers, on random torch tensors: the means,
        and the population SD (numpy's std with ddof 0), for utterances of 1, 2 and 57 frames."""
        import torch

        from affect.classical.pretrained import pool_layers

        rng = np.random.default_rng(20261020)
        for frames in (1, 2, 57):
            arrays = [rng.normal(loc=k, scale=k + 1, size=(frames, 6)).astype(np.float32) for k in range(3)]
            layers = [torch.from_numpy(a) for a in arrays]
            mean, both = pool_layers(layers, "mean"), pool_layers(layers, "mean_sd")
            self.assertEqual((mean.dtype, both.dtype, mean.shape, both.shape),
                             (np.float32, np.float32, (3, 6), (3, 12)))
            want_mean = np.stack([a.astype(np.float64).mean(axis=0) for a in arrays])
            want_sd = np.stack([a.astype(np.float64).std(axis=0) for a in arrays])
            np.testing.assert_allclose(mean, want_mean, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(both, np.concatenate([want_mean, want_sd], axis=1), rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(both[:, :6], mean)
            if frames == 1:
                np.testing.assert_array_equal(both[:, 6:], np.zeros((3, 6), dtype=np.float32))
        with self.assertRaises(ValueError):
            pool_layers(layers, "max")

    def test_commands(self):
        from affect.cli import main

        with self.assertRaises(SystemExit):
            main(["derive-pretrained", "--dataset", "iemocap", "--model", "hubert_base", "--kind", "max"])
        with self.assertRaises(SystemExit):           # --pooling is required, like every argument
            main(["extract-pretrained", "--dataset", "iemocap", "--label-set", "seven", "--subset", "all",
                  "--model", "hubert_base", "--snapshot", "x", "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()
