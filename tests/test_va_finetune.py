"""The valence and arousal probe (affect.classical.va) and EmoFilm fine-tuning (affect.classical.finetune), both added
2026-10-21, end to end on stand-ins. Needs scikit-learn.

va: a stand-in dataset whose WAV files carry each utterance's arousal in their level (so the stand-in encoder's layers
carry it too), three groups of two speakers, ratings from 1 to 5; va_probe writes its folder, chooses the penalty and
the layer on validation only (checked against validation.csv), predicts each utterance's test fold once, and its
report's test CCC equals the one computed from predictions_test.csv; CCC itself against hand cases; a rating outside 1
to 5 is refused.

finetune: the cold test's stand-in repository (tests/test_coldtest.py: a three-class source, the stand-in EmoFilm of
four films, a finished sweep and a record); emofilm_finetune with three film folds: the cold models' predictions equal
cold_test's under the language and gender z-scores, every clip is scored once by each method in its film's fold, the
report's UA equals the predictions file's, the film folds hold whole films; more folds than films is refused."""
import csv
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
from tests.test_pretrained import stand_in_encoder

catalog.load_all()


@dataclasses.dataclass(frozen=True)
class VASettings:
    label_set: Literal["four"]
    subset: Literal["all"]
    split: Literal["leave_one_session_out"]


def ratings(i: int, g: int, gender: str) -> tuple:
    """Arousal rises with i; valence and dominance follow other patterns."""
    a = 1.0 + 4.0 * (i % 12) / 11
    v = 1.0 + 4.0 * ((i * 5 + g) % 12) / 11
    d = 0.5 * (a + v)
    return round(v, 4), round(a, 4), round(d, 4)


class VADataset:
    """Three groups of two speakers, 24 utterances each, IEMOCAP's v, a and d in extra."""
    name = "fake_va"
    sample_rate = 16000

    def class_order(self, settings):
        return ("a", "b", "c", "d")

    def utterances(self, settings):
        out = []
        for g in (1, 2, 3):
            for gender in ("F", "M"):
                for i in range(24):
                    v, a, d = ratings(i, g, gender)
                    out.append(Utterance(
                        utt_id=f"G{g}_{gender}_{i:03d}", dataset=self.name, speaker=f"Ses0{g}_{gender}",
                        fold_group=f"G{g}", dialog=f"G{g}_d", kind="impro", raw_label="abcd"[i % 4],
                        label="abcd"[i % 4], audio_path=f"G{g}_{gender}_{i:03d}.wav", n_frames=1600,
                        sample_rate=16000, transcript=None,
                        extra={"speaker_gender": gender, "v": v, "a": a, "d": d}))
        return sorted(out, key=lambda u: u.utt_id)

    def folds(self, settings, utterances):
        return leave_one_out_next_validation(("G1", "G2", "G3"))

    def fold_names(self, settings):
        return ("test_G1", "test_G2", "test_G3")


if "fake_va" not in DATASETS.names():
    DATASETS.register("fake_va", settings=VASettings)(VADataset)

VA_DATA = {"label_set": "four", "subset": "all", "split": "leave_one_session_out"}


class VARepo:
    def __init__(self, bad_rating=False):
        self.bad_rating = bad_rating

    def __enter__(self):
        from affect.classical.pretrained import extract

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src" / "affect").mkdir(parents=True)
        (root / "src" / "affect" / "__init__.py").write_text("# stand-in package for code_hash\n")
        assembled = root / "data" / "fake_va" / "assembled"
        assembled.mkdir(parents=True)
        (assembled / "MANIFEST.csv").write_text("path,bytes,sha256\n")
        rng = np.random.default_rng(3)
        for u in VADataset().utterances(None):
            level = 600 * u.extra["a"] * (1.3 if u.speaker.endswith("M") else 1.0)
            samples = rng.normal(0, level, size=u.n_frames).clip(-32768, 32767).astype("<i2")
            with wave.open(str(assembled / u.audio_path), "wb") as h:
                h.setnchannels(1)
                h.setsampwidth(2)
                h.setframerate(16000)
                h.writeframes(samples.tobytes())
        snapshot = root / "cache" / "snapshots" / "0123abcd"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"stand-in weights")
        self.saved = paths.repo_root
        paths.repo_root = lambda: root
        self.root = root
        extract("fake_va", VA_DATA, "hubert_base", snapshot, "cpu", lambda m: None, encoder=stand_in_encoder())
        if self.bad_rating:
            self.saved_utts = VADataset.utterances

            def broken(ds, settings):
                utts = self.saved_utts(ds, settings)
                u = utts[5]
                utts[5] = dataclasses.replace(u, extra={**u.extra, "a": 6.5})
                return utts
            VADataset.utterances = broken
        return self

    def __exit__(self, *exc):
        if self.bad_rating:
            VADataset.utterances = self.saved_utts
        paths.repo_root = self.saved
        self.tmp.cleanup()


@needs_module("sklearn")
class TestVA(unittest.TestCase):
    def test_ccc(self):
        from affect.classical.va import ccc, pearson

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(ccc(y, y), 1.0, places=12)
        self.assertLess(ccc(y, y + 1.0), 1.0)
        self.assertLess(ccc(y, 2 * y), 1.0)
        self.assertAlmostEqual(pearson(y, 2 * y + 3), 1.0, places=12)
        self.assertAlmostEqual(ccc(y, -y + 6), -1.0, places=12)
        # 2 cov / (var y + var p + (mean y - mean p)^2), by hand for y and y + 1: 2 * 2 / (2 + 2 + 1)
        self.assertAlmostEqual(ccc(y, y + 1.0), 0.8, places=12)

    def test_probe(self):
        from affect.classical.va import ALPHAS, SCALES, ccc, va_probe

        with VARepo() as repo:
            folder = va_probe("hubert_base", VA_DATA, lambda m: None, dataset="fake_va")
            rep = json.loads((folder / "report.json").read_text())
            self.assertEqual(rep["folds"], ["test_G1", "test_G2", "test_G3"])
            with open(folder / "validation.csv", newline="") as h:
                val = list(csv.DictReader(h))
            self.assertEqual(len(val), 13 * 3 * len(SCALES) * len(ALPHAS))
            for s in SCALES:
                best = {}
                for r in val:
                    if r["scale"] != s:
                        continue
                    key = (r["layer"], r["fold"])
                    c = float(r["validation_ccc"])
                    if key not in best or c > best[key][1] + 1e-12:
                        best[key] = (float(r["alpha"]), c)
                layers = [f"hubert_base_l{k}" for k in range(13)]
                means = [np.mean([best[(layer, f)][1] for f in rep["folds"]]) for layer in layers]
                # the file holds six decimals, and the stand-in's layers are equal after z-scores up to rounding, so
                # the check is that the chosen layer is one of the best on validation, not which of equals
                chosen = layers.index(rep["chosen"][s]["layer"])
                self.assertAlmostEqual(means[chosen], max(means), places=5)
                self.assertAlmostEqual(rep["chosen"][s]["validation_ccc"], max(means), places=5)
            with open(folder / "predictions_test.csv", newline="") as h:
                rows = list(csv.DictReader(h))
            self.assertEqual(len(rows), 3 * 2 * 24)
            self.assertEqual(len({r["utt_id"] for r in rows}), len(rows))
            for s in SCALES:
                per_fold = [ccc([float(r[f"true_{s}"]) for r in rows if r["fold"] == f],
                                [float(r[f"pred_{s}"]) for r in rows if r["fold"] == f]) for f in rep["folds"]]
                self.assertAlmostEqual(rep["results"][s]["ccc"], float(np.mean(per_fold)), places=4)
            self.assertGreater(rep["results"]["a"]["ccc"], 0.8, "the stand-in audio carries arousal")
            self.assertEqual(rep["ratings_outside_1_to_5"], {"v": 0, "a": 0, "d": 0})
            self.assertTrue((repo.root / "artifacts" / "va").is_dir())

    def test_ratings_outside_1_to_5(self):
        """IEMOCAP's averages include a few at 5.5 and 0.5; they are used as given (changed 2026-10-21)."""
        from affect.classical.va import va_probe

        with VARepo() as repo:
            saved = VADataset.utterances

            def edge(ds, settings):
                utts = saved(ds, settings)
                utts[3] = dataclasses.replace(utts[3], extra={**utts[3].extra, "v": 5.5, "d": 0.5})
                return utts
            VADataset.utterances = edge
            try:
                folder = va_probe("hubert_base", VA_DATA, lambda m: None, dataset="fake_va")
            finally:
                VADataset.utterances = saved
            rep = json.loads((folder / "report.json").read_text())
            self.assertEqual(rep["ratings_outside_1_to_5"], {"v": 1, "a": 0, "d": 1})
            self.assertEqual(rep["rating_ranges"]["v"][1], 5.5)
            self.assertTrue((repo.root / "artifacts" / "va").is_dir())

    def test_refusal(self):
        from affect.classical.va import va_probe

        with VARepo(bad_rating=True) as repo:
            with self.assertRaises(ValueError):
                va_probe("hubert_base", VA_DATA, lambda m: None, dataset="fake_va")
            self.assertFalse((repo.root / "artifacts" / "va").exists())


@needs_module("sklearn")
class TestFinetune(unittest.TestCase):
    def test_finetune(self):
        from affect.classical.coldtest import cold_test
        from affect.classical.finetune import METHODS, emofilm_finetune, film_folds
        from tests.test_coldtest import TARGET, ColdRepo

        with ColdRepo():
            cold = cold_test("cold_record", "emofilm", TARGET, lambda m: None)
            folder = emofilm_finetune("cold_record", lambda m: None, n_folds=3)
            rep = json.loads((folder / "report.json").read_text())
            with open(folder / "predictions.csv", newline="") as h:
                rows = list(csv.DictReader(h))
            with open(cold / "predictions_language_gender.csv", newline="") as h:
                cold_rows = {r["utt_id"]: r for r in csv.DictReader(h)}
            names = [r["name"] for r in rep["runs"]]
            self.assertEqual(names, ["test_G1", "test_G2", "test_G3"])
            for r in rows:
                for n in names:
                    self.assertEqual(r[f"cold_{n}"], cold_rows[r["utt_id"]][f"pred_{n}"])
            self.assertEqual(len(rows), 36)
            folds = {}
            for r in rows:
                folds.setdefault(r["film"], set()).add(r["film_fold"])
            self.assertTrue(all(len(v) == 1 for v in folds.values()))
            self.assertEqual(rep["film_folds"], {f: int(next(iter(v))) for f, v in folds.items()})
            classes = rep["class_names"]
            y = np.array([classes.index(r["label"]) for r in rows])

            def ua(column):
                p = np.array([classes.index(r[column]) for r in rows])
                return float(np.mean([np.mean(p[y == c] == c) for c in range(3) if np.any(y == c)]))
            for method in METHODS:
                cols = [f"{method}_{n}" for n in names] if method != "emofilm" else ["emofilm"]
                self.assertAlmostEqual(rep["results"][method]["mean_over_models"]["ua"],
                                       float(np.mean([ua(c) for c in cols])), places=12)
            self.assertEqual(sorted(rep["emofilm_choices"]), ["0", "1", "2"])
            self.assertAlmostEqual(rep["results"]["cold"]["mean_over_models"]["ua"],
                                   json.loads((cold / "report.json").read_text())["results"]["language_gender"]
                                   ["mean_over_models"]["ua"], places=12)
            with self.assertRaises(ValueError):
                emofilm_finetune("cold_record", lambda m: None, n_folds=5)
        with self.assertRaises(ValueError):
            film_folds(["film_01", "film_02"], 3)
        with self.assertRaises(ValueError):             # two folds leave the EmoFilm-only SVM nothing to choose on
            film_folds(["a", "b", "c"], 2)
        self.assertEqual(film_folds(["b", "a", "c", "d", "a"], 3), {"a": 0, "b": 1, "c": 2, "d": 0})


if __name__ == "__main__":
    unittest.main()
