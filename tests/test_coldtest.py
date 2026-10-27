"""Cold tests (affect.classical.coldtest, added 2026-10-20), end to end on stand-ins: a three-class source dataset of
real WAV files, the stand-in EmoFilm of tests/synthetic_emofilm.py, both through the stand-in encoder of
tests/test_pretrained.py, an SVM sweep on one layer, a selection record of it, then cold_test(): the refit repeats every
run's validation UA and predictions, the target is scored under both normalizations, and the files hold what the
report says. Then the refusals, each writing nothing: a run file that differs from the refit, target features from
other encoder weights, and a record the cold test does not take. Needs scikit-learn."""
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
from tests.synthetic_emofilm import write_assembly
from tests.test_pretrained import stand_in_encoder

catalog.load_all()
CLASSES = ("hap+exc", "ang", "sad")
LEVEL = {"hap+exc": 3000, "ang": 6000, "sad": 1000}       # as the stand-in EmoFilm's gio, rab, tri


@dataclasses.dataclass(frozen=True)
class ColdSourceSettings:
    label_set: Literal["three"]


class ColdSourceDataset:
    """Three groups of two speakers, the three classes, 18 utterances per speaker, each a WAV file."""
    name = "fake_cold"
    sample_rate = 16000

    def class_order(self, settings):
        return CLASSES

    def utterances(self, settings):
        out = []
        for g in (1, 2, 3):
            for gender in ("F", "M"):
                for i in range(18):
                    out.append(Utterance(
                        utt_id=f"G{g}_{gender}_{i:03d}", dataset=self.name, speaker=f"Ses0{g}_{gender}",
                        fold_group=f"G{g}", dialog=f"G{g}_d", kind="impro", raw_label=CLASSES[i % 3],
                        label=CLASSES[i % 3], audio_path=f"G{g}_{gender}_{i:03d}.wav", n_frames=1200 + 40 * i,
                        sample_rate=16000, transcript=None, extra={}))
        return sorted(out, key=lambda u: u.utt_id)

    def folds(self, settings, utterances):
        return leave_one_out_next_validation(("G1", "G2", "G3"))

    def fold_names(self, settings):
        return ("test_G1", "test_G2", "test_G3")


if "fake_cold" not in DATASETS.names():
    DATASETS.register("fake_cold", settings=ColdSourceSettings)(ColdSourceDataset)

TARGET = {"label_set": "three_no_neutral", "subset": "all"}


class ColdRepo:
    """A repository root with both datasets, their stand-in features, a finished sweep and a selection record."""
    def __enter__(self):
        from affect.classical.pretrained import extract
        from affect.classical.runner import config_hash, run, validate_classical

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src" / "affect").mkdir(parents=True)
        (root / "src" / "affect" / "__init__.py").write_text("# stand-in package for code_hash\n")
        source = root / "data" / "fake_cold" / "assembled"
        source.mkdir(parents=True)
        (source / "MANIFEST.csv").write_text("path,bytes,sha256\n")
        rng = np.random.default_rng(1)
        for u in ColdSourceDataset().utterances(None):
            samples = rng.normal(0, LEVEL[u.label], size=u.n_frames).clip(-32768, 32767).astype("<i2")
            with wave.open(str(source / u.audio_path), "wb") as h:
                h.setnchannels(1)
                h.setsampwidth(2)
                h.setframerate(16000)
                h.writeframes(samples.tobytes())
        write_assembly(root / "data" / "emofilm" / "assembled")
        snapshot = root / "cache" / "snapshots" / "0123abcd"
        snapshot.mkdir(parents=True)
        (snapshot / "pytorch_model.bin").write_bytes(b"stand-in weights")
        self.saved = paths.repo_root
        paths.repo_root = lambda: root
        self.root = root
        for dataset, settings in (("fake_cold", {"label_set": "three"}), ("emofilm", TARGET)):
            extract(dataset, settings, "wavlm_base_plus", snapshot, "cpu", lambda m: None, encoder=stand_in_encoder())
        self.settings = {"data": {"name": "fake_cold", "label_set": "three"},
                         "functionals": {"feature_set": "wavlm_base_plus_l6", "normalization": "speaker"},
                         "method": {"name": "svm", "kernel": "rbf", "c_values": [1.0, 10.0], "gamma_scales": [1.0],
                                    "class_weight": "balanced"},
                         "protocol": {"control": "none", "refit": "train"}}
        sweep = {"sweep_id": "cold_source", "description": "test", "folds": ["test_G1", "test_G2", "test_G3"],
                 "base": self.settings, "stages": [{"name": "baseline", "kind": "base", "priority": 100, "seeds": [1]}]}
        (root / "cold_source.json").write_text(json.dumps(sweep))
        assert run(root / "cold_source.json", 1, lambda m: None)["failed"] == 0
        self.config_hash = config_hash(validate_classical(self.settings))
        self.write_selection("cold_record", self.settings, self.config_hash)
        return self

    def write_selection(self, name, settings, config_hash):
        folder = self.root / "artifacts" / "selections" / name
        folder.mkdir(parents=True)
        runs = sorted((self.root / "artifacts" / "runs" / "cold_source").glob("*/metrics_val.json"))
        ua = float(np.mean([json.loads(f.read_text())["best_val_ua"] for f in runs]))
        (folder / "selection.json").write_text(json.dumps({
            "sweep_id": "cold_source", "config_hash": config_hash, "settings": settings,
            "folds": ["test_G1", "test_G2", "test_G3"], "validation": {"ua": ua, "seeds": [1]}}))

    def coldtests(self):
        folder = self.root / "artifacts" / "coldtests"
        return sorted(p.name for p in folder.iterdir()) if folder.exists() else []

    def __exit__(self, *exc):
        paths.repo_root = self.saved
        self.tmp.cleanup()


@needs_module("sklearn")
class TestColdTest(unittest.TestCase):
    def test_end_to_end(self):
        from affect.classical.coldtest import cold_test

        with ColdRepo() as repo:
            lines = []
            done = cold_test("cold_record", "emofilm", TARGET, lines.append)
            self.assertEqual(repo.coldtests(), [done.name])
            self.assertTrue(done.name.startswith("cold_record__emofilm_"))
            rep = json.loads((done / "report.json").read_text())
            self.assertEqual(rep["class_names"], list(CLASSES))
            self.assertEqual([r["fold"] for r in rep["runs"]], ["test_G1", "test_G2", "test_G3"])
            for r in rep["runs"]:
                measured = json.loads((repo.root / "artifacts" / "runs" / "cold_source" / r["run_id"] /
                                       "metrics_val.json").read_text())
                self.assertEqual(r["validation_ua"], measured["best_val_ua"])
                self.assertEqual((r["C"], r["gamma_scale"]), (measured["chosen"]["C"],
                                                              measured["chosen"]["gamma_scale"]))
                self.assertTrue(r["validation_predictions_equal_the_run"] and r["test_predictions_equal_the_run"])
            self.assertEqual(rep["source_features"]["weights_sha256"], rep["target_features"]["weights_sha256"])
            self.assertEqual(rep["target"]["n_utterances"], 36)
            self.assertEqual(rep["target"]["support"], {"hap+exc": 12, "ang": 12, "sad": 12})
            for n in ("language_gender", "corpus"):
                res = rep["results"][n]
                self.assertEqual(sorted(res["models"]), ["test_G1", "test_G2", "test_G3"])
                uas = [m["ua"] for m in res["models"].values()]
                self.assertAlmostEqual(res["mean_over_models"]["ua"], float(np.mean(uas)), places=12)
                self.assertGreater(res["mean_over_models"]["ua"], 0.6, "the stand-in features carry the class")
                self.assertEqual(sorted(res["models"]["test_G1"]["by_language"]), ["en", "es", "it"])
                self.assertEqual(sorted(res["models"]["test_G1"]["by_gender"]), ["f", "m"])
                with open(done / f"predictions_{n}.csv", newline="") as h:
                    rows = list(csv.DictReader(h))
                self.assertEqual(len(rows), 36)
                for fold in ("test_G1", "test_G2", "test_G3", "ensemble"):
                    pred = [r[f"pred_{fold}"] for r in rows]
                    label = [r["label"] for r in rows]
                    recalls = [np.mean([p == c for p, t in zip(pred, label) if t == c]) for c in CLASSES]
                    expected = res["ensemble"]["ua"] if fold == "ensemble" else res["models"][fold]["ua"]
                    self.assertAlmostEqual(float(np.mean(recalls)), expected, places=12)
                    for r in rows:
                        probs = [float(r[f"prob_{fold}_{c}"]) for c in CLASSES]
                        self.assertAlmostEqual(sum(probs), 1.0, places=5)
                        self.assertEqual(r[f"pred_{fold}"], CLASSES[int(np.argmax(probs))])
            self.assertTrue(any("repeats the run's validation UA" in line for line in lines))

    def test_refusals(self):
        from affect.classical.coldtest import cold_test

        with ColdRepo() as repo:
            run = sorted((repo.root / "artifacts" / "runs" / "cold_source").iterdir())[1]
            path = run / "predictions_test.csv"
            text = path.read_text()
            header, first, rest = text.split("\n", 2)
            cells = first.split(",")
            cells[-1] = f"{float(cells[-1]) + 0.000001:.6f}"
            path.write_text("\n".join([header, ",".join(cells), rest]))
            with self.assertRaisesRegex(ValueError, "predictions_test.csv, row 0"):
                cold_test("cold_record", "emofilm", TARGET, lambda m: None)
            path.write_text(text)
            record = repo.root / "artifacts" / "features" / "emofilm" / "wavlm_base_plus_l6" / "record.json"
            saved = record.read_text()
            record.write_text(json.dumps({**json.loads(saved), "weights_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "weights_sha256"):
                cold_test("cold_record", "emofilm", TARGET, lambda m: None)
            record.write_text(saved)
            other = {**repo.settings, "protocol": {"control": "none", "refit": "train_and_val"}}
            repo.write_selection("refit_record", other, repo.config_hash)
            with self.assertRaisesRegex(ValueError, "refit 'train'"):
                cold_test("refit_record", "emofilm", TARGET, lambda m: None)
            self.assertEqual(repo.coldtests(), [], "a refusal writes nothing")
            done = cold_test("cold_record", "emofilm", TARGET, lambda m: None)
            self.assertEqual(repo.coldtests(), [done.name])

    def test_command(self):
        from affect.cli import main

        with ColdRepo() as repo:
            args = ["cold-test", "--selections", "cold_record", "cold_record", "--target", "emofilm",
                    "--label-set", "three_no_neutral", "--subset", "all"]
            self.assertEqual(main(args), 0)
            self.assertEqual(len(repo.coldtests()), 2, "one folder per record named")
            with self.assertRaisesRegex(SystemExit, "knows the target emofilm only"):
                main(args[:5] + ["iemocap"] + args[6:])


if __name__ == "__main__":
    unittest.main()
