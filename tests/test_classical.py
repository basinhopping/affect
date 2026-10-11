"""The classical baselines (affect.classical) without openSMILE or TPOT: settings, normalisation,
the SVM on synthetic functionals end to end into run folders that analyze reads, and the TPOT
method's use of the TPOT interface (a stand-in module records how it is called). Needs
scikit-learn; skips without it unless AFFECT_REQUIRE names it."""
import dataclasses
import json
import sys
import tempfile
import types
import unittest
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


class FakeDataset:
    """Three groups of two speakers (named like IEMOCAP's, which the analysis reads for gender), four
    classes; every utterance has a known feature vector."""
    name = "fake_classical"
    sample_rate = 16000

    def class_order(self, settings):
        return ("a", "b", "c", "d")

    def utterances(self, settings):
        out = []
        for g in (1, 2, 3):
            for gender in ("F", "M"):
                for i in range(80):
                    out.append(Utterance(
                        utt_id=f"G{g}_{gender}_{i:03d}", dataset="fake_classical", speaker=f"Ses0{g}_{gender}",
                        fold_group=f"G{g}", dialog=f"G{g}_d", kind="impro" if i % 2 else "script",
                        raw_label="abcd"[i % 4], label="abcd"[i % 4], audio_path="x.wav", n_frames=1,
                        sample_rate=16000, transcript=None, extra={"speaker_gender": gender}))
        return sorted(out, key=lambda u: u.utt_id)

    def folds(self, settings, utterances):
        return leave_one_out_next_validation(("G1", "G2", "G3"))

    def fold_names(self, settings):
        return ("test_G1", "test_G2", "test_G3")


if "fake_classical" not in DATASETS.names():
    DATASETS.register("fake_classical", settings=FakeSettings)(FakeDataset)


def write_features(root: Path, utterances: list) -> None:
    """Class-separable functionals with a per-speaker offset (which speaker z-scores remove)."""
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 2, size=(4, 12))
    offsets = {s: rng.normal(0, 3, size=12) for s in sorted({u.speaker for u in utterances})}
    rows = [centres["abcd".index(u.label)] + offsets[u.speaker] + rng.normal(0, 1, size=12) for u in utterances]
    values = np.array(rows, dtype=np.float32)
    values[:, 11] = 5.0                            # a constant feature
    values[3, 10] = np.nan                         # a missing value
    folder = root / "artifacts" / "features" / "fake_classical" / "eGeMAPSv02"
    folder.mkdir(parents=True)
    np.save(folder / "values.npy", values)
    (folder / "utt_ids.txt").write_text("".join(f"{u.utt_id}\n" for u in utterances))
    (folder / "names.txt").write_text("".join(f"f{k}\n" for k in range(12)))
    (folder / "record.json").write_text(json.dumps({"feature_set": "eGeMAPSv02", "n_features": 12}))


def sweep(sweep_id: str, method: dict) -> dict:
    return {"sweep_id": sweep_id, "description": "test", "folds": ["test_G1", "test_G2", "test_G3"],
            "base": {"data": {"name": "fake_classical", "label_set": "four"},
                     "functionals": {"feature_set": "eGeMAPSv02", "normalization": "speaker"},
                     "method": method, "protocol": {"control": "none", "refit": "train"}},
            "stages": [{"name": "baseline", "kind": "base", "priority": 100, "seeds": [1, 2]},
                       {"name": "variants", "kind": "variants", "priority": 90, "seeds": [1, 2],
                        "variants": {"global z-scores": {"functionals.normalization": "global"},
                                     "refit on train and val": {"protocol.refit": "train_and_val"},
                                     "shuffled": {"protocol.control": "shuffle_train_labels"}}}]}


SVM = {"name": "svm", "kernel": "rbf", "c_values": [0.1, 1.0, 10.0], "gamma_scales": [0.5, 1.0],
       "class_weight": "balanced"}


class TempRepo:
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src" / "affect").mkdir(parents=True)
        (root / "src" / "affect" / "__init__.py").write_text("# stand-in package for code_hash\n")
        (root / "data" / "fake_classical" / "assembled").mkdir(parents=True)
        (root / "data" / "fake_classical" / "assembled" / "MANIFEST.csv").write_text("path,bytes,sha256\n")
        self.saved = paths.repo_root
        paths.repo_root = lambda: root
        self.root = root
        return root

    def __exit__(self, *exc):
        paths.repo_root = self.saved
        self.tmp.cleanup()


@needs_module("sklearn")
class TestClassical(unittest.TestCase):
    def test_settings(self):
        from affect.classical.runner import expand, validate_classical

        job = validate_classical(sweep("x", SVM)["base"])
        self.assertEqual(job.method.c_values, (0.1, 1.0, 10.0))
        for bad in ({**SVM, "kernel": "linear"}, {**SVM, "c_values": [0.0]}, {**SVM, "gamma_scales": []},
                    {"name": "tpot", "generations": 2, "population_size": 4, "offspring_size": 4, "mutation_rate": 0.9,
                     "crossover_rate": 0.2, "operators": "light", "max_time_mins": None, "max_eval_time_mins": 1.0,
                     "n_jobs": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                validate_classical({**sweep("x", SVM)["base"], "method": bad})
        jobs, summary = expand(sweep("x", SVM))
        self.assertEqual(summary["configs"], 4)
        self.assertEqual(len(jobs), 4 * 3 * 2)
        self.assertEqual(len({j["config_hash"] for j in jobs}), 4)

    def test_normalize(self):
        from affect.classical.runner import normalize

        x = np.array([[1.0, 5.0, np.nan], [3.0, 5.0, 2.0], [10.0, 5.0, 4.0], [20.0, 5.0, 6.0]])
        speaker = np.array([0, 0, 1, 1])
        g = normalize(x, "global", np.array([0, 1]), speaker)
        np.testing.assert_allclose(g[:, 0], [-1.0, 1.0, 8.0, 18.0])
        self.assertTrue(np.all(g[:, 1] == 0), "a constant feature becomes 0")
        self.assertEqual(g[0, 2], 0.0, "a missing value becomes 0")
        s = normalize(x, "speaker", np.array([0, 1]), speaker)
        np.testing.assert_allclose(s[:, 0], [-1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(s[2:, 2], [-1.0, 1.0])
        self.assertTrue(np.isfinite(s).all())

    def test_svm_sweep_end_to_end_and_analyze(self):
        from affect.analysis.validation import analyze
        from affect.classical.runner import run

        with TempRepo() as root:
            write_features(root, FakeDataset().utterances(None))
            path = root / "svm.json"
            path.write_text(json.dumps(sweep("classical_test", SVM)))
            messages = []
            result = run(path, 3, messages.append)
            self.assertEqual((result["done"], result["failed"]), (24, 0), messages[-5:])
            again = run(path, 3, messages.append)
            self.assertEqual(again["skipped_complete"], 24, "a second run resumes and skips complete runs")
            folders = sorted((root / "artifacts" / "runs" / "classical_test").iterdir())
            self.assertEqual(len(folders), 24)
            for name in ("spec.json", "metrics_val.json", "predictions_val.csv", "metrics_test.json",
                         "predictions_test.csv", "efficiency.json", "provenance.json", "log.txt"):
                self.assertTrue((folders[0] / name).is_file(), name)
            report = analyze("classical_test", 0.95, 200, 1)
            by_label = {s["label"]: s for s in report["ranked"]}
            self.assertGreater(by_label["base"]["ua"], 0.9, "speaker z-scores remove the speaker offsets")
            self.assertLess(by_label["global z-scores"]["ua"], by_label["base"]["ua"])
            (control,) = report["controls"]
            self.assertLess(control["ua"], 0.45, "shuffled training labels give about chance")
            metrics = json.loads((folders[0] / "metrics_val.json").read_text())
            self.assertEqual(len(metrics["grid"]), 6)
            self.assertEqual(metrics["best_val_ua"], metrics["final"]["ua"])
            changed = sweep("classical_test", {**SVM, "c_values": [1.0]})
            path.write_text(json.dumps(changed))
            with self.assertRaises(FileExistsError):
                run(path, 3, messages.append)

    def test_tpot_method_uses_the_validation_split(self):
        from sklearn.linear_model import LogisticRegression

        from affect.classical.settings import TPOTSettings
        from affect.registry import CLASSICAL

        calls = {}

        class StandIn:
            def __init__(self, **kwargs):
                calls["init"] = kwargs

            def fit(self, x, y):
                calls["fit"] = (x.shape, y.shape)
                (train, val), = calls["init"]["cv"]
                calls["split"] = (train, val)
                self.fitted_pipeline_ = LogisticRegression(max_iter=500).fit(x, y)
                self._optimized_pipeline_score = 0.5
                self.evaluated_individuals_ = {"a": 1, "b": 2}
                return self

        module = types.ModuleType("tpot")
        module.TPOTClassifier = StandIn
        saved = sys.modules.get("tpot")
        sys.modules["tpot"] = module
        try:
            rng = np.random.default_rng(0)
            centres = rng.normal(0, 3, size=(4, 6))
            y_train, y_val, y_test = np.arange(80) % 4, np.arange(40) % 4, np.arange(40) % 4
            data = {"x_train": centres[y_train] + rng.normal(size=(80, 6)), "y_train": y_train,
                    "x_val": centres[y_val] + rng.normal(size=(40, 6)), "y_val": y_val,
                    "x_test": centres[y_test] + rng.normal(size=(40, 6)), "n_classes": 4}
            s = TPOTSettings(generations=3, population_size=10, offspring_size=10, mutation_rate=0.9,
                             crossover_rate=0.1, operators="light", max_time_mins=None, max_eval_time_mins=2.0, n_jobs=2)
            out = CLASSICAL.get("tpot").target(s, data, 7, "train", lambda m: None)
        finally:
            if saved is None:
                del sys.modules["tpot"]
            else:
                sys.modules["tpot"] = saved
        init = calls["init"]
        self.assertEqual(init["scoring"], "balanced_accuracy")
        self.assertEqual(init["config_dict"], "TPOT light")
        self.assertEqual(init["random_state"], 7)
        self.assertEqual((init["generations"], init["population_size"], init["n_jobs"]), (3, 10, 2))
        train, val = calls["split"]
        np.testing.assert_array_equal(train, np.arange(80))
        np.testing.assert_array_equal(val, np.arange(80, 120))
        self.assertEqual(calls["fit"], ((120, 6), (120,)))
        self.assertEqual(out["val_scores"].shape, (40, 4))
        self.assertEqual(out["chosen"]["pipelines_evaluated"], 2)
        self.assertEqual(out["chosen"]["search_val_ua"], 0.5)
        self.assertGreater(out["chosen"]["val_ua"], 0.8)


if __name__ == "__main__":
    unittest.main()
