"""The validation-only analysis of a finished sweep (affect.cli analyze), on synthetic run folders.

The fake runs are built so the answer is known: one setting is right on 80% of the validation
utterances where the baseline is right on 50%, one on 30%, and one makes exactly the baseline's
predictions. Every run folder also holds test files that are not JSON or CSV, so the analysis
fails if it ever opens one (plan 3.5: settings are chosen from validation alone).
"""
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.support import REPO, needs_module
from tests.test_sweep import cli, small_sweep

CLASSES = ["neu", "hap+exc", "ang", "sad"]
N_UTT = 200  # validation utterances per fold, 50 of each class; with 40, a bootstrap draw can miss a class


def right_of(job: dict) -> int:
    """How many of every 10 utterances the configuration gets right (the pattern shifts with the seed)."""
    s = job["settings"]
    if s["train"]["control"] != "none":
        return -1                                           # the shuffled control: see predictions()
    if job["stage"] == "one_factor":
        if s["features"]["n_mels"] == 80:
            return 8
        if s["model"]["pooling"] == "attention":
            return 3
        return 5                                            # n_mels 40: the baseline's own predictions
    if job["stage"] == "random":
        return 6
    return 5


def predictions(job: dict) -> list:
    val_session = int(job["fold"][-1]) % 5 + 1              # test S5 validates on S1
    right = right_of(job)
    rows = []
    for i in range(N_UTT):
        label = i % 4
        if right < 0:
            correct = (i + job["seed"]) % 4 == 0
        else:
            correct = (i + job["seed"]) % 10 < right
        pred = label if correct else (label + 1) % 4
        gender = "F" if (i // 4) % 2 == 0 else "M"            # every block of 4 holds each class once
        kind = "impro" if (i // 8) % 2 == 0 else "script"
        rows.append({"utt_id": f"Ses0{val_session}F_{kind}01_{gender}{i:03d}", "speaker": f"Ses0{val_session}_{gender}",
                     "kind": kind, "label": CLASSES[label], "pred": CLASSES[pred],
                     **{f"prob_{c}": 1.0 if k == pred else 0.0 for k, c in enumerate(CLASSES)}})
    return rows


def metrics(rows: list) -> dict:
    recall, f1 = {}, []
    for c in CLASSES:
        tp = sum(r["label"] == c and r["pred"] == c for r in rows)
        n_true = sum(r["label"] == c for r in rows)
        n_pred = sum(r["pred"] == c for r in rows)
        recall[c] = tp / n_true
        f1.append(0.0 if tp == 0 else 2 * tp / (n_true + n_pred))
    ua = sum(recall.values()) / len(CLASSES)
    wa = sum(r["label"] == r["pred"] for r in rows) / len(rows)
    return {"ua": ua, "wa": wa, "macro_f1": sum(f1) / len(f1), "recall": recall}


def write_run(folder: Path, job: dict) -> None:
    rows = predictions(job)
    final = metrics(rows)
    folder.mkdir(parents=True)
    (folder / "spec.json").write_text(json.dumps(job))
    (folder / "metrics_val.json").write_text(json.dumps(
        {"best_epoch": 5, "best_val_ua": final["ua"], "epochs_run": 15, "stopped_early": True, "curves": [],
         "final": final}))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (folder / "predictions_val.csv").write_text(out.getvalue())
    (folder / "metrics_test.json").write_text("test metrics must never be read by the analysis")
    (folder / "predictions_test.csv").write_text("test predictions must never be read by the analysis")


@needs_module("scipy")
class TestAnalyze(unittest.TestCase):
    def test_verdicts_controls_factor_levels_and_saved_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            (root / "sweeps").mkdir()
            sweep = small_sweep()
            for stage in sweep["stages"]:
                if stage["name"] in ("baseline", "one_factor"):
                    stage["seeds"] = [1, 2, 3]
            sweep_path = root / "sweeps" / "test_sweep.json"
            sweep_path.write_text(json.dumps(sweep))
            done = cli(root, "enqueue", "--sweep-file", str(sweep_path))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            jobs = [json.loads(p.read_text()) for p in sorted((root / "queue" / "pending").glob("*.json"))]
            runs = root / "artifacts" / "runs" / "test_sweep"
            for job in jobs:
                write_run(runs / job["run_id"], job)
            (runs / "x.failed-20260925T000000_000000Z").mkdir()     # a failed run is skipped

            report = cli(root, "analyze", "--sweep", "test_sweep", "--confidence", "0.95", "--n-boot", "400",
                         "--seed", "1", "--top", "20")
            self.assertEqual(report.returncode, 0, report.stderr[-3000:])
            text = report.stdout

            saved = sorted((root / "artifacts" / "analysis").glob("test_sweep_*"))
            self.assertEqual(len(saved), 1)
            for name in ("report.json", "report.txt", "provenance.json"):
                self.assertTrue((saved[0] / name).is_file(), name)
            result = json.loads((saved[0] / "report.json").read_text())

            self.assertEqual(result["n_runs"], len(jobs))
            by_label = {s["label"]: s for s in result["ranked"]}
            base = by_label["base"]
            self.assertEqual(base["config_hash"], result["baseline"])
            self.assertEqual(base["seeds"], [1, 2, 3])
            self.assertAlmostEqual(base["ua"], 0.5)
            self.assertAlmostEqual(by_label["features.n_mels=80"]["ua"], 0.8)
            self.assertAlmostEqual(by_label['model.pooling="attention"']["ua"], 0.3)

            verdicts = {label: s["versus_baseline"]["verdict"] for label, s in by_label.items()
                        if "versus_baseline" in s}
            self.assertEqual(verdicts, {"features.n_mels=80": "better", 'model.pooling="attention"': "worse",
                                        "features.n_mels=40": "no detectable difference"})
            same = by_label["features.n_mels=40"]["versus_baseline"]
            self.assertEqual(same["bootstrap"]["difference"], 0.0)
            self.assertEqual(same["difference_by_seed"], {"1": 0.0, "2": 0.0, "3": 0.0})
            better = by_label["features.n_mels=80"]["versus_baseline"]
            self.assertGreater(better["bootstrap"]["low"], 0.2)
            self.assertLess(better["holm_p"], 0.05)

            # Random configurations share one seed with the baseline: ranked, never compared.
            random_rows = [s for s in result["ranked"] if s["stage"] == "random"]
            self.assertEqual(len(random_rows), 3)
            self.assertFalse(any("versus_baseline" in s for s in random_rows))

            self.assertEqual([s["label"] for s in result["controls"]], ["shuffled"])     # the variant's name
            self.assertAlmostEqual(result["controls"][0]["ua"], 0.25)

            factors = {f["factor"]: f for f in result["factors"]}
            self.assertEqual(set(factors), {"features.n_mels", "model.pooling"})
            for f in factors.values():
                self.assertEqual(sum(level["n_configs"] for level in f["levels"]), 3)

            groups = base["ua_by_group"]
            self.assertEqual(set(groups["gender"]), {"F", "M"})
            self.assertEqual(set(groups["kind"]), {"impro", "script"})
            self.assertTrue(all(v is not None for field in groups.values() for v in field.values()))
            self.assertEqual(by_label["features.n_mels=40"]["ua_by_group"], groups, "identical predictions")
            self.assertIn("by gender: F ", text)

            self.assertIn("(baseline)", text)
            self.assertIn("better  | one_factor: features.n_mels=80", text)
            self.assertIn("no detectable difference  | one_factor: features.n_mels=40", text)


if __name__ == "__main__":
    unittest.main()
