"""compare-runs (affect.analysis.compare): a sweep's runs against every copy of the same run IDs in
other sweeps, on synthetic run folders. The test files of every fake run are not JSON or CSV, so
the comparison fails if it ever parses one: test files may only be hashed (plan 3.5)."""
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from affect import paths
from affect.analysis import compare
from tests.support import REPO
from tests.test_sweep import cli

OLD_CODE, NEW_CODE = "a" * 64, "b" * 64


def metrics_val(seconds: float, loss: float = 1.25) -> dict:
    curves = [{"epoch": e, "train_loss": loss / e, "val_ua": 0.5 + e / 100, "val_wa": 0.5, "val_macro_f1": 0.5,
               "val_mean_nll": 1.0, "lr": 0.001, "train_seconds": seconds * e, "eval_seconds": seconds / 3}
              for e in (1, 2, 3)]
    return {"best_epoch": 3, "best_val_ua": 0.53, "epochs_run": 3, "stopped_early": False, "curves": curves,
            "final": {"ua": 0.53, "recall": {"neu": 0.4, "ang": 0.66}}}


def write_run(root: Path, sweep: str, run_id: str, code: str, seconds: float, test_text: str = "not json or csv",
              loss: float = 1.25, predictions: str = "utt_id,label,pred\nu1,neu,neu\n") -> Path:
    folder = root / "artifacts" / "runs" / sweep / run_id
    folder.mkdir(parents=True)
    (folder / "spec.json").write_text(json.dumps({"sweep_id": sweep, "run_id": run_id}))
    (folder / "metrics_val.json").write_text(json.dumps(metrics_val(seconds, loss), indent=1))
    (folder / "predictions_val.csv").write_text(predictions)
    (folder / "metrics_test.json").write_text("test metrics: " + test_text)
    (folder / "predictions_test.csv").write_text("test predictions: " + test_text)
    (folder / "provenance.json").write_text(json.dumps({"code_sha256": code, "started_utc": str(seconds)}))
    (folder / "efficiency.json").write_text(json.dumps({"train_seconds": seconds}))
    return folder


def write_record(root: Path, sweep: str, run_ids: list) -> None:
    folder = root / "artifacts" / "sweeps" / sweep
    folder.mkdir(parents=True)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["seq", "run_id", "stage", "label", "priority", "fold", "seed", "config_hash"])
    for seq, run_id in enumerate(run_ids):
        writer.writerow([seq, run_id, "base", "base", 200, "test_S1", 1, "c" * 64])
    (folder / "jobs.csv").write_text(out.getvalue())


class TempRepo:
    def __enter__(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        shutil.copytree(REPO / "src", self.root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        self.saved = paths.repo_root
        paths.repo_root = lambda: self.root
        return self.root

    def __exit__(self, *exc):
        paths.repo_root = self.saved
        self.tmp.cleanup()


class TestCompareRuns(unittest.TestCase):
    def test_identical_copies_timing_ignored_and_test_files_never_parsed(self):
        with TempRepo() as root:
            write_record(root, "reg", ["r1", "r2"])
            for run_id in ("r1", "r2"):
                write_run(root, "reg", run_id, NEW_CODE, seconds=7.0)
                write_run(root, "night1", run_id, OLD_CODE, seconds=5.0)       # other code, other times
            write_run(root, "night4", "r1", NEW_CODE, seconds=9.0)             # same code, a third copy
            (root / "artifacts" / "runs" / "night4" / "r2.tmp").mkdir()        # unfinished: not a copy
            (root / "artifacts" / "runs" / "night4" / "r2.failed-20260927T000000_000000Z").mkdir()
            parsed = []
            original = compare._read_json

            def recording(path):
                parsed.append(Path(path).name)
                return original(path)

            compare._read_json = recording
            try:
                report = compare.compare_sweep("reg", 3)
            finally:
                compare._read_json = original
            self.assertEqual(report["verdict"], "SAME", compare.render(report))
            self.assertEqual((report["n_complete"], report["n_copies"], report["n_other_code"]), (2, 3, 2))
            self.assertEqual(report["code_versions"], {OLD_CODE: ["night1"], NEW_CODE: ["night4"]})
            self.assertFalse(set(parsed) & set(compare.TEST_FILES), "a test file was parsed")
            self.assertIn("SAME: all 2 runs are identical", compare.render(report))

    def test_differences_are_found_and_test_values_never_shown(self):
        with TempRepo() as root:
            write_record(root, "reg", ["r1", "r2", "r3"])
            write_run(root, "reg", "r1", NEW_CODE, 7.0)
            write_run(root, "night1", "r1", OLD_CODE, 5.0, test_text="secret 0.5873")   # test file differs
            write_run(root, "reg", "r2", NEW_CODE, 7.0)
            write_run(root, "night1", "r2", OLD_CODE, 5.0, loss=1.5)                    # training differs
            write_run(root, "night2", "r2", OLD_CODE, 5.0, predictions="utt_id,label,pred\nu1,neu,ang\n")
            write_run(root, "reg", "r3", NEW_CODE, 7.0)
            write_run(root, "night4", "r3", NEW_CODE, 9.0)                              # same code only
            report = compare.compare_sweep("reg", 2)
            text = compare.render(report)
            self.assertEqual(report["verdict"], "DIFFERENT", text)
            self.assertEqual(report["differing"], ["r1", "r2"])
            self.assertEqual(report["without_other_code"], ["r3"])
            r1 = report["runs"][0]["copies"][0]["problems"]
            self.assertEqual(r1, ["predictions_test.csv differs (compared by SHA-256; not read)",
                                  "metrics_test.json differs (compared by SHA-256; not read)"])
            self.assertNotIn("secret", text)
            self.assertNotIn("0.5873", json.dumps(report))
            r2 = {c["sweep"]: c["problems"] for c in report["runs"][1]["copies"]}
            self.assertEqual(r2["night1"][0], "metrics_val.curves[0].train_loss: 1.25 against 1.5")
            self.assertEqual(r2["night1"][2], "... and 1 more metric differences")   # 3 curves differ, 2 shown
            self.assertEqual(r2["night2"], ["predictions_val.csv differs"])
            self.assertIn("also differs from the copies in night2", text)
            self.assertIn("r3: no copy made by other code", text)

    def test_unfinished_sweep_and_the_command(self):
        with TempRepo() as root:
            write_record(root, "reg", ["r1", "r2"])
            write_run(root, "reg", "r1", NEW_CODE, 7.0)
            write_run(root, "night1", "r1", OLD_CODE, 5.0)
            write_run(root, "night1", "r2", OLD_CODE, 5.0)
            report = compare.compare_sweep("reg", 3)
            self.assertEqual((report["verdict"], report["missing"]), ("INCOMPLETE", ["r2"]))
            done = cli(root, "compare-runs", "--sweep", "reg", "--differences-shown", "3")
            self.assertEqual(done.returncode, 3, done.stderr[-2000:])
            self.assertIn("INCOMPLETE", done.stdout)
            self.assertFalse((root / "artifacts" / "checks").exists(), "nothing is saved before the end")

            write_run(root, "reg", "r2", NEW_CODE, 7.0)
            done = cli(root, "compare-runs", "--sweep", "reg", "--differences-shown", "3")
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-2000:])
            self.assertIn("SAME", done.stdout)
            saved = sorted((root / "artifacts" / "checks").glob("compare_runs_reg_*"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads((saved[0] / "report.json").read_text())["verdict"], "SAME")

            (root / "artifacts" / "runs" / "night1" / "r2" / "predictions_val.csv").write_text("changed\n")
            done = cli(root, "compare-runs", "--sweep", "reg", "--differences-shown", "3")
            self.assertEqual(done.returncode, 1, done.stdout + done.stderr[-2000:])
            self.assertIn("DIFFERENT: 1 runs differ", done.stdout)


if __name__ == "__main__":
    unittest.main()
