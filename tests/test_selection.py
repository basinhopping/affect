"""The test stage (affect.analysis.selection): select only after analyze, then read the chosen
configuration's test files. Synthetic runs as in test_analysis, with real test files this time."""
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.support import REPO, needs_module
from tests.test_analysis import metrics, predictions, write_run
from tests.test_sweep import cli, small_sweep


def write_test_files(folder: Path, job: dict) -> None:
    rows = predictions({**job, "fold": job["fold"]})
    final = metrics(rows)
    (folder / "metrics_test.json").write_text(json.dumps({"final": final}))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (folder / "predictions_test.csv").write_text(out.getvalue())


@needs_module("scipy")
class TestTestStage(unittest.TestCase):
    def test_select_after_analyze_then_report(self):
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
                write_test_files(runs / job["run_id"], job)
            better = next(j["config_hash"] for j in jobs if j["label"] == "features.n_mels=80")

            early = cli(root, "select", "--sweep", "test_sweep", "--config", better[:10], "--name", "pick1",
                        "--reason", "best validation UA")
            self.assertNotEqual(early.returncode, 0)
            self.assertIn("run analyze first", early.stderr)

            analyzed = cli(root, "analyze", "--sweep", "test_sweep", "--confidence", "0.95", "--n-boot", "200",
                           "--seed", "1", "--top", "3")
            self.assertEqual(analyzed.returncode, 0, analyzed.stderr[-2000:])
            short = cli(root, "select", "--sweep", "test_sweep", "--config", better[:4], "--name", "pick1",
                        "--reason", "best validation UA")
            self.assertNotEqual(short.returncode, 0)
            chosen = cli(root, "select", "--sweep", "test_sweep", "--config", better[:10], "--name", "pick1",
                         "--reason", "best validation UA")
            self.assertEqual(chosen.returncode, 0, chosen.stderr[-2000:])
            record = json.loads((root / "artifacts" / "selections" / "pick1" / "selection.json").read_text())
            self.assertEqual(record["config_hash"], better)
            self.assertAlmostEqual(record["validation"]["ua"], 0.8)
            again = cli(root, "select", "--sweep", "test_sweep", "--config", better[:10], "--name", "pick1",
                        "--reason", "again")
            self.assertNotEqual(again.returncode, 0, "a selection is never overwritten")

            report = cli(root, "test-report", "--selection", "pick1", "--confidence", "0.95")
            self.assertEqual(report.returncode, 0, report.stderr[-2000:])
            saved = sorted((root / "artifacts" / "test_reports").glob("pick1_*"))
            self.assertEqual(len(saved), 1)
            result = json.loads((saved[0] / "report.json").read_text())
            self.assertEqual(result["seeds"], [1, 2, 3])
            self.assertAlmostEqual(result["ua"]["mean"], 0.8)
            self.assertEqual(set(result["ua_by_group"]["gender"]), {"F", "M"})
            self.assertIn("test UA 0.8000", report.stdout)


if __name__ == "__main__":
    unittest.main()
