"""Sweep expansion, the file queue, enqueue and status, without torch."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from affect import catalog
from affect.sweep.design import SweepError, expand
from affect.sweep.queue import Queue, pending_name
from tests.support import REPO

catalog.load_all()
SWEEP_FILE = REPO / "sweeps" / "cnn_screen_1.json"


def small_sweep() -> dict:
    sweep = json.loads(SWEEP_FILE.read_text())
    sweep["sweep_id"] = "test_sweep"
    sweep["stages"] = [
        {"name": "baseline", "kind": "base", "priority": 100, "seeds": [1, 2]},
        {"name": "controls", "kind": "variants", "priority": 90, "seeds": [1],
         "variants": {"shuffled": {"train.control": "shuffle_train_labels"}}},
        {"name": "one_factor", "kind": "one_factor_at_a_time", "priority": 50, "seeds": [1],
         "factors": {"features.n_mels": [40, 80], "model.pooling": ["attention"]}},
        {"name": "random", "kind": "random", "priority": 10, "seeds": [1], "n_configs": 3, "sampling_seed": 7,
         "space": {"features.n_mels": [40, 64, 80], "model.pooling": ["mean", "max", "stats"]}},
    ]
    return sweep


class TestExpand(unittest.TestCase):
    def test_the_real_sweep_file_expands(self):
        jobs, summary = expand(json.loads(SWEEP_FILE.read_text()))
        self.assertEqual(summary["jobs"], len(jobs))
        self.assertEqual(len({j["run_id"] for j in jobs}), len(jobs))
        stages = {s["name"]: s["configs"] for s in summary["stages"]}
        self.assertEqual(stages["baseline"], 1)
        self.assertEqual(stages["random"], 120)

    def test_every_sweep_file_expands(self):
        from affect.classical.runner import expand as expand_classical

        ids = set()
        for path in sorted((REPO / "sweeps").glob("*.json")):
            sweep = json.loads(path.read_text())
            self.assertEqual(path.stem, sweep["sweep_id"], "a sweep file is named after its sweep_id")
            if path.stem.startswith("classical_"):
                with self.assertRaisesRegex(SweepError, "classical sweep"):
                    expand(sweep)
                jobs, summary = expand_classical(sweep)
            else:
                jobs, summary = expand(sweep)
            self.assertEqual(len({j["run_id"] for j in jobs}), summary["jobs"], path.name)
            ids.add(sweep["sweep_id"])
        self.assertIn("cnn_screen_1", ids)

    def test_configurations_that_ran_keep_their_hashes(self):
        # Hashes computed on the node when these configurations ran (nights 1 and 2); the fields added
        # on 2026-09-26 must not change them, whether the settings are written old or new style.
        jobs, summary = expand(json.loads((REPO / "sweeps" / "cnn_confirm_1.json").read_text()))
        self.assertTrue(summary["base_config_hash"].startswith("cf0495a296e9"))
        hashes = {j["config_hash"][:12] for j in jobs}
        self.assertTrue({"9e0cfdfd7f40", "7c12a8af46e1"} <= hashes, "cnn_final_1 and cnn_final_1b")
        self.assertIn("cf0495a296e9_test_S1_seed1", {j["run_id"] for j in jobs})
        # the tuned 3-block CNN (threeblock_final, run on nights 9 and 14) and cnn_final_3 (night 7), written with
        # every field of 2026-10-15, the contrastive ones at none, as the bases of the contrastive screens
        for name, recorded in (("cnn_3block_supcon_1", "e0c211fcced3"), ("cnn_final3_supcon_1", "7977141b6ba3")):
            _, summary = expand(json.loads((REPO / "sweeps" / f"{name}.json").read_text()))
            self.assertTrue(summary["base_config_hash"].startswith(recorded), name)

    def test_old_and_new_forms_of_a_job_hash_alike(self):
        from affect.runs.spec import config_hash, plain, validate_job

        jobs, _ = expand(json.loads((REPO / "sweeps" / "cnn_confirm_1.json").read_text()))
        new = jobs[0]["settings"]
        self.assertIn("speed_factors", new["augment"])
        self.assertEqual(new["train"]["eval_train"], "none")
        self.assertEqual(new["train"]["contrastive"], "none")
        self.assertEqual(new["train"]["external_test"], "none")
        self.assertEqual((new["train"]["aux_va"], new["train"]["aux_va_weight"], new["train"]["aux_va_loss"]),
                         ("none", 0.0, "none"))
        # the form of 2026-10-20 to 10-23: every field but the valence and activation ones (added 2026-10-24)
        newest_before = json.loads(json.dumps(new))
        for field in ("aux_va", "aux_va_weight", "aux_va_loss"):
            del newest_before["train"][field]
        # the form of 2026-10-15 to 10-19: also without external_test (added 2026-10-20)
        recent = json.loads(json.dumps(newest_before))
        del recent["train"]["external_test"]
        # the form of 2026-10-03 to 10-14: also without the contrastive ones (added 2026-10-15)
        late = json.loads(json.dumps(recent))
        for field in ("contrastive", "contrastive_weight", "contrastive_temperature", "contrastive_dim"):
            del late["train"][field]
        # the form of 2026-09-26 to 10-02: also without eval_train (added 2026-10-03)
        middle = json.loads(json.dumps(late))
        del middle["train"]["eval_train"]
        # the form before 2026-09-26
        old = json.loads(json.dumps(middle))
        for section, fields in (("augment", ("speed_factors", "mixup_alpha")),
                                ("train", ("aux_gender", "aux_gender_weight", "eval_segment_s", "eval_segment_hop_s")),
                                ("model", ("norm", "norm_groups", "norm_subbands", "activation"))):
            for field in fields:
                del old[section][field]
        for name, form in (("old", old), ("middle", middle), ("late", late), ("recent", recent),
                           ("newest_before", newest_before), ("new", new)):
            self.assertEqual(config_hash(validate_job(form)), jobs[0]["config_hash"], name)
            self.assertEqual(plain(validate_job(form)), new, name)
        changed = json.loads(json.dumps(new))
        changed["model"]["norm"] = "none"
        self.assertNotEqual(config_hash(validate_job(changed)), jobs[0]["config_hash"])
        measured = json.loads(json.dumps(new))
        measured["train"]["eval_train"] = "every_epoch"
        self.assertNotEqual(config_hash(validate_job(measured)), jobs[0]["config_hash"])
        contrasted = json.loads(json.dumps(new))
        contrasted["train"].update({"contrastive": "supcon", "contrastive_weight": 1.0, "contrastive_temperature": 0.1,
                                    "contrastive_dim": 128})
        self.assertNotEqual(config_hash(validate_job(contrasted)), jobs[0]["config_hash"])
        external = json.loads(json.dumps(new))
        external["train"]["external_test"] = "emofilm"
        self.assertNotEqual(config_hash(validate_job(external)), jobs[0]["config_hash"])
        rated = json.loads(json.dumps(new))
        rated["train"].update({"aux_va": "multitask", "aux_va_weight": 0.3, "aux_va_loss": "ccc"})
        self.assertNotEqual(config_hash(validate_job(rated)), jobs[0]["config_hash"])
        third = json.loads(json.dumps(new))
        del third["train"]["aux_va_loss"]            # neither the current form nor an older one
        with self.assertRaises(ValueError):
            validate_job(third)
        half = json.loads(json.dumps(new))
        del half["train"]["eval_segment_s"]          # neither the current form nor an older one
        with self.assertRaises(ValueError):
            validate_job(half)
        part = json.loads(json.dumps(new))
        del part["train"]["contrastive_dim"]         # neither, either
        with self.assertRaises(ValueError):
            validate_job(part)

    def test_counts_and_order(self):
        jobs, summary = expand(small_sweep())
        # base 1 config x 5 folds x 2 seeds; control 1 x 5; one factor 3 x 5; random 3 x 5
        self.assertEqual(len(jobs), 10 + 5 + 15 + 15)
        self.assertEqual(summary["configs"], 1 + 1 + 3 + 3)
        self.assertEqual([j["stage"] for j in jobs[:10]], ["baseline"] * 10)
        self.assertEqual({j["settings"]["train"]["control"] for j in jobs if j["stage"] == "controls"},
                         {"shuffle_train_labels"})

    def test_typo_in_an_override_stops_the_sweep(self):
        sweep = small_sweep()
        sweep["stages"][2]["factors"] = {"features.n_mel": [40]}
        with self.assertRaisesRegex(SweepError, "names no setting"):
            expand(sweep)

    def test_level_equal_to_base_stops_the_sweep(self):
        sweep = small_sweep()
        sweep["stages"][2]["factors"] = {"features.n_mels": [64]}
        with self.assertRaisesRegex(SweepError, "identical to the base"):
            expand(sweep)

    def test_wrong_type_stops_the_sweep(self):
        sweep = small_sweep()
        sweep["stages"][2]["factors"] = {"features.n_mels": ["64# comment"]}
        with self.assertRaises(ValueError):
            expand(sweep)

    def test_impossible_combination_stops_the_sweep(self):
        sweep = small_sweep()
        sweep["stages"][2]["factors"] = {"model.channels": [[8, 8, 8, 8, 8, 8, 8]]}   # 7 poolings of 64 bands
        with self.assertRaisesRegex(ValueError, "leave no bands"):
            expand(sweep)

    def test_grid_is_every_combination_but_the_base(self):
        sweep = small_sweep()
        sweep["stages"] = [
            {"name": "baseline", "kind": "base", "priority": 100, "seeds": [1]},
            {"name": "grid", "kind": "grid", "priority": 90, "seeds": [1, 2],
             "factors": {"features.n_mels": [40, 64], "model.pooling": ["mean", "stats", "max"],
                         "specaugment": [{"augment.freq_masks": 0, "augment.freq_mask_width": 0},
                                         {"augment.freq_masks": 2, "augment.freq_mask_width": 8}]}},
        ]
        jobs, summary = expand(sweep)
        grid = [s for s in summary["stages"] if s["name"] == "grid"][0]
        self.assertEqual(grid["configs"], 2 * 3 * 2 - 1)
        labels = {j["label"] for j in jobs if j["stage"] == "grid"}
        self.assertEqual(len(labels), 11)
        self.assertTrue(all(label.startswith("grid: ") for label in labels))
        self.assertEqual(len(jobs), (1 * 1 + 11 * 2) * len(sweep["folds"]))
        sweep["stages"][1]["factors"]["features.n_mels"] = [{"features.n_mels": 40, "model.pooling": "max"}]
        with self.assertRaisesRegex(SweepError, "two factors set"):
            expand(sweep)

    def test_folds_must_belong_to_the_split(self):
        sweep = small_sweep()
        sweep["base"]["data"]["split"] = "leave_one_speaker_out"
        with self.assertRaisesRegex(SweepError, "not folds of"):
            expand(sweep)
        sweep["folds"] = ["test_Ses01_F", "test_Ses01_M"]
        jobs, _ = expand(sweep)
        self.assertEqual({j["fold"] for j in jobs}, {"test_Ses01_F", "test_Ses01_M"})

    def test_one_data_protocol_per_sweep(self):
        sweep = small_sweep()
        sweep["stages"][1]["variants"]["five classes"] = {"data.label_set": "five"}
        with self.assertRaisesRegex(SweepError, "one data"):
            expand(sweep)

    def test_random_draws_are_reproducible_and_distinct(self):
        a, _ = expand(small_sweep())
        b, _ = expand(small_sweep())
        self.assertEqual([j["run_id"] for j in a], [j["run_id"] for j in b])
        random_hashes = {j["config_hash"] for j in a if j["stage"] == "random"}
        other_hashes = {j["config_hash"] for j in a if j["stage"] != "random"}
        self.assertEqual(len(random_hashes), 3)
        self.assertFalse(random_hashes & other_hashes)


class TestQueue(unittest.TestCase):
    def test_take_is_exclusive_and_ordered_by_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Queue(Path(tmp))
            q.put(pending_name(10, 0, "s", "low"), {"n": "low"})
            q.put(pending_name(100, 1, "s", "high"), {"n": "high"})
            first = q.take("w1")
            self.assertIn("high", first.name)
            second = q.take("w2")
            self.assertIn("low", second.name)
            self.assertIsNone(q.take("w3"))
            q.finish(first, "done")
            q.finish(second, "failed")
            self.assertEqual(q.counts("s"), {"pending": 0, "running": 0, "done": 1, "failed": 1})

    def test_requeue_running_restores_the_original_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Queue(Path(tmp))
            name = pending_name(50, 3, "s", "abc_test_S1_seed1")
            q.put(name, {})
            q.take("gpu0_1")
            self.assertEqual(q.requeue_running(), 1)
            self.assertTrue((Path(tmp) / "pending" / name).is_file())

    def test_duplicate_put_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Queue(Path(tmp))
            q.put("x.json", {})
            with self.assertRaises(FileExistsError):
                q.put("x.json", {})

    def test_stop_files_for_all_or_one_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Queue(Path(tmp))
            self.assertFalse(q.stop_requested())
            self.assertFalse(q.stop_requested("gpu0_1"))
            (Path(tmp) / "STOP_gpu0").touch()
            self.assertTrue(q.stop_requested("gpu0_1"))
            self.assertTrue(q.stop_requested("gpu0_3"))
            self.assertFalse(q.stop_requested("gpu1_1"), "another GPU's file stops nothing here")
            self.assertFalse(q.stop_requested("gpu2_2"))
            self.assertFalse(q.stop_requested(), "without a worker ID only queue/STOP counts")
            self.assertFalse(q.stop_requested("smoke"), "a worker ID not of the form gpu<g>_<n>")
            (Path(tmp) / "STOP_gpu0").unlink()
            self.assertFalse(q.stop_requested("gpu0_1"))
            (Path(tmp) / "STOP").touch()
            for worker in ("gpu0_1", "gpu3_3", "smoke", None):
                self.assertTrue(q.stop_requested(worker), worker)

    def test_requeue_failed_moves_only_that_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = Queue(Path(tmp))
            mine, other = pending_name(50, 1, "s", "a_test_S1_seed1"), pending_name(50, 2, "t", "b_test_S1_seed1")
            q.put(mine, {})
            q.put(other, {})
            q.finish(q.take("gpu0_1"), "failed")
            q.finish(q.take("gpu0_2"), "failed")
            self.assertEqual(q.requeue_failed("s"), 1)
            self.assertTrue((Path(tmp) / "pending" / mine).is_file())
            self.assertEqual(q.counts("t"), {"pending": 0, "running": 0, "done": 0, "failed": 1})


def cli(root: Path, *args) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    return subprocess.run([sys.executable, "-m", "affect.cli", *args], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)


class TestEnqueueAndStatus(unittest.TestCase):
    def test_enqueue_once_then_status_ranks_finished_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            (root / "sweeps").mkdir()
            sweep_path = root / "sweeps" / "test_sweep.json"
            sweep_path.write_text(json.dumps(small_sweep()))
            done = cli(root, "enqueue", "--sweep-file", str(sweep_path))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertEqual(json.loads(done.stdout)["jobs"], 45)
            pending = sorted((root / "queue" / "pending").glob("*.json"))
            self.assertEqual(len(pending), 45)
            self.assertTrue(pending[0].name.startswith("899_"))            # priority 100 first
            again = cli(root, "enqueue", "--sweep-file", str(sweep_path))
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("already enqueued", again.stderr)
            self.assertEqual(len(list((root / "queue" / "pending").glob("*.json"))), 45)

            # Fake finished runs: the baseline at 0.55 and 0.57, one factor level at 0.60,
            # the shuffled control at 0.26, and one failed run.
            jobs = [json.loads(p.read_text()) for p in pending]
            runs = root / "artifacts" / "runs" / "test_sweep"
            value = {("baseline", 1): 0.55, ("baseline", 2): 0.57, ("controls", 1): 0.26}
            for job in jobs:
                key = (job["stage"], job["seed"])
                if key in value or job["label"] == "features.n_mels=80":
                    ua = value.get(key, 0.60)
                    folder = runs / job["run_id"]
                    folder.mkdir(parents=True)
                    (folder / "spec.json").write_text(json.dumps(job))
                    (folder / "metrics_val.json").write_text(json.dumps(
                        {"best_val_ua": ua, "best_epoch": 5, "epochs_run": 15}))
                    (folder / "efficiency.json").write_text(json.dumps({"train_seconds": 120.0}))
            failed = runs / "x.failed-20260925T000000_000000Z"
            failed.mkdir()
            (failed / "error.txt").write_text("Traceback...\nRuntimeError: boom\n")
            report = cli(root, "status", "--sweep", "test_sweep", "--top", "5")
            self.assertEqual(report.returncode, 0, report.stderr[-2000:])
            out = report.stdout
            self.assertIn("baseline: 2 seeds, fold-mean validation UA mean 0.5600", out)
            self.assertIn("control shuffled seed 1: fold-mean validation UA 0.2600", out)
            self.assertIn("0.6000  vs base +0.0500  seed 1  one_factor: features.n_mels=80", out)
            self.assertIn("RuntimeError: boom", out)
            self.assertNotIn("ALERT", out)


class TestWorkerLoop(unittest.TestCase):
    """The worker's queue handling, with a stand-in for torch: jobs whose settings are broken
    fail, and the worker stops after the allowed number of failures in a row."""

    def test_stops_after_failures_in_a_row_then_requeue_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            q = Queue(root / "queue")
            for k in range(7):
                q.put(pending_name(50, k, "s", f"bad{k}_test_S1_seed1"),
                      {"sweep_id": "s", "run_id": f"bad{k}_test_S1_seed1", "stage": "x", "label": "broken",
                       "settings": {}})
            script = (
                "import sys, types\n"
                "import affect\n"
                "fake = types.ModuleType('torch')\n"
                "fake.__version__ = 'stand-in'\n"
                "fake.device = lambda *args: 'cuda:0'\n"
                "fake.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,\n"
                "                                  get_device_name=lambda i: 'stand-in GPU', empty_cache=lambda: None)\n"
                "sys.modules['torch'] = fake\n"
                "from affect.sweep.worker import run\n"
                "sys.exit(run('gpu0_1', 1, 0.0, 5))\n")
            env = dict(os.environ, PYTHONPATH=str(root / "src"))
            done = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, capture_output=True,
                                  text=True, timeout=300)
            self.assertEqual(done.returncode, 0, done.stderr[-3000:])
            self.assertEqual(done.stdout.count("JOB FAILED"), 5, done.stdout[-3000:])
            self.assertIn("5 jobs failed in a row", done.stdout)
            self.assertEqual(q.counts("s"), {"pending": 2, "running": 0, "done": 0, "failed": 5})
            failed_folders = sorted(p.name for p in (root / "artifacts" / "runs" / "s").iterdir())
            self.assertEqual(len(failed_folders), 5)
            self.assertTrue(all(".failed-" in name for name in failed_folders))
            again = cli(root, "requeue-failed", "--sweep", "s")
            self.assertEqual(again.returncode, 0, again.stderr[-2000:])
            self.assertIn("moved 5 failed jobs of s back to pending", again.stdout)
            self.assertEqual(q.counts("s"), {"pending": 7, "running": 0, "done": 0, "failed": 0})


if __name__ == "__main__":
    unittest.main()
