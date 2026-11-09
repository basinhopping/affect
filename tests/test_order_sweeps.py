"""scripts/order_sweeps.sh in a throwaway root (added 2026-10-29): the pending files of the sweeps given run one sweep
after another within each priority, in the order given; other sweeps' files, every file's content and the priority
part of every name stay as they are; a second run changes nothing; a sweep not enqueued, or with 10,000 jobs or more,
is left as it is."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from affect.sweep.queue import pending_name
from tests.support import REPO


def enqueue(root: Path, sweep_id: str, priority: int, n_jobs: int, jobs_csv_rows: int | None = None) -> dict:
    """Stand-in enqueue: the jobs.csv a real enqueue writes and one pending file per job, named as enqueue names it."""
    folder = root / "artifacts" / "sweeps" / sweep_id
    folder.mkdir(parents=True)
    rows = n_jobs if jobs_csv_rows is None else jobs_csv_rows
    lines = ["seq,run_id,stage,label,priority,fold,seed,config_hash"]
    lines += [f"{seq},cfg_{sweep_id}_{seq},record,x,{priority},test_S1,{seq},cfg" for seq in range(rows)]
    (folder / "jobs.csv").write_text("\n".join(lines) + "\n")
    contents = {}
    for seq in range(n_jobs):
        run_id = f"c0ffee{seq:06d}_test_S{seq % 5 + 1}_seed{seq}"
        text = f'{{"sweep_id": "{sweep_id}", "run_id": "{run_id}", "priority": {priority}}}\n'
        (root / "queue" / "pending" / pending_name(priority, seq, sweep_id, run_id)).write_text(text)
        contents[(sweep_id, run_id)] = text
    return contents


def run_script(root: Path, *sweeps: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, AFFECT_ROOT=str(root))
    return subprocess.run(["bash", str(root / "scripts" / "order_sweeps.sh"), *sweeps], capture_output=True,
                          text=True, env=env, timeout=60)


def pending(root: Path) -> list:
    return sorted(p.name for p in (root / "queue" / "pending").iterdir())


def owner(name: str) -> str:
    return name.split("__")[1]


@unittest.skipIf(shutil.which("bash") is None, "needs bash")
class TestOrderSweeps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "scripts").mkdir()
        shutil.copy(REPO / "scripts" / "order_sweeps.sh", self.root / "scripts" / "order_sweeps.sh")
        for state in ("pending", "running", "done", "failed"):
            (self.root / "queue" / state).mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sweeps_run_one_after_another_within_their_priority(self):
        contents = {}
        contents |= enqueue(self.root, "night_b", 147, 5)
        contents |= enqueue(self.root, "night_a", 147, 5)
        contents |= enqueue(self.root, "night_snn", 147, 3)
        contents |= enqueue(self.root, "warm_a", 146, 4)
        contents |= enqueue(self.root, "warm_snn", 146, 2)
        contents |= enqueue(self.root, "other_147", 147, 3)      # not in the list: keeps its places
        contents |= enqueue(self.root, "first_150", 150, 2)      # a higher priority: still first
        contents |= enqueue(self.root, "old_90", 90, 3)          # a lower priority: still last
        before = pending(self.root)
        interleaved = [owner(n) for n in before if n.startswith("852_")]
        self.assertEqual(interleaved[:5], ["night_a", "night_b", "night_snn", "other_147", "night_a"],
                         "the stand-in reproduces the queue's turn-taking")
        done = run_script(self.root, "night_b", "night_a", "night_snn", "warm_a", "warm_snn")
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        after = pending(self.root)
        self.assertEqual(len(after), len(before))
        self.assertEqual([owner(n) for n in after],
                         ["first_150"] * 2 + ["other_147"] * 3 + ["night_b"] * 5 + ["night_a"] * 5 + ["night_snn"] * 3
                         + ["warm_a"] * 4 + ["warm_snn"] * 2 + ["old_90"] * 3)
        for name in after:
            sweep, run_id = name.split("__")[1], name.split("__")[2][:-len(".json")]
            self.assertEqual((self.root / "queue" / "pending" / name).read_text(), contents[(sweep, run_id)])
        prefixes = {owner(n): n[:3] for n in after}
        self.assertEqual(prefixes, {"first_150": "849", "other_147": "852", "night_b": "852", "night_a": "852",
                                    "night_snn": "852", "warm_a": "853", "warm_snn": "853", "old_90": "909"})
        self.assertIn("852_010003__night_b__", " ".join(after), "place 1, job 3")
        self.assertIn("853_050001__warm_snn__", " ".join(after), "place 5, job 1")
        self.assertIn("night_b: place 1, 5 renamed, 0 already in place, 0 taken by a worker meanwhile", done.stdout)
        self.assertIn("ORDERED", done.stdout)
        self.assertIn("19 pending files renamed; first pending 849_000000__first_150__", done.stdout)
        again = run_script(self.root, "night_b", "night_a", "night_snn", "warm_a", "warm_snn")
        self.assertEqual(again.returncode, 0, again.stderr + again.stdout)
        self.assertEqual(pending(self.root), after, "a second run changes nothing")
        self.assertIn("night_b: place 1, 0 renamed, 5 already in place", again.stdout)
        self.assertIn("0 pending files renamed", again.stdout)

    def test_jobs_already_taken_and_sweeps_left_as_they_are(self):
        enqueue(self.root, "night_a", 147, 4)
        enqueue(self.root, "huge", 147, 2, jobs_csv_rows=10000)
        pending_dir = self.root / "queue" / "pending"
        loose = pending_name(147, 0, "loose", "r0")                   # pending, but never enqueued here
        (pending_dir / loose).write_text("{}\n")
        first = pending_name(147, 0, "night_a", "c0ffee000000_test_S1_seed0")
        os.rename(pending_dir / first, self.root / "queue" / "running" / (first[:-5] + ".gpu2_1.json"))
        done = run_script(self.root, "loose", "night_a", "huge")
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        self.assertIn("loose: not enqueued", done.stdout)
        self.assertIn("huge: 10000 jobs, 10,000 or more, left as it is", done.stdout)
        self.assertIn("night_a: place 2, 3 renamed", done.stdout)
        names = pending(self.root)
        self.assertIn(loose, names)
        self.assertEqual(sum(1 for n in names if "__huge__" in n and n.startswith("852_0000")), 2)
        self.assertEqual(sorted(n for n in names if "__night_a__" in n),
                         [f"852_02000{k}__night_a__c0ffee00000{k}_test_S{k + 1}_seed{k}.json" for k in (1, 2, 3)])
        self.assertEqual(len(list((self.root / "queue" / "running").iterdir())), 1, "the running job is untouched")

    def test_no_sweep_given(self):
        done = run_script(self.root)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("usage", done.stdout)


if __name__ == "__main__":
    unittest.main()
