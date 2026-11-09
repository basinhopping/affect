"""scripts/smoke_enqueue.sh and scripts/smoke_enqueue_files.sh (changed 2026-11-02) in a throwaway root with a stand-in
scripts/affect.sh: a sweep is enqueued when its own smoke test exits 0 and prints SMOKE OK, whatever the newest
logs/smoke_*.log says (another smoke test may be running at the same time, its log newer and without a verdict yet); a failed one is reported as NOT ENQUEUED
with its own log, which holds its output; a sweep already enqueued is skipped; a path that is not absolute, or whose
sweep_id is not its name, is refused."""
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tests.support import REPO

STAND_IN = r"""#!/usr/bin/env bash
# stand-in for scripts/affect.sh: smoke passes unless the sweep's name contains "bad"; it also leaves a newer
# logs/smoke_*.log without a verdict yet, as another smoke test running meanwhile would
cd "$(dirname "$0")/.."
if [ "$1" = smoke ]; then
    s=$(basename "$3" .json)
    sleep 1
    echo "sweep someone_elses: still running" > "logs/smoke_$(date -u +%Y%m%dT%H%M%S%N).log"
    echo "sweep $s: stand-in"
    case "$s" in *bad*) echo "SMOKE FAILED"; exit 1 ;; *) echo "SMOKE OK"; exit 0 ;; esac
fi
if [ "$1" = enqueue ]; then
    s=$(basename "$3" .json)
    mkdir -p "artifacts/sweeps/$s"
    printf ' "sweep_id": "%s",\n "configs": 1,\n "jobs": 5,\n' "$s"
fi
"""


@unittest.skipIf(shutil.which("bash") is None, "needs bash")
class TestSmokeEnqueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "affect"
        (self.root / "scripts").mkdir(parents=True)
        for name in ("smoke_enqueue.sh", "smoke_enqueue_files.sh"):
            shutil.copy(REPO / "scripts" / name, self.root / "scripts" / name)
        (self.root / "scripts" / "affect.sh").write_text(STAND_IN)
        for d in ("sweeps", "artifacts/sweeps", "queue/pending", "logs", "extra"):
            (self.root / d).mkdir(parents=True)
        for s in ("good_one", "bad_one", "done_one"):
            for folder in ("sweeps", "extra"):
                (self.root / folder / f"{s}.json").write_text('{"sweep_id": "%s"}\n' % s)
        (self.root / "extra" / "named_wrong.json").write_text('{"sweep_id": "other"}\n')
        (self.root / "artifacts" / "sweeps" / "done_one").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args) -> str:
        env = dict(os.environ, AFFECT_ROOT=str(self.root), SMOKE_GPU="3")
        r = subprocess.run(["bash", *args], capture_output=True, text=True, env=env, timeout=120, cwd=str(self.root))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def check(self, out: str):
        self.assertIn('"sweep_id": "good_one"', out)
        self.assertTrue((self.root / "artifacts" / "sweeps" / "good_one").is_dir())
        self.assertFalse((self.root / "artifacts" / "sweeps" / "bad_one").exists(),
                         "a failed smoke test is not enqueued, whatever the newest smoke log says")
        line = next(x for x in out.splitlines() if x.startswith("NOT ENQUEUED: bad_one, see "))
        own = self.root / line.split(", see ", 1)[1]
        self.assertIn("SMOKE FAILED", own.read_text(), "the log named is the test's own")
        self.assertIn("already enqueued: done_one", out)
        self.assertIn("ALL DONE", out)

    def test_by_id(self):
        out = self.run_script(str(self.root / "scripts" / "smoke_enqueue.sh"), "bad_one", "good_one", "done_one")
        self.check(out)

    def test_by_file(self):
        extra = self.root / "extra"
        out = self.run_script(str(self.root / "scripts" / "smoke_enqueue_files.sh"), str(extra / "bad_one.json"),
                              str(extra / "good_one.json"), str(extra / "done_one.json"), "extra/good_one.json",
                              str(extra / "named_wrong.json"))
        self.check(out)
        self.assertIn("NOT ENQUEUED: extra/good_one.json, not an absolute path", out)
        self.assertIn("its sweep_id is not its name", out)


if __name__ == "__main__":
    unittest.main()
