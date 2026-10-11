"""The resumable listing: finishes under any budget and agrees with a direct listing."""
import os
import tempfile
import unittest
from pathlib import Path

from affect.data.walk import ResumableWalk, WalkError


def direct(root: Path) -> dict:
    out = {}
    for folder, _, files in os.walk(root):
        for name in files:
            p = Path(folder) / name
            out[p.relative_to(root).as_posix()] = p.stat().st_size
    return out


class TestWalk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "tree"
        for i in range(12):
            p = self.root / f"a{i % 3}" / f"b{i % 2}" / f"f{i}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * i)
        (self.root / "top.txt").write_text("t")
        (self.root / "empty_dir").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_budget_finishes_one_folder_per_call(self):
        walk = ResumableWalk(self.root, Path(self.tmp.name) / "state")
        calls = 0
        while not walk.run(0.0):
            calls += 1
            self.assertLess(calls, 100)
        self.assertGreater(calls, 3)
        self.assertEqual(walk.sizes(), direct(self.root))
        self.assertTrue(walk.run(0.0))

    def test_rows_repeated_by_an_interrupted_call_are_harmless(self):
        state = Path(self.tmp.name) / "state"
        walk = ResumableWalk(self.root, state)
        walk.run(0.0)
        with open(state / "files.tsv", encoding="utf-8") as handle:
            rows = handle.read()
        with open(state / "files.tsv", "a", encoding="utf-8") as handle:
            handle.write(rows)
        while not walk.run(0.0):
            pass
        self.assertEqual(walk.sizes(), direct(self.root))

    def test_disagreeing_rows_are_refused(self):
        state = Path(self.tmp.name) / "state"
        walk = ResumableWalk(self.root, state)
        while not walk.run(0.0):
            pass
        with open(state / "files.tsv", "a", encoding="utf-8") as handle:
            handle.write("top.txt\t999\n")
        with self.assertRaises(WalkError):
            walk.sizes()


if __name__ == "__main__":
    unittest.main()
