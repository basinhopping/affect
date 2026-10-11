"""Output folders: <id>.tmp until complete, never overwritten, failures kept."""
import json
import tempfile
import unittest
from pathlib import Path

from affect.runs.store import AlreadyComplete, OutputFolder


class TestOutputFolder(unittest.TestCase):
    def test_complete_renames_and_blocks_a_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            folder = OutputFolder(parent, "run1").begin()
            self.assertTrue((parent / "run1.tmp").is_dir())
            folder.write_json("spec.json", {"a": 1})
            final = folder.complete()
            self.assertEqual(json.loads((final / "spec.json").read_text()), {"a": 1})
            self.assertFalse((parent / "run1.tmp").exists())
            with self.assertRaises(AlreadyComplete):
                OutputFolder(parent, "run1").begin()

    def test_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = OutputFolder(Path(tmp), "run1").begin()
            folder.write_json("m.json", {})
            with self.assertRaises(FileExistsError):
                folder.write_json("m.json", {})

    def test_failure_is_kept_with_the_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            folder = OutputFolder(parent, "run1").begin()
            try:
                raise ValueError("boom")
            except ValueError as err:
                failed = folder.fail(err)
            self.assertTrue(failed.name.startswith("run1.failed-"))
            self.assertIn("boom", (failed / "error.txt").read_text())
            self.assertFalse((parent / "run1").exists())

    def test_abandoned_tmp_is_kept_as_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / "run1.tmp").mkdir()
            (parent / "run1.tmp" / "partial.txt").write_text("x")
            OutputFolder(parent, "run1").begin()
            kept = [p.name for p in parent.iterdir() if "abandoned" in p.name]
            self.assertEqual(len(kept), 1)

    def test_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = OutputFolder(Path(tmp), "run1").begin()
            with self.assertRaises(ValueError):
                folder.path("../outside.txt")


if __name__ == "__main__":
    unittest.main()
