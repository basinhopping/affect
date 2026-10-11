"""The resumable verified copy: progress under any budget, and every failure caught."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from affect.data.manifest import read_manifest, verify_tree, write_manifest
from affect.data.transfer import CopyItem, Transfer, TransferError


def make_source(root: Path, n: int) -> list:
    items = []
    for i in range(n):
        rel = f"d{i % 3}/f{i}.bin"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytes((i * 7 + k) % 256 for k in range(100 + i))
        path.write_bytes(data)
        items.append(CopyItem(rel, "src", rel, len(data)))
    return items


def run_to_end(t: Transfer, limit: int) -> list:
    phases = []
    for _ in range(limit):
        status = t.run(0.0)
        phases.append(status["phase"])
        if status["phase"] == "done":
            return phases
    raise AssertionError(f"not done after {limit} calls: {phases[-5:]}")


class TestTransfer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.items = make_source(self.src, 7)
        self.t = Transfer(self.root / "work", self.root / "dest", {"src": self.src})
        self.t.write_plan(self.items)

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_budget_still_finishes_and_hashes_match(self):
        phases = run_to_end(self.t, 50)
        self.assertIn("copying", phases)
        self.assertIn("verifying", phases)
        rows = self.t.manifest_rows()
        for row in rows:
            data = (self.src / row["source_path"]).read_bytes()
            self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual((self.root / "dest" / row["path"]).read_bytes(), data)

    def test_corruption_between_copy_and_verify_is_caught(self):
        for _ in range(20):
            if self.t.status()["n_copied"] == len(self.items):
                break
            self.t.run(0.0)
        victim = self.root / "dest" / self.items[3].dest
        data = bytearray(victim.read_bytes())
        data[0] ^= 0xFF
        victim.write_bytes(bytes(data))
        with self.assertRaises(TransferError):
            run_to_end(self.t, 50)

    def test_truncated_copy_is_caught(self):
        for _ in range(20):
            if self.t.status()["n_copied"] == len(self.items):
                break
            self.t.run(0.0)
        (self.root / "dest" / self.items[2].dest).write_bytes(b"short")
        with self.assertRaisesRegex(TransferError, "read back 5 bytes"):
            run_to_end(self.t, 50)

    def test_interrupted_copy_is_redone(self):
        # A file written but never logged (the call died mid-copy) is copied again.
        self.t.run(0.0)
        first = self.root / "dest" / self.items[0].dest
        self.assertTrue(first.is_file())
        partial = self.root / "dest" / self.items[1].dest
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"half")
        run_to_end(self.t, 50)
        self.assertEqual(partial.read_bytes(), (self.src / self.items[1].source_path).read_bytes())
        self.assertEqual(self.t.status()["n_verified"], len(self.items))

    def test_source_change_is_caught(self):
        (self.src / self.items[0].source_path).write_bytes(b"different length")
        with self.assertRaisesRegex(TransferError, "source changed"):
            run_to_end(self.t, 50)

    def test_plan_is_written_once(self):
        with self.assertRaises(TransferError):
            self.t.write_plan(self.items)

    def test_manifest_round_trip_and_tree_verification(self):
        run_to_end(self.t, 50)
        rows = self.t.manifest_rows()
        manifest = self.root / "MANIFEST.csv"
        write_manifest(manifest, rows)
        with self.assertRaises(FileExistsError):
            write_manifest(manifest, rows)
        back = read_manifest(manifest)
        self.assertEqual([r["sha256"] for r in back], [r["sha256"] for r in sorted(rows, key=lambda r: r["path"])])
        dest = self.root / "dest"
        self.assertTrue(verify_tree(dest, back, frozenset())["ok"])
        (dest / "extra.txt").write_text("x")
        victim = dest / back[0]["path"]
        victim.write_bytes(b"\0" * victim.stat().st_size)
        report = verify_tree(dest, back, frozenset())
        self.assertFalse(report["ok"])
        self.assertEqual(report["extra"]["count"], 1)
        self.assertEqual(report["hash_mismatch"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
