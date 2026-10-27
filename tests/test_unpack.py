"""scripts/unpack.sh in a throwaway home (added 2026-10-26): a code archive's SHA-256 is checked, the old code is moved
aside into affect_incoming/replaced/ (not deleted: on the node's NFS home a file held open cannot be deleted, which
stopped the install of 2026-10-24), the new code moved in, data/, artifacts/ and queue/ untouched, the staging folder
removed, the install recorded; the next install deletes the copy moved aside before; a wrong SHA-256 changes nothing."""
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from tests.support import REPO


def archive(folder: Path, incoming: Path, name: str, readme: str) -> None:
    build = folder / f"build_{name}"
    (build / "src" / "affect").mkdir(parents=True)
    (build / "src" / "affect" / "__init__.py").write_text(f"# {readme}\n")
    (build / "scripts").mkdir()
    (build / "scripts" / "x.sh").write_text("#!/usr/bin/env bash\r\necho x\r\n")
    (build / "README.md").write_text(readme + "\n")
    path = incoming / name
    with tarfile.open(path, "w:gz") as tar:
        for item in ("src", "scripts", "README.md"):
            tar.add(build / item, arcname=item)
    (incoming / f"{name}.sha256").write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n")


class TestUnpack(unittest.TestCase):
    def run_unpack(self, home: Path, name: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, HOME=str(home), PATH=os.environ.get("PATH", ""))
        return subprocess.run(["bash", str(home / "affect_incoming" / "unpack.sh"), name, str(home / "code")],
                              capture_output=True, text=True, env=env, timeout=60)

    def test_old_code_is_moved_aside_then_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            incoming = home / "affect_incoming"
            incoming.mkdir()
            shutil.copy(REPO / "scripts" / "unpack.sh", incoming / "unpack.sh")
            for name, readme in (("affect_code_20261026T000001Z.tar.gz", "one"),
                                 ("affect_code_20261026T000002Z.tar.gz", "two"),
                                 ("affect_code_20261026T000003Z.tar.gz", "three")):
                archive(home, incoming, name, readme)
            root = home / "code"     # not $HOME/affect, whose install unpack.sh refuses while gpu sessions run
            (root / "artifacts").mkdir(parents=True)
            (root / "artifacts" / "keep.txt").write_text("kept\n")
            (root / "queue" / "pending").mkdir(parents=True)
            done = self.run_unpack(home, "affect_code_20261026T000001Z.tar.gz")
            self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
            self.assertEqual((root / "README.md").read_text(), "one\n")
            self.assertEqual((root / "scripts" / "x.sh").read_text(), "#!/usr/bin/env bash\necho x\n", "CRs removed")
            done = self.run_unpack(home, "affect_code_20261026T000002Z.tar.gz")
            self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
            self.assertEqual((root / "README.md").read_text(), "two\n")
            aside = sorted((incoming / "replaced").iterdir())
            self.assertEqual(len(aside), 1)
            self.assertEqual((aside[0] / "README.md").read_text(), "one\n", "the old code moved aside, not deleted")
            self.assertEqual((root / "artifacts" / "keep.txt").read_text(), "kept\n")
            self.assertTrue((root / "queue" / "pending").is_dir())
            self.assertEqual(list(incoming.glob("unpacking_*")), [])
            done = self.run_unpack(home, "affect_code_20261026T000003Z.tar.gz")
            self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
            aside = sorted((incoming / "replaced").iterdir())
            self.assertEqual(len(aside), 1, "the copy moved aside before is deleted by the next install")
            self.assertEqual((aside[0] / "README.md").read_text(), "two\n")
            versions = (root / "CODE_VERSIONS.log").read_text().splitlines()
            self.assertEqual([line.split("\t")[2] for line in versions],
                             [f"affect_code_20261026T00000{k}Z.tar.gz" for k in (1, 2, 3)])
            # a wrong SHA-256 stops before anything changes
            (incoming / "affect_code_20261026T000001Z.tar.gz.sha256").write_text(
                "0" * 64 + "  affect_code_20261026T000001Z.tar.gz\n")
            done = self.run_unpack(home, "affect_code_20261026T000001Z.tar.gz")
            self.assertNotEqual(done.returncode, 0)
            self.assertEqual((root / "README.md").read_text(), "three\n")
            self.assertEqual(list(incoming.glob("unpacking_*")), [])


if __name__ == "__main__":
    unittest.main()
