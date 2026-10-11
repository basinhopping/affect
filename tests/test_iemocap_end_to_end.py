"""Ingest, assemble and verify a synthetic IEMOCAP through the command line, in a throwaway
copy of the repository, so the real data/ folder is never touched.

Checks the known answer (which files are filled, the counts, the folds) and that a wrong
Phase 0 index or a changed byte is caught.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import synthetic_iemocap
from tests.support import REPO


class Repo:
    def __init__(self, root: Path):
        self.root = root
        shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))

    def cli(self, *args) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONPATH=str(self.root / "src"))
        return subprocess.run([sys.executable, "-m", "affect.cli", *args], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=300)

    def until(self, final_phase: str, *args) -> dict:
        for _ in range(400):
            done = self.cli(*args)
            status = json.loads(done.stdout) if done.returncode in (0, 3) else None
            if status is None:
                raise AssertionError(f"{args[0]} failed:\n{done.stderr[-3000:]}")
            if status["phase"] == final_phase:
                if done.returncode != 0:
                    raise AssertionError(f"{args[0]} reached {final_phase} with exit code {done.returncode}")
                return status
            # verify-data says "not finished, run again" with exit code 3; ingest and assemble with 0
            expected = 3 if args[0] == "verify-data" else 0
            if done.returncode != expected:
                raise AssertionError(f"{args[0]} in phase {status['phase']} exited {done.returncode}, not {expected}")
        raise AssertionError(f"{args[0]} did not reach {final_phase}")

    def python(self, code: str) -> str:
        env = dict(os.environ, PYTHONPATH=str(self.root / "src"))
        done = subprocess.run([sys.executable, "-c", code], cwd=self.root, env=env, capture_output=True,
                              text=True, timeout=300)
        if done.returncode != 0:
            raise AssertionError(done.stderr[-3000:])
        return done.stdout


def ingest_both(repo: Repo, src: dict) -> None:
    repo.until("finalized", "ingest", "--dataset", "iemocap", "--name", "IEMOCAP_full_release",
               "--source", str(src["release"]), "--max-seconds", "0")
    repo.until("finalized", "ingest", "--dataset", "iemocap", "--name", "collaborator_wav",
               "--source", str(src["collaborator"]), "--max-seconds", "0")


def assemble_args(src: dict) -> tuple:
    return ("assemble", "--dataset", "iemocap",
            "--source", f"IEMOCAP_full_release={src['release']}",
            "--source", f"collaborator_wav={src['collaborator']}",
            "--source", f"phase0_audit={src['index'].parent}",
            "--max-seconds", "0")


class TestEndToEnd(unittest.TestCase):
    def test_ingest_assemble_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = synthetic_iemocap.build(tmp / "received")
            repo = Repo(tmp / "repo")
            ingest_both(repo, src)
            again = repo.cli("ingest", "--dataset", "iemocap", "--name", "collaborator_wav",
                             "--source", str(src["collaborator"]), "--max-seconds", "0")
            self.assertEqual(json.loads(again.stdout)["phase"], "already_ingested")

            final = repo.until("finalized", *assemble_args(src))
            self.assertEqual(final["n_utterances"], 17)
            self.assertEqual(final["n_filled_from_collaborator"], 1)

            root = repo.root / "data" / "iemocap" / "assembled"
            record = json.loads((root / "ASSEMBLY.json").read_text())
            self.assertTrue(record["checks"]["ok"])
            self.assertEqual(record["files_by_source"],
                             {"phase0_audit": 1, "IEMOCAP_full_release": 16 + 6 + 6, "collaborator_wav": 1})
            self.assertTrue(record["plan_checks"]["ok"])
            self.assertEqual(record["plan_checks"]["n_release_sentence_wavs_empty"], 1)
            self.assertEqual(record["after_copy_checks"]["n_collaborator_files_identical_to_release"], 4)
            self.assertTrue(record["after_copy_checks"]["release_manifest_available"])
            self.assertTrue(record["after_copy_checks"]["checks"]["release_copies_match_the_release_manifest"])
            four = record["checks"]["four_class"]
            self.assertEqual(four["total"], 12)
            self.assertEqual(four["by_class"], {"neu": 5, "ang": 5, "sad": 1, "hap+exc": 1})
            filled = root / "audio" / "Session5" / "Ses05M_script01_1" / "Ses05M_script01_1_M000.wav"
            self.assertEqual(filled.read_bytes(),
                             (src["collaborator"] / "Ses05M_script01_1" / "Ses05M_script01_1_M000.wav").read_bytes())

            out = repo.python(
                "from affect import catalog\n"
                "from affect.registry import DATASETS\n"
                "from affect.settings import validate\n"
                "from affect.data.splits import assign, folds_hash\n"
                "catalog.load_all()\n"
                "e = DATASETS.get('iemocap')\n"
                "s = validate(e.settings_class, {'label_set': 'four_merged', 'subset': 'all', "
                "'split': 'leave_one_session_out'}, 'data')\n"
                "ds = e.target()\n"
                "u = ds.utterances(s)\n"
                "folds = ds.folds(s, u)\n"
                "print(len(u), [len(assign(u, f)['test']) for f in folds], ds.class_order(s), "
                "sorted({x.label for x in u}), len(folds_hash(u, folds)))\n"
            )
            self.assertEqual(out.strip(), "12 [2, 2, 2, 2, 4] ('neu', 'hap+exc', 'ang', 'sad') "
                                          "['ang', 'hap+exc', 'neu', 'sad'] 64")

            ok = repo.until("done", "verify-data", "--dataset", "iemocap", "--state", "first", "--max-seconds", "0")
            self.assertTrue(ok["ok"])
            self.assertTrue(ok["lineage"]["release_copies_match_the_release_manifest"])

            data = bytearray(filled.read_bytes())
            data[-1] ^= 0xFF
            filled.write_bytes(bytes(data))
            for _ in range(400):
                bad = repo.cli("verify-data", "--dataset", "iemocap", "--state", "second", "--max-seconds", "0")
                if json.loads(bad.stdout)["phase"] == "done":
                    break
            self.assertEqual(bad.returncode, 1)
            self.assertFalse(json.loads(bad.stdout)["ok"])
            saved = sorted((repo.root / "artifacts" / "checks").iterdir())
            self.assertEqual(len(saved), 2)
            report = json.loads((saved[-1] / "report.json").read_text())
            self.assertEqual(report["files"]["hash_mismatch"]["count"], 1)

    def test_wrong_reference_index_stops_the_assembly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = synthetic_iemocap.build(tmp / "received")
            text = src["index"].read_text(encoding="utf-8")
            src["index"].write_text(text.replace(",ang,2.5,", ",sad,2.5,", 1), encoding="utf-8")
            repo = Repo(tmp / "repo")
            ingest_both(repo, src)
            with self.assertRaisesRegex(AssertionError, "assembly checks failed"):
                repo.until("finalized", *assemble_args(src))
            self.assertFalse((repo.root / "data" / "iemocap" / "assembled" / "ASSEMBLY.json").exists())
            failed = list((repo.root / "data" / "iemocap" / "_work" / "assembly").glob("assembly_failed_*.json"))
            self.assertEqual(len(failed), 1)
            parity = json.loads(failed[0].read_text())["checks"]["parity"]
            self.assertEqual(parity["mismatch_by_field"]["label"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
