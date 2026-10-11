"""The MPS fix: importing affect clears the variables, and refuses if torch came first uncleared."""
import os
import subprocess
import sys
import unittest

from tests.support import REPO


def run(code: str, env_updates: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUDA_MPS_")}
    env.update(env_updates)
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)


class TestImportGuard(unittest.TestCase):
    def test_import_clears_the_variables(self):
        done = run("import os, affect; print(repr(os.environ['CUDA_MPS_PIPE_DIRECTORY']), "
                   "repr(os.environ['CUDA_MPS_LOG_DIRECTORY']))",
                   {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "'' ''")

    def test_torch_first_without_the_fix_raises(self):
        done = run("import sys, types; sys.modules['torch'] = types.ModuleType('torch'); import affect", {})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("torch was imported before affect", done.stderr)

    def test_torch_first_with_the_fix_already_exported_is_allowed(self):
        # scripts/common.sh exports the empty variables before Python starts.
        done = run("import sys, types; sys.modules['torch'] = types.ModuleType('torch'); import affect",
                   {"CUDA_MPS_PIPE_DIRECTORY": "", "CUDA_MPS_LOG_DIRECTORY": ""})
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
