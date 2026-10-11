"""How a result was produced, stamped into its folder at write time.

Records the code (hash of every package file, the pushed archive if known, the git commit if
the tree is a repository), the machine, Python and package versions, and the environment
variables that change behaviour on the node. Package versions are read from installed
metadata, so recording them imports nothing heavy.
"""
import datetime
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys

from affect import paths
from affect.hashing import code_hash

PACKAGES = ("torch", "numpy", "scipy", "pandas", "soundfile", "librosa", "transformers",
            "huggingface_hub", "spikingjelly", "snntorch", "papermill", "matplotlib",
            "scikit-learn", "opensmile", "tpot", "deap", "xgboost")
ENV_VARS = ("CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY",
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HOME", "PYTHONPATH")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_commit() -> str | None:
    root = paths.repo_root()
    if not (root / ".git").exists():
        return None
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def pushed_code_version() -> str | None:
    """Last line of CODE_VERSIONS.log, written on the node by scripts/unpack.sh."""
    log = paths.repo_root() / "CODE_VERSIONS.log"
    if not log.is_file():
        return None
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else None


def collect(entry_point: str) -> dict:
    return {
        "entry_point": entry_point,
        "argv": sys.argv,
        "utc": utc_now(),
        "code_sha256": code_hash(paths.repo_root() / "src" / "affect"),
        "pushed_code_version": pushed_code_version(),
        "git_commit": git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": package_versions(),
        "environment": {name: os.environ.get(name) for name in ENV_VARS},
    }
