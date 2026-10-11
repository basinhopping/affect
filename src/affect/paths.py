"""Where things live, relative to the repository root.

Every location is fixed by convention inside the repository, on both machines:
the PC (C:\\Affective Computing\\affect) and the GPU node (~/affect).
No code accepts an output path from a caller, so no run can write somewhere unexpected.
"""
from pathlib import Path


def repo_root() -> Path:
    """The repository root: the folder that holds src/affect."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "src" / "affect" / "__init__.py").is_file():
        raise RuntimeError(f"expected src/affect under {root}; is the package being run from a copy?")
    return root


def data_dir(dataset: str) -> Path:
    """data/<dataset>/: everything for one dataset."""
    return repo_root() / "data" / dataset


def raw_dir(dataset: str) -> Path:
    """data/<dataset>/raw/: verified copies of the files as received. Never edited."""
    return data_dir(dataset) / "raw"


def manifests_dir(dataset: str) -> Path:
    """data/<dataset>/manifests/: one manifest per ingested raw folder."""
    return data_dir(dataset) / "manifests"


def assembled_dir(dataset: str) -> Path:
    """data/<dataset>/assembled/: what training reads, and what goes to the node."""
    return data_dir(dataset) / "assembled"


def work_dir(dataset: str) -> Path:
    """data/<dataset>/_work/: resumable state of ingests and assemblies (logs only)."""
    return data_dir(dataset) / "_work"


def artifacts_dir() -> Path:
    """artifacts/: results. Written on the node, pulled to the PC."""
    return repo_root() / "artifacts"


def checks_dir() -> Path:
    """artifacts/checks/: one folder per data or environment check."""
    return artifacts_dir() / "checks"


def queue_dir() -> Path:
    """queue/: pending/, running/, done/, failed/ job files (node only)."""
    return repo_root() / "queue"


def sweeps_dir() -> Path:
    """artifacts/sweeps/<sweep_id>/: what a sweep was, as enqueued."""
    return artifacts_dir() / "sweeps"


def runs_dir(sweep_id: str) -> Path:
    """artifacts/runs/<sweep_id>/<run_id>/: one folder per job."""
    return artifacts_dir() / "runs" / sweep_id


def analysis_dir() -> Path:
    """artifacts/analysis/<sweep_id>_<utc>/: one folder per validation analysis of a sweep."""
    return artifacts_dir() / "analysis"


def selections_dir() -> Path:
    """artifacts/selections/<name>/: the recorded choice of a configuration, before any test result is read."""
    return artifacts_dir() / "selections"


def test_reports_dir() -> Path:
    """artifacts/test_reports/<selection>_<utc>/: test results of a selection."""
    return artifacts_dir() / "test_reports"
