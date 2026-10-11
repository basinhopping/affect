"""A job queue made of files (plan 5.1).

    queue/pending/   waiting jobs, one JSON file each; names sort by priority, then order
    queue/running/   taken by a worker: the file is moved here with the worker's name added
    queue/done/      finished (or found already complete)
    queue/failed/    failed; the run's own folder holds the error
    queue/STOP       if this file exists, workers finish their current job and exit
    queue/STOP_gpu<g>  the same for the workers of GPU g only (worker IDs gpu<g>_<n>; added 2026-10-03
                     for the weekend schedule, scripts/gpu_schedule.sh)

A worker takes a job by renaming its file from pending/ to running/. A rename is atomic, so two
workers can never take the same job: the second one's rename finds no file and it moves on.
"""
import json
import os
from pathlib import Path


def pending_name(priority: int, seq: int, sweep_id: str, run_id: str) -> str:
    return f"{999 - priority:03d}_{seq:06d}__{sweep_id}__{run_id}.json"


class Queue:
    STATES = ("pending", "running", "done", "failed")

    def __init__(self, root: Path):
        self.root = root
        for state in self.STATES:
            (root / state).mkdir(parents=True, exist_ok=True)

    def folder(self, state: str) -> Path:
        return self.root / state

    def put(self, name: str, spec: dict) -> None:
        """Add a job. Refuses a name already pending (a sweep is enqueued once; see enqueue)."""
        with open(self.folder("pending") / name, "x", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=1)

    def take(self, worker_id: str):
        for path in sorted(self.folder("pending").glob("*.json")):
            target = self.folder("running") / f"{path.stem}.{worker_id}.json"
            try:
                os.rename(path, target)
            except FileNotFoundError:
                continue
            return target
        return None

    def finish(self, claimed: Path, state: str) -> Path:
        if state not in ("done", "failed"):
            raise ValueError(state)
        target = self.folder(state) / claimed.name
        os.rename(claimed, target)
        return target

    def requeue_running(self) -> int:
        """Move every running job back to pending. Only when no worker is running."""
        moved = 0
        for path in sorted(self.folder("running").glob("*.json")):
            original = path.name.rsplit(".", 2)[0] + ".json"
            os.rename(path, self.folder("pending") / original)
            moved += 1
        return moved

    def requeue_failed(self, sweep_id: str) -> int:
        """Move one sweep's failed jobs back to pending, under their original names. Their failed
        run folders stay where they are, renamed .failed-<utc>, so the error is kept."""
        moved = 0
        for path in sorted(self.folder("failed").glob(f"*__{sweep_id}__*.json")):
            original = path.name.rsplit(".", 2)[0] + ".json"
            os.rename(path, self.folder("pending") / original)
            moved += 1
        return moved

    def stop_requested(self, worker_id: str | None = None) -> bool:
        """queue/STOP stops every worker; queue/STOP_gpu<g> stops the workers named gpu<g>_<n>."""
        if (self.root / "STOP").exists():
            return True
        if worker_id is None:
            return False
        gpu = worker_id.split("_", 1)[0]
        return gpu.startswith("gpu") and gpu[3:].isdigit() and (self.root / f"STOP_{gpu}").exists()

    def counts(self, sweep_id: str | None) -> dict:
        out = {}
        for state in self.STATES:
            files = self.folder(state).glob("*.json")
            out[state] = sum(1 for f in files if sweep_id is None or f"__{sweep_id}__" in f.name)
        return out
