"""Enqueue a sweep: expand it, record it, and write one queue file per job.

    artifacts/sweeps/<sweep_id>/sweep_file.json   the sweep file exactly as given
    artifacts/sweeps/<sweep_id>/summary.json      stages, configuration and job counts
    artifacts/sweeps/<sweep_id>/jobs.csv          run_id, stage, label, fold, seed, config hash
    queue/pending/*.json                          one job each

A sweep is enqueued once: if artifacts/sweeps/<sweep_id> exists, nothing happens. A changed
sweep gets a new sweep_id, so results of different sweeps never mix.
"""
import csv
import io
import json
from pathlib import Path

from affect import paths
from affect.hashing import code_hash, sha256_file
from affect.runs.provenance import collect
from affect.runs.store import AlreadyComplete, OutputFolder
from affect.sweep.design import expand
from affect.sweep.queue import Queue, pending_name


def load_sweep_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def enqueue(sweep_file: Path) -> dict:
    sweep = load_sweep_file(sweep_file)
    jobs, summary = expand(sweep)
    sweep_id = sweep["sweep_id"]
    try:
        folder = OutputFolder(paths.sweeps_dir(), sweep_id).begin()
    except AlreadyComplete:
        raise FileExistsError(f"sweep {sweep_id} was already enqueued (artifacts/sweeps/{sweep_id}); "
                              "give a changed sweep a new sweep_id") from None
    try:
        _, file_sha = sha256_file(sweep_file)
        summary.update({"sweep_file": str(sweep_file), "sweep_file_sha256": file_sha,
                        "code_sha256": code_hash(paths.repo_root() / "src" / "affect")})
        folder.write_json("sweep_file.json", sweep)
        folder.write_json("summary.json", summary)
        folder.write_json("provenance.json", collect("affect.cli enqueue"))
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["seq", "run_id", "stage", "label", "priority", "fold", "seed", "config_hash"])
        for seq, job in enumerate(jobs):
            writer.writerow([seq, job["run_id"], job["stage"], job["label"], job["priority"], job["fold"],
                             job["seed"], job["config_hash"]])
        folder.write_text("jobs.csv", buffer.getvalue())
        folder.complete()
    except BaseException as err:
        folder.fail(err)
        raise
    queue = Queue(paths.queue_dir())
    for seq, job in enumerate(jobs):
        queue.put(pending_name(job["priority"], seq, sweep_id, job["run_id"]), job)
    return summary
