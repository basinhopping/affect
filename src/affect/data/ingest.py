"""Ingest: copy a received folder into data/<dataset>/raw/<name>/ and record its manifest.

The received folder is only read. Run the same command until it reports "finalized"; each
call continues where the last stopped. The phases:

    1. list the received folder (every file and its size) and fix the copy plan
    2. copy every file, hashing the bytes as they are read
    3. read every copy back and check its hash
    4. list the received folder again: nothing may have changed during the copy
    5. list the copy: exactly the planned files, at the planned sizes
    6. write the manifest and a summary

Outputs:
    data/<dataset>/raw/<name>/                    the copy
    data/<dataset>/manifests/raw_<name>.csv       path, bytes, SHA-256 of every file
    data/<dataset>/manifests/raw_<name>.json      what was copied, from where, when, by which code
    data/<dataset>/_work/ingest_<name>/           resumable state and logs
"""
import datetime
import json
import platform
import re
import socket
import time
from pathlib import Path

from affect import paths
from affect.data.manifest import write_manifest
from affect.data.transfer import CopyItem, Transfer
from affect.data.walk import ResumableWalk
from affect.hashing import code_hash, sha256_file

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def ingest(dataset: str, name: str, source: Path, budget_s: float) -> dict:
    if not NAME_RE.match(name):
        raise ValueError(f"name {name!r}: use letters, digits, _ . - only")
    start = time.monotonic()

    def remaining() -> float:
        return max(budget_s - (time.monotonic() - start), 0.0)

    dest = paths.raw_dir(dataset) / name
    work = paths.work_dir(dataset) / f"ingest_{name}"
    manifest_csv = paths.manifests_dir(dataset) / f"raw_{name}.csv"
    summary_json = paths.manifests_dir(dataset) / f"raw_{name}.json"
    if summary_json.exists():
        return {"phase": "already_ingested", "summary": str(summary_json)}
    source = Path(source)
    if not source.is_dir():
        raise NotADirectoryError(source)
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "calls.log", "a", encoding="utf-8") as handle:
        handle.write(f"{_utc()}\t{socket.gethostname()}\t{source}\n")

    transfer = Transfer(work, dest, {"received": source})
    if not transfer.has_plan():
        listing = ResumableWalk(source, work / "list_source")
        if not listing.run(remaining()):
            return {"phase": "1/6 listing the received folder", **listing.progress()}
        sizes = listing.sizes()
        if not sizes:
            raise RuntimeError(f"{source} holds no files")
        transfer.write_plan([CopyItem(rel, "received", rel, size) for rel, size in sorted(sizes.items())])

    status = transfer.status()
    if status["n_verified"] < status["n_planned"]:
        status = transfer.run(remaining())
        if status["phase"] != "done":
            status["phase"] = {"copying": "2/6 copying", "verifying": "3/6 reading copies back"}[status["phase"]]
            return status

    planned = {item.dest: item.size for item in transfer.load_plan()}
    again = ResumableWalk(source, work / "list_source_again")
    if not again.run(remaining()):
        return {"phase": "4/6 listing the received folder again", **again.progress()}
    if again.sizes() != planned:
        now = again.sizes()
        changed = sorted(k for k in set(now) | set(planned) if now.get(k) != planned.get(k))
        raise RuntimeError(f"{source} changed during the ingest: {changed[:10]}")
    copy_listing = ResumableWalk(dest, work / "list_copy")
    if not copy_listing.run(remaining()):
        return {"phase": "5/6 listing the copy", **copy_listing.progress()}
    if copy_listing.sizes() != planned:
        now = copy_listing.sizes()
        differ = sorted(k for k in set(now) | set(planned) if now.get(k) != planned.get(k))
        raise RuntimeError(f"{dest} differs from the plan: {differ[:10]}")

    rows = transfer.manifest_rows()
    write_manifest(manifest_csv, rows)
    _, manifest_sha = sha256_file(manifest_csv)
    summary = {
        "dataset": dataset,
        "name": name,
        "copy": dest.relative_to(paths.repo_root()).as_posix(),
        "manifest": manifest_csv.relative_to(paths.repo_root()).as_posix(),
        "manifest_sha256": manifest_sha,
        "n_files": len(rows),
        "bytes": sum(planned.values()),
        "source_as_seen_by_the_copying_machine": str(source),
        "checks": {
            "every_copy_read_back_and_its_hash_matched": True,
            "received_folder_unchanged_during_the_ingest": True,
            "copy_holds_exactly_the_planned_files_at_the_planned_sizes": True,
        },
        "finalized_utc": _utc(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "code_sha256": code_hash(paths.repo_root() / "src" / "affect"),
    }
    with open(summary_json, "x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    return {"phase": "finalized", "summary": str(summary_json), "n_files": len(rows),
            "bytes": summary["bytes"], "manifest_sha256": manifest_sha}
