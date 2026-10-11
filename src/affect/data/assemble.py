"""Assemble a dataset's training copy, and verify it anywhere.

    data/<dataset>/assembled/                what training reads, and what goes to the node
    data/<dataset>/assembled/MANIFEST.csv    path, bytes, SHA-256 and source of every file
    data/<dataset>/assembled/ASSEMBLY.json   sources, checks, and the manifest's own hash

Sources are named and given explicitly on the command line (for IEMOCAP: the release folder,
the collaborator's wav folder, the Phase 0 audit folder), and every path given is recorded.
Run the same command until it reports "finalized"; each call continues where the last stopped:

    1. list the source folders the plan needs
    2. plan: which file comes from where; checks that need only names and sizes
    3. copy every file, hashing the bytes as they are read
    4. read every copy back and check its hash
    5. list the assembled folder: exactly the planned files at the planned sizes
    6. checks on the copy (formats, parity with the Phase 0 index, counts, lineage), then the
       manifest and ASSEMBLY.json. Nothing is written into assembled/ unless every check passes.

`verify` re-reads every file of a finished copy and repeats the checks; the pilot does not
start on the node unless it passes (plan 5.5, stop condition 1).
"""
import datetime
import json
import platform
import socket
import time
from pathlib import Path

from affect import paths
from affect.data.manifest import read_manifest, write_manifest
from affect.data.transfer import Transfer
from affect.data.walk import ResumableWalk
from affect.hashing import code_hash, sha256_file
from affect.registry import DATASETS


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _dataset(name: str):
    return DATASETS.get(name).target()


def ingested_manifests(ds, required: tuple) -> dict:
    """Manifests of ingested raw folders: every required one, plus any other present.
    Each is checked against the hash recorded when its ingest was finalized."""
    out = {}
    folder = paths.manifests_dir(ds.name)
    names = set(required)
    if folder.is_dir():
        names |= {p.stem[len("raw_"):] for p in folder.glob("raw_*.json")}
    for name in sorted(names):
        csv_path, json_path = folder / f"raw_{name}.csv", folder / f"raw_{name}.json"
        if not json_path.is_file():
            if name in required:
                raise FileNotFoundError(f"{json_path} not found: ingest {name} first (README.md)")
            continue
        with open(json_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
        _, digest = sha256_file(csv_path)
        if recorded["manifest_sha256"] != digest:
            raise RuntimeError(f"{csv_path} changed since its ingest was finalized")
        out[name] = {"rows": read_manifest(csv_path),
                     "manifest": csv_path.relative_to(paths.repo_root()).as_posix(), "manifest_sha256": digest}
    return out


def assemble(dataset: str, sources: dict, budget_s: float) -> dict:
    start = time.monotonic()

    def remaining() -> float:
        return max(budget_s - (time.monotonic() - start), 0.0)

    ds = _dataset(dataset)
    root = paths.assembled_dir(dataset)
    if (root / "ASSEMBLY.json").exists():
        return {"phase": "already_assembled", "record": str(root / "ASSEMBLY.json")}
    if set(sources) != set(ds.assembly_sources):
        raise ValueError(f"sources must be exactly {list(ds.assembly_sources)}, got {sorted(sources)}")
    sources = {k: Path(v) for k, v in sources.items()}
    for name, path in sources.items():
        if not path.is_dir():
            raise NotADirectoryError(f"source {name}: {path}")
    work = paths.work_dir(dataset) / "assembly"
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "calls.log", "a", encoding="utf-8") as handle:
        handle.write(f"{_utc()}\t{socket.gethostname()}\t{json.dumps({k: str(v) for k, v in sources.items()})}\n")
    manifests = ingested_manifests(ds, ds.manifests_required)
    transfer = Transfer(work, root, sources)

    if not transfer.has_plan():
        listings = {}
        for name, folder in ds.assembly_listings(sources).items():
            walk = ResumableWalk(folder, work / f"list_{name}")
            if not walk.run(remaining()):
                return {"phase": f"1/6 listing {name}", **walk.progress()}
            listings[name] = walk.sizes()
        items, report = ds.plan_assembly(sources, listings, manifests)
        with open(work / "plan_checks.json", "x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        if not report["ok"]:
            raise RuntimeError(f"planning checks failed; see {work / 'plan_checks.json'}")
        transfer.write_plan(items)

    status = transfer.status()
    if status["n_verified"] < status["n_planned"]:
        status = transfer.run(remaining())
        if status["phase"] != "done":
            status["phase"] = {"copying": "3/6 copying", "verifying": "4/6 reading copies back"}[status["phase"]]
            return status

    planned = {item.dest: item.size for item in transfer.load_plan()}
    listing = ResumableWalk(root, work / "list_assembled")
    if not listing.is_complete():
        if not listing.run(remaining()):
            return {"phase": "5/6 listing the assembled folder", **listing.progress()}
        return {"phase": "5/6 listed", "next": "run the same command again for the checks and the manifest"}
    present = {k: v for k, v in listing.sizes().items() if k not in ds.meta_files}
    return _finalize(ds, root, work, transfer, planned, present, manifests, sources)


def _finalize(ds, root: Path, work: Path, transfer: Transfer, planned: dict, present: dict,
              manifests: dict, sources: dict) -> dict:
    rows = transfer.manifest_rows()
    with open(work / "plan_checks.json", encoding="utf-8") as handle:
        plan_report = json.load(handle)
    after = ds.check_after_copy(rows, manifests)
    report = ds.check_assembled(root, rows)
    report["checks"]["assembled_folder_holds_exactly_the_planned_files"] = present == planned
    report["checks"]["copies_agree_with_ingested_manifests"] = after["ok"]
    report["ok"] = all(report["checks"].values())
    if not report["ok"]:
        failed = work / f"assembly_failed_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S_%fZ}.json"
        with open(failed, "x", encoding="utf-8") as handle:
            json.dump({"checks": report, "after_copy": after}, handle, indent=1)
        raise RuntimeError(f"assembly checks failed; see {failed}")
    manifest_path = root / "MANIFEST.csv"
    if manifest_path.exists():
        existing = read_manifest(manifest_path)
        if [(r["path"], r["bytes"], r["sha256"]) for r in existing] != \
                [(r["path"], r["bytes"], r["sha256"]) for r in sorted(rows, key=lambda r: r["path"])]:
            raise RuntimeError(f"{manifest_path} exists and differs from this assembly")
    else:
        write_manifest(manifest_path, rows)
    _, manifest_sha = sha256_file(manifest_path)
    record = {
        "dataset": ds.name,
        "finalized_utc": _utc(),
        "manifest_sha256": manifest_sha,
        "n_files": len(rows),
        "bytes": sum(r["bytes"] for r in rows),
        "files_by_source": {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})},
        "sources_as_seen_by_the_copying_machine": {k: str(v) for k, v in sources.items()},
        "ingested_manifests_used": {k: {"manifest": v["manifest"], "manifest_sha256": v["manifest_sha256"]}
                                    for k, v in manifests.items()},
        "plan_checks": plan_report,
        "after_copy_checks": after,
        "checks": report,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "code_sha256": code_hash(paths.repo_root() / "src" / "affect"),
    }
    with open(root / "ASSEMBLY.json", "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    return {"phase": "finalized", "manifest_sha256": manifest_sha, "n_files": len(rows),
            "bytes": record["bytes"], "n_utterances": report["n_utterances"],
            "n_filled_from_collaborator": report["n_filled_from_collaborator"],
            "files_by_source": record["files_by_source"]}


def verify(dataset: str, state_name: str, budget_s: float) -> dict:
    """Re-read every file of an assembled copy against its manifest and repeat the checks.

    Resumable, because re-reading 10,342 files through the PC bridge takes longer than one
    command may run there. State lives in data/<dataset>/_work/verify_<state_name>/; a new
    state name starts a fresh verification (scripts/check.sh uses a new one every time).
    Returns a progress dict, or the final report with `ok`.
    """
    import csv

    if not state_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"state name {state_name!r}: use letters, digits, _ and -")
    start = time.monotonic()

    def remaining() -> float:
        return max(budget_s - (time.monotonic() - start), 0.0)

    ds = _dataset(dataset)
    root = paths.assembled_dir(dataset)
    with open(root / "ASSEMBLY.json", encoding="utf-8") as handle:
        record = json.load(handle)
    manifest_path = root / "MANIFEST.csv"
    _, manifest_sha = sha256_file(manifest_path)
    rows = read_manifest(manifest_path)
    state = paths.work_dir(dataset) / f"verify_{state_name}"
    state.mkdir(parents=True, exist_ok=True)
    log_path = state / "rehash.csv"
    done = {}
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                done[r["path"]] = r
    n_new = 0
    for row in rows:
        if row["path"] in done:
            continue
        if n_new > 0 and remaining() <= 0:
            return {"phase": "1/3 re-reading files", "reread": len(done), "of": len(rows)}
        target = root / row["path"]
        n_bytes, digest = sha256_file(target) if target.is_file() else (-1, "")
        entry = {"path": row["path"], "bytes": n_bytes, "sha256": digest}
        new_file = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"), lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow(entry)
        done[row["path"]] = {k: str(v) for k, v in entry.items()}
        n_new += 1
    listing = ResumableWalk(root, state / "list")
    if not listing.is_complete():
        if not listing.run(remaining()):
            return {"phase": "2/3 listing the folder", **listing.progress()}
        return {"phase": "2/3 listed", "next": "run the same command again for the checks"}
    expected = {r["path"]: r for r in rows}
    present = {k: v for k, v in listing.sizes().items() if k not in ds.meta_files}
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    size_mismatch = sorted(p for p in expected if p in present and int(done[p]["bytes"]) != expected[p]["bytes"])
    hash_mismatch = sorted(p for p in expected if p in present and p not in size_mismatch
                           and done[p]["sha256"] != expected[p]["sha256"])
    tree = {
        "ok": not (missing or extra or size_mismatch or hash_mismatch),
        "n_manifest_rows": len(rows), "n_files_present": len(present), "n_rehashed": len(done),
        "missing": {"count": len(missing), "first": missing[:20]},
        "extra": {"count": len(extra), "first": extra[:20]},
        "size_mismatch": {"count": len(size_mismatch), "first": size_mismatch[:20]},
        "hash_mismatch": {"count": len(hash_mismatch), "first": hash_mismatch[:20]},
    }
    checks = ds.check_assembled(root, rows)
    manifests = ingested_manifests(ds, ())
    if manifests:
        lineage = ds.check_after_copy(rows, manifests)
    else:
        lineage = {"ok": True, "checks": {}, "manifests_available": [],
                   "note": "no ingested raw manifests on this machine; lineage was checked when the copy "
                           "was assembled (ASSEMBLY.json, after_copy_checks)"}
    result = {
        "phase": "done",
        "dataset": dataset,
        "state": state.relative_to(paths.repo_root()).as_posix(),
        "manifest_sha256": manifest_sha,
        "manifest_matches_assembly_record": manifest_sha == record["manifest_sha256"],
        "assembly_record_checks_ok": record["checks"]["ok"] is True,
        "files": tree,
        "checks": checks,
        "lineage": lineage,
    }
    result["ok"] = bool(result["manifest_matches_assembly_record"] and result["assembly_record_checks_ok"]
                        and tree["ok"] and checks["ok"] and lineage["ok"])
    return result
