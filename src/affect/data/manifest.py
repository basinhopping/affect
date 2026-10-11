"""Manifests: one row per file with its size, SHA-256 and where it came from.

A manifest is the data version. Its own SHA-256 is stamped into every run, so a result
traces to the exact bytes it was computed from.
"""
import csv
from pathlib import Path

from affect.hashing import sha256_file

FIELDS = ("path", "bytes", "sha256", "source", "source_path")


def write_manifest(path: Path, rows: list[dict]) -> None:
    """Write rows sorted by path. Refuses to overwrite an existing manifest."""
    if path.exists():
        raise FileExistsError(f"{path} exists; manifests are never overwritten")
    ordered = sorted(rows, key=lambda r: r["path"])
    paths = [r["path"] for r in ordered]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate paths in manifest rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row[k] for k in FIELDS})
    back = read_manifest(path)
    if [r["path"] for r in back] != paths:
        raise RuntimeError(f"{path}: re-read rows differ from the rows written")


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: columns {reader.fieldnames}, expected {list(FIELDS)}")
        rows = []
        for row in reader:
            row["bytes"] = int(row["bytes"])
            rows.append(row)
    return rows


def list_files(root: Path, exclude: frozenset) -> list[str]:
    """Every file under root as a forward-slash relative path, sorted. `exclude` holds
    relative paths to leave out (for example the manifest itself)."""
    out = []
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if rel not in exclude:
                out.append(rel)
    return sorted(out)


def verify_tree(root: Path, rows: list[dict], exclude: frozenset) -> dict:
    """Re-hash every file under root and compare with the manifest rows.

    Returns counts and the first examples of each kind of problem. `ok` is True only when
    the files on disk and the manifest match exactly: no missing, extra, resized or changed file.
    """
    expected = {r["path"]: r for r in rows}
    present = list_files(root, exclude)
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    size_mismatch, hash_mismatch = [], []
    n_checked = 0
    for rel in present:
        if rel not in expected:
            continue
        n_bytes, digest = sha256_file(root / rel)
        n_checked += 1
        if n_bytes != expected[rel]["bytes"]:
            size_mismatch.append(rel)
        elif digest != expected[rel]["sha256"]:
            hash_mismatch.append(rel)
    return {
        "ok": not (missing or extra or size_mismatch or hash_mismatch),
        "n_manifest_rows": len(rows),
        "n_files_present": len(present),
        "n_rehashed": n_checked,
        "missing": {"count": len(missing), "first": missing[:20]},
        "extra": {"count": len(extra), "first": extra[:20]},
        "size_mismatch": {"count": len(size_mismatch), "first": size_mismatch[:20]},
        "hash_mismatch": {"count": len(hash_mismatch), "first": hash_mismatch[:20]},
    }
