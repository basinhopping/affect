"""Copy a planned set of files, then verify every copy by reading it back, resumably.

Used for ingesting raw folders and for assembling a dataset. The job can stop at a time
budget and continue in a later call, because the shell on the PC side allows only a few
minutes per command. All state lives in a work folder:

    plan.csv         the files to copy, fixed at the first call
    copy_log.csv     one row per finished copy: bytes and the SHA-256 of the bytes read
    verify_log.csv   one row per copy read back from the destination and hashed again

Nothing is deleted or renamed. A partial file from an interrupted copy is overwritten when
that copy is redone. A read-back that disagrees with the copy log stops the job with an
error, because it means something changed or broke between writing and reading.
"""
import csv
import dataclasses
import hashlib
import time
from pathlib import Path


class TransferError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class CopyItem:
    dest: str          # forward-slash path relative to the destination root
    source: str        # label of the source root, a key of `source_roots`
    source_path: str   # forward-slash path relative to that source root
    size: int          # bytes when planned


PLAN_FIELDS = ("dest", "source", "source_path", "size")
COPY_FIELDS = ("dest", "bytes", "sha256")
VERIFY_FIELDS = ("dest", "bytes", "sha256", "ok")


def _append(path: Path, fields: tuple, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def _read_log(path: Path, fields: tuple) -> dict:
    """Latest row per dest."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise TransferError(f"{path}: unexpected columns {reader.fieldnames}")
        out = {}
        for row in reader:
            out[row["dest"]] = row
    return out


def _copy_hashing(src: Path, dst: Path) -> tuple[int, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    n_bytes = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            fout.write(chunk)
            n_bytes += len(chunk)
    return n_bytes, digest.hexdigest()


def _hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    n_bytes = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            n_bytes += len(chunk)
    return n_bytes, digest.hexdigest()


class Transfer:
    def __init__(self, work_dir: Path, dest_root: Path, source_roots: dict):
        self.work_dir = work_dir
        self.dest_root = dest_root
        self.source_roots = {k: Path(v) for k, v in source_roots.items()}
        self.plan_path = work_dir / "plan.csv"
        self.copy_log = work_dir / "copy_log.csv"
        self.verify_log = work_dir / "verify_log.csv"

    # plan -----------------------------------------------------------------------------
    def has_plan(self) -> bool:
        return self.plan_path.exists()

    def write_plan(self, items: list) -> None:
        if self.plan_path.exists():
            raise TransferError(f"{self.plan_path} exists; a plan is written once")
        dests = [i.dest for i in items]
        if len(set(dests)) != len(dests):
            raise TransferError("two planned files share a destination")
        for item in items:
            if item.source not in self.source_roots:
                raise TransferError(f"unknown source {item.source!r} for {item.dest}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with open(self.plan_path, "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS, lineterminator="\n")
            writer.writeheader()
            for item in sorted(items, key=lambda i: i.dest):
                writer.writerow(dataclasses.asdict(item))

    def load_plan(self) -> list:
        with open(self.plan_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PLAN_FIELDS:
                raise TransferError(f"{self.plan_path}: unexpected columns {reader.fieldnames}")
            return [CopyItem(r["dest"], r["source"], r["source_path"], int(r["size"])) for r in reader]

    # run ------------------------------------------------------------------------------
    def status(self) -> dict:
        items = self.load_plan()
        copied = _read_log(self.copy_log, COPY_FIELDS)
        verified = _read_log(self.verify_log, VERIFY_FIELDS)
        n_verified = sum(
            1 for i in items
            if i.dest in verified and verified[i.dest]["ok"] == "1"
            and i.dest in copied and verified[i.dest]["sha256"] == copied[i.dest]["sha256"]
        )
        return {
            "n_planned": len(items),
            "bytes_planned": sum(i.size for i in items),
            "n_copied": sum(1 for i in items if i.dest in copied),
            "n_verified": n_verified,
        }

    def run(self, budget_s: float) -> dict:
        """Copy, then verify, until done or the budget is spent. Returns a status dict whose
        `phase` is "copying", "verifying" or "done"."""
        start = time.monotonic()
        items = self.load_plan()
        copied = _read_log(self.copy_log, COPY_FIELDS)
        verified = _read_log(self.verify_log, VERIFY_FIELDS)

        n_new = 0
        for item in items:
            dst = self.dest_root / item.dest
            # A logged copy is not looked at again here: statting tens of thousands of files
            # through the PC bridge would spend the whole budget. The read-back below checks
            # every copy, and the ingest lists the finished copy against the plan.
            if item.dest in copied:
                continue
            # At least one file per call, so every call makes progress whatever the budget.
            if n_new > 0 and time.monotonic() - start > budget_s:
                return {"phase": "copying", "copied_this_call": n_new, **self.status()}
            if item.dest in verified:
                # This file is being written again, so an earlier read-back no longer counts.
                _append(self.verify_log, VERIFY_FIELDS, {"dest": item.dest, "bytes": 0, "sha256": "", "ok": "0"})
            src = self.source_roots[item.source] / item.source_path
            n_bytes, digest = _copy_hashing(src, dst)
            if n_bytes != item.size:
                raise TransferError(
                    f"{src}: {n_bytes} bytes read, {item.size} when planned. The source changed."
                )
            row = {"dest": item.dest, "bytes": n_bytes, "sha256": digest}
            _append(self.copy_log, COPY_FIELDS, row)
            copied[item.dest] = {k: str(v) for k, v in row.items()}
            n_new += 1

        verified = _read_log(self.verify_log, VERIFY_FIELDS)
        n_checked = 0
        for item in items:
            want = copied[item.dest]
            got = verified.get(item.dest)
            if got is not None and got["ok"] == "1" and got["sha256"] == want["sha256"]:
                continue
            if (n_new > 0 or n_checked > 0) and time.monotonic() - start > budget_s:
                return {"phase": "verifying", "copied_this_call": n_new, "verified_this_call": n_checked,
                        **self.status()}
            n_bytes, digest = _hash(self.dest_root / item.dest)
            ok = n_bytes == int(want["bytes"]) and digest == want["sha256"]
            _append(self.verify_log, VERIFY_FIELDS,
                    {"dest": item.dest, "bytes": n_bytes, "sha256": digest, "ok": "1" if ok else "0"})
            n_checked += 1
            if not ok:
                raise TransferError(
                    f"{item.dest}: read back {n_bytes} bytes sha256 {digest[:12]}, copied "
                    f"{want['bytes']} bytes sha256 {want['sha256'][:12]}"
                )
        return {"phase": "done", "copied_this_call": n_new, "verified_this_call": n_checked, **self.status()}

    def manifest_rows(self) -> list[dict]:
        """Manifest rows for a finished transfer. Refuses unless every file is verified."""
        items = self.load_plan()
        copied = _read_log(self.copy_log, COPY_FIELDS)
        verified = _read_log(self.verify_log, VERIFY_FIELDS)
        rows = []
        for item in items:
            c, v = copied.get(item.dest), verified.get(item.dest)
            if c is None or v is None or v["ok"] != "1" or v["sha256"] != c["sha256"]:
                raise TransferError(f"{item.dest} is not copied and verified yet")
            rows.append({"path": item.dest, "bytes": int(c["bytes"]), "sha256": c["sha256"],
                         "source": item.source, "source_path": item.source_path})
        return rows
