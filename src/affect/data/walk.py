"""List every file under a folder with its size, across several calls with a time budget.

Listing a large folder through the PC's shared-folder bridge costs a few milliseconds per file
(about 240 s for the 83,151 files of the IEMOCAP release, measured 2026-09-25), longer than one
command may run there. This walker scans one folder at a time and records each finished folder,
so a later call continues where the last stopped. State lives in its own folder:

    files.tsv       relative path and size of every file found so far
    dirs_done.tsv   every folder already scanned, with its subfolders
    complete.json   written once the whole tree has been scanned
"""
import json
import os
import time
from pathlib import Path


class WalkError(RuntimeError):
    pass


def _join(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


class ResumableWalk:
    def __init__(self, root: Path, state_dir: Path):
        self.root = Path(root)
        self.state_dir = state_dir
        self.files_path = state_dir / "files.tsv"
        self.dirs_path = state_dir / "dirs_done.tsv"
        self.complete_path = state_dir / "complete.json"

    def _load_dirs(self) -> dict:
        done = {}
        if self.dirs_path.exists():
            with open(self.dirs_path, encoding="utf-8") as handle:
                for line in handle:
                    rel, subdirs = line.rstrip("\n").split("\t")
                    done[rel] = json.loads(subdirs)
        return done

    def is_complete(self) -> bool:
        return self.complete_path.exists()

    def progress(self) -> dict:
        done = self._load_dirs()
        n_files = 0
        if self.files_path.exists():
            with open(self.files_path, encoding="utf-8") as handle:
                n_files = sum(1 for _ in handle)
        return {"folders_scanned": len(done), "file_rows": n_files, "complete": self.is_complete()}

    def run(self, budget_s: float) -> bool:
        """Scan until the tree is done (returns True) or the budget is spent (returns False).
        At least one folder is scanned per call, so every call makes progress."""
        if self.is_complete():
            return True
        start = time.monotonic()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        done = self._load_dirs()
        pending, seen = [""], set()
        discovered = []
        while pending:
            rel = pending.pop()
            if rel in seen:
                continue
            seen.add(rel)
            if rel in done:
                pending.extend(_join(rel, s) for s in done[rel])
            else:
                discovered.append(rel)
        n_scanned = 0
        while discovered:
            if n_scanned > 0 and time.monotonic() - start > budget_s:
                return False
            rel = discovered.pop()
            files, subdirs = [], []
            with os.scandir(self.root / rel if rel else self.root) as entries:
                for entry in entries:
                    if "\t" in entry.name or "\n" in entry.name:
                        raise WalkError(f"unsupported character in a file name under {rel!r}: {entry.name!r}")
                    if entry.is_symlink():
                        raise WalkError(f"symlink: {_join(rel, entry.name)}")
                    if entry.is_dir():
                        subdirs.append(entry.name)
                    elif entry.is_file():
                        files.append((_join(rel, entry.name), entry.stat().st_size))
                    else:
                        raise WalkError(f"neither file nor folder: {_join(rel, entry.name)}")
            with open(self.files_path, "a", encoding="utf-8") as handle:
                for path, size in files:
                    handle.write(f"{path}\t{size}\n")
            with open(self.dirs_path, "a", encoding="utf-8") as handle:
                handle.write(f"{rel}\t{json.dumps(sorted(subdirs))}\n")
            discovered.extend(_join(rel, s) for s in subdirs)
            n_scanned += 1
        sizes = self.sizes()
        with open(self.complete_path, "x", encoding="utf-8") as handle:
            json.dump({"n_files": len(sizes), "bytes": sum(sizes.values())}, handle)
        return True

    def sizes(self) -> dict:
        """Relative path -> size for every file found. Rows repeated by an interrupted call
        must agree; a disagreement means the tree changed while it was being listed."""
        out = {}
        if self.files_path.exists():
            with open(self.files_path, encoding="utf-8") as handle:
                for line in handle:
                    path, size = line.rstrip("\n").split("\t")
                    size = int(size)
                    if path in out and out[path] != size:
                        raise WalkError(f"{path} was listed twice with different sizes")
                    out[path] = size
        return out
