"""Hashes for files, settings and code. Used for manifests, run IDs and provenance."""
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> tuple[int, str]:
    """Return (number of bytes, SHA-256 hex) of a file, read in 1 MiB chunks."""
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


def canonical_json(obj) -> str:
    """JSON with sorted keys and no whitespace, so equal content always gives equal text.

    NaN and infinity are refused: they have no stable JSON form.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(canonical_json(obj))


def code_hash(package_dir: Path) -> str:
    """SHA-256 over every .py file of the package: sorted relative paths and file contents.

    Two trees give the same hash exactly when their Python files are identical.
    """
    digest = hashlib.sha256()
    files = sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    if not files:
        raise RuntimeError(f"no .py files under {package_dir}")
    for path in files:
        rel = path.relative_to(package_dir).as_posix()
        _, file_digest = sha256_file(path)
        digest.update(f"{rel}\0{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()
