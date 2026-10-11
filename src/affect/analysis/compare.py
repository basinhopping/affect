"""Compare the runs of one sweep with every other copy of the same runs (added 2026-09-27).

A run ID is the configuration hash, the fold and the seed, so two folders with the same run ID
in different sweeps trained the same settings on the same data from the same seed. Training here
is deterministic: night 1's repeated baseline runs and night 2's 160 repeats of night-1 runs were
identical (VERIFICATION_LOG entry 12, the night-2 report). So every copy must agree exactly.
cnn_regression_1 uses this to show that the code of 2026-09-26 reproduces the runs made by the
code of nights 1 to 3.

For every complete run folder of the sweep, against every complete folder of the same name under
artifacts/runs/<other sweep>/:
  predictions_val.csv   byte for byte (SHA-256)
  metrics_val.json      as JSON, after removing each epoch's train_seconds and eval_seconds
                        (wall-clock times, which differ between any two runs)
  predictions_test.csv  byte for byte (SHA-256). Test files are only hashed, never parsed, so no
  metrics_test.json     test value is read or shown (plan 3.5).
spec.json, efficiency.json, provenance.json and log.txt are not compared: they hold the sweep's
name, times and versions, which differ by design. Each copy's code version (code_sha256 in its
provenance.json) is reported, and a run counts as reproduced only when at least one copy was made
by other code.

Verdicts: SAME (every run of the sweep complete, identical to every copy, and each with a copy made
by other code), DIFFERENT (some copy differs, or some run has no copy made by other code), and
INCOMPLETE (no difference so far, but some runs of the sweep are not finished: run it again later).
"""
import csv
import json

from affect import paths
from affect.hashing import sha256_file

HASHED = ("predictions_val.csv", "predictions_test.csv", "metrics_test.json")
TEST_FILES = ("predictions_test.csv", "metrics_test.json")
TIMING_KEYS = ("train_seconds", "eval_seconds", "train_eval_seconds")   # the last with eval_train (2026-10-03)


def _read_json(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _val_metrics(folder) -> dict:
    """metrics_val.json without the per-epoch wall-clock times."""
    metrics = _read_json(folder / "metrics_val.json")
    for curve in metrics.get("curves", []):
        for key in TIMING_KEYS:
            curve.pop(key, None)
    return metrics


def _code_of(folder):
    try:
        return _read_json(folder / "provenance.json").get("code_sha256")
    except (OSError, ValueError):
        return None


def differences(a, b, where: str) -> list:
    """Where two parsed JSON values differ. Numbers must match exactly, in value and type."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in b:
                out.append(f"{where}.{key} only in this run")
            elif key not in a:
                out.append(f"{where}.{key} only in the copy")
            else:
                out += differences(a[key], b[key], f"{where}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} entries against {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{where}[{i}]")
        return out
    if type(a) is type(b) and a == b:
        return []
    return [f"{where}: {a!r} against {b!r}"]


def expected_runs(sweep_id: str) -> list:
    """The sweep's run IDs, from its record (artifacts/sweeps/<sweep_id>/jobs.csv)."""
    with open(paths.sweeps_dir() / sweep_id / "jobs.csv", newline="", encoding="utf-8") as handle:
        return [row["run_id"] for row in csv.DictReader(handle)]


def compare_run(sweep_id: str, run_id: str, others: list, shown: int) -> dict:
    """One run of the sweep against its copies in the other sweeps. shown: how many metric
    differences to list per copy."""
    mine = paths.runs_dir(sweep_id) / run_id
    code = _code_of(mine)
    hashes = {name: sha256_file(mine / name)[1] for name in HASHED}
    metrics = _val_metrics(mine)
    copies = []
    for other in others:
        folder = paths.runs_dir(other) / run_id      # only a finished run has exactly this name
        if not folder.is_dir():
            continue
        problems = []
        for name in HASHED:
            if not (folder / name).is_file():
                problems.append(f"{name} missing in the copy")
            elif sha256_file(folder / name)[1] != hashes[name]:
                problems.append(f"{name} differs" + (" (compared by SHA-256; not read)" if name in TEST_FILES else ""))
        found = differences(metrics, _val_metrics(folder), "metrics_val")
        problems += found[:shown] + ([f"... and {len(found) - shown} more metric differences"] if len(found) > shown else [])
        other_code = _code_of(folder)
        copies.append({"sweep": other, "code_sha256": other_code, "same_code": other_code == code,
                       "identical": not problems, "problems": problems})
    return {"run_id": run_id, "code_sha256": code, "copies": copies,
            "n_copies": len(copies), "n_other_code": sum(not c["same_code"] for c in copies),
            "identical": all(c["identical"] for c in copies)}


def compare_sweep(sweep_id: str, shown: int) -> dict:
    expected = expected_runs(sweep_id)
    runs_root = paths.artifacts_dir() / "runs"
    others = sorted(p.name for p in runs_root.iterdir() if p.is_dir() and p.name != sweep_id)
    results, missing = [], []
    for run_id in expected:
        if (paths.runs_dir(sweep_id) / run_id).is_dir():
            results.append(compare_run(sweep_id, run_id, others, shown))
        else:
            missing.append(run_id)
    differing = [r["run_id"] for r in results if not r["identical"]]
    without_other_code = [r["run_id"] for r in results if r["n_other_code"] == 0]
    if differing or without_other_code:
        verdict = "DIFFERENT"
    elif missing:
        verdict = "INCOMPLETE"
    else:
        verdict = "SAME"
    versions = {}
    for r in results:
        for c in r["copies"]:
            versions.setdefault(c["code_sha256"], set()).add(c["sweep"])
    return {"sweep_id": sweep_id, "verdict": verdict, "n_expected": len(expected), "n_complete": len(results),
            "missing": missing, "differing": differing, "without_other_code": without_other_code,
            "n_copies": sum(r["n_copies"] for r in results),
            "n_other_code": sum(r["n_other_code"] for r in results),
            "code_versions": {str(k): sorted(v) for k, v in versions.items()},
            "runs": results}


def render(report: dict) -> str:
    lines = [f"compare-runs {report['sweep_id']}: {report['n_complete']} of {report['n_expected']} runs complete, "
             f"{report['n_copies']} copies in other sweeps ({report['n_other_code']} made by other code)"]
    for code, sweeps in sorted(report["code_versions"].items()):
        lines.append(f"  copies made by code {code[:12]}: {', '.join(sweeps)}")
    lines.append("  run                                copies  by other code  identical")
    for r in report["runs"]:
        lines.append(f"  {r['run_id']:<34} {r['n_copies']:>6}  {r['n_other_code']:>13}  "
                     f"{'all' if r['identical'] else 'NO'}")
        differ = [c for c in r["copies"] if not c["identical"]]
        if differ:
            first = differ[0]
            for problem in first["problems"]:
                lines.append(f"    vs {first['sweep']} (code {str(first['code_sha256'])[:12]}): {problem}")
            if len(differ) > 1:
                lines.append(f"    also differs from the copies in {', '.join(c['sweep'] for c in differ[1:])}")
    if report["missing"]:
        lines.append(f"  not finished yet: {len(report['missing'])} runs, for example {report['missing'][0]}")
    for run_id in report["without_other_code"]:
        lines.append(f"  {run_id}: no copy made by other code, so nothing shows it reproduces earlier code")
    if report["verdict"] == "SAME":
        lines.append(f"SAME: all {report['n_complete']} runs are identical to every copy ({report['n_copies']} "
                     f"copies, {report['n_other_code']} made by other code). Test files compared by SHA-256 only.")
    elif report["verdict"] == "INCOMPLETE":
        lines.append(f"INCOMPLETE: no difference so far, but {len(report['missing'])} runs are not finished; "
                     "run the same command again later")
    else:
        lines.append(f"DIFFERENT: {len(report['differing'])} runs differ from a copy; "
                     f"{len(report['without_other_code'])} have no copy made by other code")
    return "\n".join(lines)
