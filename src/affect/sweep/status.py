"""The morning check: queue counts, failures, the ranking on validation UA, and the pilot's stop
conditions (plan 5.5). Reads queue files and finished run folders only, never logs, and never a
test metric (plan 3.5).

The screening number of a configuration is its fold-mean validation UA: the mean over the five
folds of the best validation UA each reached. Configurations missing a fold are listed as partial.
"""
import json
import statistics
from pathlib import Path

from affect import paths
from affect.runs.spec import control_of
from affect.sweep.queue import Queue


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def collect_runs(sweep_id: str) -> dict:
    root = paths.runs_dir(sweep_id)
    complete, failed, in_progress = [], [], []
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if folder.name.endswith(".tmp"):
                in_progress.append(folder.name)
            elif ".failed-" in folder.name:
                error = (folder / "error.txt").read_text(encoding="utf-8").strip().splitlines() \
                    if (folder / "error.txt").is_file() else ["(no error.txt)"]
                failed.append({"folder": folder.name, "error": error[-1] if error else ""})
            elif (folder / "metrics_val.json").is_file():
                spec = _read(folder / "spec.json")
                metrics = _read(folder / "metrics_val.json")
                efficiency = _read(folder / "efficiency.json")
                complete.append({"run_id": spec["run_id"], "stage": spec["stage"], "label": spec["label"],
                                 "config_hash": spec["config_hash"], "fold": spec["fold"], "seed": spec["seed"],
                                 "control": control_of(spec["settings"]),
                                 "best_val_ua": metrics["best_val_ua"], "best_epoch": metrics["best_epoch"],
                                 "epochs_run": metrics["epochs_run"], "train_seconds": efficiency["train_seconds"]})
    return {"complete": complete, "failed": failed, "in_progress": in_progress}


def summarise(sweep_id: str) -> dict:
    sweep_file = _read(paths.sweeps_dir() / sweep_id / "sweep_file.json")
    folds = sweep_file["folds"]
    base_stages = {s["name"] for s in sweep_file["stages"] if s["kind"] == "base"}
    runs = collect_runs(sweep_id)
    groups = {}
    for r in runs["complete"]:
        key = (r["config_hash"], r["seed"])
        g = groups.setdefault(key, {"stage": r["stage"], "label": r["label"], "control": r["control"],
                                    "seed": r["seed"], "config_hash": r["config_hash"], "folds": {}})
        g["folds"][r["fold"]] = r["best_val_ua"]
    for g in groups.values():
        g["complete"] = all(f in g["folds"] for f in folds)
        g["fold_mean_val_ua"] = statistics.fmean(g["folds"][f] for f in folds) if g["complete"] else None
    base = [g for g in groups.values() if g["stage"] in base_stages and g["complete"]]
    base_by_seed = {g["seed"]: g["fold_mean_val_ua"] for g in base}
    baseline = None
    if base:
        values = [g["fold_mean_val_ua"] for g in base]
        baseline = {"seeds": len(values), "mean": statistics.fmean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else None,
                    "min": min(values), "max": max(values),
                    "per_fold_mean": {f: statistics.fmean(g["folds"][f] for g in base) for f in folds}}
    ranked = sorted((g for g in groups.values() if g["complete"] and g["control"] == "none"),
                    key=lambda g: -g["fold_mean_val_ua"])
    for g in ranked:
        paired = base_by_seed.get(g["seed"])
        g["minus_base_same_seed"] = g["fold_mean_val_ua"] - paired if paired is not None else None
    controls = [g for g in groups.values() if g["control"] != "none" and g["complete"]]
    alerts = []
    if baseline and baseline["mean"] < 0.50:
        alerts.append(f"STOP CONDITION 3: baseline fold-mean validation UA {baseline['mean']:.4f} is below 0.50")
    for g in controls:
        if g["fold_mean_val_ua"] > 0.30:
            alerts.append(f"STOP CONDITION 4: control '{g['label']}' seed {g['seed']} reaches validation UA "
                          f"{g['fold_mean_val_ua']:.4f}, above 0.30")
    for r in runs["complete"]:
        if r["best_val_ua"] > 0.80:
            alerts.append(f"CONDITION 5: {r['run_id']} reaches validation UA {r['best_val_ua']:.4f}; investigate")
    seconds = [r["train_seconds"] for r in runs["complete"]]
    return {"sweep_id": sweep_id, "folds": folds, "queue": Queue(paths.queue_dir()).counts(sweep_id),
            "runs_complete": len(runs["complete"]), "runs_failed": runs["failed"],
            "runs_in_progress": runs["in_progress"], "median_train_seconds": statistics.median(seconds) if seconds else None,
            "baseline": baseline, "controls": controls, "ranked": ranked,
            "partial_configs": sum(1 for g in groups.values() if not g["complete"]), "alerts": alerts}


def render(s: dict, top: int) -> str:
    runs = (f"runs: {s['runs_complete']} complete, {len(s['runs_failed'])} failed, "
            f"{len(s['runs_in_progress'])} in progress")
    if s["median_train_seconds"] is not None:
        runs += f"; median training time per run {s['median_train_seconds']:.0f}s"
    lines = [f"sweep {s['sweep_id']}", f"queue: {s['queue']}", runs]
    for alert in s["alerts"]:
        lines.append("ALERT " + alert)
    for f in s["runs_failed"][:10]:
        lines.append(f"failed {f['folder']}: {f['error']}")
    b = s["baseline"]
    if b:
        sd = f"{b['sd']:.4f}" if b["sd"] is not None else "n/a"
        lines.append(f"baseline: {b['seeds']} seeds, fold-mean validation UA mean {b['mean']:.4f}, SD {sd}, "
                     f"range {b['min']:.4f} to {b['max']:.4f}")
        lines.append("baseline per fold: " + ", ".join(f"{f} {v:.4f}" for f, v in b["per_fold_mean"].items()))
    for g in s["controls"]:
        lines.append(f"control {g['label']} seed {g['seed']}: fold-mean validation UA {g['fold_mean_val_ua']:.4f}")
    lines.append(f"top {top} of {len(s['ranked'])} complete configurations ({s['partial_configs']} partial), "
                 "by fold-mean validation UA:")
    for g in s["ranked"][:top]:
        delta = f"{g['minus_base_same_seed']:+.4f}" if g["minus_base_same_seed"] is not None else "n/a"
        lines.append(f"  {g['fold_mean_val_ua']:.4f}  vs base {delta}  seed {g['seed']}  {g['stage']}: {g['label']}")
    return "\n".join(lines)
