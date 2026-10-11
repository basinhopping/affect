"""The test stage (plan 3.5): record the choice first, then read the test results of that choice.

    select        writes artifacts/selections/<name>/selection.json: the chosen configuration of a
                  sweep, the reason, and the validation analysis the choice was made from. It refuses
                  unless a finished `analyze` report of that sweep exists, and never reads a test file.
    test_report   reads the selection, then the test files (metrics_test.json, predictions_test.csv)
                  of the chosen configuration's runs only: per seed the fold means, a t interval over
                  seeds, per-class recall, and UA by speaker gender and by kind.

Nothing here retrains. The runs' test files come from the models whose epoch was chosen on the
validation session (train on 3 sessions, pick the epoch on the 4th, test on the 5th), the scheme of
Pepino et al. 2021. Whether final numbers should instead come from retraining on 4 sessions is an
open decision recorded in HANDOFF.md.
"""
import csv
import json

import numpy as np

from affect import paths
from affect.analysis.groups import group_ua
from affect.analysis.validation import _configs, _read_json, load
from affect.eval.stats import t_interval
from affect.hashing import sha256_file


def latest_analysis(sweep_id: str):
    """(folder, report) of the newest finished analyze output for the sweep, or (None, None)."""
    root = paths.analysis_dir()
    folders = sorted(p for p in root.glob(f"{sweep_id}_*") if p.is_dir() and not p.name.endswith(".tmp")
                     and ".failed-" not in p.name and (p / "report.json").is_file()) if root.is_dir() else []
    folders = [p for p in folders if p.name[len(sweep_id) + 1:][:1].isdigit()]     # not another sweep's prefix
    if not folders:
        return None, None
    return folders[-1], _read_json(folders[-1] / "report.json")


def selection_record(sweep_id: str, config_prefix: str, reason: str) -> dict:
    """What select writes. Validation files and the analysis report only."""
    if len(config_prefix) < 8:
        raise ValueError("give at least 8 characters of the configuration hash")
    if not reason.strip():
        raise ValueError("give the reason for the choice")
    folder, report = latest_analysis(sweep_id)
    if report is None:
        raise ValueError(f"no finished analyze report for {sweep_id}: run analyze first, then choose")
    sweep_file, runs = load(sweep_id)
    configs = _configs(sweep_file, runs)
    matches = [h for h in configs if h.startswith(config_prefix)]
    if len(matches) != 1:
        raise ValueError(f"{config_prefix!r} matches {len(matches)} configurations of {sweep_id}")
    chosen = configs[matches[0]]
    summary = next((s for s in report["ranked"] if s["config_hash"] == matches[0]), None)
    if summary is None:
        raise ValueError(f"{matches[0][:12]} is not ranked in {folder.name} (a control, or no complete seed)")
    _, report_sha = sha256_file(folder / "report.json")
    return {"sweep_id": sweep_id, "config_hash": matches[0], "stage": chosen["stage"], "label": chosen["label"],
            "settings": chosen["settings"], "reason": reason.strip(),
            "validation": {k: summary[k] for k in ("ua", "interval", "seeds", "wa", "macro_f1", "recall")},
            "analysis_report": {"folder": folder.name, "sha256": report_sha},
            "baseline": report["baseline"], "folds": report["folds"]}


def _rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_report(selection: dict, confidence: float) -> dict:
    """Test results of the selected configuration: every seed with all folds complete."""
    sweep_file, runs = load(selection["sweep_id"])
    config = _configs(sweep_file, runs)[selection["config_hash"]]
    folds = list(selection["folds"])
    per_seed, incomplete = {}, []
    for seed, by_fold in sorted(config["runs"].items()):
        if not all(f in by_fold for f in folds):
            incomplete.append(seed)
            continue
        finals, groups = [], {"gender": {}, "kind": {}}
        for f in folds:
            folder = by_fold[f]["folder"]
            finals.append(_read_json(folder / "metrics_test.json")["final"])
            rows = _rows(folder / "predictions_test.csv")
            classes = tuple(k[len("prob_"):] for k in rows[0] if k.startswith("prob_"))
            for field in groups:
                for group, ua in group_ua(rows, field, classes).items():
                    groups[field].setdefault(group, []).append(ua)
        classes = list(finals[0]["recall"])
        per_seed[seed] = {
            "ua": float(np.mean([x["ua"] for x in finals])), "wa": float(np.mean([x["wa"] for x in finals])),
            "macro_f1": float(np.mean([x["macro_f1"] for x in finals])),
            "recall": {c: float(np.mean([x["recall"][c] for x in finals])) for c in classes},
            "ua_by_fold": {f: x["ua"] for f, x in zip(folds, finals)},
            "ua_by_group": {field: {g: (None if None in v or len(v) != len(folds) else float(np.mean(v)))
                                    for g, v in values.items()} for field, values in groups.items()}}
    if not per_seed:
        raise ValueError("no seed of the selected configuration has every fold complete")
    seeds = sorted(per_seed)
    out = {"selection": {k: selection[k] for k in ("sweep_id", "config_hash", "label", "reason")},
           "confidence": confidence, "seeds": seeds, "incomplete_seeds": incomplete, "per_seed": per_seed}
    for metric in ("ua", "wa", "macro_f1"):
        values = [per_seed[s][metric] for s in seeds]
        out[metric] = {"mean": float(np.mean(values)),
                       "interval": t_interval(values, confidence) if len(values) >= 2 else None}
    out["recall"] = {c: float(np.mean([per_seed[s]["recall"][c] for s in seeds])) for c in per_seed[seeds[0]]["recall"]}
    out["ua_by_group"] = {field: {g: (None if any(per_seed[s]["ua_by_group"][field].get(g) is None for s in seeds)
                                      else float(np.mean([per_seed[s]["ua_by_group"][field][g] for s in seeds])))
                                  for g in per_seed[seeds[0]]["ua_by_group"][field]}
                          for field in ("gender", "kind")}
    return out


def render(report: dict) -> str:
    sel, ua = report["selection"], report["ua"]
    lines = [f"test results of {sel['sweep_id']} / {sel['config_hash'][:12]} ({sel['label']}), seeds {report['seeds']}",
             f"chosen because: {sel['reason']}"]
    if report["incomplete_seeds"]:
        lines.append(f"left out, not every fold complete: seeds {report['incomplete_seeds']}")
    for metric in ("ua", "wa", "macro_f1"):
        m = report[metric]
        ci = m["interval"]
        lines.append(f"  test {metric.upper() if metric != 'macro_f1' else 'macro-F1'} {m['mean']:.4f}"
                     + (f" [{ci['low']:.4f}, {ci['high']:.4f}] ({round(report['confidence'] * 100)}% t interval "
                        f"over seeds), SD {ci['sd']:.4f}" if ci else ""))
    lines.append("  recall " + "  ".join(f"{c} {v:.3f}" for c, v in report["recall"].items()))
    for field in ("gender", "kind"):
        lines.append(f"  UA by {field}: " + "  ".join(f"{g} {'undefined' if v is None else f'{v:.4f}'}"
                                                   for g, v in report["ua_by_group"][field].items()))
    for seed in report["seeds"]:
        s = report["per_seed"][seed]
        lines.append(f"  seed {seed}: UA {s['ua']:.4f} | " + "  ".join(f"{f} {v:.4f}" for f, v in s["ua_by_fold"].items()))
    return "\n".join(lines)
