"""Validation-only analysis of a finished sweep (plan 3.3 to 3.5).

Reads spec.json, metrics_val.json and predictions_val.csv of complete run folders, and the sweep
record under artifacts/sweeps/. It never opens a test file: settings are chosen from validation
alone, and test results are read only after a selection file records the choice (plan 3.5).

For every configuration: the fold-mean validation UA of each seed with a t interval over seeds,
and the mean WA, macro-F1 and per-class recall over its runs.

For every configuration that shares at least two seeds with the sweep's baseline, the comparison
with the baseline on those seeds (plan 3.3):
  - the per-seed differences with a t interval (seed noise only);
  - a hierarchical paired bootstrap of the difference in fold-mean UA, which resamples seeds and
    validation utterances and shares each utterance resample between the two;
  - P(configuration beats baseline) over the seeds, with a bootstrap interval;
  - Holm's adjustment of the bootstrap p-values across every compared configuration.
The verdict is better or worse when the Holm-adjusted p is below 1 - confidence, and otherwise
"no detectable difference".

For random stages: the mean fold-mean UA of every level of every factor, over the configurations
drawn with that level.

For every configuration: validation UA within each speaker gender and within improvised and
scripted speech (affect.analysis.groups), as the mean over seeds of the fold means.
"""
import csv
import json

import numpy as np

from affect import paths
from affect.analysis.groups import group_ua
from affect.eval.stats import holm, paired_bootstrap_ua_difference, probability_of_outperforming_interval, t_interval
from affect.runs.spec import control_of
from affect.sweep.design import label_of, levels_of


def _read_json(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_predictions(path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    classes = [key[len("prob_"):] for key in rows[0] if key.startswith("prob_")]
    index = {c: i for i, c in enumerate(classes)}
    return {"classes": classes, "utt_ids": [r["utt_id"] for r in rows],
            "y_true": np.array([index[r["label"]] for r in rows]),
            "y_pred": np.array([index[r["pred"]] for r in rows])}


def _read_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_scores(config: dict, folds: list) -> dict:
    """{"gender": {group: UA}, "kind": {group: UA}}: per seed the mean over folds of the UA within the
    group, then the mean over seeds. None where UA is undefined in some fold (a class is missing)."""
    per_seed = {"gender": {}, "kind": {}}
    for seed, by_fold in sorted(config["runs"].items()):
        if not all(f in by_fold for f in folds):
            continue
        values = {"gender": {}, "kind": {}}
        for f in folds:
            rows = _read_rows(by_fold[f]["folder"] / "predictions_val.csv")
            classes = tuple(k[len("prob_"):] for k in rows[0] if k.startswith("prob_"))
            for field in values:
                for group, ua in group_ua(rows, field, classes).items():
                    values[field].setdefault(group, []).append(ua)
        for field, groups in values.items():
            per_seed[field][seed] = {g: (None if None in v or len(v) != len(folds) else float(np.mean(v)))
                                     for g, v in groups.items()}
    out = {}
    for field, seeds in per_seed.items():
        groups = sorted({g for v in seeds.values() for g in v})
        out[field] = {g: (None if any(seeds[s].get(g) is None for s in seeds)
                          else float(np.mean([seeds[s][g] for s in seeds]))) for g in groups}
    return out


def load(sweep_id: str) -> tuple:
    """(the enqueued sweep file, the complete runs as dicts). Validation files only."""
    sweep_file = _read_json(paths.sweeps_dir() / sweep_id / "sweep_file.json")
    root = paths.runs_dir(sweep_id)
    runs = []
    for folder in sorted(root.iterdir()) if root.is_dir() else []:
        if not folder.is_dir() or folder.name.endswith(".tmp") or ".failed-" in folder.name:
            continue
        spec = _read_json(folder / "spec.json")
        metrics = _read_json(folder / "metrics_val.json")
        runs.append({"folder": folder, "spec": spec, "metrics": metrics})
    return sweep_file, runs


def _configs(sweep_file: dict, runs: list) -> dict:
    kinds = {s["name"]: s["kind"] for s in sweep_file["stages"]}
    out = {}
    for r in runs:
        s = r["spec"]
        c = out.setdefault(s["config_hash"], {"config_hash": s["config_hash"], "stage": s["stage"],
                                              "kind": kinds[s["stage"]], "label": s["label"],
                                              "control": control_of(s["settings"]),
                                              "settings": s["settings"], "runs": {}})
        c["runs"].setdefault(s["seed"], {})[s["fold"]] = r
    return out


def _seed_scores(config: dict, folds: list) -> dict:
    """seed -> fold-mean validation UA, for the seeds with every fold complete."""
    out = {}
    for seed, by_fold in sorted(config["runs"].items()):
        if all(f in by_fold for f in folds):
            out[seed] = float(np.mean([by_fold[f]["metrics"]["final"]["ua"] for f in folds]))
    return out


def _summary(config: dict, folds: list, confidence: float) -> dict:
    scores = _seed_scores(config, folds)
    runs = [r for by_fold in config["runs"].values() for r in by_fold.values()]
    finals = [r["metrics"]["final"] for r in runs]
    classes = list(finals[0]["recall"])
    out = {"config_hash": config["config_hash"], "stage": config["stage"], "label": config["label"],
           "control": config["control"], "seeds": sorted(scores), "ua_by_seed": scores,
           "n_runs": len(runs),
           "wa": float(np.mean([f["wa"] for f in finals])),
           "macro_f1": float(np.mean([f["macro_f1"] for f in finals])),
           "recall": {c: float(np.mean([f["recall"][c] for f in finals])) for c in classes},
           "best_minus_final_max": float(max(abs(r["metrics"]["best_val_ua"] - r["metrics"]["final"]["ua"])
                                             for r in runs))}
    values = list(scores.values())
    out["ua"] = float(np.mean(values)) if values else None
    out["interval"] = t_interval(values, confidence) if len(values) >= 2 else None
    out["ua_by_group"] = _group_scores(config, folds) if values else None
    return out


def _fold_arrays(config: dict, seeds: list, folds: list) -> list:
    """Per fold: the validation utterances, their labels, and seeds x utterances correctness."""
    out = []
    for fold in folds:
        preds = [_read_predictions(config["runs"][s][fold]["folder"] / "predictions_val.csv") for s in seeds]
        first = preds[0]
        for p in preds[1:]:
            if p["utt_ids"] != first["utt_ids"] or not np.array_equal(p["y_true"], first["y_true"]):
                raise ValueError(f"{config['config_hash']} {fold}: seeds disagree on the validation utterances")
        out.append({"utt_ids": first["utt_ids"], "y_true": first["y_true"],
                    "correct": np.stack([p["y_pred"] == p["y_true"] for p in preds]),
                    "n_classes": len(first["classes"])})
    return out


def _versus(config: dict, base: dict, folds: list, confidence: float, n_boot: int, rng) -> dict:
    a, b = _seed_scores(config, folds), _seed_scores(base, folds)
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    fa, fb = _fold_arrays(config, seeds, folds), _fold_arrays(base, seeds, folds)
    boot = paired_bootstrap_ua_difference(fa, fb, fa[0]["n_classes"], n_boot, confidence, rng, 250)
    poo = probability_of_outperforming_interval([a[s] for s in seeds], [b[s] for s in seeds], n_boot, confidence,
                                                rng, 0.75)
    return {"seeds": seeds, "difference_by_seed": dict(zip(seeds, diffs)),
            "seed_interval": t_interval(diffs, confidence), "bootstrap": boot, "p_better": poo}


def factor_levels(sweep_file: dict, configs: dict, folds: list) -> list:
    """For each random stage: every factor's levels with the mean UA of the configurations drawn with it."""
    out = []
    for stage in sweep_file["stages"]:
        if stage["kind"] != "random":
            continue
        members = [c for c in configs.values() if c["stage"] == stage["name"]]
        for factor, raw_levels in stage["space"].items():
            rows = []
            for level in levels_of(factor, raw_levels):
                def matches(c, level=level):
                    return all(c["settings"][k.partition(".")[0]][k.partition(".")[2]] == v for k, v in level.items())
                per_config = [list(_seed_scores(c, folds).values()) for c in members if matches(c)]
                scores = [float(np.mean(v)) for v in per_config if v]
                rows.append({"level": label_of(level), "n_configs": len(scores),
                             "mean_ua": float(np.mean(scores)) if scores else None})
            means = [r["mean_ua"] for r in rows if r["mean_ua"] is not None]
            out.append({"stage": stage["name"], "factor": factor, "levels": rows,
                        "spread": float(max(means) - min(means)) if len(means) > 1 else 0.0})
    out.sort(key=lambda e: -e["spread"])
    return out


def analyze(sweep_id: str, confidence: float, n_boot: int, seed: int) -> dict:
    sweep_file, runs = load(sweep_id)
    folds = list(sweep_file["folds"])
    configs = _configs(sweep_file, runs)
    base_hashes = [h for h, c in configs.items() if c["kind"] == "base"]
    if len(base_hashes) > 1:
        raise ValueError(f"{sweep_id}: expected at most one baseline configuration, found {len(base_hashes)}")
    base = configs[base_hashes[0]] if base_hashes else None
    rng = np.random.default_rng(seed)
    summaries = {h: _summary(c, folds, confidence) for h, c in configs.items()}
    compared = []
    for h, c in configs.items():
        if base is None or h == base["config_hash"] or c["control"] != "none":
            continue
        shared = set(_seed_scores(c, folds)) & set(_seed_scores(base, folds))
        if len(shared) >= 2:
            compared.append((h, _versus(c, base, folds, confidence, n_boot, rng)))
    if compared:
        adjusted = holm([v["bootstrap"]["p_two_sided"] for _, v in compared])
        alpha = 1 - confidence
        for (h, v), p in zip(compared, adjusted):
            v["holm_p"] = p
            d = v["bootstrap"]["difference"]
            v["verdict"] = ("better" if d > 0 else "worse") if p < alpha else "no detectable difference"
            summaries[h]["versus_baseline"] = v
    ranked = sorted((s for s in summaries.values() if s["ua"] is not None and s["control"] == "none"),
                    key=lambda s: -s["ua"])
    return {"sweep_id": sweep_id, "folds": folds, "confidence": confidence, "n_boot": n_boot, "seed": seed,
            "n_runs": len(runs), "n_configs": len(configs), "baseline": base["config_hash"] if base else None,
            "ranked": ranked, "controls": [s for s in summaries.values() if s["control"] != "none"],
            "factors": factor_levels(sweep_file, configs, folds)}


def render(report: dict, top: int) -> str:
    lines = [f"sweep {report['sweep_id']}: {report['n_configs']} configurations, {report['n_runs']} complete runs "
             f"(validation only; {round(report['confidence'] * 100)}% intervals, {report['n_boot']} bootstrap draws)"]
    worst = max(s["best_minus_final_max"] for s in report["ranked"] + report["controls"])
    if worst > 1e-9:
        lines.append(f"NOTE reloading the best epoch changed validation UA by up to {worst:.2e} in some runs")
    base = next((s for s in report["ranked"] if s["config_hash"] == report["baseline"]), None)
    if base is None:
        lines.append("no baseline in this sweep: configurations are ranked but not compared")
    else:
        bi = base["interval"]
        lines.append(f"baseline {base['config_hash'][:12]}: seeds {base['seeds']}, UA {base['ua']:.4f}"
                     + (f" [{bi['low']:.4f}, {bi['high']:.4f}], SD {bi['sd']:.4f}" if bi else "")
                     + f", WA {base['wa']:.4f}, macro-F1 {base['macro_f1']:.4f}")
        lines.append("  recall " + "  ".join(f"{c} {v:.3f}" for c, v in base["recall"].items()))
    for s in report["controls"]:
        lines.append(f"control {s['label']}: UA {s['ua']:.4f} over seeds {s['seeds']}")
    lines.append(f"top {top} of {len(report['ranked'])} by mean validation UA; difference = configuration minus "
                 "baseline on the same seeds:")
    for s in report["ranked"][:top]:
        v = s.get("versus_baseline")
        head = f"  {s['ua']:.4f}"
        if s["interval"]:
            head += f" [{s['interval']['low']:.4f}, {s['interval']['high']:.4f}]"
        if v:
            b, si, pb = v["bootstrap"], v["seed_interval"], v["p_better"]
            head += (f"  diff {b['difference']:+.4f} bootstrap [{b['low']:+.4f}, {b['high']:+.4f}]"
                     f" seeds [{si['low']:+.4f}, {si['high']:+.4f}]  P(better) {pb['p_a_beats_b']:.2f}"
                     f"  Holm p {v['holm_p']:.3f}  {v['verdict']}")
        elif s["config_hash"] == report["baseline"]:
            head += "  (baseline)"
        lines.append(head + f"  | {s['stage']}: {s['label']}")
        if s.get("ua_by_group"):
            parts = []
            for field in ("gender", "kind"):
                parts.append(" ".join(f"{g} {'undefined' if v is None else f'{v:.4f}'}"
                                      for g, v in s["ua_by_group"][field].items()))
            lines.append(f"      by gender: {parts[0]}; by kind: {parts[1]}")
    if report["factors"]:
        lines.append("factor levels in random stages (mean UA of the configurations drawn with each level):")
        for f in report["factors"]:
            levels = "  ".join(f"{r['level']}: {r['mean_ua']:.4f} (n={r['n_configs']})" for r in f["levels"]
                               if r["mean_ua"] is not None)
            lines.append(f"  {f['factor']} (spread {f['spread']:.4f}): {levels}")
    return "\n".join(lines)
