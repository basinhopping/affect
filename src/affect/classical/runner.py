"""Classical sweeps: expand, record, run (added 2026-09-26).

A classical sweep file has the shape of the neural sweep files: sweep_id, description, folds, base
(the four sections of a classical job, affect.classical.settings) and stages of kind base,
variants or grid (as in affect.sweep.design), each with a priority and seeds. run() expands it,
records it under artifacts/sweeps/<sweep_id>/ as enqueue does, and runs its jobs in this process,
highest priority first, each into artifacts/runs/<sweep_id>/<run_id>/. A job whose folder is
complete is skipped, so running the same file again resumes it; a changed file with the same
sweep_id is refused. There is no queue and no GPU: a TPOT job runs n_jobs processes itself.

Per job: the fold's utterances come from the dataset's split scheme (the neural models' folds),
the functionals are loaded (affect.classical.functionals) and z-scored (global: training
statistics; speaker: each speaker's own statistics), the method picks its settings on validation
UA, and the run folder gets the files the neural runs have, so analyze, select and test-report
read it unchanged.
"""
import csv
import dataclasses
import io
import itertools
import json
import time
import warnings

import numpy as np

from affect import paths
from affect.classical import functionals
from affect.classical.settings import FunctionalSettings, ProtocolSettings, check_svm, check_tpot
from affect.data.splits import assign, folds_hash
from affect.eval.metrics import report, report_by_group
from affect.hashing import code_hash, sha256_file, sha256_json
from affect.registry import CLASSICAL, DATASETS
from affect.runs.provenance import collect, utc_now
from affect.runs.store import AlreadyComplete, OutputFolder
from affect.settings import SettingsError, canonical_plain, to_plain, validate
from affect.sweep.design import SweepError, apply, label_of, levels_of
from affect.train.data import speaker_index

SECTIONS = ("data", "functionals", "method", "protocol")
TOP_KEYS = ("sweep_id", "description", "folds", "base", "stages")


@dataclasses.dataclass(frozen=True)
class ClassicalJob:
    data_name: str
    data: object
    functionals: FunctionalSettings
    method_name: str
    method: object
    protocol: ProtocolSettings


def validate_classical(raw) -> ClassicalJob:
    if type(raw) is not dict:
        raise SettingsError(f"classical job settings must be a mapping, got {type(raw).__name__}")
    missing = [s for s in SECTIONS if s not in raw]
    unknown = sorted(k for k in raw if k not in SECTIONS)
    if missing or unknown:
        raise SettingsError(f"classical job settings: missing sections {missing}, unknown sections {unknown}")
    data_entry, data = DATASETS.validate_section(raw["data"], "data")
    feats = validate(FunctionalSettings, raw["functionals"], "functionals")
    method_entry, method = CLASSICAL.validate_section(raw["method"], "method")
    protocol = validate(ProtocolSettings, raw["protocol"], "protocol")
    job = ClassicalJob(data_entry.name, data, feats, method_entry.name, method, protocol)
    {"svm": check_svm, "tpot": check_tpot}[job.method_name](job.method)
    return job


def plain(job: ClassicalJob) -> dict:
    return {"data": {"name": job.data_name, **to_plain(job.data)}, "functionals": to_plain(job.functionals),
            "method": {"name": job.method_name, **to_plain(job.method)}, "protocol": to_plain(job.protocol)}


def config_hash(job: ClassicalJob) -> str:
    return sha256_json({"data": {"name": job.data_name, **canonical_plain(job.data)},
                        "functionals": canonical_plain(job.functionals),
                        "method": {"name": job.method_name, **canonical_plain(job.method)},
                        "protocol": canonical_plain(job.protocol)})


def run_id(job: ClassicalJob, fold: str, seed: int) -> str:
    return f"{config_hash(job)[:12]}_{fold}_seed{seed}"


def expand(sweep: dict) -> tuple:
    """(jobs, summary), like affect.sweep.design.expand, for stage kinds base, variants and grid."""
    missing = [k for k in TOP_KEYS if k not in sweep]
    unknown = sorted(k for k in sweep if k not in TOP_KEYS)
    if missing or unknown:
        raise SweepError(f"sweep file: missing {missing}, unknown {unknown}")
    sweep_id = sweep["sweep_id"]
    if not sweep_id or not all(c.isalnum() or c in "_-" for c in sweep_id):
        raise SweepError(f"sweep_id {sweep_id!r}: letters, digits, _ and - only")
    base_job = validate_classical(sweep["base"])
    base_hash = config_hash(base_job)
    folds = sweep["folds"]
    known = DATASETS.get(base_job.data_name).target().fold_names(base_job.data)
    if type(folds) is not list or not folds or len(set(folds)) != len(folds) or any(f not in known for f in folds):
        raise SweepError(f"folds must be distinct folds of the base configuration's split: {list(known)}")
    configs, seen = [], {}
    summary = {"sweep_id": sweep_id, "stages": []}
    names = [s.get("name") for s in sweep["stages"]]
    if len(set(names)) != len(names):
        raise SweepError(f"stage names repeat: {names}")
    for stage in sweep["stages"]:
        name, kind = stage.get("name"), stage.get("kind")
        allowed = {"base": {"name", "kind", "priority", "seeds"},
                   "variants": {"name", "kind", "priority", "seeds", "variants"},
                   "grid": {"name", "kind", "priority", "seeds", "factors"}}.get(kind)
        if allowed is None:
            raise SweepError(f"stage {name}: kind {kind!r} is not one of base, variants, grid")
        if set(stage) != allowed:
            raise SweepError(f"stage {name}: keys must be {sorted(allowed)}, got {sorted(stage)}")
        priority, seeds = stage["priority"], stage["seeds"]
        if type(priority) is not int or not 0 <= priority <= 999:
            raise SweepError(f"stage {name}: priority must be a whole number from 0 to 999")
        if type(seeds) is not list or not seeds or any(type(s) is not int for s in seeds):
            raise SweepError(f"stage {name}: seeds must be a list of whole numbers")
        if kind == "base":
            stage_configs = [("base", {})]
        elif kind == "variants":
            stage_configs = list(stage["variants"].items())
        else:
            lists = [levels_of(factor, levels) for factor, levels in stage["factors"].items()]
            stage_configs = []
            for combination in itertools.product(*lists):
                overrides = {}
                for level in combination:
                    overrides.update(level)
                changed = {k: v for k, v in overrides.items()
                           if sweep["base"][k.partition(".")[0]][k.partition(".")[2]] != v}
                if changed:
                    stage_configs.append((f"grid: {label_of(changed)}", overrides))
        for label, overrides in stage_configs:
            job = validate_classical(apply(sweep["base"], overrides))
            h = config_hash(job)
            if kind != "base" and h == base_hash:
                raise SweepError(f"stage {name}, {label}: identical to the base configuration")
            if h in seen and kind != "base":
                raise SweepError(f"stage {name}, {label}: identical to {seen[h]}")
            if plain(job)["data"] != plain(base_job)["data"]:
                raise SweepError(f"stage {name}, {label}: changes the data settings; one data protocol per sweep")
            seen.setdefault(h, f"{name}/{label}")
            configs.append((name, label, overrides, job, priority, seeds))
        summary["stages"].append({"name": name, "kind": kind, "priority": priority, "seeds": seeds,
                                  "configs": len(stage_configs), "invalid_random_draws_skipped": 0})
    jobs, ids = [], set()
    for stage, label, overrides, job, priority, seeds in configs:
        for fold in folds:
            for seed in seeds:
                rid = run_id(job, fold, seed)
                if rid in ids:
                    raise SweepError(f"run {rid} appears twice")
                ids.add(rid)
                jobs.append({"sweep_id": sweep_id, "stage": stage, "label": label, "overrides": overrides,
                             "priority": priority, "config_hash": config_hash(job), "run_id": rid, "fold": fold,
                             "seed": seed, "settings": plain(job), "kind": "classical"})
    summary.update({"configs": len(configs), "jobs": len(jobs), "base_config_hash": base_hash})
    return jobs, summary


def normalize(values: np.ndarray, mode: str, train_rows: np.ndarray, speaker_of: np.ndarray) -> np.ndarray:
    """z-scores per feature: global with the training rows' statistics, speaker with each speaker's
    own rows. A feature with SD 0 (or no finite value) becomes 0, and so does a missing value."""
    if mode not in ("global", "speaker"):
        raise ValueError(mode)
    out = np.array(values, dtype=np.float64, copy=True)
    if mode == "global":
        groups = [(train_rows, np.arange(out.shape[0]))]
    else:
        groups = [(np.flatnonzero(speaker_of == s), np.flatnonzero(speaker_of == s)) for s in np.unique(speaker_of)]
    result = np.zeros_like(out)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for stats_rows, apply_rows in groups:
            mean = np.nanmean(out[stats_rows], axis=0)
            sd = np.nanstd(out[stats_rows], axis=0)
            usable = np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
            scaled = (out[apply_rows] - np.where(usable, mean, 0.0)) / np.where(usable, sd, 1.0)
            scaled[:, ~usable] = 0.0
            result[apply_rows] = scaled
    result[~np.isfinite(result)] = 0.0
    return result


class _Data:
    """What every job of one data protocol shares: utterances, labels, folds, functionals."""

    def __init__(self, job: ClassicalJob):
        ds = DATASETS.get(job.data_name).target()
        self.utterances = ds.utterances(job.data)
        self.class_names = tuple(ds.class_order(job.data))
        self.labels = np.array([self.class_names.index(u.label) for u in self.utterances], dtype=np.int64)
        self.kinds = np.array([u.kind for u in self.utterances])
        self.folds = {f.name: f for f in ds.folds(job.data, self.utterances)}
        self.folds_hash = folds_hash(self.utterances, tuple(self.folds.values()))
        self.position = {u.utt_id: i for i, u in enumerate(self.utterances)}
        self.speaker_of = speaker_index(self.utterances)[1]
        _, self.manifest_sha256 = sha256_file(paths.assembled_dir(job.data_name) / "MANIFEST.csv")
        self.features = {}

    def functionals(self, dataset: str, feature_set: str) -> tuple:
        if feature_set not in self.features:
            self.features[feature_set] = functionals.load(dataset, feature_set, [u.utt_id for u in self.utterances])
        return self.features[feature_set]


def _evaluate(data: _Data, rows: np.ndarray, scores: np.ndarray) -> dict:
    y_true, y_pred = data.labels[rows], scores.argmax(axis=1)
    out = report(y_true, y_pred, data.class_names)
    out["by_kind"] = report_by_group(y_true, y_pred, data.kinds[rows], data.class_names)
    out["mean_nll"] = float(-np.mean(np.log(np.clip(scores[np.arange(rows.shape[0]), y_true], 1e-12, None))))
    return out


def _predictions_csv(data: _Data, rows: np.ndarray, scores: np.ndarray) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utt_id", "speaker", "kind", "label", "pred"] + [f"prob_{c}" for c in data.class_names])
    for k, row in enumerate(rows):
        u = data.utterances[row]
        writer.writerow([u.utt_id, u.speaker, u.kind, u.label, data.class_names[int(scores[k].argmax())]]
                        + [f"{p:.6f}" for p in scores[k]])
    return buffer.getvalue()


def run_job(spec: dict, data: _Data, log) -> dict:
    """One classical job. Returns every file of its run folder except spec.json and log.txt."""
    started = utc_now()
    job = validate_classical(spec["settings"])
    if config_hash(job) != spec["config_hash"]:
        raise ValueError("the settings do not hash to the job's config_hash; the job was edited")
    if spec["fold"] not in data.folds:
        raise ValueError(f"no fold {spec['fold']!r}")
    roles = assign(data.utterances, data.folds[spec["fold"]])
    rows = {role: np.array([data.position[u] for u in roles[role]], dtype=np.int64) for role in ("train", "val", "test")}
    seed = int(spec["seed"])
    values, names, record = data.functionals(job.data_name, job.functionals.feature_set)
    x = normalize(values, job.functionals.normalization, rows["train"], data.speaker_of)
    y_train = data.labels[rows["train"]]
    if job.protocol.control == "shuffle_train_labels":
        y_train = y_train[np.random.default_rng(seed + 1_000_003).permutation(y_train.shape[0])]
    inputs = {"x_train": x[rows["train"]], "y_train": y_train, "x_val": x[rows["val"]], "y_val": data.labels[rows["val"]],
              "x_test": x[rows["test"]], "n_classes": len(data.class_names)}
    log(f"run {spec['run_id']}: fold {spec['fold']}, seed {seed}, train {rows['train'].shape[0]}, val "
        f"{rows['val'].shape[0]}, test {rows['test'].shape[0]}, {x.shape[1]} {job.functionals.feature_set} features "
        f"({job.functionals.normalization} z-scores), {job.method_name}")
    start = time.perf_counter()
    result = CLASSICAL.get(job.method_name).target(job.method, inputs, seed, job.protocol.refit, log)
    seconds = time.perf_counter() - start
    val = _evaluate(data, rows["val"], result["val_scores"])
    test = _evaluate(data, rows["test"], result["test_scores"])
    log(f"  chosen {result['chosen']}; validation UA {val['ua']:.4f}; {seconds:.0f}s")
    provenance = collect("affect.cli classical")
    provenance.update({"started_utc": started, "finished_utc": utc_now(), "folds_hash": data.folds_hash,
                       "data_manifest_sha256": data.manifest_sha256,
                       "functionals": {k: record.get(k) for k in ("feature_set", "n_features", "names_sha256",
                                                                   "opensmile_version", "non_finite_values")}})
    return {
        "metrics_val.json": {"best_epoch": None, "best_val_ua": val["ua"], "epochs_run": None, "stopped_early": None,
                             "curves": [], "final": val, "chosen": result["chosen"], "grid": result["grid"]},
        "predictions_val.csv": _predictions_csv(data, rows["val"], result["val_scores"]),
        "metrics_test.json": {"final": test, "refit": job.protocol.refit},
        "predictions_test.csv": _predictions_csv(data, rows["test"], result["test_scores"]),
        "efficiency.json": {"train_seconds": round(seconds, 2), "n_features": int(x.shape[1]),
                            "n_train": int(rows["train"].shape[0])},
        "provenance.json": provenance,
    }


def _record(sweep_file, sweep: dict, jobs: list, summary: dict, log) -> None:
    folder_path = paths.sweeps_dir() / sweep["sweep_id"]
    if folder_path.exists():
        recorded = json.loads((folder_path / "sweep_file.json").read_text(encoding="utf-8"))
        if recorded != sweep:
            raise FileExistsError(f"sweep {sweep['sweep_id']} was recorded with a different file; give a changed "
                                  "sweep a new sweep_id")
        log(f"sweep {sweep['sweep_id']} was recorded before; resuming (complete runs are skipped)")
        return
    folder = OutputFolder(paths.sweeps_dir(), sweep["sweep_id"]).begin()
    try:
        _, file_sha = sha256_file(sweep_file)
        folder.write_json("sweep_file.json", sweep)
        folder.write_json("summary.json", {**summary, "sweep_file": str(sweep_file), "sweep_file_sha256": file_sha,
                                           "code_sha256": code_hash(paths.repo_root() / "src" / "affect")})
        folder.write_json("provenance.json", collect("affect.cli classical"))
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["seq", "run_id", "stage", "label", "priority", "fold", "seed", "config_hash"])
        for seq, job in enumerate(jobs):
            writer.writerow([seq, job["run_id"], job["stage"], job["label"], job["priority"], job["fold"],
                             job["seed"], job["config_hash"]])
        folder.write_text("jobs.csv", buffer.getvalue())
        folder.complete()
    except BaseException as err:
        folder.fail(err)
        raise


def run(sweep_file, max_consecutive_failures: int, log) -> dict:
    """Run every job of a classical sweep file that has no complete run folder yet."""
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be at least 1")
    with open(sweep_file, encoding="utf-8") as handle:
        sweep = json.load(handle)
    jobs, summary = expand(sweep)
    _record(sweep_file, sweep, jobs, summary, log)
    order = sorted(range(len(jobs)), key=lambda k: (-jobs[k]["priority"], k))
    data = None
    done = skipped = failed = in_a_row = 0
    for n, k in enumerate(order, 1):
        spec = jobs[k]
        folder = OutputFolder(paths.runs_dir(spec["sweep_id"]), spec["run_id"])
        try:
            folder.begin()
        except AlreadyComplete:
            skipped += 1
            continue
        lines = []

        def tee(message: str) -> None:
            lines.append(message)
            log(message)

        try:
            folder.write_json("spec.json", spec)
            if data is None:
                data = _Data(validate_classical(spec["settings"]))
            files = run_job(spec, data, tee)
            for name, content in files.items():
                (folder.write_json if name.endswith(".json") else folder.write_text)(name, content)
            folder.write_text("log.txt", "\n".join(lines) + "\n")
            folder.complete()
        except Exception as err:
            try:
                folder.write_text("log.txt", "\n".join(lines) + "\n")
            except FileExistsError:
                pass
            folder.fail(err)
            failed += 1
            in_a_row += 1
            log(f"JOB FAILED {spec['run_id']} ({n} of {len(jobs)}): {type(err).__name__}: {err}")
            if in_a_row >= max_consecutive_failures:
                log(f"{in_a_row} jobs failed in a row; stopping")
                break
            continue
        done += 1
        in_a_row = 0
        log(f"JOB DONE {spec['run_id']} ({n} of {len(jobs)}) validation UA "
            f"{files['metrics_val.json']['best_val_ua']:.4f}")
    result = {"sweep_id": sweep["sweep_id"], "jobs": len(jobs), "done": done, "skipped_complete": skipped,
              "failed": failed}
    log(json.dumps(result))
    return result
