"""Expand a sweep file into jobs: one job per configuration, fold and seed (plan 5.2).

A sweep file is JSON with a complete base configuration and a list of stages:

    base          every setting, in the five sections of affect.runs.spec
    folds         the folds every configuration runs on
    stages        each with a name, a priority (higher runs first), the seeds, and one kind:
      "base"          the base configuration alone (for example with 5 seeds, to measure seed noise)
      "variants"      named configurations, each a set of overrides of the base (controls)
      "one_factor_at_a_time"  each level of each factor changed alone from the base
      "random"        n_configs configurations drawing every factor's level at random
      "grid"          every combination of the factors' levels (a full grid search); the combination
                      that equals the base configuration is left out, since it is the base itself

An override names a setting by section and field, for example {"features.n_mels": 80}; a level
may change several settings together, for example window, hop and n_fft. A factor given as
{"features.n_mels": [40, 80]} is short for levels [{"features.n_mels": 40}, {"features.n_mels": 80}].

Every configuration is validated and checked before anything is queued, so a typo or an
impossible value stops the enqueue instead of failing jobs overnight.
"""
import copy
import itertools
import json
import random

from affect.registry import DATASETS
from affect.runs.spec import check_job, config_hash, plain, run_id, validate_job

TOP_KEYS = ("sweep_id", "description", "folds", "base", "stages")
STAGE_KINDS = ("base", "variants", "one_factor_at_a_time", "random", "grid")


class SweepError(ValueError):
    pass


def set_dotted(settings: dict, key: str, value) -> None:
    section, _, field = key.partition(".")
    if not field or section not in settings or field not in settings[section]:
        raise SweepError(f"override {key!r} names no setting of the base configuration")
    settings[section][field] = value


def apply(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        set_dotted(out, key, value)
    return out


def levels_of(factor: str, levels) -> list:
    """A factor's levels as override dicts, expanding the {"section.field": [values]} shorthand."""
    if type(levels) is not list or not levels:
        raise SweepError(f"factor {factor!r}: levels must be a non-empty list")
    if all(type(level) is dict for level in levels):
        return levels
    if "." in factor and not any(type(level) is dict for level in levels):
        return [{factor: level} for level in levels]
    raise SweepError(f"factor {factor!r}: give every level as overrides, or name one setting and list values")


def _base_value(base: dict, key: str):
    section, _, field = key.partition(".")
    return base[section][field]


def label_of(overrides: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in overrides.items()) or "base"


def _config(settings: dict, sample_rate_of):
    job = validate_job(settings)
    check_job(job, sample_rate_of(job))
    return job


def expand(sweep: dict) -> tuple:
    """(jobs, summary). Raises SweepError or a settings error for any problem."""
    missing = [k for k in TOP_KEYS if k not in sweep]
    unknown = sorted(k for k in sweep if k not in TOP_KEYS)
    if missing or unknown:
        raise SweepError(f"sweep file: missing {missing}, unknown {unknown}")
    sweep_id = sweep["sweep_id"]
    if not sweep_id or not all(c.isalnum() or c in "_-" for c in sweep_id):
        raise SweepError(f"sweep_id {sweep_id!r}: letters, digits, _ and - only")

    def sample_rate_of(job):
        return DATASETS.get(job.data_name).target().sample_rate

    if type(sweep["base"]) is dict and {"functionals", "method", "protocol"} <= set(sweep["base"]):
        raise SweepError(f"{sweep_id} is a classical sweep (functionals, method, protocol): it runs on CPUs with "
                         "scripts/classical.sh classical, not through smoke, enqueue and the GPU workers")
    base_job = _config(sweep["base"], sample_rate_of)
    base_hash = config_hash(base_job)
    folds = sweep["folds"]
    if type(folds) is not list or not folds or len(set(folds)) != len(folds):
        raise SweepError("folds must be a non-empty list without repeats")
    known_folds = DATASETS.get(base_job.data_name).target().fold_names(base_job.data)
    strangers = [f for f in folds if f not in known_folds]
    if strangers:
        raise SweepError(f"folds {strangers} are not folds of the base configuration's data settings "
                         f"{plain(base_job)['data']}; its folds are {list(known_folds)}")
    names = [stage.get("name") for stage in sweep["stages"]]
    if len(set(names)) != len(names):
        raise SweepError(f"stage names repeat: {names}")
    configs = []          # (stage, label, overrides, job, priority, seeds)
    seen = {}
    summary = {"sweep_id": sweep_id, "stages": []}
    for stage in sweep["stages"]:
        for key in ("name", "kind", "priority", "seeds"):
            if key not in stage:
                raise SweepError(f"stage {stage.get('name', '?')}: missing {key!r}")
        kind, name, priority, seeds = stage["kind"], stage["name"], stage["priority"], stage["seeds"]
        if kind not in STAGE_KINDS:
            raise SweepError(f"stage {name}: kind {kind!r} is not one of {STAGE_KINDS}")
        if type(priority) is not int or not 0 <= priority <= 999:
            raise SweepError(f"stage {name}: priority must be a whole number from 0 to 999")
        if type(seeds) is not list or not seeds or any(type(s) is not int for s in seeds):
            raise SweepError(f"stage {name}: seeds must be a list of whole numbers")
        allowed = {"base": {"name", "kind", "priority", "seeds"},
                   "variants": {"name", "kind", "priority", "seeds", "variants"},
                   "one_factor_at_a_time": {"name", "kind", "priority", "seeds", "factors"},
                   "random": {"name", "kind", "priority", "seeds", "space", "n_configs", "sampling_seed"},
                   "grid": {"name", "kind", "priority", "seeds", "factors"}}[kind]
        extra = sorted(set(stage) - allowed)
        missing_keys = sorted(allowed - set(stage))
        if extra or missing_keys:
            raise SweepError(f"stage {name}: unknown keys {extra}, missing keys {missing_keys}")
        stage_configs = []
        if kind == "base":
            stage_configs.append(("base", {}))
        elif kind == "variants":
            for vname, overrides in stage["variants"].items():
                stage_configs.append((vname, overrides))
        elif kind == "grid":
            lists = [levels_of(factor, levels) for factor, levels in stage["factors"].items()]
            if not lists:
                raise SweepError(f"stage {name}: a grid needs at least one factor")
            for combination in itertools.product(*lists):
                overrides = {}
                for level in combination:
                    overlap = set(overrides) & set(level)
                    if overlap:
                        raise SweepError(f"stage {name}: two factors set {sorted(overlap)}")
                    overrides.update(level)
                changed = {k: v for k, v in overrides.items() if _base_value(sweep["base"], k) != v}
                if changed:
                    stage_configs.append((f"grid: {label_of(changed)}", overrides))
        elif kind == "one_factor_at_a_time":
            for factor, levels in stage["factors"].items():
                for level in levels_of(factor, levels):
                    label = label_of(level) if list(level) == [factor] else f"{factor}: {label_of(level)}"
                    stage_configs.append((label, level))
        n_skipped = 0
        if kind == "random":
            space = {f: levels_of(f, levels) for f, levels in stage["space"].items()}
            for factor, levels in space.items():
                for level in levels:
                    try:
                        validate_job(apply(sweep["base"], level))
                    except ValueError as err:
                        raise SweepError(f"stage {name}, factor {factor}, level {level}: {err}") from None
            rng = random.Random(stage["sampling_seed"])
            target = stage["n_configs"]
            attempts = 0
            drawn, drawn_hashes = [], set()
            while len(drawn) < target:
                attempts += 1
                if attempts > 50 * target:
                    raise SweepError(f"stage {name}: only {len(drawn)} distinct valid configurations in "
                                     f"{attempts - 1} draws")
                overrides = {}
                for levels in space.values():
                    overrides.update(rng.choice(levels))
                try:
                    job = _config(apply(sweep["base"], overrides), sample_rate_of)
                except ValueError:
                    n_skipped += 1
                    continue
                h = config_hash(job)
                if h in seen or h in drawn_hashes:
                    continue
                drawn_hashes.add(h)
                changed = {k: v for k, v in overrides.items() if _base_value(sweep["base"], k) != v}
                drawn.append((f"random {len(drawn) + 1}: {label_of(changed)}", overrides, h, job))
            for label, overrides, _, job in drawn:
                configs.append((name, label, overrides, job, priority, seeds))
                seen[config_hash(job)] = f"{name}/{label}"
        for label, overrides in stage_configs:
            job = _config(apply(sweep["base"], overrides), sample_rate_of)
            h = config_hash(job)
            if kind != "base" and h == base_hash:
                raise SweepError(f"stage {name}, {label}: identical to the base configuration")
            if h in seen and kind != "base":
                raise SweepError(f"stage {name}, {label}: identical to {seen[h]}")
            seen.setdefault(h, f"{name}/{label}")
            configs.append((name, label, overrides, job, priority, seeds))
        summary["stages"].append({"name": name, "kind": kind, "priority": priority, "seeds": seeds,
                                  "configs": sum(1 for c in configs if c[0] == name),
                                  "invalid_random_draws_skipped": n_skipped})
    # One data protocol per sweep: validation sets differ between protocols, so configurations are
    # compared only against a baseline on the same dataset, label set, subset and split.
    for stage, label, overrides, job, priority, seeds in configs:
        if plain(job)["data"] != plain(base_job)["data"]:
            raise SweepError(f"stage {stage}, {label}: changes the data settings; a sweep holds one data "
                             "protocol, so put another label set, subset or split in its own sweep file")
    jobs, run_ids = [], set()
    for stage, label, overrides, job, priority, seeds in configs:
        settings = plain(job)
        h = config_hash(job)
        for fold in folds:
            for seed in seeds:
                rid = run_id(job, fold, seed)
                if rid in run_ids:
                    raise SweepError(f"run {rid} appears twice (stage {stage}, {label})")
                run_ids.add(rid)
                jobs.append({"sweep_id": sweep_id, "stage": stage, "label": label, "overrides": overrides,
                             "priority": priority, "config_hash": h, "run_id": rid, "fold": fold,
                             "seed": seed, "settings": settings})
    summary["configs"] = len(configs)
    summary["jobs"] = len(jobs)
    summary["base_config_hash"] = base_hash
    return jobs, summary
