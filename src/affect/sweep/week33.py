"""Week 33's automatic stages (added 2026-11-09, after the advisor meeting of that day), exactly as RULE_week33.txt fixes
them (artifacts/analyses/week33_prep_20261109, fixed before any run of these sweeps). Dr. Luitel and Harrison,
2026-11-09: speaker-specific folds are the main protocol and the validation speaker makes the choice that is made there;
every model is kept until this week's results are in (Whisper large-v3's four heads and the six networks trained from
scratch); the four setups stay (four and three classes, each speaker-specific and pooled).

It reads validation files only: each sweep's record (artifacts/sweeps/<id>/sweep_file.json and jobs.csv) and the
spec.json and metrics_val.json of complete run folders; it never opens a test file. What it writes goes to
artifacts/advance_week33/: the sweep files (sweeps/), one decision file per sweep with what it was made from
(decisions/) and a log (log.txt). It never smoke-tests or queues anything itself: scripts/advance_week33.sh queues the
sweeps it lists as ENQUEUE <path>, with scripts/smoke_enqueue_files.sh.

    python -m affect.sweep.week33 advance     every sweep that is due is written; then ENQUEUE <file> for each sweep file
                                              written and neither queued nor marked as failed, highest priority first
    python -m affect.sweep.week33 status      each chain's state, one line per model

(on the node with the environment of scripts/common.sh: affect_python -m affect.sweep.week33 ...)

The models (RULE_week33.txt part 1): each one's week-32 record exactly as its speaker-specific refit ran it, read from
the queue's record of that sweep (artifacts/sweeps/<id>/sweep_file.json): heads32w_<head>_spk for Whisper large-v3's
heads (pool, rnn, transformer, lif; tags wpool, wrnn, wtransformer, wlif) and spk32_<net> for the networks (base, cnn,
transformer, threeblock, lif, crnn). The chains (part 2), for each model and setup:
  validation   w33_<tag>_<setup>v, written at the first pass: the record's settings with the setup's data settings and
               without the refit (train.refit none, train.refit_epochs none). Each run trains on its fold's training
               speakers (or groups), stops early on the validation one, and records its best epoch.
  refit        w33_<tag>_<setup>, once the validation sweep has every run and no failed job: the same settings refitted
               on training plus validation (train.refit train_and_val) for the median of the validation runs'
               best_epoch, rounded half up, on the record's own learning-rate schedule; tested once with its last
               weights.
The setups: spk4 and spk3, speaker-specific folds (leave_one_speaker_out: one speaker tests, the other speaker of the
same session validates, the other eight train), four classes (four_merged) and three (three_no_neutral), seeds 11 to
15, 50 runs a sweep, for all ten models; pooled4 and pooled3, session-agnostic folds (utterance_5fold), seeds 6 to 10 as
the heads' week-32 setups, 25 runs a sweep, for the six networks only (Whisper's heads have theirs from week 32).
Nothing else differs from the record. A sweep with a failed job is not read: its chain waits, and status says so.

Addendum 1 (added 2026-11-10; RULE_week33_addendum1.txt, fixed before any of its runs) adds three setups, made the same
way (a validation sweep, then the refit for the median chosen epoch), with seeds 11 to 15:
  impro4   all ten models: four classes, speaker-specific folds whose training and validation sides hold improvised
           utterances only (leave_one_speaker_out_train_impro); the whole held-out speaker is tested, so one run scores
           the improvised utterances and the scripted ones, whose scripts no training utterance holds; priorities 180
           and 181.
  film     all ten models trained on EmoFilm: its three classes (happiness, anger, sadness as hap+exc, ang, sad) in five
           film folds (data emofilm, split film_5fold: one fold tests, the next validates, three train; the refit
           trains on four), 25 runs a sweep; the refit also scores IEMOCAP's 3,823 utterances of the three classes cold
           (train.external_test iemocap) and adapted speaker by speaker (train.warm_start iemocap: for each of the ten
           speakers, 10 epochs at a tenth of the record's peak rate on the other nine speakers' utterances, then that
           speaker scored); priorities 178 and 179.
  wmel4    the six networks on Whisper large-v3's own log-mel spectrogram (feature whisper_logmel: 128 bands, Whisper's
           window, filters and 80 dB floor, the record's normalization), the frequency masks' width scaled to 128 bands
           (rounded half up), four classes, speaker-specific folds; priorities 176 and 177.

Addendum 5 (added 2026-11-13; RULE_week33_addendum5.txt, fixed before any of its runs): the films' validation sweeps of
addendum 1, w33_<tag>_filmv on split film_5fold, each lost its folds test_F2 and test_F3 (film fold F3 holds no sadness
clip: "classes [2] have no examples"), so no film chain could go on. They are superseded (SUPERSEDED), kept as they are
and never read. The film chain's validation sweep is now w33_<tag>_film2v, the same settings on split film_5fold_strat
(the films dealt so that every fold holds all three classes; affect.data.emofilm); its refit keeps the name
w33_<tag>_film and everything addendum 1 fixed for it; priorities 178 and 179 as before.

Addendum 8 (added 2026-11-14; RULE_week33_addendum8.txt, fixed before any of its runs): the corpus's file list spells 13
films' titles in two to four ways, so film_5fold_strat, which deals film IDs, put 11 of the 37 films in more than one
fold (RULE_week33_addendum7.txt held the films). The film chain's validation sweep is now w33_<tag>_film3v, the same
settings on split film_5fold_whole (film_5fold_strat's dealing over whole films; affect.data.emofilm); its refit keeps
the name w33_<tag>_film. The film2v files written on the node by addendum 5's first pass (never queued) are superseded
with the filmv ones (SUPERSEDED): kept, never queued (to_enqueue skips them) and never read.
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from affect import catalog, paths
from affect.data.iemocap_splits import SPEAKER_OUT, UTTERANCE_5
from affect.registry import DATASETS
from affect.runs.spec import plain, validate_job
from affect.sweep import design

catalog.load_all()

NAME = "week33"
RULE = "How the results are read is fixed in RULE_week33.txt (analyses/week33_prep_20261109) before any run."
RULE_A1 = ("How the results are read is fixed in RULE_week33_addendum1.txt (analyses/week33_prep_20261109) before any "
           "run.")
RULE_A5 = ("How the results are read is fixed in RULE_week33_addendum1.txt and RULE_week33_addendum5.txt "
           "(analyses/week33_prep_20261109) before any run.")
RULE_A8 = ("How the results are read is fixed in RULE_week33_addendum1.txt, RULE_week33_addendum5.txt and "
           "RULE_week33_addendum8.txt (analyses/week33_prep_20261109) before any run.")
WHY = ("Week 33 (2026-11-09 to 2026-11-16; Dr. Luitel, 2026-11-09: speaker-specific folds are the main protocol, the "
       "validation speaker chooses, every model kept until this week's results are in)")
# tag -> (what it is, the week-32 speaker-specific refit sweep whose queued record holds the record's settings)
MODELS = {"wpool": ("my pooling head on Whisper large-v3's layer-31 frames", "heads32w_pool_spk"),
          "wrnn": ("my recurrent head on Whisper large-v3's layer-31 frames", "heads32w_rnn_spk"),
          "wtransformer": ("my Transformer head on Whisper large-v3's layer-31 frames", "heads32w_transformer_spk"),
          "wlif": ("my LIF head on Whisper large-v3's layer-31 frames", "heads32w_lif_spk"),
          "base": ("the baseline CNN", "spk32_base"),
          "cnn": ("the tuned CNN", "spk32_cnn"),
          "transformer": ("the CNN + Transformer", "spk32_transformer"),
          "threeblock": ("the 3-block CNN", "spk32_threeblock"),
          "lif": ("the CNN + LIF", "spk32_lif"),
          "crnn": ("the CNN + LSTM", "spk32_crnn")}
HEADS = ("wpool", "wrnn", "wtransformer", "wlif")
NETWORKS = ("base", "cnn", "transformer", "threeblock", "lif", "crnn")
SEEDS_SPEAKER = [11, 12, 13, 14, 15]
SEEDS_POOLED = [6, 7, 8, 9, 10]
# setup -> (data settings, words, seeds, the models it is run for, validation priority, refit priority)
SETUPS = {"spk4": ({"data.split": SPEAKER_OUT, "data.label_set": "four_merged"},
                   "four classes, speaker-specific folds (one speaker tests, the other speaker of the same session "
                   "validates, the other eight train)", SEEDS_SPEAKER, tuple(MODELS), 188, 189),
          "spk3": ({"data.split": SPEAKER_OUT, "data.label_set": "three_no_neutral"},
                   "three classes (no neutral), speaker-specific folds (one speaker tests, the other speaker of the "
                   "same session validates, the other eight train)", SEEDS_SPEAKER, tuple(MODELS), 186, 187),
          "pooled4": ({"data.split": UTTERANCE_5, "data.label_set": "four_merged"},
                      "four classes, session-agnostic folds (utterance_5fold)", SEEDS_POOLED, NETWORKS, 184, 185),
          "pooled3": ({"data.split": UTTERANCE_5, "data.label_set": "three_no_neutral"},
                      "three classes (no neutral), session-agnostic folds (utterance_5fold)", SEEDS_POOLED, NETWORKS,
                      182, 183),
          # addendum 1 (added 2026-11-10, RULE_week33_addendum1.txt)
          "impro4": ({"data.split": "leave_one_speaker_out_train_impro", "data.label_set": "four_merged"},
                     "four classes, speaker-specific folds trained and validated on improvised utterances only (one "
                     "speaker tests, all of its utterances; the other speaker of the same session validates on its "
                     "improvised ones; the other eight train on theirs)", SEEDS_SPEAKER, tuple(MODELS), 180, 181),
          # its split film_5fold_whole since addendum 8 (2026-11-14, RULE_week33_addendum8.txt); film_5fold_strat
          # from addendum 5 (2026-11-13), film_5fold before
          "film": ({"data.name": "emofilm", "data.label_set": "three_no_neutral", "data.subset": "all",
                    "data.split": "film_5fold_whole"},
                   "EmoFilm's three classes (happiness, anger and sadness as hap+exc, ang and sad), five folds of whole "
                   "films (a film's IDs under its title's several spellings counted as one film), dealt so that every "
                   "fold holds all three classes (one film fold tests, the next validates, the other three train)",
                   SEEDS_SPEAKER, tuple(MODELS), 178, 179),
          "wmel4": ({"data.split": SPEAKER_OUT, "data.label_set": "four_merged"},
                    "four classes, speaker-specific folds, on Whisper large-v3's own log-mel spectrogram (128 bands)",
                    SEEDS_SPEAKER, NETWORKS, 176, 177)}
# the film refit's tests on IEMOCAP (addendum 1): cold, then adapted to each of its ten speakers in turn for this many
# epochs at the record's peak rate divided by lr_divisor (written without float noise), on the other nine
FILM_WARM = {"folds": 10, "epochs": 10, "lr_divisor": 10}
# Whisper's log-mel for the networks (addendum 1): its bands, which the frequency masks' width is scaled to
WHISPER_MEL = {"n_mels": 128}


def root() -> Path:
    """The tree whose sweeps, artifacts and queue are read and written (affect.paths.repo_root)."""
    return paths.repo_root()


def out_dir() -> Path:
    return root() / "artifacts" / f"advance_{NAME}"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(text: str) -> None:
    out_dir().mkdir(parents=True, exist_ok=True)
    with open(out_dir() / "log.txt", "a", encoding="utf-8") as handle:
        handle.write(f"{now_utc()} {text}\n")
    print(text)


# addendum 5: the film chain's validation sweep has its own name, so that the superseded w33_<tag>_filmv (split
# film_5fold, folds test_F2 and test_F3 failed) stay as they are; its refit keeps w33_<tag>_film. Addendum 8: the same
# again for w33_<tag>_film2v (split film_5fold_strat, written by addendum 5's first pass, never queued)
VALIDATION_IDS = {"film": "film3v"}
SUPERSEDED = tuple(f"w33_{tag}_filmv" for tag in MODELS) + tuple(f"w33_{tag}_film2v" for tag in MODELS)
SUPERSEDED_BY = {"filmv": "RULE_week33_addendum5.txt", "film2v": "RULE_week33_addendum8.txt"}


def ids(tag: str, setup: str) -> tuple:
    """(the validation sweep's ID, the refit's ID) of one model in one setup."""
    return f"w33_{tag}_{VALIDATION_IDS.get(setup, setup + 'v')}", f"w33_{tag}_{setup}"


def chains() -> list:
    """(tag, setup) for every chain, in the order status lists them."""
    return [(tag, setup) for tag in MODELS for setup in SETUPS if tag in SETUPS[setup][3]]


# ---------------------------------------------------------------------------------------------------------------
# Reading sweeps and runs (validation files only), as week 32 reads them

def sweep_record(sweep_id: str):
    """(the enqueued sweep file, its jobs from jobs.csv in order) or (None, None) when it is not enqueued."""
    folder = root() / "artifacts" / "sweeps" / sweep_id
    if not (folder / "jobs.csv").is_file() or not (folder / "sweep_file.json").is_file():
        return None, None
    with open(folder / "jobs.csv", newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle))
    return json.loads((folder / "sweep_file.json").read_text(encoding="utf-8")), jobs


def complete_runs(sweep_id: str) -> dict:
    """run_id -> folder for the sweep's complete runs (spec.json, metrics_val.json and predictions_val.csv; not a
    .tmp or .failed- folder)."""
    base = root() / "artifacts" / "runs" / sweep_id
    out = {}
    if base.is_dir():
        for d in base.iterdir():
            if (d.is_dir() and not d.name.endswith(".tmp") and ".failed-" not in d.name
                    and (d / "spec.json").is_file() and (d / "metrics_val.json").is_file()
                    and (d / "predictions_val.csv").is_file()):
                out[d.name] = d
    return out


def failed_jobs(sweep_id: str) -> list:
    folder = root() / "queue" / "failed"
    return sorted(p.name for p in folder.glob(f"*__{sweep_id}__*.json")) if folder.is_dir() else []


def state(sweep_id: str) -> dict:
    """enqueued, jobs, done, failed and complete (every job has a complete run and none failed)."""
    sweep, jobs = sweep_record(sweep_id)
    if sweep is None:
        return {"enqueued": False, "jobs": 0, "done": 0, "failed": [], "complete": False}
    runs = complete_runs(sweep_id)
    done = sum(1 for j in jobs if j["run_id"] in runs)
    failed = failed_jobs(sweep_id)
    return {"enqueued": True, "jobs": len(jobs), "done": done, "failed": failed,
            "complete": done == len(jobs) and not failed and len(jobs) > 0}


def read_runs(sweep_id: str) -> tuple:
    """(sweep file, jobs, {run_id: (spec, metrics_val)}) of a complete sweep; every run's spec must match its job."""
    sweep, jobs = sweep_record(sweep_id)
    runs = complete_runs(sweep_id)
    out = {}
    for j in jobs:
        d = runs[j["run_id"]]
        spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
        metrics = json.loads((d / "metrics_val.json").read_text(encoding="utf-8"))
        if (spec["config_hash"] != j["config_hash"] or spec["fold"] != j["fold"] or int(spec["seed"]) != int(j["seed"])
                or spec["sweep_id"] != sweep_id):
            raise SystemExit(f"{d}: spec.json does not match its job in jobs.csv")
        out[j["run_id"]] = (spec, metrics)
    return sweep, jobs, out


def mean_validation_ua(sweep_id: str) -> tuple:
    """(the mean validation macro accuracy of a complete one-configuration sweep: each seed's mean over its folds of
    metrics_val.json's final UA, then the mean over the seeds, as analyze computes it; {seed: that seed's mean})."""
    sweep, jobs, runs = read_runs(sweep_id)
    got = {(int(j["seed"]), j["fold"]): float(runs[j["run_id"]][1]["final"]["ua"]) for j in jobs}
    seeds = sorted({s for s, _ in got})
    by_seed = {s: float(np.mean([got[(s, f)] for f in sweep["folds"]])) for s in seeds}
    return float(np.mean(list(by_seed.values()))), by_seed


def median_epoch(sweep_id: str, expected: int) -> tuple:
    """(the median of the best_epoch of every run of a complete one-configuration sweep, rounded half up; the epochs),
    as analyses/night30_prep_20261029/make_night30.py and week 32 compute the refit epochs."""
    _, jobs, runs = read_runs(sweep_id)
    if len({j["config_hash"] for j in jobs}) != 1:
        raise SystemExit(f"{sweep_id}: more than one configuration")
    epochs = sorted(int(runs[j["run_id"]][1]["best_epoch"]) for j in jobs)
    if len(epochs) != expected:
        raise SystemExit(f"{sweep_id}: {len(epochs)} runs, {expected} expected")
    m = statistics.median(epochs)
    return (int(m) if m == int(m) else int(m + 0.5)), epochs


# ---------------------------------------------------------------------------------------------------------------
# Writing sweeps

def expand(sweep: dict) -> tuple:
    return design.expand(sweep)


def folds_of(settings: dict) -> list:
    """The fold names of a configuration's data settings (IEMOCAP's split, or EmoFilm's film folds since addendum 1)."""
    job = validate_job(settings)
    return list(DATASETS.get(job.data_name).target().fold_names(job.data))


def written(sweep_id: str) -> bool:
    return (out_dir() / "sweeps" / f"{sweep_id}.json").is_file()


def write_atomically(path: Path, text: str) -> None:
    """The whole text or nothing: written beside the target, then renamed onto it, so that a reader running at the
    same time never reads half a file (and a half-written file never matches *.json)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_EXISTING = None


def existing_run_ids() -> set:
    global _EXISTING
    if _EXISTING is None:
        base = root() / "artifacts" / "runs"
        _EXISTING = {d.name for s in base.iterdir() if s.is_dir() for d in s.iterdir()} if base.is_dir() else set()
    return _EXISTING


def write_sweep(sweep: dict, n_jobs: int, decision: dict) -> Path:
    """Expand the sweep with the repository's own code, check its size (n_jobs jobs, one configuration), write it and
    its decision once into artifacts/advance_week33/sweeps/ and decisions/. A file already written is kept as it is (it
    must be the same sweep)."""
    jobs, summary = expand(sweep)
    if len(jobs) != n_jobs or summary["configs"] != 1 or len({j["run_id"] for j in jobs}) != n_jobs:
        raise SystemExit(f"{sweep['sweep_id']}: {len(jobs)} jobs and {summary['configs']} configurations, expected "
                         f"{n_jobs} and 1")
    folder = out_dir() / "sweeps"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sweep['sweep_id']}.json"
    text = json.dumps(sweep, indent=1) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} exists with other content; not overwritten")
        return path
    elsewhere = sorted({j["run_id"] for j in jobs} & existing_run_ids())
    if elsewhere:
        raise SystemExit(f"{sweep['sweep_id']}: {len(elsewhere)} of its run IDs exist already (e.g. {elsewhere[0]})")
    write_atomically(path, text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    decision = {"sweep_id": sweep["sweep_id"], "made_utc": now_utc(), "jobs": n_jobs, "configs": 1,
                "priority": sorted({j["priority"] for j in jobs}), "sweep_file_sha256": digest, **decision}
    (out_dir() / "decisions").mkdir(parents=True, exist_ok=True)
    write_atomically(out_dir() / "decisions" / f"{sweep['sweep_id']}.json", json.dumps(decision, indent=1) + "\n")
    log(f"written {sweep['sweep_id']}: {n_jobs} jobs, SHA-256 {digest[:16]}")
    return path


def with_overrides(settings: dict, dotted: dict) -> dict:
    out = json.loads(json.dumps(settings))
    for key, value in dotted.items():
        section, _, field = key.partition(".")
        if section not in out or field not in out[section]:
            raise SystemExit(f"no setting {key}")
        out[section][field] = value
    return out


def changed(before: dict, after: dict) -> list:
    """The dotted keys whose values differ between two settings of the same sections and keys."""
    out = []
    for section in before:
        if set(before[section]) != set(after[section]):
            raise SystemExit(f"section {section} has other keys")
        out += [f"{section}.{k}" for k in before[section] if before[section][k] != after[section][k]]
    return sorted(out)


def record_settings(tag: str) -> dict:
    """A model's week-32 record as its speaker-specific refit ran it (the queue's record of that sweep, one
    configuration), without the refit: train.refit none, train.refit_epochs none."""
    source = MODELS[tag][1]
    sweep, jobs = sweep_record(source)
    if sweep is None:
        raise SystemExit(f"{source} is not in artifacts/sweeps: the record of {tag} cannot be read")
    if len(sweep["stages"]) != 1 or sweep["stages"][0]["kind"] != "base" or len({j["config_hash"] for j in jobs}) != 1:
        raise SystemExit(f"{source}: one base configuration expected")
    settings = plain(validate_job(sweep["base"]))
    if settings["train"]["refit"] != "train_and_val" or settings["data"]["split"] != SPEAKER_OUT:
        raise SystemExit(f"{source}: not a speaker-specific refit")
    out = with_overrides(settings, {"train.refit": "none", "train.refit_epochs": None})
    return plain(validate_job(out))


def whisper_mel(settings: dict) -> dict:
    """A network's settings on Whisper's own log-mel (addendum 1): the features whisper_logmel with WHISPER_MEL's bands
    and the record's normalization, and the frequency masks' width scaled from the record's bands to those, rounded half
    up; nothing else changes."""
    if settings["features"]["name"] != "logmel":
        raise SystemExit("Whisper's log-mel replaces a network's own log-mel")
    out = json.loads(json.dumps(settings))
    bands = int(settings["features"]["n_mels"])
    out["features"] = {"name": "whisper_logmel", "n_mels": WHISPER_MEL["n_mels"],
                       "normalization": settings["features"]["normalization"]}
    width = int(settings["augment"]["freq_mask_width"])
    out["augment"]["freq_mask_width"] = int(width * WHISPER_MEL["n_mels"] / bands + 0.5) if width else 0
    return out


def setup_settings(tag: str, setup: str) -> tuple:
    """(the record's settings without the refit, the setup's validation settings, the dotted keys that differ, the
    sections replaced whole): the setup's data settings, and for wmel4 Whisper's log-mel (whisper_mel)."""
    dotted = SETUPS[setup][0]
    record = record_settings(tag)
    out = with_overrides(record, dotted)
    replaced = []
    if setup == "wmel4":
        out = whisper_mel(out)
        replaced = ["features"]
    base = plain(validate_job(out))
    differ = changed({k: v for k, v in record.items() if k not in replaced},
                     {k: v for k, v in base.items() if k not in replaced})
    allowed = set(dotted) | ({"augment.freq_mask_width"} if setup == "wmel4" else set())
    if not set(differ) <= allowed:
        raise SystemExit(f"{tag} {setup}: {differ} differ from the record, only {sorted(allowed)} may")
    return record, base, differ, replaced


def rule_of(setup: str) -> str:
    """The rule a setup's sweeps follow, as their decisions record it."""
    if setup in ("spk4", "spk3", "pooled4", "pooled3"):
        return "RULE_week33.txt part 2"
    return "RULE_week33_addendum8.txt" if setup == "film" else "RULE_week33_addendum1.txt"


def rule_words(setup: str) -> str:
    """The sentence a setup's sweep descriptions end with, naming the rule its reading follows."""
    return {"RULE_week33.txt part 2": RULE, "RULE_week33_addendum8.txt": RULE_A8}.get(rule_of(setup), RULE_A1)


def make_validation(tag: str, setup: str) -> Path:
    dotted, words, seeds, _, p_val, _ = SETUPS[setup]
    v_id, _ = ids(tag, setup)
    record, base, differ, replaced = setup_settings(tag, setup)
    folds = folds_of(base)
    source = MODELS[tag][1]
    what = f"the data settings {', '.join(sorted(dotted))}"
    if replaced:
        what += (f", the features (whisper_logmel, {WHISPER_MEL['n_mels']} bands, the record's normalization) and "
                 f"augment.freq_mask_width (scaled to {WHISPER_MEL['n_mels']} bands)")
    rule = rule_of(setup)
    sweep = {"sweep_id": v_id,
             "description": (f"{WHY}: {MODELS[tag][0]}, its week-32 record as {source} ran it, in {words}, not "
                             f"refitted: each run trains on its fold's training side and stops early on its validation "
                             f"side, which chooses the refit's epochs ({rule}). Only {what} and train.refit and "
                             f"train.refit_epochs differ from the record. Seeds {seeds[0]} to {seeds[-1]}. "
                             f"{rule_words(setup)} Priority {p_val}."),
             "folds": folds, "base": base,
             "stages": [{"name": "validation", "kind": "base", "priority": p_val, "seeds": list(seeds)}]}
    decision = {"rule": rule, "from": source, "setup": setup, "changed_from_record": differ}
    if replaced:
        decision["replaced_from_record"] = {k: record[k] for k in replaced}
    return write_sweep(sweep, len(folds) * len(seeds), decision)


def film_tests(settings: dict) -> dict:
    """The film refit's tests on IEMOCAP (addendum 1): train.external_test iemocap and train.warm_start iemocap, ten
    speakers held out in turn, FILM_WARM's epochs at the record's peak rate divided by its lr_divisor (written without
    float noise, as week 32's films write theirs)."""
    lr = float(f"{float(settings['train']['lr']) / FILM_WARM['lr_divisor']:.6g}")
    return {"train.external_test": "iemocap", "train.warm_start": "iemocap", "train.warm_start_folds": FILM_WARM["folds"],
            "train.warm_start_epochs": FILM_WARM["epochs"], "train.warm_start_lr": lr}


def make_refit(tag: str, setup: str) -> Path:
    _, words, seeds, _, _, p_refit = SETUPS[setup]
    v_id, r_id = ids(tag, setup)
    sweep, jobs, _ = read_runs(v_id)
    epochs, best = median_epoch(v_id, len(sweep["folds"]) * len(seeds))
    ua, by_seed = mean_validation_ua(v_id)
    settings = plain(validate_job(sweep["base"]))
    extra = film_tests(settings) if setup == "film" else {}
    base = with_overrides(settings, {"train.refit": "train_and_val", "train.refit_epochs": int(epochs),
                                     "train.eval_train": "none", **extra})
    base = plain(validate_job(base))
    assert set(changed(settings, base)) <= {"train.refit", "train.refit_epochs", "train.eval_train", *extra}
    rule = rule_of(setup)
    tests = ""
    if extra:
        tests = (f" Its last weights also score IEMOCAP's utterances of the three classes, cold (train.external_test "
                 f"iemocap), then adapted to each of IEMOCAP's ten speakers in turn: {FILM_WARM['epochs']} epochs at "
                 f"{extra['train.warm_start_lr']:g}, a tenth of the record's peak rate, on the other nine speakers' "
                 f"utterances, then that speaker scored (train.warm_start iemocap).")
    out = {"sweep_id": r_id,
           "description": (f"{WHY}: {MODELS[tag][0]} in {words}, refitted on each fold's training and validation sides "
                           f"together for {epochs} epochs, the median of the epochs its {len(best)} runs of {v_id} "
                           f"chose on validation ({rule}), on its own learning-rate schedule; tested once with its last "
                           f"weights.{tests} Seeds {seeds[0]} to {seeds[-1]}. "
                           f"{rule_words(setup)} Priority {p_refit}."),
           "folds": list(sweep["folds"]), "base": base,
           "stages": [{"name": "refit", "kind": "base", "priority": p_refit, "seeds": list(seeds)}]}
    decision = {"rule": rule, "from": v_id, "refit_epochs": epochs, "best_epochs": best, "validation_ua": ua,
                "validation_ua_by_seed": {str(k): v for k, v in by_seed.items()}}
    if extra:
        decision["film_tests"] = extra
    return write_sweep(out, len(sweep["folds"]) * len(seeds), decision)


# ---------------------------------------------------------------------------------------------------------------
# One pass

def advance_chain(tag: str, setup: str) -> str:
    """Write whatever is due in one chain; return a one-line state."""
    v_id, r_id = ids(tag, setup)
    if not written(v_id) and not state(v_id)["enqueued"]:
        make_validation(tag, setup)
    v = state(v_id)
    if not v["enqueued"]:
        return f"{setup}: validation written, not yet queued"
    if v["failed"]:
        return f"{setup}: WAITING, validation has {len(v['failed'])} failed jobs (e.g. {v['failed'][0]})"
    if not v["complete"]:
        return f"{setup}: validation {v['done']}/{v['jobs']}"
    if not written(r_id) and not state(r_id)["enqueued"]:
        make_refit(tag, setup)
    r = state(r_id)
    if not r["enqueued"]:
        return f"{setup}: validation done, refit written, not yet queued"
    return (f"{setup}: validation done, refit {'FAILED ' if r['failed'] else ''}{r['done']}/{r['jobs']}"
            + (" (complete)" if r["complete"] else ""))


def to_enqueue() -> list:
    """The sweep files written here and not queued, without a smoke failure marked, highest priority first; a tie in
    the order of MODELS (Whisper's heads first, then the networks), then of SETUPS."""
    folder = out_dir() / "sweeps"
    order = {sweep_id: i for i, (tag, setup) in enumerate(chains()) for sweep_id in ids(tag, setup)}
    files = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        sweep_id = path.stem
        if sweep_id in SUPERSEDED:                 # addendum 8: kept, never queued
            continue
        if state(sweep_id)["enqueued"] or (out_dir() / "smoke_failed" / f"{sweep_id}.txt").exists():
            continue
        try:
            sweep = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            print(f"ERROR: {path.name} unreadable ({err}); not queued")
            continue
        files.append((-max(st["priority"] for st in sweep["stages"]), order.get(sweep_id, len(order)), sweep_id, path))
    return [p for _, _, _, p in sorted(files)]


def cmd_advance(args) -> int:
    """Each chain on its own: one that fails (an unreadable file, a check that does not hold) is reported as ERROR and
    logged, and the others go on."""
    for tag, setup in chains():
        try:
            print(f"w33_{tag}: {advance_chain(tag, setup)}")
        except (Exception, SystemExit) as err:
            log(f"w33_{tag}: {setup}: ERROR {type(err).__name__}: {err}")        # log prints it too
    for path in to_enqueue():
        print(f"ENQUEUE {path}")
    return 0


def cmd_status(args) -> int:
    for tag in MODELS:
        parts = []
        for setup in SETUPS:
            if tag not in SETUPS[setup][3]:
                continue
            for sweep_id in ids(tag, setup):
                key = sweep_id[len(f"w33_{tag}_"):]
                s = state(sweep_id)
                if s["enqueued"]:
                    parts.append(f"{key} {s['done']}/{s['jobs']}" + (f" ({len(s['failed'])} failed)" if s["failed"]
                                                                     else ""))
                elif written(sweep_id):
                    parts.append(f"{key} written")
        print(f"w33_{tag}: " + (", ".join(parts) or "nothing written"))
    for kind, rule in SUPERSEDED_BY.items():
        old = [f"{s[len('w33_'):]} " + (f"{state(s)['done']}/{state(s)['jobs']}" if state(s)["enqueued"] else "written")
               for s in SUPERSEDED if s.endswith(f"_{kind}") and (state(s)["enqueued"] or written(s))]
        if old:
            print(f"superseded ({rule}; kept, never read): {', '.join(old)}")
    marks = sorted((out_dir() / "smoke_failed").glob("*.txt")) if (out_dir() / "smoke_failed").is_dir() else []
    for m in marks:
        print(f"SMOKE FAILED, not queued: {m.stem} ({m.read_text(encoding='utf-8').strip()})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("advance").set_defaults(func=cmd_advance)
    sub.add_parser("status").set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
