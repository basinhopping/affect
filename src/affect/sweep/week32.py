"""Week 32's automatic stages and the new encoders' sweeps (added 2026-11-02), exactly as RULE_week32.txt fixes them
(artifacts/analyses/week32_prep_20261102, fixed before any run of these sweeps). It reads validation files only: each
sweep's record (artifacts/sweeps/<id>/sweep_file.json and jobs.csv) and the spec.json and metrics_val.json of complete
run folders; it never opens a test file. What it writes goes to artifacts/advance_week32/: the stages' sweep files
(sweeps/), the new encoders' sweep files (encoders/), one decision file per sweep with what the sweep was made from
(decisions/), the layers chosen for an encoder (layers_<model>.json) and a log (log.txt). It never smoke-tests or
queues anything itself: scripts/advance_week.sh queues the stages it lists as ENQUEUE <path>, and scripts/enc32.sh the
screens it lists as SCREEN <path>, both with scripts/smoke_enqueue_files.sh.

    python -m affect.sweep.week32 advance          every stage sweep that is due is written; then ENQUEUE <file> for
                                                   each sweep file written and neither queued nor marked as failed
    python -m affect.sweep.week32 status           each chain's state, one line per head and encoder
    python -m affect.sweep.week32 choose-layers --model w2v_bert_2       (or whisper_small, bud_e_whisper,
                                                                          whisper_medium)
    python -m affect.sweep.week32 encoder-sweeps --model ast_audioset    (or w2v_bert_2, whisper_small, bud_e_whisper
                                                                          or whisper_medium, once its layers are
                                                                          chosen)
    python -m affect.sweep.week32 check-frames --model <model> --layer <k> [--dataset emofilm]

(on the node with the environment of scripts/common.sh: affect_python -m affect.sweep.week32 ...). The week's numbers
(priorities, seeds, the screens' size, AST's layer, the films' warm start) are WEEK32, a settings object (WeekSettings).

The chains (RULE part 2): for each encoder (whisper_large_v3 "w", ast_audioset "a", w2v_bert_2 "v") and head (pool,
rnn, transformer, lif), the screen heads32<x>_<head>_screen, then
  confirm      once the screen has every run: its two configurations with the highest mean validation macro accuracy
               (each seed's mean over the 5 folds of metrics_val.json's final UA, then the mean over the seeds), a tie
               to the configuration listed first in the screen's jobs.csv; seeds 6 to 10; priority 190
  record       once the confirmation has every run: of its two configurations, the higher mean validation macro accuracy
               over seeds 6 to 10, a tie to rank 1; its refit epochs, the median of its 25 runs' best_epoch rounded
               half up. Then, seeds 6 to 10: refit (training plus validation, priority 195); spk, script, oscript
               (speaker folds and the script splits, refitted for the same epochs, priority 192); pooled4, so3, pooled3
               (the record's configuration in that setup, not refitted, priority 191)
  setup refit  once a setup sweep has every run: refitted for the median of its own 25 runs' best_epoch (priority 191)
A sweep with a failed job is not read: its chain waits, and status says so.

The new encoders (RULE part 6): AST's frames are those of layer 8, the layer its four-class session-specific probe chose
on validation on night 30 (probe30_ast_audioset_so4); w2v-BERT 2.0's those of the layer its four-class
session-specific probe chooses here (choose-layers: the highest mean validation macro accuracy over the 5 folds, seed
1, a tie to the lower layer, as night 30 chose). Each encoder's head screens are Whisper's: the base of
heads31w_<head>_so4 and 30 random configurations of heads27_<head>'s space with sampling seed 20261102, the encoder and
layer changed; on AST, whose frames come 10 a second instead of 50, the time masks' widths (counted in frames) are
divided by 5 so that they span the same seconds. Priorities: AST 165, w2v-BERT 2.0 164.

Added 2026-11-05, as RULE_week32_addendum7.txt fixes them (before any of their runs):
  Whisper Small    a fourth encoder, letter "s" (openai/whisper-small: 12 layers of 768 and its input embedding, 50
                   frames a second), done as w2v-BERT 2.0 is: its probe sweeps in the four setups, each setup's layer
                   chosen on validation (choose-layers), its probe refits and checks, its frames at the so4 layer,
                   checked, and its four head screens (Whisper's search, priority 163), then part 2's stages.
  film             once a chain's refit is queued and EmoFilm's frames of its encoder and layer have passed check-frames
                   (artifacts/advance_week32/frames_checked/emofilm_<model>_l<k>.json): heads32<x>_<head>_film, the refit
                   with train.external_test emofilm (its last weights score EmoFilm's 726 clips, cold) and
                   train.warm_start emofilm (then 5 film folds, 10 epochs each at a tenth of the head's learning rate,
                   as warm30), seeds 6 to 10, priority 194. Nothing is chosen from these runs.
  network films    film32_<net>_so4: the same for the networks' final records (refit30_<net>_so4 for base, cnn,
                   transformer, threeblock and lif, refit32_crnn_so4), each from the sweep file the queue holds, priority
                   168; written by advance once those refits are queued.

Added 2026-11-07, as RULE_week32_addendum9.txt part N fixes them (before any of their runs):
  BUD-E-Whisper    a fifth encoder, letter "b" (laion/BUD-E-Whisper: Whisper Small fine-tuned to caption emotional
                   speech, Whisper Small's 13 hidden states of 768), and
  Whisper medium   a sixth, letter "m" (openai/whisper-medium: 24 layers of 1024 and its input embedding), each done as
                   Whisper Small is (scripts/enc32x.sh): its probe sweeps, layers, probe refits and checks, its frames
                   checked, EmoFilm's frames, and its four head screens (Whisper's search; priority 162 BUD-E-Whisper,
                   161 Whisper medium), then part 2's stages and the films.
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from affect import catalog, paths
from affect.data.iemocap_splits import OTHER_SCRIPT_HELD_OUT, SCRIPT_HELD_OUT, SPEAKER_OUT, UTTERANCE_5, fold_names
from affect.runs.spec import config_hash, plain, validate_job
from affect.sweep import design

catalog.load_all()


@dataclasses.dataclass(frozen=True)
class WeekSettings:
    name: str                          # the folder artifacts/advance_<name>/
    prefix: str                        # the chains' sweep IDs: <prefix><letter>_<head>_<stage>
    letters: tuple[str, ...]           # one letter per encoder, in the order of models
    models: tuple[str, ...]            # the encoders (affect.classical.settings.PRETRAINED_MODELS)
    screen_priorities: tuple[int, ...]  # each encoder's screen priority, in the order of models
    heads: tuple[str, ...]             # the heads, in the order a tie between them is broken
    ast_layer: int                     # the layer of AST's frames
    seeds_screen: tuple[int, ...]
    seeds_record: tuple[int, ...]
    screen_configs: int                # configurations in a screen: the base and the random ones
    random_configs: int
    sampling_seed: int                 # the screens' random draw
    priority_confirm: int
    priority_refit: int
    priority_protocol: int             # speaker folds and script splits
    priority_setup: int                # the other setups and their refits
    time_mask_divisor: int             # on AST: 50 frames a second over 10
    # added 2026-11-05 (RULE_week32_addendum7.txt): the films of the final records
    priority_film: int                 # the heads' films (heads32<x>_<head>_film)
    priority_network_film: int         # the networks' films (film32_<net>_so4)
    film_folds: int                    # the warm start's film folds
    film_epochs: int                   # its epochs on the other folds' clips
    film_lr_divisor: int               # its constant learning rate: the record's peak rate over this


WEEK32 = WeekSettings(name="week32", prefix="heads32", letters=("w", "a", "v", "s", "b", "m"),
                      models=("whisper_large_v3", "ast_audioset", "w2v_bert_2", "whisper_small", "bud_e_whisper",
                              "whisper_medium"),
                      screen_priorities=(185, 165, 164, 163, 162, 161),
                      heads=("pool", "rnn", "transformer", "lif"), ast_layer=8, seeds_screen=(1, 2, 3),
                      seeds_record=(6, 7, 8, 9, 10), screen_configs=31, random_configs=30, sampling_seed=20261102,
                      priority_confirm=190, priority_refit=195, priority_protocol=192, priority_setup=191,
                      time_mask_divisor=5, priority_film=194, priority_network_film=168, film_folds=5, film_epochs=10,
                      film_lr_divisor=10)
RULE = "How the results are read is fixed in RULE_week32.txt (analyses/week32_prep_20261102) before any run."
RULE7 = ("How the results are read is fixed in RULE_week32_addendum7.txt (analyses/week32_prep_20261102) before any "
         "run.")
RULE9 = ("How the results are read is fixed in RULE_week32_addendum9.txt (analyses/week32_prep_20261102) before any "
         "run.")
# the networks' final records whose films are run (RULE_week32_addendum7.txt part F): tag -> its refit sweep
NETWORK_FILMS = (("base", "refit30_base_so4"), ("cnn", "refit30_cnn_so4"), ("transformer", "refit30_transformer_so4"),
                 ("threeblock", "refit30_threeblock_so4"), ("lif", "refit30_lif_so4"), ("crnn", "refit32_crnn_so4"))
WHY = ("Week 32 (2026-11-02 to 2026-11-09; Dr. Luitel, 2026-11-02: spectrogram models only, only refits reported, "
       "speaker-specific against session-specific, scripted against improvised)")
HEADS = WEEK32.heads
ENCODERS = dict(zip(WEEK32.letters, WEEK32.models))
NAMES = {"whisper_large_v3": "Whisper large-v3", "ast_audioset": "AST", "w2v_bert_2": "w2v-BERT 2.0",
         "whisper_small": "Whisper Small", "bud_e_whisper": "BUD-E-Whisper", "whisper_medium": "Whisper medium"}
LETTER = {m: x for x, m in ENCODERS.items()}
SCREEN_PRIORITY = dict(zip(WEEK32.models, WEEK32.screen_priorities))
AST_LAYER = WEEK32.ast_layer                # probe30_ast_audioset_so4's layer, chosen on validation on night 30
SEEDS_SCREEN = list(WEEK32.seeds_screen)
SEEDS_RECORD = list(WEEK32.seeds_record)
SAMPLING_SEED = WEEK32.sampling_seed
P_CONFIRM, P_REFIT = WEEK32.priority_confirm, WEEK32.priority_refit
P_PROTOCOL, P_SETUP = WEEK32.priority_protocol, WEEK32.priority_setup
P_FILM, P_NETWORK_FILM = WEEK32.priority_film, WEEK32.priority_network_film
# the encoders that choose their layers here, and the rule each one's own sweeps cite (Whisper Small: added 2026-11-05;
# BUD-E-Whisper and Whisper medium: added 2026-11-07)
CHOSEN_HERE = {"w2v_bert_2": "RULE_week32.txt part 6", "whisper_small": "RULE_week32_addendum7.txt part S",
               "bud_e_whisper": "RULE_week32_addendum9.txt part N", "whisper_medium": "RULE_week32_addendum9.txt part N"}
SHORT = {"w2v_bert_2": "w2v", "whisper_small": "wsmall",     # in the classical sweeps' IDs (classical_spk32_<short>)
         "bud_e_whisper": "bude", "whisper_medium": "wmed"}
# the rule an encoder's screens and probe refits cite, and the closing sentence of their descriptions: AST's and w2v-BERT
# 2.0's as written on 2026-11-02, Whisper Small's as on 2026-11-05 (the bytes of those sweeps do not change)
SWEEP_RULES = {"whisper_small": ("RULE_week32_addendum7.txt part S", RULE7),
               "bud_e_whisper": ("RULE_week32_addendum9.txt part N", RULE9),
               "whisper_medium": ("RULE_week32_addendum9.txt part N", RULE9)}
ENCODER_SWEEP_MODELS = ("ast_audioset", "w2v_bert_2", "whisper_small", "bud_e_whisper", "whisper_medium")
SPLITS = {"spk": (SPEAKER_OUT, "speaker-specific folds (one speaker tests, the other speaker of the same session "
                               "validates, the other eight speakers train)"),
          "script": (SCRIPT_HELD_OUT, "the script-held-out folds (each test session's utterances of one script, that "
                                      "script kept out of training and validation)"),
          "oscript": (OTHER_SCRIPT_HELD_OUT, "the control of the script-held-out folds (the same test utterances, "
                                             "another script kept out of training and validation)")}
SETUPS = {"pooled4": ({"data.split": UTTERANCE_5}, "four classes, session-agnostic folds (utterance_5fold)"),
          "so3": ({"data.label_set": "three_no_neutral"}, "three classes (no neutral), session-specific folds"),
          "pooled3": ({"data.label_set": "three_no_neutral", "data.split": UTTERANCE_5},
                      "three classes (no neutral), session-agnostic folds")}
PROBE_SETUPS = ("so4", "pooled4", "so3", "pooled3")


def root() -> Path:
    """The tree whose sweeps, artifacts and queue are read and written (affect.paths.repo_root)."""
    return paths.repo_root()


def out_dir() -> Path:
    return root() / "artifacts" / f"advance_{WEEK32.name}"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(text: str) -> None:
    out_dir().mkdir(parents=True, exist_ok=True)
    with open(out_dir() / "log.txt", "a", encoding="utf-8") as handle:
        handle.write(f"{now_utc()} {text}\n")
    print(text)


# ---------------------------------------------------------------------------------------------------------------
# Reading sweeps and runs (validation files only)

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


def rank(sweep_id: str, seeds: list) -> list:
    """Every configuration of a complete sweep, highest mean validation macro accuracy first, as analyze computes it
    (affect.analysis.validation: each seed's mean over the sweep's folds of metrics_val.json's final UA, then the mean
    over the seeds); a tie goes to the configuration whose first job comes first in jobs.csv."""
    sweep, jobs, runs = read_runs(sweep_id)
    folds = list(sweep["folds"])
    first, by_config = {}, {}
    for j in jobs:
        h = j["config_hash"]
        first.setdefault(h, int(j["seq"]))
        spec, metrics = runs[j["run_id"]]
        entry = by_config.setdefault(h, {"config_hash": h, "stage": j["stage"], "label": j["label"],
                                         "settings": spec["settings"], "ua_by_run": {}})
        entry["ua_by_run"][(int(j["seed"]), j["fold"])] = float(metrics["final"]["ua"])
    table = []
    for h, entry in by_config.items():
        got = entry.pop("ua_by_run")
        if sorted(got) != sorted((s, f) for s in seeds for f in folds):
            raise SystemExit(f"{sweep_id}: configuration {h[:12]} has runs {sorted(got)[:3]}..., not seeds {seeds} x "
                             f"folds {folds}")
        by_seed = {s: float(np.mean([got[(s, f)] for f in folds])) for s in sorted(seeds)}
        entry.update({"ua": float(np.mean(list(by_seed.values()))), "ua_by_seed": by_seed, "first_seq": first[h]})
        table.append(entry)
    table.sort(key=lambda e: (-e["ua"], e["first_seq"]))
    return table


def median_epoch(sweep_id: str, config: str) -> tuple:
    """(the median of the best_epoch of a configuration's runs in a complete sweep, rounded half up, the epochs), as
    analyses/night30_prep_20261029/make_night30.py computes the refit epochs."""
    _, jobs, runs = read_runs(sweep_id)
    epochs = sorted(int(runs[j["run_id"]][1]["best_epoch"]) for j in jobs if j["config_hash"] == config)
    if len(epochs) != 25:
        raise SystemExit(f"{sweep_id}: {len(epochs)} runs of {config[:12]}, 25 expected")
    m = statistics.median(epochs)
    return (int(m) if m == int(m) else int(m + 0.5)), epochs


# ---------------------------------------------------------------------------------------------------------------
# Writing sweeps

def overrides(base: dict, other: dict) -> dict:
    """The dotted keys at which other's settings differ from base's (both validated; the same kinds of section)."""
    a, b = plain(validate_job(base)), plain(validate_job(other))
    out = {}
    for section in a:
        if a[section].get("name") != b[section].get("name") or set(a[section]) != set(b[section]):
            raise SystemExit(f"section {section} differs in kind or keys between the two configurations")
        for k in a[section]:
            if a[section][k] != b[section][k]:
                out[f"{section}.{k}"] = b[section][k]
    return out


def changed(before: dict, after: dict) -> list:
    return sorted(overrides(before, after))


def is_classical(sweep: dict) -> bool:
    return "functionals" in sweep["base"]


def expand(sweep: dict) -> tuple:
    if is_classical(sweep):
        from affect.classical import runner
        return runner.expand(sweep)
    return design.expand(sweep)


def written(sweep_id: str):
    path = out_dir() / "sweeps" / f"{sweep_id}.json"
    return path if path.is_file() else None


def write_atomically(path: Path, text: str) -> None:
    """The whole text or nothing: written beside the target, then renamed onto it, so that a reader running at the
    same time (an advance pass while enc32 writes its sweeps) never reads half a file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_sweep(sweep: dict, n_jobs: int, n_configs: int, decision: dict, where: str = "sweeps") -> Path:
    """Expand the sweep with the repository's own code, check its size, write it and its decision once, into
    artifacts/advance_week32/<where>/: sweeps/ for the stages (advance_week.sh queues them), encoders/ for the new
    encoders' sweeps (enc32.sh queues their screens once their frames are checked). A file already written is kept as
    it is (it must be the same sweep)."""
    jobs, summary = expand(sweep)
    if len(jobs) != n_jobs or summary["configs"] != n_configs or len({j["run_id"] for j in jobs}) != n_jobs:
        raise SystemExit(f"{sweep['sweep_id']}: {len(jobs)} jobs and {summary['configs']} configurations, expected "
                         f"{n_jobs} and {n_configs}")
    folder = out_dir() / where
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sweep['sweep_id']}.json"
    text = json.dumps(sweep, indent=1) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} exists with other content; not overwritten")
        return path
    elsewhere = sorted({j["run_id"] for j in jobs} & existing_run_ids())
    write_atomically(path, text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    decision = {"sweep_id": sweep["sweep_id"], "made_utc": now_utc(), "jobs": n_jobs, "configs": n_configs,
                "priority": sorted({j["priority"] for j in jobs}), "sweep_file_sha256": digest,
                "runs_with_the_same_id_elsewhere": len(elsewhere), **decision}
    (out_dir() / "decisions").mkdir(parents=True, exist_ok=True)
    write_atomically(out_dir() / "decisions" / f"{sweep['sweep_id']}.json", json.dumps(decision, indent=1) + "\n")
    note = (f"; {len(elsewhere)} of its run IDs exist in other sweeps (the same configuration, fold and seed: a rerun "
            f"that checks the runs repeat)" if elsewhere else "")
    log(f"written {sweep['sweep_id']}: {n_jobs} jobs, {n_configs} configurations, SHA-256 {digest[:16]}{note}")
    return path


_EXISTING = None


def existing_run_ids() -> set:
    global _EXISTING
    if _EXISTING is None:
        base = root() / "artifacts" / "runs"
        _EXISTING = {d.name for s in base.iterdir() if s.is_dir() for d in s.iterdir()} if base.is_dir() else set()
    return _EXISTING


def refit_of(settings: dict, epochs: int) -> dict:
    out = json.loads(json.dumps(settings))
    out["train"].update({"refit": "train_and_val", "refit_epochs": int(epochs), "eval_train": "none"})
    return out


def with_overrides(settings: dict, dotted: dict) -> dict:
    out = json.loads(json.dumps(settings))
    for key, value in dotted.items():
        section, _, field = key.partition(".")
        if section not in out or field not in out[section]:
            raise SystemExit(f"no setting {key}")
        out[section][field] = value
    return out


# ---------------------------------------------------------------------------------------------------------------
# The stages of one chain

def chain_ids(letter: str, head: str) -> dict:
    p = f"heads32{letter}_{head}"
    ids = {"screen": f"{p}_screen", "confirm": f"{p}_confirm", "refit": f"{p}_refit"}
    ids.update({k: f"{p}_{k}" for k in SPLITS})
    ids.update({k: f"{p}_{k}" for k in SETUPS})
    ids.update({f"{k}_refit": f"{p}_{k}_refit" for k in SETUPS})
    ids["film"] = f"{p}_film"                  # added 2026-11-05 (RULE_week32_addendum7.txt part F)
    return ids


def make_confirm(letter: str, head: str, ids: dict):
    table = rank(ids["screen"], SEEDS_SCREEN)
    if len(table) != WEEK32.screen_configs:
        raise SystemExit(f"{ids['screen']}: {len(table)} configurations, {WEEK32.screen_configs} expected")
    first, second = table[0], table[1]
    base = plain(validate_job(first["settings"]))
    if config_hash(validate_job(base)) != first["config_hash"]:
        raise SystemExit(f"{ids['screen']}: rank 1's settings do not give its configuration hash")
    variant = overrides(base, second["settings"])
    if config_hash(validate_job(design.apply(base, variant))) != second["config_hash"]:
        raise SystemExit(f"{ids['screen']}: rank 2's overrides do not give its configuration hash")
    model = NAMES[ENCODERS[letter]]
    name2 = f"rank 2: {ids['screen']} {second['label']} ({second['config_hash'][:12]})"
    sweep = {"sweep_id": ids["confirm"],
             "description": (f"{WHY}: the confirmation of my {head} head on {model}'s frames (RULE_week32.txt part "
                             f"2a). The two configurations of {ids['screen']} with the highest mean validation macro "
                             f"accuracy over its 5 folds and seeds 1 to 3, each exactly as the screen ran it: rank 1, "
                             f"{first['label']} ({first['config_hash'][:12]}, {first['ua']:.4f}), as the base, and "
                             f"rank 2, {second['label']} ({second['config_hash'][:12]}, {second['ua']:.4f}), as the "
                             f"variant; seeds 6 to 10, which no screen used. The record is the one with the higher mean "
                             f"validation macro accuracy over these seeds, a tie to rank 1, chosen before any test file "
                             f"is read. {RULE} Priority {P_CONFIRM}."),
             "folds": [f"test_S{k}" for k in range(1, 6)], "base": base,
             "stages": [{"name": "rank_1", "kind": "base", "priority": P_CONFIRM, "seeds": SEEDS_RECORD},
                        {"name": "rank_2", "kind": "variants", "priority": P_CONFIRM, "seeds": SEEDS_RECORD,
                         "variants": {name2: variant}}]}
    ranking = [{k: e[k] for k in ("config_hash", "stage", "label", "ua", "ua_by_seed", "first_seq")} for e in table]
    return write_sweep(sweep, 50, 2, {"rule": "RULE_week32.txt part 2a", "from": ids["screen"],
                                      "chosen": [first["config_hash"], second["config_hash"]], "ranking": ranking})


def record_of(ids: dict) -> dict:
    """The record of a complete confirmation (part 2b): its configuration, settings, validation and refit epochs."""
    table = rank(ids["confirm"], SEEDS_RECORD)
    if len(table) != 2:
        raise SystemExit(f"{ids['confirm']}: {len(table)} configurations, 2 expected")
    by_stage = {e["stage"]: e for e in table}
    r1, r2 = by_stage["rank_1"], by_stage["rank_2"]
    chosen = r2 if r2["ua"] > r1["ua"] else r1               # a tie to rank 1
    epochs, best = median_epoch(ids["confirm"], chosen["config_hash"])
    return {"config_hash": chosen["config_hash"], "stage": chosen["stage"], "label": chosen["label"],
            "settings": plain(validate_job(chosen["settings"])), "ua": chosen["ua"], "ua_by_seed": chosen["ua_by_seed"],
            "other": {"config_hash": (r1 if chosen is r2 else r2)["config_hash"],
                      "ua": (r1 if chosen is r2 else r2)["ua"]},
            "refit_epochs": epochs, "best_epochs": best}


def make_stage2(letter: str, head: str, ids: dict, rec: dict) -> list:
    model = NAMES[ENCODERS[letter]]
    settings, h, e = rec["settings"], rec["config_hash"], rec["refit_epochs"]
    words = (f"the record of my {head} head on {model}'s frames, {ids['confirm']} {rec['stage']} ({h[:12]}, validation "
             f"macro accuracy {rec['ua']:.4f} over seeds 6 to 10; RULE_week32.txt part 2b)")
    decision = {"rule": "RULE_week32.txt part 2b", "from": ids["confirm"], "record": rec}
    made = []
    base = refit_of(settings, e)
    assert set(changed(settings, base)) <= {"train.refit", "train.refit_epochs", "train.eval_train"}
    made.append(write_sweep(
        {"sweep_id": ids["refit"],
         "description": (f"{WHY}: {words}, refitted on each fold's training and validation sessions together for {e} "
                         f"epochs (the median of the epochs its 25 runs chose on validation), on its own learning-rate "
                         f"schedule; tested once with its last weights. Seeds 6 to 10, four classes, session-specific. "
                         f"{RULE} Priority {P_REFIT}."),
         "folds": [f"test_S{k}" for k in range(1, 6)], "base": base,
         "stages": [{"name": "refit", "kind": "base", "priority": P_REFIT, "seeds": SEEDS_RECORD}]},
        25, 1, decision))
    for key, (split, text) in SPLITS.items():
        base = refit_of(settings, e)
        base["data"]["split"] = split
        assert set(changed(settings, base)) <= {"data.split", "train.refit", "train.refit_epochs", "train.eval_train"}
        folds = list(fold_names(split))
        made.append(write_sweep(
            {"sweep_id": ids[key],
             "description": (f"{WHY}: {words}, in {text}, refitted on its training and validation speakers together for "
                             f"{e} epochs, its session-specific refit epochs, so that nothing is chosen on these folds. "
                             f"Only data.split, train.refit, train.refit_epochs and train.eval_train differ from the "
                             f"record. Seeds 6 to 10. {RULE} Priority {P_PROTOCOL}."),
             "folds": folds, "base": base,
             "stages": [{"name": "refit", "kind": "base", "priority": P_PROTOCOL, "seeds": SEEDS_RECORD}]},
            len(folds) * 5, 1, decision))
    for key, (dotted, text) in SETUPS.items():
        base = with_overrides(settings, dotted)
        assert set(changed(settings, base)) == set(dotted)
        split = base["data"]["split"]
        made.append(write_sweep(
            {"sweep_id": ids[key],
             "description": (f"{WHY}: {words}, its configuration in {text}, not refitted (the setup's own refit "
                             f"follows once these runs exist, for the median epoch they choose). Seeds 6 to 10. "
                             f"{RULE} Priority {P_SETUP}."),
             "folds": list(fold_names(split)), "base": base,
             "stages": [{"name": "record", "kind": "base", "priority": P_SETUP, "seeds": SEEDS_RECORD}]},
            25, 1, decision))
    return made


def make_setup_refit(letter: str, head: str, ids: dict, key: str):
    sweep, jobs, _ = read_runs(ids[key])
    configs = {j["config_hash"] for j in jobs}
    if len(configs) != 1:
        raise SystemExit(f"{ids[key]}: {len(configs)} configurations, 1 expected")
    h = configs.pop()
    e, best = median_epoch(ids[key], h)
    settings = plain(validate_job(sweep["base"]))
    base = refit_of(settings, e)
    model = NAMES[ENCODERS[letter]]
    return write_sweep(
        {"sweep_id": ids[f"{key}_refit"],
         "description": (f"{WHY}: my {head} head's record on {model}'s frames in {SETUPS[key][1]} ({ids[key]}, "
                         f"{h[:12]}), refitted on training plus validation for {e} epochs, the median of the epochs its "
                         f"25 runs there chose on validation (RULE_week32.txt part 2c). Seeds 6 to 10. {RULE} Priority "
                         f"{P_SETUP}."),
         "folds": list(sweep["folds"]), "base": base,
         "stages": [{"name": "refit", "kind": "base", "priority": P_SETUP, "seeds": SEEDS_RECORD}]},
        25, 1, {"rule": "RULE_week32.txt part 2c", "from": ids[key], "config_hash": h, "refit_epochs": e,
                "best_epochs": best})


def frames_checked_path(dataset: str, model: str, layer: int) -> Path:
    """The record check-frames writes when a frame set passes (added 2026-11-05)."""
    return out_dir() / "frames_checked" / f"{dataset}_{model}_l{int(layer)}.json"


def film_settings(refit_settings: dict) -> tuple:
    """(the films' settings, the warm start's learning rate): a final record's refit with the external test and the warm
    start on EmoFilm (RULE_week32_addendum7.txt part F), the rate a tenth of the record's peak rate (warm30's rule,
    written without float noise); every other setting as the refit's."""
    from affect.runs.spec import check_job

    base = plain(validate_job(refit_settings))
    if base["train"]["refit"] != "train_and_val" or base["train"]["external_test"] != "none" \
            or base["train"]["warm_start"] != "none":
        raise SystemExit("the films start from a refit without an external test or a warm start")
    lr = float(f"{float(base['train']['lr']) / WEEK32.film_lr_divisor:.6g}")
    settings = json.loads(json.dumps(base))
    settings["train"].update({"external_test": "emofilm", "warm_start": "emofilm",
                              "warm_start_folds": WEEK32.film_folds, "warm_start_epochs": WEEK32.film_epochs,
                              "warm_start_lr": lr})
    check_job(validate_job(settings), 16000)
    if changed(base, settings) != ["train.external_test", "train.warm_start", "train.warm_start_epochs",
                                   "train.warm_start_folds", "train.warm_start_lr"]:
        raise SystemExit(f"the films differ from the refit in {changed(base, settings)}")
    return settings, lr


def film_words(lr: float) -> str:
    return (f"with train.external_test emofilm (the refit's last weights score EmoFilm's 726 happiness, anger and "
            f"sadness clips, cold, on those three classes) and train.warm_start emofilm (then the same weights trained "
            f"for {WEEK32.film_epochs} epochs at {lr:g}, a tenth of the record's peak rate, on the clips of "
            f"{WEEK32.film_folds - 1} of {WEEK32.film_folds} film folds, each fold's clips scored by the model adapted "
            f"on the others; fixed in advance, as warm30, and not tuned). Every other setting is the refit's, so its "
            f"IEMOCAP files should repeat the refit's runs. Nothing is chosen from these runs")


def make_film(letter: str, head: str, ids: dict) -> Path:
    """A chain's films (RULE_week32_addendum7.txt part F), from its refit's sweep file as the queue holds it."""
    sweep, jobs = sweep_record(ids["refit"])
    if sweep is None:
        raise SystemExit(f"{ids['refit']} is not queued")
    settings, lr = film_settings(sweep["base"])
    if settings["features"]["model"] != ENCODERS[letter]:
        raise SystemExit(f"{ids['refit']} reads {settings['features']['model']}'s frames, not {ENCODERS[letter]}'s")
    layer = int(settings["features"]["layer"])
    model = NAMES[ENCODERS[letter]]
    checked = frames_checked_path("emofilm", ENCODERS[letter], layer)
    _, refit_sha = sha256_of(root() / "artifacts" / "sweeps" / ids["refit"] / "sweep_file.json")
    return write_sweep(
        {"sweep_id": ids["film"],
         "description": (f"{WHY}: the films of my {head} head on {model}'s layer-{layer} frames (RULE_week32_addendum7"
                         f".txt part F): {ids['refit']}'s configuration, refitted as it is, {film_words(lr)}. EmoFilm's "
                         f"frames are its layer {layer}, extracted like IEMOCAP's and checked against EmoFilm's probe "
                         f"means of that layer. Seeds 6 to 10, four classes, session-specific. {RULE7} Priority "
                         f"{P_FILM}."),
         "folds": list(sweep["folds"]), "base": settings,
         "stages": [{"name": "film", "kind": "base", "priority": P_FILM, "seeds": SEEDS_RECORD}]},
        len(sweep["folds"]) * len(SEEDS_RECORD), 1,
        {"rule": "RULE_week32_addendum7.txt part F", "from": ids["refit"], "refit_sweep_file_sha256": refit_sha,
         "warm_start_lr": lr, "emofilm_frames": json.loads(checked.read_text(encoding="utf-8"))})


def advance_film(letter: str, head: str, ids: dict) -> str:
    """The film stage of a chain (added 2026-11-05): written once its refit is queued and EmoFilm's frames of the
    refit's encoder and layer have passed check-frames; returns its state."""
    s = state(ids["film"])
    if s["enqueued"]:
        return f"film {'FAILED ' if s['failed'] else ''}{s['done']}/{s['jobs']}"
    if written(ids["film"]):
        return "film written"
    sweep, _ = sweep_record(ids["refit"])
    if sweep is None:
        return "film -"
    layer = int(sweep["base"]["features"]["layer"])
    if not frames_checked_path("emofilm", ENCODERS[letter], layer).is_file():
        return f"film waits for EmoFilm's checked frames ({ENCODERS[letter]} layer {layer})"
    make_film(letter, head, ids)
    return "film written"


def sha256_of(path: Path) -> tuple:
    from affect.hashing import sha256_file
    return sha256_file(path)


def network_film_id(tag: str) -> str:
    return f"film32_{tag}_so4"


def make_network_film(tag: str, refit_id: str) -> Path:
    """A network's films (RULE_week32_addendum7.txt part F): its final record's refit as the queue holds it, with the
    external test and the warm start on EmoFilm."""
    sweep, jobs = sweep_record(refit_id)
    if sweep is None:
        raise SystemExit(f"{refit_id} is not queued")
    settings, lr = film_settings(sweep["base"])
    if settings["features"]["name"] != "logmel":
        raise SystemExit(f"{refit_id} does not read log-mel features")
    _, refit_sha = sha256_of(root() / "artifacts" / "sweeps" / refit_id / "sweep_file.json")
    return write_sweep(
        {"sweep_id": network_film_id(tag),
         "description": (f"{WHY}: the films of the {tag} network's final record (RULE_week32_addendum7.txt part F): "
                         f"{refit_id}'s configuration, refitted as it is, {film_words(lr)}. Its folds and seeds as "
                         f"{refit_id}'s, four classes, session-specific. {RULE7} Priority {P_NETWORK_FILM}."),
         "folds": list(sweep["folds"]), "base": settings,
         "stages": [{"name": "film", "kind": "base", "priority": P_NETWORK_FILM,
                     "seeds": list(sweep["stages"][0]["seeds"])}]},
        len(jobs), 1,
        {"rule": "RULE_week32_addendum7.txt part F", "from": refit_id, "refit_sweep_file_sha256": refit_sha,
         "warm_start_lr": lr})


def advance_network_films() -> list:
    """The networks' films (added 2026-11-05): each written once its refit is queued; returns one state per network."""
    out = []
    for tag, refit_id in NETWORK_FILMS:
        film_id = network_film_id(tag)
        try:
            s = state(film_id)
            if s["enqueued"]:
                out.append(f"{film_id} {'FAILED ' if s['failed'] else ''}{s['done']}/{s['jobs']}")
            elif written(film_id):
                out.append(f"{film_id} written")
            elif not state(refit_id)["enqueued"]:
                out.append(f"{film_id} waits for {refit_id} to be queued")
            else:
                sweep, _ = sweep_record(refit_id)
                if len(sweep["stages"]) != 1 or sweep["stages"][0]["kind"] != "base":
                    raise SystemExit(f"{refit_id}: one base stage expected")
                make_network_film(tag, refit_id)
                out.append(f"{film_id} written")
        except (Exception, SystemExit) as err:            # one network's error leaves the others to go on
            out.append(f"{film_id} ERROR {type(err).__name__}: {err}")
            log(f"{film_id}: ERROR {type(err).__name__}: {err}")
    return out


def advance_chain(letter: str, head: str) -> str:
    """Write whatever stage of one chain is due; return a one-line state."""
    ids = chain_ids(letter, head)
    screen = state(ids["screen"])
    if not screen["enqueued"]:
        return "screen not queued"
    if screen["failed"]:
        return f"WAITING: screen has {len(screen['failed'])} failed jobs (e.g. {screen['failed'][0]})"
    if not screen["complete"]:
        return f"screen {screen['done']} of {screen['jobs']}"
    if not written(ids["confirm"]) and not state(ids["confirm"])["enqueued"]:
        make_confirm(letter, head, ids)
    confirm = state(ids["confirm"])
    if not confirm["enqueued"]:
        return "confirmation written, not yet queued"
    if confirm["failed"]:
        return f"WAITING: confirmation has {len(confirm['failed'])} failed jobs"
    if not confirm["complete"]:
        return f"confirmation {confirm['done']} of {confirm['jobs']}"
    stage2 = ["refit", *SPLITS, *SETUPS]
    if any(not written(ids[k]) and not state(ids[k])["enqueued"] for k in stage2):
        make_stage2(letter, head, ids, record_of(ids))
    parts = []
    for key in stage2:
        s = state(ids[key])
        if key in SETUPS and s["complete"] and not written(ids[f"{key}_refit"]) and \
                not state(ids[f"{key}_refit"])["enqueued"]:
            make_setup_refit(letter, head, ids, key)
        parts.append(f"{key} {'FAILED ' if s['failed'] else ''}{s['done']}/{s['jobs']}" if s["enqueued"]
                     else f"{key} not queued")
    for key in SETUPS:
        s = state(ids[f"{key}_refit"])
        parts.append(f"{key}_refit {s['done']}/{s['jobs']}" if s["enqueued"] else f"{key}_refit -")
    parts.append(advance_film(letter, head, ids))           # added 2026-11-05 (RULE_week32_addendum7.txt part F)
    return ", ".join(parts)


def to_enqueue() -> list:
    """The stages' sweep files (sweeps/) not queued, without a smoke failure marked, highest priority first. A screen
    is never listed here: the new encoders' screens are enc32.sh's to queue, after their frames are checked."""
    folder = out_dir() / "sweeps"
    files = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        sweep_id = path.stem
        if sweep_id.endswith("_screen") or state(sweep_id)["enqueued"] \
                or (out_dir() / "smoke_failed" / f"{sweep_id}.txt").exists():
            continue
        try:
            sweep = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            print(f"ERROR: {path.name} unreadable ({err}); not queued")
            continue
        if is_classical(sweep):
            continue
        files.append((-max(st["priority"] for st in sweep["stages"]), sweep_id, path))
    return [p for _, _, p in sorted(files)]


def cmd_advance(args) -> int:
    """Each chain on its own: one that fails (an unreadable file, a check that does not hold) is reported as ERROR and
    logged, and the others go on."""
    for letter in ENCODERS:
        for head in HEADS:
            try:
                text = advance_chain(letter, head)
            except (Exception, SystemExit) as err:
                text = f"ERROR {type(err).__name__}: {err}"
                log(f"heads32{letter}_{head}: {text}")
            print(f"heads32{letter}_{head}: {text}")
    print("network films: " + "; ".join(advance_network_films()))      # added 2026-11-05
    for path in to_enqueue():
        print(f"ENQUEUE {path}")
    return 0


def cmd_status(args) -> int:
    for letter in ENCODERS:
        for head in HEADS:
            ids = chain_ids(letter, head)
            parts = []
            for key, sweep_id in ids.items():
                s = state(sweep_id)
                if s["enqueued"]:
                    parts.append(f"{key} {s['done']}/{s['jobs']}" + (f" ({len(s['failed'])} failed)" if s["failed"]
                                                                     else ""))
                elif written(sweep_id) or (out_dir() / "encoders" / f"{sweep_id}.json").is_file():
                    parts.append(f"{key} written")
            print(f"heads32{letter}_{head}: " + (", ".join(parts) or "nothing queued"))
    films = []                                              # added 2026-11-05: the networks' films
    for tag, _ in NETWORK_FILMS:
        s = state(network_film_id(tag))
        films.append(f"{tag} {s['done']}/{s['jobs']}" + (f" ({len(s['failed'])} failed)" if s["failed"] else "")
                     if s["enqueued"] else f"{tag} {'written' if written(network_film_id(tag)) else '-'}")
    print("film32 (networks): " + ", ".join(films))
    checked = sorted(p.stem for p in (out_dir() / "frames_checked").glob("*.json")) \
        if (out_dir() / "frames_checked").is_dir() else []
    print("frames checked: " + (" ".join(checked) or "none recorded"))
    marks = sorted((out_dir() / "smoke_failed").glob("*.txt")) if (out_dir() / "smoke_failed").is_dir() else []
    for m in marks:
        print(f"SMOKE FAILED, not queued: {m.stem} ({m.read_text(encoding='utf-8').strip()})")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# The new encoders

def probe_sweep_id(model: str, setup: str) -> str:
    return f"classical_pretrained_{model}_{setup}_1"


def cmd_choose_layers(args) -> int:
    """Each setup's layer of the encoder's probe sweeps (RULE part 6): the highest mean validation macro accuracy over
    the 5 folds (seed 1), a tie to the lower layer. Writes layers_<model>.json once."""
    from affect.classical.settings import PRETRAINED_MODELS

    model = args.model
    target = out_dir() / f"layers_{model}.json"
    if target.exists():
        print(target.read_text(encoding="utf-8"), end="")
        return 0
    n_layers = PRETRAINED_MODELS[model][0]
    out = {"model": model, "rule": CHOSEN_HERE[model], "made_utc": now_utc(), "setups": {}}
    for setup in PROBE_SETUPS:
        sweep_id = probe_sweep_id(model, setup)
        s = state(sweep_id)
        if not s["complete"]:
            print(f"{sweep_id}: {s['done']} of {s['jobs']} runs, {len(s['failed'])} failed: not every run exists")
            return 1
        table = rank(sweep_id, [1])
        layer_of = {e["config_hash"]: e["settings"]["functionals"]["feature_set"] for e in table}
        names = [f"{model}_l{k}" for k in range(n_layers)]
        if sorted(layer_of.values()) != sorted(names):
            raise SystemExit(f"{sweep_id}: layers {sorted(layer_of.values())[:3]}..., not {model}'s {n_layers}")
        best = sorted(table, key=lambda e: (-e["ua"], names.index(layer_of[e["config_hash"]])))[0]
        layer = int(layer_of[best["config_hash"]].rsplit("_l", 1)[1])
        out["setups"][setup] = {"sweep": sweep_id, "layer": layer, "config_hash": best["config_hash"],
                                "validation_ua": best["ua"], "settings": best["settings"],
                                "profile": {layer_of[e["config_hash"]]: e["ua"] for e in table}}
        log(f"{model} {setup}: layer {layer} ({best['config_hash'][:12]}), validation macro accuracy {best['ua']:.4f}")
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


def heads_layer(model: str) -> int:
    if model == "ast_audioset":
        return AST_LAYER
    path = out_dir() / f"layers_{model}.json"
    if not path.exists():
        raise SystemExit(f"no {path}: choose-layers first")
    return int(json.loads(path.read_text(encoding="utf-8"))["setups"]["so4"]["layer"])


def ast_masks(level: dict) -> dict:
    """AST's frames come 10 a second, not 50: a time mask's width counted in frames is divided by 5, so that it spans the
    same seconds (part 6)."""
    out = dict(level)
    if "augment.time_mask_width" in out:
        if out["augment.time_mask_width"] % WEEK32.time_mask_divisor:
            raise SystemExit(f"a time mask of {out['augment.time_mask_width']} frames is not a whole number at 10 a second")
        out["augment.time_mask_width"] //= WEEK32.time_mask_divisor
    return out


def screen_sweep(model: str, head: str, layer: int) -> dict:
    sweeps = root() / "sweeps"
    base_sweep = json.loads((sweeps / f"heads31w_{head}_so4.json").read_text(encoding="utf-8"))
    space = [st for st in json.loads((sweeps / f"heads27_{head}.json").read_text(encoding="utf-8"))["stages"]
             if st["kind"] == "random"]
    if len(space) != 1:
        raise SystemExit(f"heads27_{head}: {len(space)} random stages")
    space = json.loads(json.dumps(space[0]["space"]))
    base = json.loads(json.dumps(base_sweep["base"]))
    base["features"].update({"model": model, "layer": layer})
    note = ""
    if model == "ast_audioset":
        base["augment"]["time_mask_width"] = ast_masks({"augment.time_mask_width": base["augment"]["time_mask_width"]}
                                                       )["augment.time_mask_width"]
        space = {f: ([ast_masks(v) if isinstance(v, dict) else v for v in levels] if f == "time_masks" else levels)
                 for f, levels in space.items()}
        if "augment.time_mask_width" in space:
            raise SystemExit("a bare time_mask_width factor; expected the time_masks levels")
        note = (" On AST, whose frames come 10 a second, the time masks' widths (in frames) are divided by 5 so that "
                "they span the seconds they span on Whisper's frames.")
    priority = SCREEN_PRIORITY[model]
    # Whisper Small's screens (added 2026-11-05), BUD-E-Whisper's and Whisper medium's (2026-11-07) cite the addendum
    # that adds them; AST's and w2v-BERT 2.0's text is as it was written on 2026-11-02
    rule_part, rule = SWEEP_RULES.get(model, ("RULE_week32.txt part 6", RULE))
    return {"sweep_id": f"heads32{LETTER[model]}_{head}_screen",
            "description": (f"{WHY}: my {head} head tuned on {NAMES[model]}'s layer-{layer} frames ({rule_part}), "
                            f"with Whisper's search: the base is heads31w_{head}_so4's configuration, and 30 "
                            f"random configurations of the head and its training from heads27_{head}'s space, sampling "
                            f"seed {SAMPLING_SEED}, the encoder and layer changed.{note} Seeds 1 to 3, four classes, "
                            f"session-specific. Validation only: the two configurations with the highest mean "
                            f"validation macro accuracy go to a confirmation on seeds 6 to 10. {rule} Priority "
                            f"{priority} for every job, one head after another (order_sweeps.sh)."),
            "folds": base_sweep["folds"], "base": base,
            "stages": [{"name": "base", "kind": "base", "priority": priority, "seeds": SEEDS_SCREEN},
                       {"name": "random", "kind": "random", "priority": priority, "seeds": SEEDS_SCREEN,
                        "n_configs": WEEK32.random_configs, "sampling_seed": SAMPLING_SEED, "space": space}]}


def probe_sweeps(model: str) -> list:
    """w2v-BERT 2.0's (and since 2026-11-05 Whisper Small's, since 2026-11-07 BUD-E-Whisper's and Whisper medium's) probe
    refits in the four setups and the protocol checks of its four-class session-specific probe, as make_week32.py makes
    them for Whisper and AST (CPU)."""
    chosen = json.loads((out_dir() / f"layers_{model}.json").read_text(encoding="utf-8"))["setups"]
    from affect.classical import runner

    def classical_changed(a, b):
        pa, pb = runner.plain(runner.validate_classical(a)), runner.plain(runner.validate_classical(b))
        return sorted(f"{s}.{k}" for s in set(pa) | set(pb) for k in set(pa.get(s, {})) | set(pb.get(s, {}))
                      if pa.get(s, {}).get(k) != pb.get(s, {}).get(k))

    short = SHORT[model]
    rule_part, rule = CHOSEN_HERE[model], SWEEP_RULES.get(model, (None, RULE))[1]     # w2v-BERT 2.0's as written
    out = []
    for setup in PROBE_SETUPS:
        c = chosen[setup]
        base = json.loads(json.dumps(c["settings"]))
        base["protocol"]["refit"] = "train_and_val"
        assert classical_changed(c["settings"], base) == ["protocol.refit"]
        folds = list(fold_names(base["data"]["split"]))
        out.append(({"sweep_id": f"classical_refit32_probe32_{model}_{setup}",
                     "description": (f"{WHY}: the SVM probe of {NAMES[model]} in {setup} ({c['sweep']}, layer "
                                     f"{c['layer']}, {c['config_hash'][:12]}, chosen on validation by {rule_part}) with "
                                     f"only protocol.refit changed to train_and_val: C and gamma are chosen "
                                     f"on validation as before, then the chosen SVM is fitted again on training plus "
                                     f"validation before it predicts the test fold. Seed 1. {rule} Runs on CPUs."),
                     "folds": folds, "base": base,
                     "stages": [{"name": "refit", "kind": "base", "priority": 100, "seeds": [1]}]}, len(folds)))
    c = chosen["so4"]
    for prefix, key in (("spk32", "spk"), ("script32", "script"), ("oscript32", "oscript")):
        split, text = SPLITS[key]
        base = json.loads(json.dumps(c["settings"]))
        base["protocol"]["refit"] = "train_and_val"
        base["data"]["split"] = split
        assert classical_changed(c["settings"], base) == ["data.split", "protocol.refit"]
        folds = list(fold_names(split))
        out.append(({"sweep_id": f"classical_{prefix}_{short}",
                     "description": (f"{WHY}: the SVM probe of {NAMES[model]} in so4 (layer {c['layer']}, "
                                     f"{c['config_hash'][:12]}), its layer fixed, in {text}, refitted on training plus "
                                     f"validation; C and gamma are chosen on each fold's validation set as the probe "
                                     f"chooses them. Seed 1. {rule} Runs on CPUs."),
                     "folds": folds, "base": base,
                     "stages": [{"name": "refit", "kind": "base", "priority": 100, "seeds": [1]}]}, len(folds)))
    base = json.loads(json.dumps(c["settings"]))
    base["protocol"]["refit"] = "train_and_val"
    base["functionals"]["normalization"] = "global"
    assert classical_changed(c["settings"], base) == ["functionals.normalization", "protocol.refit"]
    out.append(({"sweep_id": f"classical_gnorm32_{short}",
                 "description": (f"{WHY}: the SVM probe of {NAMES[model]} in so4 (layer {c['layer']}, "
                                 f"{c['config_hash'][:12]}) with each feature standardized on the training sessions "
                                 f"(normalization global) instead of on each speaker's own utterances, refitted on "
                                 f"training plus validation: the check that the probe gains nothing from seeing the "
                                 f"test speaker's unlabeled frames. Seed 1. {rule} Runs on CPUs."),
                 "folds": [f"test_S{k}" for k in range(1, 6)], "base": base,
                 "stages": [{"name": "refit", "kind": "base", "priority": 100, "seeds": [1]}]}, 5))
    return out


def cmd_encoder_sweeps(args) -> int:
    model = args.model
    if model not in ENCODER_SWEEP_MODELS:
        raise SystemExit(f"{model}: only {', '.join(ENCODER_SWEEP_MODELS)} get their sweeps here")
    layer = heads_layer(model)
    rule_part = CHOSEN_HERE.get(model, "RULE_week32.txt part 6")
    for head in HEADS:
        sweep = screen_sweep(model, head, layer)
        n_configs = WEEK32.screen_configs
        path = write_sweep(sweep, n_configs * 5 * len(SEEDS_SCREEN), n_configs,
                           {"rule": rule_part, "model": model, "layer": layer,
                            "base_from": f"heads31w_{head}_so4", "space_from": f"heads27_{head}"}, where="encoders")
        print(f"SCREEN {path}")
    if model in CHOSEN_HERE:
        for sweep, n in probe_sweeps(model):
            path = write_sweep(sweep, n, 1, {"rule": rule_part, "model": model}, where="encoders")
            print(f"CLASSICAL {path}")
    return 0


def cmd_check_frames(args) -> int:
    """The frames of a layer against the probe's mean vectors of the same layer: the same utterances in the same order,
    and for every 50th utterance the largest difference between the mean of its frames (float16, read as float64)
    and its probe vector (float32), against the largest value; it passes when that difference is at most 0.002 times
    the largest value, plus 0.002 (float16 keeps 11 significant bits). Since 2026-11-05 a test corpus's frames too
    (--dataset emofilm, every clip checked), and a frame set that passes is recorded in
    artifacts/advance_week32/frames_checked/<dataset>_<model>_l<layer>.json, which the film stage waits for."""
    base = root() / "artifacts" / "features" / args.dataset
    frames_dir, probe_dir = base / f"{args.model}_l{args.layer}_frames", base / f"{args.model}_l{args.layer}"
    step = 50 if args.dataset == "iemocap" else 1
    rec = json.loads((frames_dir / "record.json").read_text(encoding="utf-8"))
    fr = np.load(frames_dir / "frames.npy", mmap_mode="r")
    off = np.load(frames_dir / "offsets.npy")
    ids = (frames_dir / "utt_ids.txt").read_text(encoding="utf-8").split()
    probe = np.load(probe_dir / "values.npy")
    probe_ids = (probe_dir / "utt_ids.txt").read_text(encoding="utf-8").split()
    index = {u: i for i, u in enumerate(probe_ids)}
    missing = [u for u in ids if u not in index]
    diffs, scale = [], 0.0
    for i in range(0, len(ids), step):
        mean = fr[off[i]:off[i + 1]].astype(np.float64).mean(axis=0)
        p = probe[index[ids[i]]].astype(np.float64)
        diffs.append(float(np.abs(mean - p).max()))
        scale = max(scale, float(np.abs(p).max()))
    ok = (not missing and len(ids) == len(probe_ids) and rec["width"] == probe.shape[1]
          and max(diffs) <= 0.002 * scale + 0.002)
    where = "" if args.dataset == "iemocap" else f" ({args.dataset})"
    print(f"{'FRAMES OK' if ok else 'FRAMES DIFFER'}: {args.model} layer {args.layer}{where}, {len(ids)} utterances "
          f"(probe {len(probe_ids)}, {len(missing)} missing), {rec['total_frames']:,} frames of {rec['width']}, "
          f"{rec['frame_rate']} a second; {len(diffs)} checked: largest difference of the frames' mean from the probe's "
          f"{max(diffs):.5f}, values up to {scale:.2f}")
    target = frames_checked_path(args.dataset, args.model, args.layer)
    if ok and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(target, json.dumps(
            {"dataset": args.dataset, "model": args.model, "layer": int(args.layer), "utterances": len(ids),
             "checked": len(diffs), "every": step, "largest_difference": max(diffs), "largest_value": scale,
             "total_frames": rec["total_frames"], "weights_sha256": rec.get("weights_sha256"),
             "frames_record_sha256": sha256_of(frames_dir / "record.json")[1],
             "probe_record_sha256": (sha256_of(probe_dir / "record.json")[1] if (probe_dir / "record.json").is_file()
                                     else None), "checked_utc": now_utc()},
            indent=1) + "\n")
        log(f"frames checked: {args.dataset} {args.model} layer {args.layer} ({len(diffs)} utterances, largest "
            f"difference {max(diffs):.5f} of values up to {scale:.2f})")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("advance").set_defaults(func=cmd_advance)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("choose-layers")
    p.add_argument("--model", required=True, choices=sorted(CHOSEN_HERE))
    p.set_defaults(func=cmd_choose_layers)
    p = sub.add_parser("encoder-sweeps")
    p.add_argument("--model", required=True, choices=list(ENCODER_SWEEP_MODELS))
    p.set_defaults(func=cmd_encoder_sweeps)
    p = sub.add_parser("check-frames")
    p.add_argument("--model", required=True)
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--dataset", default="iemocap", choices=["iemocap", "emofilm"])    # added 2026-11-05
    p.set_defaults(func=cmd_check_frames)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
