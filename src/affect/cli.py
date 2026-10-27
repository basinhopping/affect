"""Command line: python -m affect.cli <command> ...

    ingest        copy a received folder into data/<dataset>/raw/<name>/ with a manifest
    assemble      build data/<dataset>/assembled/ from named source folders
    verify-data   re-read an assembled copy against its manifest and repeat its checks
    data-summary  class counts of an assembled copy, per session and per kind
    env-check     machine, packages, GPUs, network; one training step on every GPU
    smoke         before a sweep: every configuration of a sweep file for one step, one short
                  real job, and a timing of the base configuration (one GPU)
    enqueue       expand a sweep file and put its jobs in the queue
    worker        take jobs from the queue until it is empty (started by scripts/worker.sh)
    status        queue counts, failures, stop conditions and the validation ranking of a sweep
    analyze       every configuration of a finished sweep against its baseline, with intervals,
                  Holm-corrected verdicts and factor-level means (validation only; plan 3.3)
    select        record the chosen configuration of an analyzed sweep, before any test result is read
    test-report   the test results of a selection (reads test files of the chosen configuration only)
    compare-runs  a sweep's runs against every copy of the same runs in other sweeps: validation
                  predictions and metrics exactly, test files by SHA-256 only (cnn_regression_1)
    requeue-running   move jobs left in queue/running back to pending, after all workers stopped
    requeue-failed    move a sweep's failed jobs back to pending (after fixing what failed them)
    extract-functionals  openSMILE functionals of every utterance of a label set, once per feature
                  set, into artifacts/features/ (classical environment: scripts/classical.sh)
    classical     run a classical sweep file (SVM or TPOT on functionals) into run folders, in this
                  process; running it again resumes (classical environment)
    classical-selftest  openSMILE, the SVM and a tiny TPOT search, once after building the classical
                  environment
    download-pretrained  a pretrained speech encoder's files from the Hugging Face hub into the cache
                  (needs huggingface_hub; added 2026-10-19)
    extract-pretrained  every utterance of a label set through a pretrained encoder, one feature set per
                  hidden layer, into artifacts/features/ (needs torch and transformers; added 2026-10-19)
    cold-test     a classical record's SVMs fitted again on IEMOCAP (checked against its runs) and scored on
                  another corpus, EmoFilm, into artifacts/coldtests/ (classical environment; added 2026-10-20)
    derive-pretrained  a feature set made from extracted ones: layer_average, the mean of an encoder's layers'
                  mean vectors (added 2026-10-20)
    extract-frames  every frame of chosen hidden layers of a pretrained encoder, for the heads trained on them
                  (features pretrained_frames, models frames_*; needs torch and transformers; added 2026-10-26)
    va-probe      valence, activation and dominance from each layer of a pretrained encoder: ridge regressions,
                  penalty and layer chosen on validation by CCC, then test (classical environment; added 2026-10-21)
    emofilm-finetune  a classical record's SVMs fitted again with EmoFilm's clips of the other film folds added,
                  and an SVM on EmoFilm alone, each scoring held-out films, beside the cold models (added 2026-10-21)

Every argument is required: nothing here has a default. Commands that copy data take
--max-seconds, stop cleanly when it is spent, and continue where they stopped when run again.
verify-data exits 0 when its checks passed, 1 when they failed, and 3 when it is not finished
yet (it always stops once after listing the folder): run the same command again.
Results of verify-data and env-check are written to artifacts/checks/<check>_<utc>/.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import affect  # noqa: F401  applies the MPS fix before anything imports torch
from affect import catalog, paths
from affect.runs.provenance import collect
from affect.runs.store import OutputFolder


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def _stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _save_check(name: str, report: dict) -> Path:
    folder = OutputFolder(paths.checks_dir(), f"{name}_{_stamp()}").begin()
    try:
        folder.write_json("report.json", report)
        folder.write_json("provenance.json", collect(f"affect.cli {name}"))
    except BaseException as err:
        folder.fail(err)
        raise
    return folder.complete()


def cmd_ingest(args) -> int:
    from affect.data.ingest import ingest
    from affect.registry import DATASETS

    DATASETS.get(args.dataset)
    _print(ingest(args.dataset, args.name, Path(args.source), args.max_seconds))
    return 0


def cmd_assemble(args) -> int:
    from affect.data.assemble import assemble

    sources = {}
    for item in args.source:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--source expects NAME=PATH, got {item!r}")
        if name in sources:
            raise SystemExit(f"--source {name} given twice")
        sources[name] = Path(path)
    _print(assemble(args.dataset, sources, args.max_seconds))
    return 0


def cmd_verify_data(args) -> int:
    from affect.data.assemble import verify

    report = verify(args.dataset, args.state, args.max_seconds)
    if report["phase"] != "done":
        _print(report)
        return 3   # not finished: run the same command again (scripts/check.sh loops on this)
    saved = _save_check(f"verify_data_{args.dataset}", report)
    checks = report["checks"]
    _print({"phase": "done", "ok": report["ok"],
            "files": {k: report["files"][k] for k in ("ok", "n_manifest_rows", "n_rehashed")},
            "manifest_matches_assembly_record": report["manifest_matches_assembly_record"],
            "checks": checks["checks"], "lineage": report["lineage"]["checks"],
            "n_utterances": checks["n_utterances"],
            "four_class_total": checks["four_class"]["total"], "saved": str(saved)})
    return 0 if report["ok"] else 1


def cmd_data_summary(args) -> int:
    from affect.data.iemocap import class_counts, load_records

    if args.dataset != "iemocap":
        raise SystemExit(f"data-summary knows iemocap only, got {args.dataset!r}")
    records, _, _ = load_records(paths.assembled_dir("iemocap"), "assembled")
    counts = class_counts(records, "four_merged")
    classes = ("neu", "hap+exc", "ang", "sad")
    print("| Session | " + " | ".join(classes) + " | Total |")
    print("|---|" + "---|" * (len(classes) + 1))
    for session, row in counts["by_session"].items():
        print(f"| {session} | " + " | ".join(str(row.get(c, 0)) for c in classes) + f" | {sum(row.values())} |")
    print("| All | " + " | ".join(str(counts["by_class"].get(c, 0)) for c in classes) + f" | {counts['total']} |")
    print(json.dumps({"by_kind": counts["by_kind"], "kind_totals": counts["kind_totals"]}))
    return 0


def cmd_env_check(args) -> int:
    from affect.env import environment_report, gpu_report

    report = {"environment": environment_report()}
    if args.expected_gpus > 0:
        report["gpus"] = gpu_report(args.expected_gpus)
        ok = report["gpus"]["ok"]
    else:
        report["gpus"] = "not checked: --expected-gpus 0"
        ok = True
    report["ok"] = ok
    saved = _save_check("env_check", report)
    summary = {"ok": ok, "saved": str(saved)}
    if isinstance(report["gpus"], dict):
        summary["gpus"] = {k: report["gpus"][k] for k in ("torch", "torch_cuda", "cuda_available", "device_count")}
        summary["devices"] = [{k: d.get(k) for k in ("index", "name", "ok", "train_step_ms", "error")}
                              for d in report["gpus"]["devices"]]
    summary["network"] = report["environment"]["network"]
    summary["disk_at_repo"] = report["environment"]["disk_at_repo"]
    _print(summary)
    return 0 if ok else 1


def cmd_smoke(args) -> int:
    import torch

    from affect.env import require_gpus
    from affect.sweep.smoke import smoke

    require_gpus(args.expected_gpus)
    return 0 if smoke(Path(args.sweep_file), torch.device("cuda", 0)) else 1


def cmd_enqueue(args) -> int:
    from affect.sweep.enqueue import enqueue

    summary = enqueue(Path(args.sweep_file))
    _print({k: summary[k] for k in ("sweep_id", "configs", "jobs", "stages", "base_config_hash")})
    return 0


def cmd_worker(args) -> int:
    from affect.sweep.worker import run

    return run(args.worker_id, args.expected_gpus, args.min_free_gb, args.max_consecutive_failures)


def cmd_status(args) -> int:
    from affect.sweep.status import render, summarise

    print(render(summarise(args.sweep), args.top))
    return 0


def cmd_analyze(args) -> int:
    from affect.analysis.validation import analyze, render

    report = analyze(args.sweep, args.confidence, args.n_boot, args.seed)
    text = render(report, args.top)
    print(text)
    folder = OutputFolder(paths.analysis_dir(), f"{args.sweep}_{_stamp()}").begin()
    try:
        folder.write_json("report.json", report)
        folder.write_text("report.txt", text + "\n")
        folder.write_json("provenance.json", collect("affect.cli analyze"))
    except BaseException as err:
        folder.fail(err)
        raise
    print(f"saved {folder.complete()}")
    return 0


def cmd_select(args) -> int:
    from affect.analysis.selection import selection_record

    record = selection_record(args.sweep, args.config, args.reason)
    folder = OutputFolder(paths.selections_dir(), args.name).begin()
    try:
        folder.write_json("selection.json", record)
        folder.write_json("provenance.json", collect("affect.cli select"))
    except BaseException as err:
        folder.fail(err)
        raise
    saved = folder.complete()
    v = record["validation"]
    print(f"selected {record['config_hash'][:12]} of {record['sweep_id']} ({record['label']}), validation UA "
          f"{v['ua']:.4f} over seeds {v['seeds']}, from {record['analysis_report']['folder']}")
    print(f"saved {saved}")
    return 0


def cmd_test_report(args) -> int:
    from affect.analysis.selection import render, test_report

    selection = json.loads((paths.selections_dir() / args.selection / "selection.json").read_text(encoding="utf-8"))
    report = test_report(selection, args.confidence)
    text = render(report)
    print(text)
    folder = OutputFolder(paths.test_reports_dir(), f"{args.selection}_{_stamp()}").begin()
    try:
        folder.write_json("report.json", report)
        folder.write_text("report.txt", text + "\n")
        folder.write_json("provenance.json", collect("affect.cli test-report"))
    except BaseException as err:
        folder.fail(err)
        raise
    print(f"saved {folder.complete()}")
    return 0


def cmd_compare_runs(args) -> int:
    from affect.analysis.compare import compare_sweep, render

    report = compare_sweep(args.sweep, args.differences_shown)
    print(render(report))
    if report["verdict"] == "INCOMPLETE":
        return 3   # not finished: run the same command again later
    print(f"saved {_save_check(f'compare_runs_{args.sweep}', report)}")
    return 0 if report["verdict"] == "SAME" else 1


def cmd_requeue_running(args) -> int:
    from affect.sweep.queue import Queue

    if not args.all_workers_stopped:
        raise SystemExit("add --all-workers-stopped once no worker is running (tmux ls shows none)")
    print(f"moved {Queue(paths.queue_dir()).requeue_running()} jobs from running back to pending")
    return 0


def cmd_requeue_failed(args) -> int:
    from affect.sweep.queue import Queue

    print(f"moved {Queue(paths.queue_dir()).requeue_failed(args.sweep)} failed jobs of {args.sweep} back to pending")
    return 0


def _iemocap_all(args) -> dict:
    """IEMOCAP data settings covering every utterance of a label set (the split does not change them)."""
    if args.dataset != "iemocap":
        raise SystemExit(f"the classical commands know iemocap only, got {args.dataset!r}")
    return {"label_set": args.label_set, "subset": args.subset, "split": "leave_one_session_out"}


def _all_utterances(args) -> dict:
    """Data settings covering every utterance of a label set, for the feature extractions: IEMOCAP's, or EmoFilm's
    (a test corpus, with no split; its subset is the language; added 2026-10-20)."""
    if args.dataset == "emofilm":
        return {"label_set": args.label_set, "subset": args.subset}
    return _iemocap_all(args)


def cmd_extract_functionals(args) -> int:
    from affect.classical.functionals import extract

    extract(args.dataset, _all_utterances(args), args.feature_set, args.workers, lambda m: print(m, flush=True))
    return 0


def cmd_download_pretrained(args) -> int:
    from affect.classical.pretrained import download

    snapshot = download(args.model, Path(args.cache_dir).expanduser(), lambda m: print(m, flush=True))
    print(f"snapshot {snapshot}")
    return 0


def cmd_extract_pretrained(args) -> int:
    from affect.classical.pretrained import extract

    extract(args.dataset, _all_utterances(args), args.model, Path(args.snapshot).expanduser(), args.device,
            lambda m: print(m, flush=True), pooling=args.pooling)
    return 0


def cmd_extract_frames(args) -> int:
    from affect.classical.pretrained import extract_frames

    layers = tuple(int(k) for k in args.layers.split(","))
    extract_frames(args.dataset, _all_utterances(args), args.model, layers, Path(args.snapshot).expanduser(),
                   args.device, lambda m: print(m, flush=True))
    return 0


def cmd_derive_pretrained(args) -> int:
    from affect.classical.pretrained import derive_layer_average

    if args.kind != "layer_average":
        raise SystemExit(f"derive-pretrained knows layer_average only, got {args.kind!r}")
    derive_layer_average(args.dataset, args.model, lambda m: print(m, flush=True))
    return 0


def cmd_cold_test(args) -> int:
    from affect.classical.coldtest import cold_test

    if args.target != "emofilm":
        raise SystemExit(f"cold-test knows the target emofilm only, got {args.target!r}")
    cache = {}
    for name in args.selections:
        cold_test(name, args.target, {"label_set": args.label_set, "subset": args.subset},
                  lambda m: print(m, flush=True), cache)
    return 0


def cmd_va_probe(args) -> int:
    from affect.classical.va import va_probe

    settings = {"label_set": args.label_set, "subset": args.subset, "split": args.split}
    for model in args.models:
        va_probe(model, settings, lambda m: print(m, flush=True), dataset=args.dataset)
    return 0


def cmd_emofilm_finetune(args) -> int:
    from affect.classical.finetune import emofilm_finetune

    for name in args.selections:
        emofilm_finetune(name, lambda m: print(m, flush=True), n_folds=args.film_folds)
    return 0


def cmd_classical(args) -> int:
    from affect.classical.runner import run

    result = run(Path(args.sweep_file), args.max_consecutive_failures, lambda m: print(m, flush=True))
    return 0 if result["failed"] == 0 else 1


def cmd_classical_selftest(args) -> int:
    from affect.classical.selftest import selftest

    return 0 if selftest(args.dataset, _iemocap_all(args), args.n_jobs) else 1


def main(argv=None) -> int:
    catalog.load_all()
    parser = argparse.ArgumentParser(prog="python -m affect.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="copy a received folder into data/<dataset>/raw/<name>/")
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--max-seconds", type=float, required=True)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("assemble", help="build data/<dataset>/assembled/")
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", action="append", required=True,
                   help="NAME=PATH, once per source the dataset names (iemocap: IEMOCAP_full_release, "
                        "collaborator_wav, phase0_audit)")
    p.add_argument("--max-seconds", type=float, required=True)
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("verify-data", help="re-read an assembled copy and repeat its checks")
    p.add_argument("--dataset", required=True)
    p.add_argument("--state", required=True,
                   help="name of this verification; a new name starts fresh, the same name resumes")
    p.add_argument("--max-seconds", type=float, required=True)
    p.set_defaults(func=cmd_verify_data)

    p = sub.add_parser("data-summary", help="class counts of an assembled copy")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_data_summary)

    p = sub.add_parser("env-check", help="machine, packages, GPUs and network")
    p.add_argument("--expected-gpus", type=int, required=True)
    p.set_defaults(func=cmd_env_check)

    p = sub.add_parser("smoke", help="try every configuration of a sweep file before enqueueing it")
    p.add_argument("--sweep-file", required=True)
    p.add_argument("--expected-gpus", type=int, required=True)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("enqueue", help="expand a sweep file into queued jobs")
    p.add_argument("--sweep-file", required=True)
    p.set_defaults(func=cmd_enqueue)

    p = sub.add_parser("worker", help="run queued jobs until the queue is empty")
    p.add_argument("--worker-id", required=True)
    p.add_argument("--expected-gpus", type=int, required=True)
    p.add_argument("--min-free-gb", type=float, required=True)
    p.add_argument("--max-consecutive-failures", type=int, required=True,
                   help="stop this worker after this many failed jobs in a row")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("status", help="progress and validation ranking of a sweep")
    p.add_argument("--sweep", required=True)
    p.add_argument("--top", type=int, required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("analyze", help="validation-only comparison of a sweep's configurations with its baseline")
    p.add_argument("--sweep", required=True)
    p.add_argument("--confidence", type=float, required=True, help="for example 0.95")
    p.add_argument("--n-boot", type=int, required=True, help="bootstrap draws, for example 2000")
    p.add_argument("--seed", type=int, required=True, help="seed of the bootstrap's random generator")
    p.add_argument("--top", type=int, required=True)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("select", help="record the chosen configuration of an analyzed sweep")
    p.add_argument("--sweep", required=True)
    p.add_argument("--config", required=True, help="the configuration hash, or its first 8 or more characters")
    p.add_argument("--name", required=True, help="name of this selection, for example cnn_final_1")
    p.add_argument("--reason", required=True, help="why this configuration, in one sentence")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("test-report", help="test results of a selection")
    p.add_argument("--selection", required=True)
    p.add_argument("--confidence", type=float, required=True, help="for example 0.95")
    p.set_defaults(func=cmd_test_report)

    p = sub.add_parser("compare-runs", help="a sweep's runs against every copy of the same runs in other sweeps")
    p.add_argument("--sweep", required=True)
    p.add_argument("--differences-shown", type=int, required=True,
                   help="metric differences listed per differing copy, for example 3")
    p.set_defaults(func=cmd_compare_runs)

    p = sub.add_parser("requeue-running", help="put jobs left running back in the queue")
    p.add_argument("--all-workers-stopped", action="store_true")
    p.set_defaults(func=cmd_requeue_running)

    p = sub.add_parser("requeue-failed", help="put a sweep's failed jobs back in the queue")
    p.add_argument("--sweep", required=True)
    p.set_defaults(func=cmd_requeue_failed)

    p = sub.add_parser("extract-functionals", help="openSMILE functionals of every utterance of a label set")
    p.add_argument("--dataset", required=True)
    p.add_argument("--label-set", required=True, help="take the largest (iemocap: seven; emofilm: three_no_neutral)")
    p.add_argument("--subset", required=True, help="take all (iemocap: all, impro or script; emofilm: the language)")
    p.add_argument("--feature-set", required=True, help="eGeMAPSv02, ComParE_2016 or emobase (openSMILE's)")
    p.add_argument("--workers", type=int, required=True, help="CPU processes")
    p.set_defaults(func=cmd_extract_functionals)

    p = sub.add_parser("download-pretrained", help="a pretrained speech encoder's files into the Hugging Face cache")
    p.add_argument("--model", required=True, help="a name in affect.classical.settings.PRETRAINED_MODELS")
    p.add_argument("--cache-dir", required=True, help="the cache folder (the node: ~/cache/huggingface)")
    p.set_defaults(func=cmd_download_pretrained)

    p = sub.add_parser("extract-pretrained", help="a pretrained encoder's hidden layers for every utterance")
    p.add_argument("--dataset", required=True)
    p.add_argument("--label-set", required=True, help="take the largest (iemocap: seven; emofilm: three_no_neutral)")
    p.add_argument("--subset", required=True, help="take all (iemocap: all, impro or script; emofilm: the language)")
    p.add_argument("--model", required=True, help="a name in affect.classical.settings.PRETRAINED_MODELS")
    p.add_argument("--snapshot", required=True, help="the folder download-pretrained printed")
    p.add_argument("--device", required=True, help="cuda or cpu")
    p.add_argument("--pooling", required=True, help="mean (<model>_l<k>) or mean_sd (<model>_l<k>_msd)")
    p.set_defaults(func=cmd_extract_pretrained)

    p = sub.add_parser("extract-frames", help="every frame of chosen layers of a pretrained encoder, for the heads")
    p.add_argument("--dataset", required=True)
    p.add_argument("--label-set", required=True, help="iemocap: four_merged holds every utterance the heads train on")
    p.add_argument("--subset", required=True, help="all")
    p.add_argument("--model", required=True, help="a name in affect.classical.settings.PRETRAINED_MODELS")
    p.add_argument("--layers", required=True, help="comma-separated layer numbers, for example 20")
    p.add_argument("--snapshot", required=True, help="the folder download-pretrained printed")
    p.add_argument("--device", required=True, help="cuda or cpu")
    p.set_defaults(func=cmd_extract_frames)

    p = sub.add_parser("derive-pretrained", help="a feature set made from an encoder's extracted layers")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="a name in affect.classical.settings.PRETRAINED_MODELS")
    p.add_argument("--kind", required=True, help="layer_average (<model>_lavg)")
    p.set_defaults(func=cmd_derive_pretrained)

    p = sub.add_parser("classical", help="run a classical sweep file (SVM or TPOT on functionals)")
    p.add_argument("--sweep-file", required=True)
    p.add_argument("--max-consecutive-failures", type=int, required=True)
    p.set_defaults(func=cmd_classical)

    p = sub.add_parser("cold-test", help="a classical record's SVMs refitted, checked and scored on another corpus")
    p.add_argument("--selections", nargs="+", required=True, help="selection records (artifacts/selections/<name>)")
    p.add_argument("--target", required=True, help="the corpus scored: emofilm")
    p.add_argument("--label-set", required=True, help="the target's label set (emofilm: three_no_neutral)")
    p.add_argument("--subset", required=True, help="emofilm: all, en, es or it (the language)")
    p.set_defaults(func=cmd_cold_test)

    p = sub.add_parser("va-probe", help="valence, activation and dominance from a pretrained encoder's layers")
    p.add_argument("--models", nargs="+", required=True, help="pretrained encoders whose <model>_l<k> sets exist")
    p.add_argument("--dataset", required=True, help="iemocap")
    p.add_argument("--label-set", required=True, help="the utterances used, for example four_merged")
    p.add_argument("--subset", required=True)
    p.add_argument("--split", required=True, help="leave_one_session_out or utterance_5fold")
    p.set_defaults(func=cmd_va_probe)

    p = sub.add_parser("emofilm-finetune", help="a classical record's SVMs fitted again with EmoFilm's other films")
    p.add_argument("--selections", nargs="+", required=True, help="selection records (artifacts/selections/<name>)")
    p.add_argument("--film-folds", type=int, required=True, help="EmoFilm's films in this many folds, for example 5")
    p.set_defaults(func=cmd_emofilm_finetune)

    p = sub.add_parser("classical-selftest", help="check openSMILE, the SVM and TPOT in the classical environment")
    p.add_argument("--dataset", required=True)
    p.add_argument("--label-set", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--n-jobs", type=int, required=True, help="CPU processes for the tiny TPOT search")
    p.set_defaults(func=cmd_classical_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
