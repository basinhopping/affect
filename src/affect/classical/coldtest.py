"""Cold tests: a classical record's models, trained on one corpus, scored on another (added 2026-10-20).

Dr. Luitel's third request of 2026-10-12: the best three-class models, trained on IEMOCAP, tested cold on EmoFilm
(affect.data.emofilm). A classical run keeps its predictions, not its fitted model, but an SVM fit is deterministic: the
same z-scored features, settings and (C, gamma) give the same model. So for each run of a selection record (one per
fold and seed), cold_test() fits again the SVM that the run chose on validation, on that fold's training utterances
made exactly as affect.classical.runner.run_job made them, and checks the fit against the run: its validation UA must
equal the run's, and its validation and test predictions the run's predictions_val.csv and predictions_test.csv (the
predicted class and every class score at the six decimals the files hold). Only then does it score the target corpus,
with the same feature set extracted from the target's audio (artifacts/features/<target>/<feature set>/: the same
feature names, and the same encoder weights or openSMILE version, checked), z-scored within the groups of each
normalization, without labels:

    language_gender   each (language, gender) group of the target with its own mean and SD. The models were trained
                      with each speaker's own z-scores; EmoFilm's actors have 1 to 18 clips (median 3), too few for
                      statistics of their own, so the six groups of about 120 clips stand in for them, and they take
                      out each language's dubbing studio and the main difference between women's and men's voices
    corpus            the whole target with one mean and SD

Both are scored every time. The target's class names must be the model's, in the same order. Only SVM records without
a control and with refit "train" are accepted (every record of 2026-10-20 is one).

    cold_test(selection, target, target_settings, log, cache=None) -> the output folder

writes artifacts/coldtests/<selection>__<target>_<utc>/ once (affect.runs.store):
    report.json                      the record, both feature records, each run's checks, and per normalization:
                                     each fold model's metrics on the target (all of it, by language, by gender), their
                                     means over the fold models, and the ensemble (the mean of the fold models' class
                                     scores)
    predictions_<normalization>.csv  one row per target utterance: its speaker, film, language, gender and label, then
                                     each fold model's predicted class and class scores, then the ensemble's
    provenance.json
Nothing is written when a check fails. cache (a dict) keeps each data protocol's utterances and features between
records of one command.
"""
import csv
import datetime
import io
import json

import numpy as np

from affect import paths
from affect.classical import functionals
from affect.classical.methods import softmax, unweighted_accuracy
from affect.classical.runner import _Data, config_hash, normalize, run_id, validate_classical
from affect.data.splits import assign
from affect.eval.metrics import report, report_by_group
from affect.registry import DATASETS
from affect.runs.provenance import collect, utc_now
from affect.runs.store import OutputFolder
from affect.settings import validate

NORMALIZATIONS = ("language_gender", "corpus")
RECORD_KEYS = ("dataset", "feature_set", "n_features", "names_sha256", "model", "repo", "commit", "weights_file",
               "weights_sha256", "layer", "opensmile_version", "n_utterances", "data_manifest_sha256")
MUST_MATCH = ("feature_set", "n_features", "names_sha256", "model", "repo", "commit", "weights_sha256", "layer",
              "opensmile_version")


def coldtests_dir():
    return paths.artifacts_dir() / "coldtests"


def _scores(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    decision = model.decision_function(x)
    if decision.ndim != 2 or decision.shape[1] != n_classes:
        raise ValueError(f"SVM decision values have shape {decision.shape}, expected (n, {n_classes})")
    return softmax(decision)


def _same_as_run(path, utterances: list, scores: np.ndarray, class_names: tuple) -> None:
    """The refit's predictions against a run's predictions file, utterance by utterance."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(utterances):
        raise ValueError(f"{path}: {len(rows)} rows; the refit predicts {len(utterances)} utterances")
    for k, (row, u) in enumerate(zip(rows, utterances)):
        pred = class_names[int(scores[k].argmax())]
        probs = [f"{p:.6f}" for p in scores[k]]
        theirs = [row[f"prob_{c}"] for c in class_names]
        if row["utt_id"] != u.utt_id or row["pred"] != pred or theirs != probs:
            raise ValueError(f"{path}, row {k}: the run has {row['utt_id']} {row['pred']} {theirs}; the refit "
                             f"{u.utt_id} {pred} {probs}")


def _metrics(y: np.ndarray, pred: np.ndarray, groups: dict, class_names: tuple) -> dict:
    out = report(y, pred, class_names)
    for name, values in groups.items():
        out[f"by_{name}"] = report_by_group(y, pred, values, class_names)
    return out


def _predictions_csv(utterances: list, class_names: tuple, labels: list, per_run: list, ensemble: np.ndarray) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    header = ["utt_id", "speaker", "film", "language", "gender", "label"]
    for name, _ in per_run:
        header += [f"pred_{name}"] + [f"prob_{name}_{c}" for c in class_names]
    header += ["pred_ensemble"] + [f"prob_ensemble_{c}" for c in class_names]
    writer.writerow(header)
    for i, u in enumerate(utterances):
        row = [u.utt_id, u.speaker, u.extra["film"], u.extra["language"], u.extra["gender"], labels[i]]
        for _, scores in per_run:
            row += [class_names[int(scores[i].argmax())]] + [f"{p:.6f}" for p in scores[i]]
        row += [class_names[int(ensemble[i].argmax())]] + [f"{p:.6f}" for p in ensemble[i]]
        writer.writerow(row)
    return buffer.getvalue()


def cold_test(selection: str, target: str, target_settings: dict, log, cache: dict | None = None):
    from sklearn.svm import SVC

    started = utc_now()
    sel = json.loads((paths.artifacts_dir() / "selections" / selection / "selection.json").read_text(encoding="utf-8"))
    job = validate_classical(sel["settings"])
    if job.method_name != "svm" or job.protocol.control != "none" or job.protocol.refit != "train":
        raise ValueError(f"{selection}: a cold test takes an SVM record without a control and with refit 'train'; "
                         f"this one is {job.method_name}, control {job.protocol.control}, refit {job.protocol.refit}")
    if config_hash(job) != sel["config_hash"]:
        raise ValueError(f"{selection}: its settings do not hash to its config_hash")
    cache = {} if cache is None else cache
    key = json.dumps(sel["settings"]["data"], sort_keys=True)
    if key not in cache:
        cache[key] = _Data(job)
    data = cache[key]
    class_names = tuple(data.class_names)
    n_classes = len(class_names)
    feature_set = job.functionals.feature_set
    values, names, source_record = data.functionals(job.data_name, feature_set)

    entry = DATASETS.get(target)
    tsettings = validate(entry.settings_class, target_settings, "target")
    tds = entry.target()
    tutts = tds.utterances(tsettings)
    if tuple(tds.class_order(tsettings)) != class_names:
        raise ValueError(f"{target}'s classes {tds.class_order(tsettings)} are not the model's {class_names}")
    missing = [k for k in ("film", "language", "gender") if any(k not in u.extra for u in tutts)]
    if missing:
        raise ValueError(f"{target}'s utterances lack {missing}")
    tvalues, tnames, target_record = functionals.load(target, feature_set, [u.utt_id for u in tutts])
    if tnames != names:
        raise ValueError(f"{feature_set}: the target's feature names are not the model's")
    differ = [k for k in MUST_MATCH if source_record.get(k) != target_record.get(k)]
    if differ:
        raise ValueError(f"{feature_set}: the target's features differ from the model's in {differ}")
    ty = np.array([class_names.index(u.label) for u in tutts], dtype=np.int64)
    language = np.array([u.extra["language"] for u in tutts])
    gender = np.array([u.extra["gender"] for u in tutts])
    groups = {"language_gender": np.unique(np.char.add(np.char.add(language, "_"), gender), return_inverse=True)[1],
              "corpus": np.zeros(len(tutts), dtype=np.int64)}
    every_row = np.arange(len(tutts))
    xt = {n: normalize(tvalues, "speaker", every_row, groups[n]) for n in NORMALIZATIONS}
    by = {"language": language, "gender": gender}
    log(f"{selection}: {feature_set}, {len(data.utterances)} {job.data_name} utterances; {target} "
        f"{tsettings.label_set} {tsettings.subset}: {len(tutts)} utterances")

    runs, per_run = [], {n: [] for n in NORMALIZATIONS}
    for fold in sel["folds"]:
        for seed in sel["validation"]["seeds"]:
            rid = run_id(job, fold, seed)
            folder = paths.runs_dir(sel["sweep_id"]) / rid
            spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
            if (spec["config_hash"], spec["fold"], int(spec["seed"])) != (sel["config_hash"], fold, seed):
                raise ValueError(f"{folder}: spec.json is another run")
            measured = json.loads((folder / "metrics_val.json").read_text(encoding="utf-8"))
            c, g = measured["chosen"]["C"], measured["chosen"]["gamma_scale"]
            roles = assign(data.utterances, data.folds[fold])
            rows = {r: np.array([data.position[u] for u in roles[r]], dtype=np.int64) for r in ("train", "val", "test")}
            x = normalize(values, job.functionals.normalization, rows["train"], data.speaker_of)
            model = SVC(kernel=job.method.kernel, C=c, gamma="scale" if g is None else g / x.shape[1],
                        class_weight="balanced" if job.method.class_weight == "balanced" else None,
                        decision_function_shape="ovr", random_state=seed)
            model.fit(x[rows["train"]], data.labels[rows["train"]])
            val_scores = _scores(model, x[rows["val"]], n_classes)
            val_ua = unweighted_accuracy(data.labels[rows["val"]], val_scores.argmax(axis=1), n_classes)
            if abs(val_ua - measured["best_val_ua"]) > 1e-12:
                raise ValueError(f"{rid}: the refit's validation UA {val_ua!r} is not the run's "
                                 f"{measured['best_val_ua']!r}")
            _same_as_run(folder / "predictions_val.csv", [data.utterances[i] for i in rows["val"]], val_scores,
                         class_names)
            _same_as_run(folder / "predictions_test.csv", [data.utterances[i] for i in rows["test"]],
                         _scores(model, x[rows["test"]], n_classes), class_names)
            name = fold if len(sel["validation"]["seeds"]) == 1 else f"{fold}_seed{seed}"
            runs.append({"name": name, "run_id": rid, "fold": fold, "seed": seed, "C": c, "gamma_scale": g,
                         "n_train": int(rows["train"].shape[0]), "validation_ua": val_ua,
                         "validation_predictions_equal_the_run": True, "test_predictions_equal_the_run": True})
            for n in NORMALIZATIONS:
                per_run[n].append((name, _scores(model, xt[n], n_classes)))
            log(f"  {rid}: C {c}, gamma scale {g}; the refit repeats the run's validation UA {val_ua:.4f} and its "
                f"validation and test predictions")

    results, ensembles = {}, {}
    for n in NORMALIZATIONS:
        folds_out = {name: _metrics(ty, s.argmax(axis=1), by, class_names) for name, s in per_run[n]}
        ensembles[n] = np.mean([s for _, s in per_run[n]], axis=0)
        results[n] = {"models": folds_out,
                      "mean_over_models": {k: float(np.mean([m[k] for m in folds_out.values()]))
                                           for k in ("ua", "wa", "macro_f1")},
                      "ensemble": _metrics(ty, ensembles[n].argmax(axis=1), by, class_names)}
        log(f"  {target}, {n} z-scores: UA {results[n]['mean_over_models']['ua']:.4f} (mean over the "
            f"{len(folds_out)} models: " + ", ".join(f"{m['ua']:.4f}" for m in folds_out.values())
            + f"); ensemble {results[n]['ensemble']['ua']:.4f}")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = OutputFolder(coldtests_dir(), f"{selection}__{target}_{stamp}").begin()
    try:
        for n in NORMALIZATIONS:
            folder.write_text(f"predictions_{n}.csv", _predictions_csv(tutts, class_names, [u.label for u in tutts],
                                                                       per_run[n], ensembles[n]))
        folder.write_json("report.json", {
            "selection": selection, "sweep_id": sel["sweep_id"], "config_hash": sel["config_hash"],
            "settings": sel["settings"], "selection_validation_ua": sel["validation"]["ua"],
            "class_names": list(class_names),
            "target": {"dataset": target, "settings": target_settings, "n_utterances": len(tutts),
                       "support": {c: int((ty == i).sum()) for i, c in enumerate(class_names)},
                       "normalizations": list(NORMALIZATIONS)},
            "source_features": {k: source_record.get(k) for k in RECORD_KEYS},
            "target_features": {k: target_record.get(k) for k in RECORD_KEYS},
            "runs": runs, "results": results, "started_utc": started, "finished_utc": utc_now()})
        folder.write_json("provenance.json", collect("affect.cli cold-test"))
    except BaseException as err:
        folder.fail(err)
        raise
    done = folder.complete()
    log(f"  written {done}")
    return done
