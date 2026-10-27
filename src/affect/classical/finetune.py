"""EmoFilm fine-tuning of a classical record (added 2026-10-21).

Dr. Luitel's third request of 2026-10-12, its second half: the best models tested cold on EmoFilm, "then a little
fine-tuning if the cold results are poor". Night 20's cold test (affect.classical.coldtest) left WavLM Large 20 and 22
points below its IEMOCAP test UA. A classical record keeps no fitted model, but an SVM fit is deterministic, so this
module fits the record's models again exactly as cold_test does (and checks each against its run), then adds EmoFilm.

EmoFilm's films are split into n_folds film folds: the films sorted by name, the k-th film (from 0) in fold k mod
n_folds. Every clip of a film, in every language, is then in one fold, so a held-out fold shares no scene, actor or
dubbing with the clips a model learned from. For each film fold j, held out in turn:
    cold      each of the record's fold models (fitted on IEMOCAP only, night 20's models) scores fold j's clips
    adapted   each fold model's SVM (its run's C and gamma, the record's kernel and class weights) fitted again on the
              run's IEMOCAP training utterances together with the clips of the other film folds, scores fold j's clips
    emofilm   one SVM fitted on the other film folds' clips alone: its C and gamma chosen from the record's own grid by
              the mean UA over those folds, each held out in turn while it trains on the rest (a tie to the first pair
              in the grid's order), then fitted on all of them, scores fold j's clips
So every clip is scored once by each method: by cold and adapted once per IEMOCAP fold model, by emofilm once.
EmoFilm's features are z-scored within each (language, gender) group over the whole corpus, without labels, as in
night 20's primary normalization; IEMOCAP's per speaker, as in the record's runs. Predictions are the highest of the
softmax over the SVM's one-against-rest decision values, as everywhere in the classical pipeline.

    emofilm_finetune(selection, log, n_folds=5) -> the output folder
writes artifacts/finetune/<selection>__emofilm_<utc>/ once (affect.runs.store):
    report.json      the record, the film folds, each fold model's checks and settings, the emofilm method's choices,
                     and per method each fold model's metrics on all clips (UA, WA, macro-F1, recall, by language and
                     by gender) with their means over the fold models
    predictions.csv  one row per clip: its speaker, film, film fold, language, gender and label, then the class each
                     method predicted (cold and adapted once per fold model)
    provenance.json
"""
import csv
import datetime
import io
import json

import numpy as np

from affect import paths
from affect.classical import functionals
from affect.classical.coldtest import MUST_MATCH, RECORD_KEYS, _metrics, _same_as_run, _scores
from affect.classical.methods import unweighted_accuracy
from affect.classical.runner import _Data, config_hash, normalize, run_id, validate_classical
from affect.data.splits import assign
from affect.registry import DATASETS
from affect.runs.provenance import collect, utc_now
from affect.runs.store import OutputFolder
from affect.settings import validate

METHODS = ("cold", "adapted", "emofilm")


def finetune_dir():
    return paths.artifacts_dir() / "finetune"


def film_folds(films, n_folds: int) -> dict:
    """film -> fold: the films sorted by name, the k-th in fold k mod n_folds."""
    films = sorted(set(films))
    if n_folds < 3 or len(films) < n_folds:
        raise ValueError(f"{len(films)} films cannot make {n_folds} film folds (at least 3, so that the EmoFilm-only "
                         f"SVM can hold out each training fold in turn)")
    return {film: k % n_folds for k, film in enumerate(films)}


def emofilm_finetune(selection: str, log, n_folds: int = 5, target: str = "emofilm",
                     target_settings: dict | None = None):
    from sklearn.svm import SVC

    started = utc_now()
    target_settings = target_settings or {"label_set": "three_no_neutral", "subset": "all"}
    sel = json.loads((paths.artifacts_dir() / "selections" / selection / "selection.json").read_text(encoding="utf-8"))
    job = validate_classical(sel["settings"])
    if job.method_name != "svm" or job.protocol.control != "none" or job.protocol.refit != "train":
        raise ValueError(f"{selection}: fine-tuning takes an SVM record without a control and with refit 'train'")
    if config_hash(job) != sel["config_hash"]:
        raise ValueError(f"{selection}: its settings do not hash to its config_hash")
    data = _Data(job)
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
    tvalues, tnames, target_record = functionals.load(target, feature_set, [u.utt_id for u in tutts])
    if tnames != names:
        raise ValueError(f"{feature_set}: the target's feature names are not the model's")
    differ = [k for k in MUST_MATCH if source_record.get(k) != target_record.get(k)]
    if differ:
        raise ValueError(f"{feature_set}: the target's features differ from the model's in {differ}")
    ty = np.array([class_names.index(u.label) for u in tutts], dtype=np.int64)
    language = np.array([u.extra["language"] for u in tutts])
    gender = np.array([u.extra["gender"] for u in tutts])
    groups = np.unique(np.char.add(np.char.add(language, "_"), gender), return_inverse=True)[1]
    xt = normalize(tvalues, "speaker", np.arange(len(tutts)), groups)
    folds_of = film_folds([u.extra["film"] for u in tutts], n_folds)
    fold = np.array([folds_of[u.extra["film"]] for u in tutts])
    by = {"language": language, "gender": gender}
    n_features = values.shape[1]

    def svm(c, g, seed):
        return SVC(kernel=job.method.kernel, C=c, gamma="scale" if g is None else g / n_features,
                   class_weight="balanced" if job.method.class_weight == "balanced" else None,
                   decision_function_shape="ovr", random_state=seed)

    log(f"{selection}: {feature_set}, {len(data.utterances)} {job.data_name} utterances; {target}: {len(tutts)} clips "
        f"in {len(folds_of)} films, {n_folds} film folds of " + ", ".join(str(int((fold == j).sum()))
                                                                        for j in range(n_folds)) + " clips")
    runs, preds = [], {"cold": [], "adapted": []}
    for f in sel["folds"]:
        for seed in sel["validation"]["seeds"]:
            rid = run_id(job, f, seed)
            folder = paths.runs_dir(sel["sweep_id"]) / rid
            spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
            if (spec["config_hash"], spec["fold"], int(spec["seed"])) != (sel["config_hash"], f, seed):
                raise ValueError(f"{folder}: spec.json is another run")
            measured = json.loads((folder / "metrics_val.json").read_text(encoding="utf-8"))
            c, g = measured["chosen"]["C"], measured["chosen"]["gamma_scale"]
            roles = assign(data.utterances, data.folds[f])
            rows = {r: np.array([data.position[u] for u in roles[r]], dtype=np.int64) for r in ("train", "val", "test")}
            x = normalize(values, job.functionals.normalization, rows["train"], data.speaker_of)
            model = svm(c, g, seed).fit(x[rows["train"]], data.labels[rows["train"]])
            val_scores = _scores(model, x[rows["val"]], n_classes)
            val_ua = unweighted_accuracy(data.labels[rows["val"]], val_scores.argmax(axis=1), n_classes)
            if abs(val_ua - measured["best_val_ua"]) > 1e-12:
                raise ValueError(f"{rid}: the refit's validation UA {val_ua!r} is not the run's "
                                 f"{measured['best_val_ua']!r}")
            _same_as_run(folder / "predictions_val.csv", [data.utterances[i] for i in rows["val"]], val_scores,
                         class_names)
            _same_as_run(folder / "predictions_test.csv", [data.utterances[i] for i in rows["test"]],
                         _scores(model, x[rows["test"]], n_classes), class_names)
            name = f if len(sel["validation"]["seeds"]) == 1 else f"{f}_seed{seed}"
            cold = _scores(model, xt, n_classes).argmax(axis=1)
            adapted = np.empty(len(tutts), dtype=np.int64)
            for j in range(n_folds):
                keep = fold != j
                both = np.vstack([x[rows["train"]], xt[keep]])
                labels = np.concatenate([data.labels[rows["train"]], ty[keep]])
                adapted[~keep] = _scores(svm(c, g, seed).fit(both, labels), xt[~keep], n_classes).argmax(axis=1)
            preds["cold"].append((name, cold))
            preds["adapted"].append((name, adapted))
            runs.append({"name": name, "run_id": rid, "fold": f, "seed": seed, "C": c, "gamma_scale": g,
                         "n_iemocap_train": int(rows["train"].shape[0]), "validation_ua": val_ua,
                         "validation_predictions_equal_the_run": True, "test_predictions_equal_the_run": True})
            log(f"  {rid}: C {c}, gamma scale {g}; the refit repeats the run; cold UA "
                f"{unweighted_accuracy(ty, cold, n_classes):.4f}, adapted {unweighted_accuracy(ty, adapted, n_classes):.4f}")

    grid = [(c, g) for c in job.method.c_values for g in job.method.gamma_scales]
    emofilm = np.empty(len(tutts), dtype=np.int64)
    choices = {}
    for j in range(n_folds):
        inner = [k for k in range(n_folds) if k != j]
        means = []
        for c, g in grid:
            uas = []
            for k in inner:
                tr = np.isin(fold, [q for q in inner if q != k])
                va = fold == k
                p = _scores(svm(c, g, 1).fit(xt[tr], ty[tr]), xt[va], n_classes).argmax(axis=1)
                uas.append(unweighted_accuracy(ty[va], p, n_classes))
            means.append(float(np.mean(uas)))
        best = int(np.argmax(means))
        c, g = grid[best]
        keep = fold != j
        emofilm[~keep] = _scores(svm(c, g, 1).fit(xt[keep], ty[keep]), xt[~keep], n_classes).argmax(axis=1)
        choices[str(j)] = {"C": c, "gamma_scale": g, "inner_ua": means[best],
                           "inner_ua_by_pair": {f"C={a}, gamma scale={b}": m for (a, b), m in zip(grid, means)}}
        log(f"  film fold {j}: emofilm SVM C {c}, gamma scale {g} (inner UA {means[best]:.4f})")
    preds["emofilm"] = [("emofilm", emofilm)]

    results = {}
    for method in METHODS:
        models = {name: _metrics(ty, p, by, class_names) for name, p in preds[method]}
        results[method] = {"models": models,
                           "mean_over_models": {k: float(np.mean([m[k] for m in models.values()]))
                                                for k in ("ua", "wa", "macro_f1")}}
        log(f"  {method}: UA {results[method]['mean_over_models']['ua']:.4f} (" +
            ", ".join(f"{m['ua']:.4f}" for m in models.values()) + ")")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    header = ["utt_id", "speaker", "film", "film_fold", "language", "gender", "label"]
    columns = [(method, name) for method in METHODS for name, _ in preds[method]]
    header += [f"{method}_{name}" if method != "emofilm" else "emofilm" for method, name in columns]
    writer.writerow(header)
    for i, u in enumerate(tutts):
        writer.writerow([u.utt_id, u.speaker, u.extra["film"], int(fold[i]), u.extra["language"], u.extra["gender"],
                         u.label] + [class_names[int(p[i])] for method in METHODS for _, p in preds[method]])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = OutputFolder(finetune_dir(), f"{selection}__{target}_{stamp}").begin()
    try:
        out.write_text("predictions.csv", buffer.getvalue())
        out.write_json("report.json", {
            "selection": selection, "sweep_id": sel["sweep_id"], "config_hash": sel["config_hash"],
            "settings": sel["settings"], "class_names": list(class_names),
            "target": {"dataset": target, "settings": target_settings, "n_utterances": len(tutts),
                       "support": {c: int((ty == i).sum()) for i, c in enumerate(class_names)},
                       "normalization": "language_gender"},
            "n_folds": n_folds, "film_folds": folds_of,
            "clips_per_film_fold": {str(j): int((fold == j).sum()) for j in range(n_folds)},
            "source_features": {k: source_record.get(k) for k in RECORD_KEYS},
            "target_features": {k: target_record.get(k) for k in RECORD_KEYS},
            "runs": runs, "emofilm_choices": choices, "grid": [list(p) for p in grid], "results": results,
            "started_utc": started, "finished_utc": utc_now()})
        out.write_json("provenance.json", collect("affect.cli emofilm-finetune"))
    except BaseException as err:
        out.fail(err)
        raise
    done = out.complete()
    log(f"  written {done}")
    return done
