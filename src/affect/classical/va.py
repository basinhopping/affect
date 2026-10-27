"""Valence, activation (arousal) and dominance from a frozen encoder's layers (added 2026-10-21).

Dr. Luitel's fourth request of 2026-10-12: a model of IEMOCAP's dimensional ratings. Every utterance's EmoEvaluation
summary line gives the evaluators' mean rating on three scales from 1 to 5, valence (v), activation (a, arousal in
most papers) and dominance (d); affect.data.iemocap carries them in Utterance.extra. A ridge regression on one layer's
utterance vectors (the mean of the layer's frames, the feature sets of nights 19b and 19c, <model>_l<k>), z-scored
per speaker as the classification probe does (affect.classical.runner.normalize), predicts the three scales at once.
Scored by Lin's concordance correlation coefficient (CCC), the usual measure for dimensional emotion: it is 1 only when
the predictions equal the ratings, and it counts a shift or a shrinkage of the predictions against them, which
Pearson's r does not.

The choices, on validation only, in a fixed order:
  1. for each layer, fold and scale, the ridge penalty with the highest validation CCC (ALPHAS; a tie to the smaller);
  2. for each scale, the layer with the highest mean over the folds of step 1's validation CCC (a tie to the lower).
Only then are the chosen layers fitted again with their penalties and the test utterances predicted: the test set of
any layer not chosen is never predicted.

    va_probe(model, data_settings, log, dataset="iemocap") -> the output folder
writes artifacts/va/<model>_<split>_<utc>/ once (affect.runs.store):
    validation.csv        every layer, fold, scale and penalty: the validation CCC (the record of every choice)
    report.json           the choices, and per scale each fold's test CCC, Pearson's r and mean squared error, and
                          their means over the folds
    predictions_test.csv  one row per test utterance (each utterance is in one fold's test set): its fold, speaker,
                          gender, class and ratings, and each scale's prediction from its chosen layer
    provenance.json
"""
import csv
import datetime
import io

import numpy as np

from affect import paths
from affect.classical.pretrained import feature_sets
from affect.classical.runner import _Data, normalize, validate_classical
from affect.data.splits import assign
from affect.runs.provenance import collect, utc_now
from affect.runs.store import OutputFolder

SCALES = ("v", "a", "d")
NAMES = {"v": "valence", "a": "activation (arousal)", "d": "dominance"}
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def va_dir():
    return paths.artifacts_dir() / "va"


def ccc(y: np.ndarray, p: np.ndarray) -> float:
    """Lin's concordance correlation coefficient, with population moments."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = np.mean((y - my) * (p - mp))
    den = vy + vp + (my - mp) ** 2
    return float(2 * cov / den) if den > 0 else 0.0


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.std() == 0 or p.std() == 0:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def _job(data_settings: dict, dataset: str, feature_set: str):
    """A classical job of these data settings, only to make the utterances, folds and speakers exactly as the
    classification probe makes them (affect.classical.runner._Data)."""
    return validate_classical({"data": {"name": dataset, **data_settings},
                               "functionals": {"feature_set": feature_set, "normalization": "speaker"},
                               "method": {"name": "svm", "kernel": "rbf", "c_values": [1.0], "gamma_scales": [1.0],
                                          "class_weight": "balanced"},
                               "protocol": {"control": "none", "refit": "train"}})


def va_probe(model: str, data_settings: dict, log, dataset: str = "iemocap", alphas: tuple = ALPHAS):
    from sklearn.linear_model import Ridge

    started = utc_now()
    layers = feature_sets(model)
    data = _Data(_job(data_settings, dataset, layers[0]))
    utts = data.utterances
    # The scale is 1 to 5, but a few of IEMOCAP's averages lie just outside it (valence up to 5.5, dominance down to
    # 0.5: night 19's first look, analyses/night19_20261019/va_first_look_output.txt), so the ratings are used as the
    # files give them, and only a value that is not a number from 0 to 6 is refused (changed 2026-10-21, after the
    # first start refused the real data; no probe had been fitted).
    for u in utts:
        if any(not isinstance(u.extra.get(s), (int, float)) or not np.isfinite(u.extra[s]) or not 0 <= u.extra[s] <= 6
               for s in SCALES):
            raise ValueError(f"{u.utt_id}: ratings {[u.extra.get(s) for s in SCALES]}, not three numbers from 0 to 6")
    y = np.array([[float(u.extra[s]) for s in SCALES] for u in utts])
    folds = list(data.folds)
    rows = {}
    for f in folds:
        roles = assign(utts, data.folds[f])
        rows[f] = {r: np.array([data.position[u] for u in roles[r]], dtype=np.int64) for r in ("train", "val", "test")}
    tested = np.concatenate([rows[f]["test"] for f in folds])
    if len(set(tested.tolist())) != len(tested):
        raise ValueError("an utterance is in more than one fold's test set")
    log(f"{model}: {len(layers)} layers, {len(utts)} {dataset} utterances ({data_settings}), folds {folds}, "
        f"penalties {list(alphas)}")

    records, val_rows = {}, []
    best = {}                    # (layer, fold, scale) -> (alpha, validation CCC)
    for layer in layers:
        values, _, record = data.functionals(dataset, layer)
        records[layer] = {k: record.get(k) for k in ("feature_set", "n_features", "model", "commit",
                                                     "weights_sha256", "layer", "n_utterances")}
        x = normalize(values, "speaker", np.arange(len(utts)), data.speaker_of)
        for f in folds:
            tr, va = rows[f]["train"], rows[f]["val"]
            for alpha in alphas:
                pred = Ridge(alpha=alpha).fit(x[tr], y[tr]).predict(x[va])
                for j, s in enumerate(SCALES):
                    c = ccc(y[va, j], pred[:, j])
                    val_rows.append((layer, f, s, alpha, c))
                    if (layer, f, s) not in best or c > best[(layer, f, s)][1]:
                        best[(layer, f, s)] = (alpha, c)
        log(f"  {layer}: validation CCC (best penalty per fold, mean over folds) "
            + ", ".join(f"{s} {np.mean([best[(layer, f, s)][1] for f in folds]):.4f}" for s in SCALES))
    chosen = {}
    for s in SCALES:
        means = [float(np.mean([best[(layer, f, s)][1] for f in folds])) for layer in layers]
        k = int(np.argmax(means))                    # the first of the highest: the lower layer on a tie
        chosen[s] = {"layer": layers[k], "validation_ccc": means[k],
                     "alpha_by_fold": {f: best[(layers[k], f, s)][0] for f in folds},
                     "validation_ccc_by_fold": {f: best[(layers[k], f, s)][1] for f in folds},
                     "validation_ccc_by_layer": dict(zip(layers, means))}
        log(f"  chosen for {NAMES[s]}: {layers[k]} (validation CCC {means[k]:.4f})")

    pred_test = np.full((len(utts), len(SCALES)), np.nan)
    fold_of = np.empty(len(utts), dtype=object)
    results = {}
    for j, s in enumerate(SCALES):
        layer = chosen[s]["layer"]
        values, _, _ = data.functionals(dataset, layer)
        x = normalize(values, "speaker", np.arange(len(utts)), data.speaker_of)
        per_fold = {}
        for f in folds:
            tr, te = rows[f]["train"], rows[f]["test"]
            p = Ridge(alpha=chosen[s]["alpha_by_fold"][f]).fit(x[tr], y[tr]).predict(x[te])[:, j]
            pred_test[te, j] = p
            fold_of[te] = f
            per_fold[f] = {"ccc": ccc(y[te, j], p), "pearson": pearson(y[te, j], p),
                           "mse": float(np.mean((y[te, j] - p) ** 2)), "n": int(te.shape[0])}
        results[s] = {"scale": NAMES[s], "layer": layer, "by_fold": per_fold,
                      **{k: float(np.mean([v[k] for v in per_fold.values()])) for k in ("ccc", "pearson", "mse")}}
        log(f"  test {NAMES[s]}: CCC {results[s]['ccc']:.4f} (mean over the folds: "
            + ", ".join(f"{v['ccc']:.4f}" for v in per_fold.values()) + f"), Pearson {results[s]['pearson']:.4f}")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["utt_id", "fold", "speaker", "gender", "label"] + [f"true_{s}" for s in SCALES]
                    + [f"pred_{s}" for s in SCALES])
    for i in tested.tolist():
        u = utts[i]
        writer.writerow([u.utt_id, fold_of[i], u.speaker, u.extra.get("speaker_gender", ""), u.label]
                        + [f"{y[i, j]:.4f}" for j in range(len(SCALES))]
                        + [f"{pred_test[i, j]:.6f}" for j in range(len(SCALES))])
    val_buffer = io.StringIO()
    vw = csv.writer(val_buffer, lineterminator="\n")
    vw.writerow(["layer", "fold", "scale", "alpha", "validation_ccc"])
    for layer, f, s, alpha, c in val_rows:
        vw.writerow([layer, f, s, alpha, f"{c:.6f}"])
    split = data_settings.get("split", "split")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = OutputFolder(va_dir(), f"{model}_{split}_{stamp}").begin()
    try:
        folder.write_text("validation.csv", val_buffer.getvalue())
        folder.write_text("predictions_test.csv", buffer.getvalue())
        folder.write_json("report.json", {
            "model": model, "dataset": dataset, "data_settings": data_settings, "n_utterances": len(utts),
            "folds": folds, "alphas": list(alphas), "scales": {s: NAMES[s] for s in SCALES},
            "rating_means": {s: float(y[:, j].mean()) for j, s in enumerate(SCALES)},
            "rating_ranges": {s: [float(y[:, j].min()), float(y[:, j].max())] for j, s in enumerate(SCALES)},
            "ratings_outside_1_to_5": {s: int(((y[:, j] < 1) | (y[:, j] > 5)).sum()) for j, s in enumerate(SCALES)},
            "rating_sds": {s: float(y[:, j].std()) for j, s in enumerate(SCALES)},
            "feature_records": records, "chosen": chosen, "results": results, "folds_hash": data.folds_hash,
            "data_manifest_sha256": data.manifest_sha256, "started_utc": started, "finished_utc": utc_now()})
        folder.write_json("provenance.json", collect("affect.cli va-probe"))
    except BaseException as err:
        folder.fail(err)
        raise
    done = folder.complete()
    log(f"  written {done}")
    return done
