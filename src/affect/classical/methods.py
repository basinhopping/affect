"""The classical methods, registered in affect.registry.CLASSICAL (added 2026-09-26).

Each method gets z-scored feature matrices for the fold's training, validation and test sets and
the training labels, picks its settings on validation UA alone, and returns class scores for the
validation set (from the model fitted on training alone) and the test set (from the model the
protocol's refit setting asks for), plus a record of what it chose. Scores are probabilities
where the model gives them; for the SVM they are the softmax of its one-vs-rest decision values,
whose largest entry is its prediction.

svm   scikit-learn's SVC (libsvm) on every C (and, for the rbf kernel, every gamma) listed.
tpot  TPOT 0.12 (Olson et al. 2016): genetic programming over scikit-learn pipelines; every
      pipeline is fitted on the training set and scored by UA on the validation set (one
      predefined split), the setting of every other model here.
"""
import numpy as np

from affect.classical.settings import SVMSettings, TPOTSettings, check_svm, check_tpot
from affect.registry import register_classical


def unweighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = [np.mean(y_pred[y_true == c] == c) for c in range(n_classes) if np.any(y_true == c)]
    return float(np.mean(recalls))


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def _svm_scores(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    decision = model.decision_function(x)
    if decision.ndim != 2 or decision.shape[1] != n_classes:
        raise ValueError(f"SVM decision values have shape {decision.shape}, expected (n, {n_classes})")
    return softmax(decision)


@register_classical("svm", settings=SVMSettings)
def fit_svm(settings: SVMSettings, data: dict, seed: int, refit: str, log) -> dict:
    from sklearn.svm import SVC

    check_svm(settings)
    n_classes, n_features = data["n_classes"], data["x_train"].shape[1]
    grid = []
    best = None
    gammas = settings.gamma_scales if settings.kernel == "rbf" else (None,)
    for c in settings.c_values:
        for g in gammas:
            model = SVC(kernel=settings.kernel, C=c, gamma="scale" if g is None else g / n_features,
                        class_weight="balanced" if settings.class_weight == "balanced" else None,
                        decision_function_shape="ovr", random_state=seed)
            model.fit(data["x_train"], data["y_train"])
            val_scores = _svm_scores(model, data["x_val"], n_classes)
            ua = unweighted_accuracy(data["y_val"], val_scores.argmax(axis=1), n_classes)
            grid.append({"C": c, "gamma_scale": g, "val_ua": ua})
            log(f"  svm {settings.kernel} C {c} gamma scale {g}: validation UA {ua:.4f}")
            if best is None or ua > best[0]:
                best = (ua, c, g, model, val_scores)
    ua, c, g, model, val_scores = best
    if refit == "train_and_val":
        final = SVC(kernel=settings.kernel, C=c, gamma="scale" if g is None else g / n_features,
                    class_weight="balanced" if settings.class_weight == "balanced" else None,
                    decision_function_shape="ovr", random_state=seed)
        final.fit(np.vstack([data["x_train"], data["x_val"]]), np.concatenate([data["y_train"], data["y_val"]]))
    else:
        final = model
    return {"val_scores": val_scores, "test_scores": _svm_scores(final, data["x_test"], n_classes),
            "chosen": {"C": c, "gamma_scale": g, "val_ua": ua}, "grid": grid}


def _number(value):
    """A float for JSON, or None when TPOT did not record one (JSON has no NaN)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _pipeline_scores(pipeline, x: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Class probabilities from predict_proba when the pipeline has it, else one-hot predictions."""
    out = np.zeros((x.shape[0], n_classes))
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(x)
            out[:, np.asarray(getattr(pipeline, "classes_", classes))] = probs
            return out
        except (AttributeError, NotImplementedError):
            pass
    out[np.arange(x.shape[0]), np.asarray(pipeline.predict(x), dtype=np.int64)] = 1.0
    return out


@register_classical("tpot", settings=TPOTSettings)
def fit_tpot(settings: TPOTSettings, data: dict, seed: int, refit: str, log) -> dict:
    from sklearn.base import clone
    from tpot import TPOTClassifier

    check_tpot(settings)
    n_classes = data["n_classes"]
    n_train, n_val = data["x_train"].shape[0], data["x_val"].shape[0]
    x = np.vstack([data["x_train"], data["x_val"]])
    y = np.concatenate([data["y_train"], data["y_val"]])
    split = [(np.arange(n_train), np.arange(n_train, n_train + n_val))]
    search = TPOTClassifier(
        generations=settings.generations, population_size=settings.population_size,
        offspring_size=settings.offspring_size, mutation_rate=settings.mutation_rate,
        crossover_rate=settings.crossover_rate, scoring="balanced_accuracy", cv=split, n_jobs=settings.n_jobs,
        max_time_mins=settings.max_time_mins, max_eval_time_mins=settings.max_eval_time_mins,
        random_state=seed, config_dict=None if settings.operators == "default" else "TPOT light",
        verbosity=2, disable_update_check=True)
    search.fit(x, y)
    found = search.fitted_pipeline_                  # TPOT refits the winner on everything passed to fit
    on_train = clone(found).fit(data["x_train"], data["y_train"])
    classes = np.arange(n_classes)
    val_scores = _pipeline_scores(on_train, data["x_val"], classes, n_classes)
    final = found if refit == "train_and_val" else on_train
    ua = unweighted_accuracy(data["y_val"], val_scores.argmax(axis=1), n_classes)
    log(f"  tpot: best pipeline {found}, validation UA refitted on training {ua:.4f}")
    return {"val_scores": val_scores, "test_scores": _pipeline_scores(final, data["x_test"], classes, n_classes),
            "chosen": {"pipeline": str(found), "search_val_ua": _number(getattr(search, "_optimized_pipeline_score", None)),
                       "val_ua": ua, "pipelines_evaluated": len(getattr(search, "evaluated_individuals_", {}) or {})},
            "grid": []}
