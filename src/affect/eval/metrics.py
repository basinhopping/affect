"""Classification metrics (plan section 2).

UA  unweighted accuracy: the mean of per-class recall. The primary metric.
WA  weighted accuracy: correct predictions over all predictions.
Macro F1: the mean of per-class F1. A class that is never predicted has precision 0 and F1 0.

The confusion matrix has true classes as rows and predicted classes as columns. Every metric
is computed from it, so all metrics of one run agree with each other by construction.
"""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(f"expected two 1-D arrays of equal length, got {y_true.shape} and {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("no predictions")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{name} must hold class indices, got dtype {arr.dtype}")
        if arr.min() < 0 or arr.max() >= n_classes:
            raise ValueError(f"{name} holds indices outside 0..{n_classes - 1}")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    if (support == 0).any():
        raise ValueError(f"classes {np.flatnonzero(support == 0).tolist()} have no examples; recall is undefined")
    return np.diag(cm) / support


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    predicted = cm.sum(axis=0)
    out = np.zeros(cm.shape[0], dtype=float)
    nonzero = predicted > 0
    out[nonzero] = np.diag(cm)[nonzero] / predicted[nonzero]
    return out


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    p = per_class_precision(cm)
    r = per_class_recall(cm)
    out = np.zeros_like(p)
    nonzero = (p + r) > 0
    out[nonzero] = 2 * p[nonzero] * r[nonzero] / (p[nonzero] + r[nonzero])
    return out


def unweighted_accuracy(cm: np.ndarray) -> float:
    return float(per_class_recall(cm).mean())


def weighted_accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum())


def macro_f1(cm: np.ndarray) -> float:
    return float(per_class_f1(cm).mean())


def report(y_true, y_pred, class_names: tuple) -> dict:
    """Every metric of one set of predictions, as plain JSON types."""
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    return {
        "class_names": list(class_names),
        "n": int(cm.sum()),
        "support": dict(zip(class_names, cm.sum(axis=1).tolist())),
        "ua": unweighted_accuracy(cm),
        "wa": weighted_accuracy(cm),
        "macro_f1": macro_f1(cm),
        "recall": dict(zip(class_names, per_class_recall(cm).tolist())),
        "precision": dict(zip(class_names, per_class_precision(cm).tolist())),
        "f1": dict(zip(class_names, per_class_f1(cm).tolist())),
        "confusion_true_by_predicted": cm.tolist(),
    }


def report_by_group(y_true, y_pred, groups, class_names: tuple) -> dict:
    """`report` per group (for example improvised against scripted). A group missing a class
    gets its counts and per-class recall where defined, and UA marked undefined."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    out = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        cm = confusion_matrix(y_true[mask], y_pred[mask], len(class_names))
        support = cm.sum(axis=1)
        if (support > 0).all():
            out[group] = report(y_true[mask], y_pred[mask], class_names)
        else:
            recall = {c: (float(cm[i, i] / support[i]) if support[i] else None) for i, c in enumerate(class_names)}
            out[group] = {"class_names": list(class_names), "n": int(cm.sum()),
                          "support": dict(zip(class_names, support.tolist())), "ua": None,
                          "ua_undefined_because": "a class has no examples in this group",
                          "wa": weighted_accuracy(cm), "recall": recall,
                          "confusion_true_by_predicted": cm.tolist()}
    return out
