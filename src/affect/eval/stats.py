"""Statistics for backed results (plan section 3).

- t_interval: mean, SD and t confidence interval over seeds. With 5 seeds, df = 4 and the
  95% critical value is 2.776, so the half-width is 1.24 times the SD.
- paired_bootstrap_ua_difference: the difference in fold-mean UA between two models, with a
  percentile interval from a hierarchical bootstrap. Each draw resamples each model's seeds,
  and within every fold resamples the test utterances once, using the same resample for
  both models. The shared utterance resample is the pairing: a wrong implementation that
  resamples utterances separately per model gives much wider intervals, and the tests
  check that.
- probability_of_outperforming: P(A > B) over all pairs of runs, ties counted half, with a
  percentile bootstrap interval (Bouthillier et al. 2021: significant if the lower end is above
  0.5, meaningful if the upper end is above 0.75).
- holm: Holm's step-down adjustment of p-values for a family of comparisons.

Every function takes an explicit numpy Generator, so results are reproducible.
"""
import numpy as np


def t_interval(values, confidence: float) -> dict:
    from scipy import stats

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("need at least two values")
    n = x.size
    sd = float(x.std(ddof=1))
    t_crit = float(stats.t.ppf((1 + confidence) / 2, n - 1))
    half = t_crit * sd / np.sqrt(n)
    mean = float(x.mean())
    return {"n": n, "mean": mean, "sd": sd, "df": n - 1, "confidence": confidence, "t_critical": t_crit,
            "half_width": float(half), "low": mean - float(half), "high": mean + float(half),
            "min": float(x.min()), "max": float(x.max())}


def _check_fold(fold: dict, name: str) -> None:
    for key in ("utt_ids", "y_true", "correct"):
        if key not in fold:
            raise ValueError(f"{name}: fold is missing {key!r}")
    n = len(fold["utt_ids"])
    if np.asarray(fold["y_true"]).shape != (n,):
        raise ValueError(f"{name}: y_true does not match utt_ids")
    correct = np.asarray(fold["correct"])
    if correct.ndim != 2 or correct.shape[1] != n:
        raise ValueError(f"{name}: correct must be seeds x utterances, got {correct.shape}")


def _fold_ua_per_seed(correct: np.ndarray, onehot: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """UA of every seed under utterance weights. correct: [S, N]; onehot: [N, C];
    weights: [B, N] (counts of each utterance in each draw). Returns [B, S]."""
    weighted_class = weights[:, :, None] * onehot[None, :, :]              # [B, N, C]
    denom = weighted_class.sum(axis=1)                                   # [B, C]
    if (denom == 0).any():
        raise ValueError("a bootstrap draw holds no example of some class; the fold is too small")
    numer = np.einsum("sn,bnc->bsc", correct.astype(float), weighted_class)   # [B, S, C]
    return (numer / denom[:, None, :]).mean(axis=2)


def paired_bootstrap_ua_difference(model_a: list, model_b: list, n_classes: int, n_boot: int,
                                   confidence: float, rng: np.random.Generator, chunk: int) -> dict:
    """Difference in fold-mean UA, A minus B.

    model_a, model_b: one dict per fold, in the same fold order, each with
      utt_ids  the test utterances, in a fixed order
      y_true   their class indices
      correct  seeds x utterances, True where that seed's prediction was right
    The two models must hold the same utterances in the same order in every fold.
    """
    if len(model_a) != len(model_b) or not model_a:
        raise ValueError("both models need the same, nonzero number of folds")
    folds = []
    for i, (fa, fb) in enumerate(zip(model_a, model_b)):
        _check_fold(fa, f"A fold {i}")
        _check_fold(fb, f"B fold {i}")
        if list(fa["utt_ids"]) != list(fb["utt_ids"]):
            raise ValueError(f"fold {i}: the models' test utterances differ; the comparison cannot be paired")
        if not np.array_equal(np.asarray(fa["y_true"]), np.asarray(fb["y_true"])):
            raise ValueError(f"fold {i}: the models' labels differ")
        y = np.asarray(fa["y_true"])
        onehot = np.eye(n_classes)[y]
        folds.append((onehot, np.asarray(fa["correct"], dtype=bool), np.asarray(fb["correct"], dtype=bool)))

    def observed(which: int) -> float:
        per_fold = []
        for onehot, ca, cb in folds:
            c = ca if which == 0 else cb
            w = np.ones((1, onehot.shape[0]))
            per_fold.append(_fold_ua_per_seed(c, onehot, w)[0].mean())
        return float(np.mean(per_fold))

    obs_a, obs_b = observed(0), observed(1)
    s_a, s_b = folds[0][1].shape[0], folds[0][2].shape[0]
    diffs = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        seed_w_a = rng.multinomial(s_a, np.full(s_a, 1 / s_a), size=b) / s_a    # [b, S_a]
        seed_w_b = rng.multinomial(s_b, np.full(s_b, 1 / s_b), size=b) / s_b
        ua_a = np.zeros(b)
        ua_b = np.zeros(b)
        for onehot, ca, cb in folds:
            n = onehot.shape[0]
            w = rng.multinomial(n, np.full(n, 1 / n), size=b).astype(float)   # shared by both models
            ua_a += (_fold_ua_per_seed(ca, onehot, w) * seed_w_a).sum(axis=1)
            ua_b += (_fold_ua_per_seed(cb, onehot, w) * seed_w_b).sum(axis=1)
        diffs[done:done + b] = (ua_a - ua_b) / len(folds)
        done += b
    alpha = 1 - confidence
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_two_sided = min(1.0, 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean())))
    return {"ua_a": obs_a, "ua_b": obs_b, "difference": obs_a - obs_b, "low": float(low), "high": float(high),
            "confidence": confidence, "n_boot": n_boot, "p_two_sided": p_two_sided,
            "n_folds": len(folds), "n_seeds_a": s_a, "n_seeds_b": s_b}


def probability_of_outperforming(a, b) -> float:
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


def probability_of_outperforming_interval(a, b, n_boot: int, confidence: float, rng: np.random.Generator,
                                          meaningful_threshold: float) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = probability_of_outperforming(rng.choice(a, a.size), rng.choice(b, b.size))
    alpha = 1 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"p_a_beats_b": probability_of_outperforming(a, b), "low": float(low), "high": float(high),
            "confidence": confidence, "n_boot": n_boot,
            "significant": bool(low > 0.5), "meaningful": bool(high > meaningful_threshold),
            "meaningful_threshold": meaningful_threshold}


def holm(pvalues) -> list:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.size == 0 or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a non-empty list of numbers in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()
