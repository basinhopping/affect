"""Settings of a classical job (added 2026-09-26). Four sections, each validated like every other
setting in the project (affect.settings: no defaults, checked both ways, hashed):

    data         {"name": <registered dataset>, ...its settings}, exactly as for the neural models
    functionals  which openSMILE functionals, and how they are normalised
    method       {"name": "svm" or "tpot", ...its settings}
    protocol     the leak-test control, and what the chosen model is fitted on before it predicts
"""
import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class FunctionalSettings:
    feature_set: Literal["eGeMAPSv02", "ComParE_2016", "emobase"]
    # openSMILE functionals, one vector per utterance: eGeMAPS v02 (88 features; Eyben et al. 2016),
    # the ComParE 2016 set (6,373) or emobase (988)
    normalization: Literal["global", "speaker"]
    # global: each feature z-scored with the mean and SD of the training utterances. speaker: each
    # feature z-scored with the mean and SD of all of that speaker's utterances (audio only, labels
    # unused), which for validation and test speakers uses their unlabeled audio, as the CNN's
    # speaker normalization does. A feature with SD 0 becomes 0; a missing value (NaN) becomes 0.


@dataclasses.dataclass(frozen=True)
class ProtocolSettings:
    control: Literal["none", "shuffle_train_labels"]   # shuffle_train_labels: the leak test, expect chance
    refit: Literal["train", "train_and_val"]
    # the model that predicts the test set: the chosen setting fitted on the training utterances (the
    # neural models' protocol, where validation only selects), or refitted on training plus validation.
    # Validation predictions always come from the model fitted on training alone.


@dataclasses.dataclass(frozen=True)
class SVMSettings:
    kernel: Literal["linear", "rbf"]
    c_values: tuple[float, ...]       # the C values tried; the one with the best validation UA is kept
    gamma_scales: tuple[float, ...]
    # rbf: the gamma values tried, as multiples of 1 / (number of features), which is scikit-learn's
    # "scale" for z-scored features; every (C, gamma) pair is tried. linear: [] (no gamma)
    class_weight: Literal["none", "balanced"]   # balanced: each class weighted by n / (classes x count)


@dataclasses.dataclass(frozen=True)
class TPOTSettings:
    generations: int                  # generations of the genetic programming search (Olson et al. 2016: 100)
    population_size: int              # pipelines kept per generation (Olson et al. 2016: 100)
    offspring_size: int               # new pipelines made per generation (TPOT's default: population_size)
    mutation_rate: float              # share of offspring made by mutation (TPOT's default 0.9)
    crossover_rate: float             # share made by crossover (TPOT's default 0.1); the two add to at most 1
    operators: Literal["default", "light"]
    # TPOT's operator sets: "default" (every classifier and preprocessor TPOT knows) or "light"
    # (TPOT's fast subset, without the slow ensemble methods)
    max_time_mins: float | None       # stop one search after this many minutes; None: run every generation
    max_eval_time_mins: float         # give up on one pipeline after this many minutes
    n_jobs: int                       # CPU processes evaluating pipelines at once
    # Every pipeline is scored by UA (scikit-learn's balanced_accuracy) on the validation set after
    # fitting on the training set: the same selection rule as every other model here.


def check_svm(settings: SVMSettings) -> None:
    if not settings.c_values or any(not c > 0 for c in settings.c_values):
        raise ValueError("method.c_values must list positive numbers")
    if settings.kernel == "linear" and settings.gamma_scales:
        raise ValueError("method.gamma_scales must be [] for the linear kernel")
    if settings.kernel == "rbf" and (not settings.gamma_scales or any(not g > 0 for g in settings.gamma_scales)):
        raise ValueError("method.gamma_scales must list positive numbers for the rbf kernel")
    for name in ("c_values", "gamma_scales"):
        values = getattr(settings, name)
        if len(set(values)) != len(values):
            raise ValueError(f"method.{name} repeat")


def check_tpot(settings: TPOTSettings) -> None:
    if settings.generations < 1 or settings.population_size < 2 or settings.offspring_size < 1:
        raise ValueError("method: generations >= 1, population_size >= 2 and offspring_size >= 1 are required")
    if not (0 <= settings.mutation_rate <= 1 and 0 <= settings.crossover_rate <= 1
            and settings.mutation_rate + settings.crossover_rate <= 1):
        raise ValueError("method: mutation_rate and crossover_rate must be in [0, 1] and add to at most 1")
    if settings.max_time_mins is not None and not settings.max_time_mins > 0:
        raise ValueError("method.max_time_mins must be positive or None")
    if not settings.max_eval_time_mins > 0 or settings.n_jobs < 1:
        raise ValueError("method: max_eval_time_mins must be positive and n_jobs at least 1")
