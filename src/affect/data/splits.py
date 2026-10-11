"""Fold schemes. Defined once here, hashed, and read by every model (plan 6.7, last row).

Primary protocol (plan section 2): leave one group out. For test group k the validation group
is the next one in order (the last group uses the first), and the remaining groups train.
With IEMOCAP's five sessions: test S1 validates on S2 and trains on S3, S4, S5; test S5
validates on S1. Final runs retrain on train plus validation (four sessions).

The other schemes (added 2026-09-26; the project page "IEMOCAP protocols and split plan" gives the
reasons) are RoleFold objects: explicit sets of utterance IDs per role, built from the utterance
table by the functions below and by each dataset's own scheme table. An utterance in no role of a
RoleFold is left out of that run, for example the other gender in a gender-specific scheme.
Where a scheme needs a random-looking order it uses stable_order, which sorts by SHA-256 of a
fixed text and the item, so the folds are the same on every machine and Python version. Both
kinds of fold are hashed into every run (folds_hash).
"""
import dataclasses
import hashlib

from affect.hashing import sha256_json


@dataclasses.dataclass(frozen=True)
class Fold:
    name: str
    test_group: str
    val_group: str
    train_groups: tuple


@dataclasses.dataclass(frozen=True)
class RoleFold:
    """A fold given as explicit sets of utterance IDs. `val_group` names the validation set in logs.

    The three roles never share an utterance, with one labeled exception: selection_on_test=True
    marks a fold whose validation set IS its test set, so that the epoch is picked on the test data
    (the split plan's test-as-validation measurement of how much that inflates a result; it is
    never reported as a result). Such a fold must have val exactly equal to test.
    """

    name: str
    val_group: str
    train: frozenset
    val: frozenset
    test: frozenset
    selection_on_test: bool = False

    def __post_init__(self):
        for role in ("train", "val", "test"):
            if type(getattr(self, role)) is not frozenset:
                raise TypeError(f"{self.name}: {role} must be a frozenset of utterance IDs")
        if type(self.selection_on_test) is not bool:
            raise TypeError(f"{self.name}: selection_on_test must be True or False")
        pairs = (("train", "val"), ("train", "test"), ("val", "test"))
        if self.selection_on_test:
            if self.val != self.test:
                raise ValueError(f"{self.name}: selection_on_test needs the validation set to equal the test set")
            pairs = (("train", "val"),)
        for a, b in pairs:
            both = getattr(self, a) & getattr(self, b)
            if both:
                raise ValueError(f"{self.name}: {len(both)} utterances in both {a} and {b}, e.g. {sorted(both)[:3]}")


def leave_one_out_next_validation(groups: tuple) -> tuple:
    """One fold per group, in the given order."""
    if len(groups) < 3:
        raise ValueError(f"need at least 3 groups for train, validation and test, got {groups}")
    if len(set(groups)) != len(groups):
        raise ValueError(f"groups repeat: {groups}")
    folds = []
    for i, test in enumerate(groups):
        val = groups[(i + 1) % len(groups)]
        train = tuple(g for g in groups if g not in (test, val))
        folds.append(Fold(name=f"test_{test}", test_group=test, val_group=val, train_groups=train))
    return tuple(folds)


def assign(utterances: list, fold) -> dict:
    """utt_ids per role. `train_final` is train plus validation, for the final retraining."""
    if isinstance(fold, RoleFold):
        return _assign_role_fold(utterances, fold)
    roles = {"train": [], "val": [], "test": []}
    for u in utterances:
        if u.fold_group == fold.test_group:
            roles["test"].append(u.utt_id)
        elif u.fold_group == fold.val_group:
            roles["val"].append(u.utt_id)
        elif u.fold_group in fold.train_groups:
            roles["train"].append(u.utt_id)
        else:
            raise ValueError(f"{u.utt_id}: fold group {u.fold_group!r} is in no role of {fold.name}")
    for role, ids in roles.items():
        if not ids:
            raise ValueError(f"{fold.name}: no utterances in {role}")
    roles = {role: sorted(ids) for role, ids in roles.items()}
    roles["train_final"] = sorted(roles["train"] + roles["val"])
    return roles


def _assign_role_fold(utterances: list, fold: RoleFold) -> dict:
    present = {u.utt_id for u in utterances}
    roles = {}
    for role in ("train", "val", "test"):
        members = getattr(fold, role)
        if not members:
            raise ValueError(f"{fold.name}: no utterances in {role}")
        missing = members - present
        if missing:
            raise ValueError(f"{fold.name}: {len(missing)} {role} utterances are not in the utterance table, "
                             f"e.g. {sorted(missing)[:3]}; the fold was built from a different table")
        roles[role] = sorted(members)
    # Final retraining on train plus validation would train on the test set when they are one set.
    roles["train_final"] = sorted(roles["train"]) if fold.selection_on_test else sorted(roles["train"] + roles["val"])
    return roles


def folds_hash(utterances: list, folds: tuple) -> str:
    """SHA-256 of every fold's role assignment. Stamped into each run's provenance."""
    payload = {}
    for f in folds:
        if isinstance(f, RoleFold):
            payload[f.name] = {"val_group": f.val_group, **assign(utterances, f)}
            if f.selection_on_test:
                payload[f.name]["selection_on_test"] = True
        else:
            payload[f.name] = {"test_group": f.test_group, "val_group": f.val_group,
                               "train_groups": list(f.train_groups), **assign(utterances, f)}
    return sha256_json(payload)


# Building blocks for RoleFold schemes ----------------------------------------------------------

def stable_order(items, salt: str) -> list:
    """The items sorted by SHA-256 of salt and item: a fixed shuffle, the same on every machine."""
    return sorted(items, key=lambda item: (hashlib.sha256(f"{salt}|{item}".encode("utf-8")).hexdigest(), str(item)))


def deal(units: list, stratum_of, k: int, salt: str) -> dict:
    """Unit -> group number 1..k, dealt round-robin within each stratum.

    Strata are taken in sorted order and the units of each in stable_order; the deal continues
    where the previous stratum stopped, so every group gets an equal share of every stratum
    (within one) and the group sizes differ by at most one.
    """
    if k < 2:
        raise ValueError(f"need at least 2 groups, got {k}")
    if len(set(units)) != len(units):
        raise ValueError("units repeat")
    by_stratum = {}
    for unit in units:
        by_stratum.setdefault(stratum_of(unit), []).append(unit)
    group, position = {}, 0
    for stratum in sorted(by_stratum, key=str):
        for unit in stable_order(by_stratum[stratum], f"{salt}|{stratum}"):
            group[unit] = position % k + 1
            position += 1
    return group


def next_group_folds(order: tuple, members: dict, fold_name, describe=str) -> tuple:
    """Leave one group out, the next group in `order` validating (the last uses the first).

    members: group -> frozenset of utterance IDs. fold_name(test group) names each fold.
    """
    if len(order) < 3:
        raise ValueError(f"need at least 3 groups, got {order}")
    folds = []
    for i, test in enumerate(order):
        val = order[(i + 1) % len(order)]
        train = frozenset().union(*(members[g] for g in order if g not in (test, val)))
        folds.append(RoleFold(name=fold_name(test), val_group=describe(val), train=train,
                              val=frozenset(members[val]), test=frozenset(members[test])))
    return tuple(folds)
