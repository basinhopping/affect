"""Folds: leave one session out, validation on the next session, 5 validates on 1."""
import unittest

from affect.data.base import Utterance
from affect.data.splits import assign, folds_hash, leave_one_out_next_validation


def utt(uid: str, group: str) -> Utterance:
    return Utterance(uid, "x", "spk", group, "d", "impro", "neu", "neu", "a.wav", 16000, 16000, None, {})


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.groups = ("S1", "S2", "S3", "S4", "S5")
        self.folds = leave_one_out_next_validation(self.groups)
        self.utts = [utt(f"{g}_{i}", g) for g in self.groups for i in range(3)]

    def test_validation_is_the_next_session(self):
        pairs = [(f.test_group, f.val_group) for f in self.folds]
        self.assertEqual(pairs, [("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("S4", "S5"), ("S5", "S1")])
        self.assertEqual(self.folds[0].train_groups, ("S3", "S4", "S5"))
        self.assertEqual(self.folds[4].train_groups, ("S2", "S3", "S4"))

    def test_roles_are_disjoint_and_cover_everything(self):
        for fold in self.folds:
            roles = assign(self.utts, fold)
            train, val, test = set(roles["train"]), set(roles["val"]), set(roles["test"])
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, {u.utt_id for u in self.utts})
            self.assertEqual(set(roles["train_final"]), train | val)

    def test_each_session_is_tested_once(self):
        tested = [assign(self.utts, f)["test"] for f in self.folds]
        flat = [x for t in tested for x in t]
        self.assertEqual(sorted(flat), sorted(u.utt_id for u in self.utts))

    def test_hash_changes_with_assignment(self):
        h1 = folds_hash(self.utts, self.folds)
        moved = self.utts[:-1] + [utt(self.utts[-1].utt_id, "S1")]
        self.assertNotEqual(h1, folds_hash(moved, self.folds))
        self.assertEqual(h1, folds_hash(list(reversed(self.utts)), self.folds))

    def test_too_few_groups(self):
        with self.assertRaises(ValueError):
            leave_one_out_next_validation(("S1", "S2"))


if __name__ == "__main__":
    unittest.main()
