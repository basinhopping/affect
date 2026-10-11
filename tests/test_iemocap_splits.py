"""The IEMOCAP fold schemes (affect.data.iemocap_splits) on a synthetic IEMOCAP-shaped table.

Five sessions, one female and one male speaker each, six dialogs per session (three improvised,
three scripted: one performance of each of IEMOCAP's three scripts), both speakers in every
dialog, two utterances of every class per speaker and dialog. Every property a scheme promises is
checked on the folds it builds: which speakers, dialogs, sessions, genders, kinds and scripts may
meet across train, validation and test.
"""
import collections
import typing
import unittest

from affect.data import iemocap_splits as S
from affect.data.base import Utterance
from affect.data.iemocap import CLASS_ORDER, LABEL_SETS, SUBSETS, IEMOCAPSettings
from affect.data.splits import RoleFold, assign, deal, folds_hash, stable_order

CLASSES = ("neu", "hap+exc", "ang", "sad")


SCENARIOS = ("impro01", "impro02", "impro03", "script01_1", "script02_1", "script03_2")


def table() -> list:
    rows = []
    for s in range(1, 6):
        for d, scenario in enumerate(SCENARIOS, 1):
            kind = scenario[:6] if scenario.startswith("script") else scenario[:5]
            dialog = f"Ses0{s}{'F' if d % 2 else 'M'}_{scenario}"
            for g in ("F", "M"):
                for c, label in enumerate(CLASSES):
                    for k in range(2):
                        turn = 10 * c + k
                        rows.append(Utterance(
                            utt_id=f"{dialog}_{g}{turn:03d}", dataset="iemocap", speaker=f"Ses0{s}_{g}",
                            fold_group=f"S{s}", dialog=dialog, kind=kind, raw_label=label, label=label,
                            audio_path="x.wav", n_frames=16000, sample_rate=16000, transcript=None,
                            extra={"speaker_gender": g, "scenario": scenario}))
    return rows


# Schemes whose test sets cover every utterance once; the others test a chosen part on purpose.
PARTIAL_TEST = {S.SESSION5, S.SCRIPT_HELD_OUT, S.OTHER_SCRIPT_HELD_OUT, S.ENROLL_QUARTER}


class TestSchemes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utts = table()
        cls.by_id = {u.utt_id: u for u in cls.utts}
        cls.folds = {split: S.build(split, cls.utts) for split in S.SPLITS}

    def roles(self, split):
        return [(f, assign(self.utts, f)) for f in self.folds[split]]

    def attr(self, ids, field):
        if field == "gender":
            return {self.by_id[i].extra["speaker_gender"] for i in ids}
        return {getattr(self.by_id[i], field) for i in ids}

    def test_settings_list_every_scheme_label_set_and_subset(self):
        fields = {f.name: f.type for f in IEMOCAPSettings.__dataclass_fields__.values()}
        self.assertEqual(typing.get_args(fields["split"]), S.SPLITS)
        self.assertEqual(set(typing.get_args(fields["label_set"])), set(LABEL_SETS))
        self.assertEqual(set(typing.get_args(fields["subset"])), set(SUBSETS))

    def test_label_sets_are_consistent_and_exclude_no_agreement(self):
        for name, mapping in LABEL_SETS.items():
            self.assertEqual(set(mapping.values()), set(CLASS_ORDER[name]), name)
            self.assertEqual(len(set(CLASS_ORDER[name])), len(CLASS_ORDER[name]), name)
            self.assertNotIn("xxx", mapping)
            self.assertNotIn("oth", mapping)

    def test_fold_names_match_and_every_role_is_filled(self):
        for split, folds in self.folds.items():
            self.assertEqual(tuple(f.name for f in folds), S.fold_names(split), split)
            for fold, roles in self.roles(split):
                for role in ("train", "val", "test"):
                    self.assertTrue(roles[role], f"{split} {fold.name} {role}")
                self.assertFalse(set(roles["train"]) & set(roles["val"]))
                self.assertFalse(set(roles["train"]) & set(roles["test"]))
                if split == S.TEST_AS_VAL:
                    self.assertEqual(roles["val"], roles["test"])
                else:
                    self.assertFalse(set(roles["val"]) & set(roles["test"]))
                self.assertEqual(fold.selection_on_test if isinstance(fold, RoleFold) else False,
                                 split == S.TEST_AS_VAL, "only test_as_val selects on the test set")

    def test_every_utterance_is_tested_exactly_once(self):
        for split in S.SPLITS:
            if split in PARTIAL_TEST:
                continue
            tested = collections.Counter(i for _, roles in self.roles(split) for i in roles["test"])
            self.assertEqual(set(tested), set(self.by_id), split)
            self.assertEqual(set(tested.values()), {1}, split)
        for split in PARTIAL_TEST:
            tested = collections.Counter(i for _, roles in self.roles(split) for i in roles["test"])
            if split != S.OTHER_SCRIPT_HELD_OUT:
                self.assertEqual(set(tested.values()), {1}, f"{split}: nothing is tested twice")

    def test_session_schemes_keep_the_test_session_out_of_training(self):
        loso_family = [S.LOSO, S.HALF, *S.RESTRICTED_LOSO]
        for split in loso_family:
            for fold, roles in self.roles(split):
                test_sessions = self.attr(roles["test"], "fold_group")
                self.assertEqual(len(test_sessions), 1)
                self.assertFalse(test_sessions & self.attr(roles["train"] + roles["val"], "fold_group"), split)
                self.assertEqual(len(self.attr(roles["test"], "speaker")), 2, "the whole session is tested")

    def test_speaker_out_with_partner_validating(self):
        for fold, roles in self.roles(S.SPEAKER_OUT):
            (speaker,) = self.attr(roles["test"], "speaker")
            (partner,) = self.attr(roles["val"], "speaker")
            self.assertEqual(speaker[:5], partner[:5], "the partner comes from the same session")
            self.assertNotEqual(speaker, partner)
            self.assertFalse({speaker, partner} & self.attr(roles["train"], "speaker"))
            self.assertEqual(len(self.attr(roles["train"], "speaker")), 8)

    def test_speaker_out_with_partner_in_training(self):
        for fold, roles in self.roles(S.SPEAKER_OUT_PARTNER_TRAIN):
            (speaker,) = self.attr(roles["test"], "speaker")
            (val_speaker,) = self.attr(roles["val"], "speaker")
            partner = f"{speaker[:6]}{'M' if speaker.endswith('F') else 'F'}"
            self.assertNotEqual(speaker[:5], val_speaker[:5], "validation comes from another session")
            train_speakers = self.attr(roles["train"], "speaker")
            self.assertIn(partner, train_speakers)
            self.assertNotIn(speaker, train_speakers)
            self.assertEqual(len(train_speakers), 8)

    def test_dialog_folds_never_split_a_dialog(self):
        for fold, roles in self.roles(S.DIALOG_5):
            test_dialogs = self.attr(roles["test"], "dialog")
            self.assertFalse(test_dialogs & self.attr(roles["train"] + roles["val"], "dialog"))
            self.assertEqual(self.attr(roles["test"], "speaker") - self.attr(roles["train"], "speaker"), set(),
                             "speakers overlap by design")

    def test_pooled_folds_share_speakers_and_balance_classes(self):
        for fold, roles in self.roles(S.UTTERANCE_5):
            self.assertEqual(self.attr(roles["test"], "speaker"), self.attr(roles["train"], "speaker"))
            counts = collections.Counter(self.by_id[i].label for i in roles["test"])
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 10)

    def test_within_session_folds_stay_inside_one_session(self):
        for fold, roles in self.roles(S.WITHIN_5):
            sessions = self.attr(roles["train"] + roles["val"] + roles["test"], "fold_group")
            self.assertEqual(len(sessions), 1)
            self.assertTrue(fold.name.startswith(sessions.pop()))

    def test_cross_speaker_trains_on_one_actor_and_tests_the_partner(self):
        for fold, roles in self.roles(S.CROSS_SPEAKER):
            (source,) = self.attr(roles["train"] + roles["val"], "speaker")
            (target,) = self.attr(roles["test"], "speaker")
            self.assertEqual(source[:5], target[:5])
            self.assertNotEqual(source, target)
            n_source = len(roles["train"]) + len(roles["val"])
            self.assertAlmostEqual(len(roles["val"]) / n_source, 0.2, delta=0.05)

    def test_restricted_schemes(self):
        for split, (field, value, _) in S.RESTRICTED_LOSO.items():
            for fold, roles in self.roles(split):
                self.assertEqual(self.attr(roles["train"] + roles["val"], field), {value}, split)
                other = {"gender": {"F", "M"}, "kind": {"impro", "script"}}[field]
                self.assertEqual(self.attr(roles["test"], field), other, f"{split}: the whole session is tested")

    def test_half_train_is_a_stratified_half(self):
        for fold, roles in self.roles(S.HALF):
            full = next(r for f, r in self.roles(S.LOSO) if f.name == fold.name)
            self.assertEqual(roles["val"], full["val"])
            self.assertEqual(roles["test"], full["test"])
            self.assertTrue(set(roles["train"]) < set(full["train"]))
            self.assertEqual(len(roles["train"]), len(full["train"]) // 2)
            half = collections.Counter((self.by_id[i].speaker, self.by_id[i].label) for i in roles["train"])
            whole = collections.Counter((self.by_id[i].speaker, self.by_id[i].label) for i in full["train"])
            for key, n in whole.items():
                self.assertLessEqual(abs(half[key] - n / 2), 1)

    def test_random_validation_is_a_fifth_of_the_other_sessions(self):
        for fold, roles in self.roles(S.RANDOM_VAL):
            (test_session,) = self.attr(roles["test"], "fold_group")
            self.assertEqual(self.attr(roles["val"], "fold_group") | self.attr(roles["train"], "fold_group"),
                             set(S.GROUPS) - {test_session})
            n = len(roles["train"]) + len(roles["val"])
            self.assertAlmostEqual(len(roles["val"]) / n, 0.2, delta=0.01)
            self.assertEqual(len(self.attr(roles["val"], "speaker")), 8, "every training speaker validates")
            self.assertEqual(len(self.attr(roles["test"], "speaker")), 2)

    def test_test_as_validation_is_labeled_and_selects_on_the_test_session(self):
        for fold, roles in self.roles(S.TEST_AS_VAL):
            self.assertTrue(fold.selection_on_test)
            (test_session,) = self.attr(roles["test"], "fold_group")
            self.assertEqual(roles["val"], roles["test"])
            self.assertEqual(self.attr(roles["train"], "fold_group"), set(S.GROUPS) - {test_session})
            self.assertEqual(roles["train_final"], roles["train"], "never retrain on the test set")

    def test_session5_test_validates_on_whole_training_dialogs(self):
        (fold,) = self.folds[S.SESSION5]
        roles = assign(self.utts, fold)
        self.assertEqual(self.attr(roles["test"], "fold_group"), {"S5"})
        self.assertEqual(set(roles["test"]), {u.utt_id for u in self.utts if u.fold_group == "S5"})
        val_dialogs, train_dialogs = self.attr(roles["val"], "dialog"), self.attr(roles["train"], "dialog")
        self.assertFalse(val_dialogs & train_dialogs)
        self.assertEqual(len(val_dialogs) + len(train_dialogs), 24)
        self.assertEqual(len(val_dialogs), 3, "a tenth of 24 dialogs, dealt round-robin within sessions")
        self.assertNotIn("S5", self.attr(roles["val"] + roles["train"], "fold_group"))

    def test_training_share_curve_is_nested(self):
        loso = {f.name: r for f, r in self.roles(S.LOSO)}
        shares = {split: {f.name: r for f, r in self.roles(split)}
                  for split in (S.QUARTER, S.HALF, S.THREE_QUARTER)}
        for name, full in loso.items():
            quarter, half, three = (set(shares[x][name]["train"]) for x in (S.QUARTER, S.HALF, S.THREE_QUARTER))
            self.assertTrue(quarter < half < three < set(full["train"]))
            for part, share in ((quarter, 0.25), (half, 0.5), (three, 0.75)):
                self.assertAlmostEqual(len(part) / len(full["train"]), share, delta=0.02)
            for x in (S.QUARTER, S.THREE_QUARTER):
                self.assertEqual(shares[x][name]["val"], full["val"])
                self.assertEqual(shares[x][name]["test"], full["test"])

    def test_script_held_out_and_its_control(self):
        def script(i):
            u = self.by_id[i]
            return u.extra["scenario"].split("_")[0] if u.kind == "script" else None

        held = dict(self.roles(S.SCRIPT_HELD_OUT))
        control = dict(self.roles(S.OTHER_SCRIPT_HELD_OUT))
        removed_by_arm = {"held": collections.Counter(), "control": collections.Counter()}
        for (f_held, r_held), (f_control, r_control) in zip(held.items(), control.items()):
            self.assertEqual(f_held.name, f_control.name)
            session, tested = f_held.name.split("_")[1], f_held.name.split("_")[2]
            self.assertEqual(r_held["test"], r_control["test"])
            self.assertEqual({script(i) for i in r_held["test"]}, {tested})
            self.assertEqual(self.attr(r_held["test"], "fold_group"), {session})
            self.assertNotIn(tested, {script(i) for i in r_held["train"] + r_held["val"]})
            self.assertIn(tested, {script(i) for i in r_control["train"]}, "the control keeps the tested script")
            for arm, roles in (("held", r_held), ("control", r_control)):
                present = {script(i) for i in roles["train"]} - {None}
                (gone,) = set(S.SCRIPTS) - present
                removed_by_arm[arm][(session, gone)] += 1
                self.assertEqual(len(self.attr(roles["train"], "fold_group")), 3)
        for arm, removed in removed_by_arm.items():
            self.assertEqual(set(removed.values()), {1}, f"{arm}: each script removed once per session")
            self.assertEqual(len(removed), 15)

    def test_within_speaker_folds_use_one_speaker(self):
        per_speaker = collections.Counter()
        for fold, roles in self.roles(S.WITHIN_SPEAKER):
            (speaker,) = self.attr(roles["train"] + roles["val"] + roles["test"], "speaker")
            self.assertTrue(fold.name.startswith(speaker))
            per_speaker[speaker] += 1
        self.assertEqual(set(per_speaker.values()), {5})
        self.assertEqual(len(per_speaker), 10)

    def test_enrollment_moves_a_quarter_of_the_test_speaker_into_training(self):
        speaker_out = dict((f.name, r) for f, r in self.roles(S.SPEAKER_OUT))
        for fold, roles in self.roles(S.ENROLL_QUARTER):
            base = speaker_out[fold.name]
            speaker = fold.name[len("test_"):]
            enrolled = set(roles["train"]) - set(base["train"])
            self.assertEqual(self.attr(list(enrolled), "speaker"), {speaker})
            self.assertEqual(enrolled | set(roles["test"]), set(base["test"]))
            self.assertAlmostEqual(len(enrolled) / len(base["test"]), 0.25, delta=0.02)
            self.assertEqual(roles["val"], base["val"])
            labels = collections.Counter(self.by_id[i].label for i in enrolled)
            self.assertLessEqual(max(labels.values()) - min(labels.values()), 1, "dealt by class")

    def test_folds_are_deterministic_and_independent_of_input_order(self):
        shuffled = stable_order(self.utts, "any order")
        for split in S.SPLITS:
            self.assertEqual(folds_hash(self.utts, S.build(split, self.utts)),
                             folds_hash(shuffled, S.build(split, shuffled)), split)

    def test_schemes_differ(self):
        hashes = {split: folds_hash(self.utts, folds) for split, folds in self.folds.items()}
        self.assertEqual(len(set(hashes.values())), len(hashes))

    def test_a_foreign_table_is_refused(self):
        bad = self.utts[:-1] + [Utterance(**{**self.utts[-1].__dict__, "speaker": "Ses01_F"})]
        with self.assertRaises(ValueError):
            S.build(S.DIALOG_5, bad)
        with self.assertRaises(ValueError):
            S.fold_names("random_split")


class TestBuildingBlocks(unittest.TestCase):
    def test_deal_balances_groups_and_strata(self):
        units = [f"u{i}" for i in range(103)]
        group = deal(units, lambda u: int(u[1:]) % 7, 5, "salt")
        sizes = collections.Counter(group.values())
        self.assertEqual(set(sizes), {1, 2, 3, 4, 5})
        self.assertLessEqual(max(sizes.values()) - min(sizes.values()), 1)
        for stratum in range(7):
            share = collections.Counter(g for u, g in group.items() if int(u[1:]) % 7 == stratum)
            self.assertLessEqual(max(share.values()) - min(share.values()), 1)
        self.assertEqual(group, deal(list(reversed(units)), lambda u: int(u[1:]) % 7, 5, "salt"))
        self.assertNotEqual(group, deal(units, lambda u: int(u[1:]) % 7, 5, "other salt"))
        with self.assertRaises(ValueError):
            deal(units, str, 1, "salt")
        with self.assertRaises(ValueError):
            deal(units + ["u1"], str, 5, "salt")

    def test_role_fold_refuses_overlap_and_plain_sets(self):
        with self.assertRaises(ValueError):
            RoleFold("f", "v", frozenset({"a"}), frozenset({"a"}), frozenset({"b"}))
        with self.assertRaises(TypeError):
            RoleFold("f", "v", {"a"}, frozenset({"b"}), frozenset({"c"}))
        with self.assertRaises(ValueError):          # validation and test overlap without the label
            RoleFold("f", "v", frozenset({"a"}), frozenset({"b"}), frozenset({"b"}))

    def test_selection_on_test_needs_equal_sets_and_is_hashed(self):
        a, b = frozenset({"a"}), frozenset({"b"})
        fold = RoleFold("f", "v", a, b, b, selection_on_test=True)
        self.assertTrue(fold.selection_on_test)
        with self.assertRaises(ValueError):          # validation must be exactly the test set
            RoleFold("f", "v", a, frozenset({"b", "c"}), b, selection_on_test=True)
        with self.assertRaises(ValueError):          # training still may not touch it
            RoleFold("f", "v", frozenset({"a", "b"}), b, b, selection_on_test=True)
        with self.assertRaises(TypeError):
            RoleFold("f", "v", a, b, b, selection_on_test=1)
        from affect.hashing import sha256_json

        utts = table()[:2]
        ids = [u.utt_id for u in utts]
        flagged = RoleFold("f", "v", frozenset(ids[:1]), frozenset(ids[1:]), frozenset(ids[1:]), selection_on_test=True)
        roles = assign(utts, flagged)
        self.assertEqual(roles["train_final"], sorted(ids[:1]), "never retrain on the test set")
        with_flag = sha256_json({"f": {"val_group": "v", **roles, "selection_on_test": True}})
        self.assertEqual(folds_hash(utts, (flagged,)), with_flag)
        self.assertNotEqual(with_flag, sha256_json({"f": {"val_group": "v", **roles}}))

    def test_assign_refuses_missing_or_empty_roles(self):
        utts = table()[:3]
        ids = [u.utt_id for u in utts]
        ok = RoleFold("f", "v", frozenset(ids[:1]), frozenset(ids[1:2]), frozenset(ids[2:]))
        self.assertEqual(assign(utts, ok)["train_final"], sorted(ids[:2]))
        with self.assertRaises(ValueError):
            assign(utts, RoleFold("f", "v", frozenset(["nope"]), frozenset(ids[1:2]), frozenset(ids[2:])))
        with self.assertRaises(ValueError):
            assign(utts, RoleFold("f", "v", frozenset(), frozenset(ids[1:2]), frozenset(ids[2:])))


class TestSharedAudio(unittest.TestCase):
    def test_audio_is_read_once_per_utterance_list(self):
        import numpy as np

        from affect.train import data

        reads = []
        original = data.read_int16
        data.read_int16 = lambda path, n: reads.append(str(path)) or np.zeros(n, dtype=np.int16)
        saved = dict(data._AUDIO)
        try:
            from pathlib import Path

            utts = table()[:4]
            first = data._audio(Path("/root"), utts)
            again = data._audio(Path("/root"), utts)
            self.assertIs(first, again)
            self.assertEqual(len(reads), 4)
            data._audio(Path("/root"), utts[:2])
            self.assertEqual(len(reads), 6)
        finally:
            data.read_int16 = original
            data._AUDIO.clear()
            data._AUDIO.update(saved)


if __name__ == "__main__":
    unittest.main()
