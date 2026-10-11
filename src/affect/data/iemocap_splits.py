"""IEMOCAP fold schemes: the data setting `split` (split plan 2026-09-26).

Every scheme has a name, a fixed list of fold names that needs no data (fold_names, used to check
sweep files at enqueue), and a builder that makes the folds from the utterance table (build).
Validation always comes from the training side of a split, never from the test side.

    leave_one_session_out                   test one session, validate on the next, train on 3 (5 folds)
    leave_one_speaker_out                   test one speaker, validate on the partner from the same
                                            session, train on the other 4 sessions (10 folds; the
                                            literature's default, Antoniou et al. 2023)
    leave_one_speaker_out_partner_in_train  test one speaker, validate on a speaker from another
                                            session, train on the other 8 speakers, the test
                                            speaker's partner included (10 folds; a leakage rung)
    dialog_5fold                            dialogs dealt into 5 groups within each session; test one
                                            group, validate on the next (speakers overlap, dialogs do not)
    utterance_5fold                         utterances dealt into 5 groups by session, speaker and
                                            class; test one group, validate on the next (the pooled
                                            random split: speakers and dialogs overlap)
    within_session_5fold                    inside each session, utterances dealt into 5 groups by
                                            speaker and class; test one, validate on the next (25 folds)
    within_session_cross_speaker            inside each session, train on one actor (a fifth of it,
                                            dealt by class, validates) and test on the partner (10 folds)
    leave_one_session_out_train_female      session folds; training and validation use female speakers
    leave_one_session_out_train_male        only (or male only); the test is the whole held-out session,
                                            so one run gives same-gender, cross-gender and all
    leave_one_session_out_train_impro       session folds; training and validation use improvised (or
    leave_one_session_out_train_script      scripted) utterances only; the test is the whole session
    leave_one_session_out_half_train        session folds; training uses half of the training
                                            utterances, dealt by session, speaker and class (the control
                                            for gender models, which see about half the data)

Added 2026-09-26 from the literature review (section 12 of claude/IEMOCAP literature review.md):

    leave_one_session_out_random_val        test one session; train on the other 4 minus a fifth of
                                            them, dealt by session, speaker and class, which validates
                                            (the pretrained-model convention: emotion2vec, EmoBox)
    leave_one_session_out_test_as_val       test one session; train on the other 4; the epoch is picked
                                            on the test session itself. A labeled measurement of how
                                            much test-set selection inflates a result, never a result
    session5_test                           one fold: Sessions 1 to 4 train, a tenth of their dialogs
                                            (dealt within each session) validates, Session 5 tests (the
                                            conversation convention: DialogueGCN, DialogueRNN)
    leave_one_session_out_quarter_train     session folds; training keeps a quarter (or three quarters)
    leave_one_session_out_three_quarter_train  of the training utterances; with half_train a nested
                                            training-size curve: quarter within half within three quarters
    leave_one_session_out_script_held_out   15 folds: test one script (script01, 02, 03) of one session,
                                            with that script removed from training and validation
                                            (Pepino 2020's script-disjoint test)
    leave_one_session_out_other_script_held_out  the same 15 test sets, with a different script
                                            removed instead (the next one: 01 -> 02 -> 03 -> 01), so the
                                            training data shrinks about as much but still holds the
                                            tested script: the control for the held-out split
    within_speaker_5fold                    inside each speaker, utterances dealt into 5 groups by class;
                                            test one, validate on the next, train on the other 3, all from
                                            that speaker alone (50 folds; the speaker-dependent case)
    leave_one_speaker_out_enroll_quarter    speaker out with the partner validating, plus a quarter of the
                                            test speaker's utterances (dealt by class) moved into training;
                                            the other three quarters test ("enroll the user" first)

An utterance in no role of a fold is left out of that run. Group assignments are deals in a fixed
SHA-256 order (splits.deal), the same on every machine.
"""
from affect.data.splits import deal, leave_one_out_next_validation, next_group_folds, RoleFold

GROUPS = ("S1", "S2", "S3", "S4", "S5")
GENDERS = ("F", "M")
SPEAKERS = tuple(f"Ses0{s}_{g}" for s in range(1, 6) for g in GENDERS)          # fold order
NEXT_SPEAKER_ORDER = tuple(f"Ses0{s}_{g}" for g in GENDERS for s in range(1, 6))  # next is another session
FIVE = ("G1", "G2", "G3", "G4", "G5")

LOSO = "leave_one_session_out"
RESTRICTED_LOSO = {
    "leave_one_session_out_train_female": ("gender", "F", "female"),
    "leave_one_session_out_train_male": ("gender", "M", "male"),
    "leave_one_session_out_train_impro": ("kind", "impro", "improvised"),
    "leave_one_session_out_train_script": ("kind", "script", "scripted"),
}
HALF = "leave_one_session_out_half_train"
SPEAKER_OUT = "leave_one_speaker_out"
SPEAKER_OUT_PARTNER_TRAIN = "leave_one_speaker_out_partner_in_train"
DIALOG_5 = "dialog_5fold"
UTTERANCE_5 = "utterance_5fold"
WITHIN_5 = "within_session_5fold"
CROSS_SPEAKER = "within_session_cross_speaker"
# added 2026-09-26
RANDOM_VAL = "leave_one_session_out_random_val"
TEST_AS_VAL = "leave_one_session_out_test_as_val"
SESSION5 = "session5_test"
QUARTER = "leave_one_session_out_quarter_train"
THREE_QUARTER = "leave_one_session_out_three_quarter_train"
SCRIPT_HELD_OUT = "leave_one_session_out_script_held_out"
OTHER_SCRIPT_HELD_OUT = "leave_one_session_out_other_script_held_out"
WITHIN_SPEAKER = "within_speaker_5fold"
ENROLL_QUARTER = "leave_one_speaker_out_enroll_quarter"
SCRIPTS = ("script01", "script02", "script03")

SPLITS = (LOSO, SPEAKER_OUT, SPEAKER_OUT_PARTNER_TRAIN, DIALOG_5, UTTERANCE_5, WITHIN_5, CROSS_SPEAKER,
          *RESTRICTED_LOSO, HALF, RANDOM_VAL, TEST_AS_VAL, SESSION5, QUARTER, THREE_QUARTER, SCRIPT_HELD_OUT,
          OTHER_SCRIPT_HELD_OUT, WITHIN_SPEAKER, ENROLL_QUARTER)


def fold_names(split: str) -> tuple:
    if split in (LOSO, *RESTRICTED_LOSO, HALF, RANDOM_VAL, TEST_AS_VAL, QUARTER, THREE_QUARTER):
        return tuple(f"test_{g}" for g in GROUPS)
    if split in (SPEAKER_OUT, SPEAKER_OUT_PARTNER_TRAIN, ENROLL_QUARTER):
        return tuple(f"test_{s}" for s in SPEAKERS)
    if split in (DIALOG_5, UTTERANCE_5):
        return tuple(f"test_{g}" for g in FIVE)
    if split == WITHIN_5:
        return tuple(f"{s}_test_{g}" for s in GROUPS for g in FIVE)
    if split == CROSS_SPEAKER:
        return tuple(f"{s}_{a}_to_{b}" for s in GROUPS for a, b in (("F", "M"), ("M", "F")))
    if split == SESSION5:
        return ("test_S5",)
    if split in (SCRIPT_HELD_OUT, OTHER_SCRIPT_HELD_OUT):
        return tuple(f"test_{g}_{k}" for g in GROUPS for k in SCRIPTS)
    if split == WITHIN_SPEAKER:
        return tuple(f"{s}_test_{g}" for s in SPEAKERS for g in FIVE)
    raise ValueError(f"unknown IEMOCAP split {split!r}; known: {SPLITS}")


def _gender(u) -> str:
    return u.extra["speaker_gender"]


def _script(u):
    """"script01", "script02" or "script03" for a scripted utterance (its scenario, e.g.
    "script01_1b", up to the underscore); None for an improvised one."""
    if u.kind != "script":
        return None
    script = u.extra["scenario"].split("_")[0]
    if script not in SCRIPTS:
        raise ValueError(f"{u.utt_id}: scenario {u.extra['scenario']!r} names no known script")
    return script


def _check_table(utterances: list) -> None:
    ids = [u.utt_id for u in utterances]
    if len(set(ids)) != len(ids):
        raise ValueError("utterance IDs repeat")
    for u in utterances:
        if u.fold_group not in GROUPS or u.speaker not in SPEAKERS:
            raise ValueError(f"{u.utt_id}: session {u.fold_group!r} or speaker {u.speaker!r} is not IEMOCAP's")
        if u.speaker != f"Ses0{u.fold_group[1]}_{_gender(u)}":
            raise ValueError(f"{u.utt_id}: speaker {u.speaker} does not match session {u.fold_group} "
                             f"and gender {_gender(u)}")


def _ids(utterances: list, keep) -> frozenset:
    return frozenset(u.utt_id for u in utterances if keep(u))


def _restricted(utterances: list, field: str, value: str, word: str) -> tuple:
    def keep(u):
        return (_gender(u) if field == "gender" else u.kind) == value

    folds = []
    for f in leave_one_out_next_validation(GROUPS):
        folds.append(RoleFold(
            name=f.name, val_group=f"{f.val_group} {word}",
            train=_ids(utterances, lambda u: u.fold_group in f.train_groups and keep(u)),
            val=_ids(utterances, lambda u: u.fold_group == f.val_group and keep(u)),
            test=_ids(utterances, lambda u: u.fold_group == f.test_group)))
    return tuple(folds)


def _half(utterances: list) -> tuple:
    by_id = {u.utt_id: u for u in utterances}
    group = deal(sorted(by_id), lambda i: (by_id[i].fold_group, by_id[i].speaker, by_id[i].label), 2,
                 "iemocap half_train")
    folds = []
    for f in leave_one_out_next_validation(GROUPS):
        folds.append(RoleFold(
            name=f.name, val_group=f.val_group,
            train=_ids(utterances, lambda u: u.fold_group in f.train_groups and group[u.utt_id] == 1),
            val=_ids(utterances, lambda u: u.fold_group == f.val_group),
            test=_ids(utterances, lambda u: u.fold_group == f.test_group)))
    return tuple(folds)


def _speaker_out(utterances: list, partner_validates: bool) -> tuple:
    folds = []
    for speaker in SPEAKERS:
        session = f"S{speaker[4]}"
        partner = f"{speaker[:6]}{'M' if speaker.endswith('F') else 'F'}"
        if partner_validates:
            val_speaker = partner
            train = _ids(utterances, lambda u: u.fold_group != session)
        else:
            val_speaker = NEXT_SPEAKER_ORDER[(NEXT_SPEAKER_ORDER.index(speaker) + 1) % len(NEXT_SPEAKER_ORDER)]
            train = _ids(utterances, lambda u: u.speaker not in (speaker, val_speaker))
        folds.append(RoleFold(name=f"test_{speaker}", val_group=val_speaker, train=train,
                              val=_ids(utterances, lambda u: u.speaker == val_speaker),
                              test=_ids(utterances, lambda u: u.speaker == speaker)))
    return tuple(folds)


def _dialog_5(utterances: list) -> tuple:
    session_of = {u.dialog: u.fold_group for u in utterances}
    group = deal(sorted(session_of), lambda d: session_of[d], 5, "iemocap dialog_5fold")
    members = {g: _ids(utterances, lambda u, g=g: f"G{group[u.dialog]}" == g) for g in FIVE}
    return next_group_folds(FIVE, members, lambda g: f"test_{g}")


def _utterance_5(utterances: list) -> tuple:
    by_id = {u.utt_id: u for u in utterances}
    group = deal(sorted(by_id), lambda i: (by_id[i].fold_group, by_id[i].speaker, by_id[i].label), 5,
                 "iemocap utterance_5fold")
    members = {g: frozenset(i for i in by_id if f"G{group[i]}" == g) for g in FIVE}
    return next_group_folds(FIVE, members, lambda g: f"test_{g}")


def _within_5(utterances: list) -> tuple:
    folds = []
    for session in GROUPS:
        by_id = {u.utt_id: u for u in utterances if u.fold_group == session}
        group = deal(sorted(by_id), lambda i: (by_id[i].speaker, by_id[i].label), 5,
                     f"iemocap within_session_5fold {session}")
        members = {g: frozenset(i for i in by_id if f"G{group[i]}" == g) for g in FIVE}
        folds.extend(next_group_folds(FIVE, members, lambda g, s=session: f"{s}_test_{g}",
                                      lambda g, s=session: f"{s} {g}"))
    return tuple(folds)


def _cross_speaker(utterances: list) -> tuple:
    folds = []
    for session in GROUPS:
        for a, b in (("F", "M"), ("M", "F")):
            source, target = f"Ses0{session[1]}_{a}", f"Ses0{session[1]}_{b}"
            by_id = {u.utt_id: u for u in utterances if u.speaker == source}
            group = deal(sorted(by_id), lambda i: by_id[i].label, 5, f"iemocap cross_speaker {source}")
            folds.append(RoleFold(
                name=f"{session}_{a}_to_{b}", val_group=f"{source} fifth 1",
                train=frozenset(i for i in by_id if group[i] != 1),
                val=frozenset(i for i in by_id if group[i] == 1),
                test=_ids(utterances, lambda u: u.speaker == target)))
    return tuple(folds)


def _random_val(utterances: list) -> tuple:
    by_id = {u.utt_id: u for u in utterances}
    group = deal(sorted(by_id), lambda i: (by_id[i].fold_group, by_id[i].speaker, by_id[i].label), 5,
                 "iemocap random_val")
    folds = []
    for test in GROUPS:
        folds.append(RoleFold(
            name=f"test_{test}", val_group="a fifth of the other 4 sessions",
            train=_ids(utterances, lambda u: u.fold_group != test and group[u.utt_id] != 1),
            val=_ids(utterances, lambda u: u.fold_group != test and group[u.utt_id] == 1),
            test=_ids(utterances, lambda u: u.fold_group == test)))
    return tuple(folds)


def _test_as_val(utterances: list) -> tuple:
    folds = []
    for test in GROUPS:
        members = _ids(utterances, lambda u: u.fold_group == test)
        folds.append(RoleFold(
            name=f"test_{test}", val_group=f"{test}, the test session itself", selection_on_test=True,
            train=_ids(utterances, lambda u: u.fold_group != test), val=members, test=members))
    return tuple(folds)


def _session5(utterances: list) -> tuple:
    session_of = {u.dialog: u.fold_group for u in utterances if u.fold_group != "S5"}
    group = deal(sorted(session_of), lambda d: session_of[d], 10, "iemocap session5_test")
    return (RoleFold(
        name="test_S5", val_group="a tenth of the dialogs of Sessions 1 to 4",
        train=_ids(utterances, lambda u: u.fold_group != "S5" and group[u.dialog] != 1),
        val=_ids(utterances, lambda u: u.fold_group != "S5" and group[u.dialog] == 1),
        test=_ids(utterances, lambda u: u.fold_group == "S5")),)


def _training_share(utterances: list, share: str) -> tuple:
    """quarter: half_train's half, halved again; three_quarter: that half plus half of the other.
    Each cut is a deal by session, speaker and class, so quarter < half < three quarters."""
    by_id = {u.utt_id: u for u in utterances}
    stratum = lambda i: (by_id[i].fold_group, by_id[i].speaker, by_id[i].label)   # noqa: E731
    half = deal(sorted(by_id), stratum, 2, "iemocap half_train")
    if share == "quarter":
        inside = sorted(i for i in by_id if half[i] == 1)
        second = deal(inside, stratum, 2, "iemocap quarter_train")
        keep = {i for i in inside if second[i] == 1}
    else:
        outside = sorted(i for i in by_id if half[i] == 2)
        second = deal(outside, stratum, 2, "iemocap three_quarter_train")
        keep = {i for i in by_id if half[i] == 1} | {i for i in outside if second[i] == 1}
    folds = []
    for f in leave_one_out_next_validation(GROUPS):
        folds.append(RoleFold(
            name=f.name, val_group=f.val_group,
            train=_ids(utterances, lambda u: u.fold_group in f.train_groups and u.utt_id in keep),
            val=_ids(utterances, lambda u: u.fold_group == f.val_group),
            test=_ids(utterances, lambda u: u.fold_group == f.test_group)))
    return tuple(folds)


def _scripts_held_out(utterances: list, remove_the_tested_script: bool) -> tuple:
    folds = []
    for f in leave_one_out_next_validation(GROUPS):
        for k, script in enumerate(SCRIPTS):
            removed = script if remove_the_tested_script else SCRIPTS[(k + 1) % len(SCRIPTS)]
            folds.append(RoleFold(
                name=f"test_{f.test_group}_{script}", val_group=f"{f.val_group} without {removed}",
                train=_ids(utterances, lambda u: u.fold_group in f.train_groups and _script(u) != removed),
                val=_ids(utterances, lambda u: u.fold_group == f.val_group and _script(u) != removed),
                test=_ids(utterances, lambda u: u.fold_group == f.test_group and _script(u) == script)))
    return tuple(folds)


def _within_speaker(utterances: list) -> tuple:
    folds = []
    for speaker in SPEAKERS:
        by_id = {u.utt_id: u for u in utterances if u.speaker == speaker}
        group = deal(sorted(by_id), lambda i: by_id[i].label, 5, f"iemocap within_speaker_5fold {speaker}")
        members = {g: frozenset(i for i in by_id if f"G{group[i]}" == g) for g in FIVE}
        folds.extend(next_group_folds(FIVE, members, lambda g, s=speaker: f"{s}_test_{g}",
                                      lambda g, s=speaker: f"{s} {g}"))
    return tuple(folds)


def _enroll_quarter(utterances: list) -> tuple:
    folds = []
    for f in _speaker_out(utterances, partner_validates=True):
        speaker = f.name[len("test_"):]
        by_id = {u.utt_id: u for u in utterances if u.speaker == speaker}
        group = deal(sorted(by_id), lambda i: by_id[i].label, 4, f"iemocap enroll_quarter {speaker}")
        enrolled = frozenset(i for i in by_id if group[i] == 1)
        folds.append(RoleFold(name=f.name, val_group=f.val_group, train=f.train | enrolled, val=f.val,
                              test=f.test - enrolled))
    return tuple(folds)


def build(split: str, utterances: list) -> tuple:
    """The folds of a scheme, in fold_names order."""
    _check_table(utterances)
    if split == LOSO:
        folds = leave_one_out_next_validation(GROUPS)
    elif split in RESTRICTED_LOSO:
        folds = _restricted(utterances, *RESTRICTED_LOSO[split])
    elif split == HALF:
        folds = _half(utterances)
    elif split == SPEAKER_OUT:
        folds = _speaker_out(utterances, partner_validates=True)
    elif split == SPEAKER_OUT_PARTNER_TRAIN:
        folds = _speaker_out(utterances, partner_validates=False)
    elif split == DIALOG_5:
        folds = _dialog_5(utterances)
    elif split == UTTERANCE_5:
        folds = _utterance_5(utterances)
    elif split == WITHIN_5:
        folds = _within_5(utterances)
    elif split == CROSS_SPEAKER:
        folds = _cross_speaker(utterances)
    elif split == RANDOM_VAL:
        folds = _random_val(utterances)
    elif split == TEST_AS_VAL:
        folds = _test_as_val(utterances)
    elif split == SESSION5:
        folds = _session5(utterances)
    elif split == QUARTER:
        folds = _training_share(utterances, "quarter")
    elif split == THREE_QUARTER:
        folds = _training_share(utterances, "three_quarter")
    elif split == SCRIPT_HELD_OUT:
        folds = _scripts_held_out(utterances, remove_the_tested_script=True)
    elif split == OTHER_SCRIPT_HELD_OUT:
        folds = _scripts_held_out(utterances, remove_the_tested_script=False)
    elif split == WITHIN_SPEAKER:
        folds = _within_speaker(utterances)
    elif split == ENROLL_QUARTER:
        folds = _enroll_quarter(utterances)
    else:
        raise ValueError(f"unknown IEMOCAP split {split!r}; known: {SPLITS}")
    if tuple(f.name for f in folds) != fold_names(split):
        raise RuntimeError(f"{split}: built folds {[f.name for f in folds]} differ from {fold_names(split)}")
    return folds
