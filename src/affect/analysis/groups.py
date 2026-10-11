"""Results broken down by who spoke (gender) and what was said (improvised or scripted), and the
gender-routed system: the male model answers for male speakers, the female model for female ones.

Works on prediction rows as the runs write them (predictions_val.csv, predictions_test.csv): dicts
with utt_id, speaker, kind, label and pred. The speaker's gender is the letter at the end of an
IEMOCAP speaker ID (Ses01_F is the Session 1 female).
"""
import re

from affect.eval.metrics import report_by_group

IEMOCAP_SPEAKER = re.compile(r"^Ses0[1-5]_([FM])$")


def speaker_gender(speaker: str) -> str:
    match = IEMOCAP_SPEAKER.match(speaker)
    if not match:
        raise ValueError(f"{speaker!r} is not an IEMOCAP speaker ID like Ses01_F")
    return match.group(1)


def group_of(row: dict, field: str) -> str:
    if field == "gender":
        return speaker_gender(row["speaker"])
    if field in ("kind", "speaker"):
        return row[field]
    raise ValueError(f"no grouping {field!r}; use gender, kind or speaker")


def group_ua(rows: list, field: str, class_names: tuple) -> dict:
    """UA per group, None where a group lacks a class (UA is undefined there)."""
    index = {c: i for i, c in enumerate(class_names)}
    y_true = [index[r["label"]] for r in rows]
    y_pred = [index[r["pred"]] for r in rows]
    groups = [group_of(r, field) for r in rows]
    return {group: rep["ua"] for group, rep in report_by_group(y_true, y_pred, groups, class_names).items()}


def route_by_gender(rows_female_model: list, rows_male_model: list) -> list:
    """The routed system's predictions: each utterance answered by the model of its speaker's gender.

    Both models must have predicted the same utterances with the same labels (the whole held-out
    session, as the leave_one_session_out_train_female / _male schemes test it).
    """
    female = {r["utt_id"]: r for r in rows_female_model}
    male = {r["utt_id"]: r for r in rows_male_model}
    if len(female) != len(rows_female_model) or len(male) != len(rows_male_model):
        raise ValueError("an utterance appears twice in a model's predictions")
    if set(female) != set(male):
        raise ValueError(f"the two models predicted different utterances: {len(set(female) ^ set(male))} differ")
    routed = []
    for utt_id in sorted(female):
        f, m = female[utt_id], male[utt_id]
        if (f["label"], f["speaker"]) != (m["label"], m["speaker"]):
            raise ValueError(f"{utt_id}: the two prediction files disagree on its label or speaker")
        routed.append(dict(f if speaker_gender(f["speaker"]) == "F" else m))
    return routed
