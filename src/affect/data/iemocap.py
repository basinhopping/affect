"""IEMOCAP (Busso et al. 2008).

What this module relies on, each verified on disk in Phase 0
(C:\\Affective Computing\\_phase0_audit\\README.md; plan section 1):

- Labels come from one summary file per dialog, SessionN/dialog/EmoEvaluation/<dialog>.txt.
  The release README says "Evaluation/"; the folder is "EmoEvaluation/".
- The per-evaluator files in EmoEvaluation/Attribute/ store valence and activation reversed
  (6 minus the value) relative to the summary files. This module never reads them.
- The released label is the unique plurality over the three external evaluators' categories
  for all 10,039 utterances, so vote counts from the same lines are consistent with it.
- The speaker is the session plus the letter in the utterance ID's last field:
  Ses01F_impro01_M000 is the Session 1 male. The letter after SesXX marks who wore the markers.
- 501 Session 5 sentence wavs in Harrison's copy are empty (0 bytes). The collaborator's wav/
  folder holds copies of all 2,170 Session 5 sentence wavs; 1,669 are byte-identical to the
  non-empty ones and 501 fill the gap, each a sample-exact cut of its dialog recording.

Parsing below is a port of phase0_iemocap_audit.py (SHA-256 60dda282bdae), and the assembly
checks that it reproduces the Phase 0 index for every utterance.
"""
import collections
import csv
import dataclasses
import json
import re
from pathlib import Path
from typing import Literal

from affect import paths
from affect.data.audio import read_wav_info
from affect.data.base import Dataset, Utterance
from affect.data import iemocap_splits
from affect.data.splits import leave_one_out_next_validation
from affect.registry import register_dataset

HEADER_RE = re.compile(
    r"^\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]\s+(\S+)\s+(\S+)\s+\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*$"
)
EVAL_RE = re.compile(r"^([CA])-([A-Za-z])(\d+):\s*(.*)$")
UTT_RE = re.compile(r"^Ses(\d\d)([FM])_(.+)_([FM])(\d+)$")
DIM_RE = re.compile(r"val\s+(\d+);\s*act\s+(\d+);\s*dom\s+(\d+)")
TRANS_RE = re.compile(r"^(\S+)\s+\[([\d.]+)-([\d.]+)\]:\s?(.*)$")
CAT_MAP = {
    "Neutral": "neu", "Happiness": "hap", "Excited": "exc", "Sadness": "sad", "Anger": "ang",
    "Frustration": "fru", "Surprise": "sur", "Fear": "fea", "Disgust": "dis", "Other": "oth",
}

# Named sources of an assembly (given on the command line), and the names of the folders
# ingested under data/iemocap/raw/. The manifest's "source" column holds these names.
RAW_RELEASE = "IEMOCAP_full_release"
RAW_FILL = "collaborator_wav"
SOURCE_PHASE0 = "phase0_audit"
REFERENCE_NAME = "iemocap_index.csv"
REFERENCE_REL = "reference/phase0_iemocap_index.csv"
META_FILES = frozenset({"MANIFEST.csv", "ASSEMBLY.json"})

# Label sets: raw label -> class. Utterances whose raw label is not a key are left out; "xxx"
# (no majority among the evaluators) is in no set. Counts over the whole corpus (Phase 0 index):
# four_merged 5,531; four_without_excited 4,490; five 5,531; six 7,380; seven 7,487.
# Added 2026-10-03, from Dr. Luitel's meeting of 2026-10-02 (happy and excited stay merged, frustration
# added, with and without neutral): five_with_frustration 7,380; four_with_frustration_no_neutral 5,672;
# three_no_neutral 3,823 (the last separates dropping neutral from adding frustration).
LABEL_SETS = {
    "four_merged": {"neu": "neu", "hap": "hap+exc", "exc": "hap+exc", "ang": "ang", "sad": "sad"},
    "four_without_excited": {"neu": "neu", "hap": "hap", "ang": "ang", "sad": "sad"},
    "five": {"neu": "neu", "hap": "hap", "exc": "exc", "ang": "ang", "sad": "sad"},
    "six": {"neu": "neu", "hap": "hap", "exc": "exc", "ang": "ang", "sad": "sad", "fru": "fru"},
    "seven": {"neu": "neu", "hap": "hap", "exc": "exc", "ang": "ang", "sad": "sad", "fru": "fru", "sur": "sur"},
    "five_with_frustration": {"neu": "neu", "hap": "hap+exc", "exc": "hap+exc", "ang": "ang", "sad": "sad",
                              "fru": "fru"},
    "four_with_frustration_no_neutral": {"hap": "hap+exc", "exc": "hap+exc", "ang": "ang", "sad": "sad",
                                         "fru": "fru"},
    "three_no_neutral": {"hap": "hap+exc", "exc": "hap+exc", "ang": "ang", "sad": "sad"},
}
CLASS_ORDER = {
    "four_merged": ("neu", "hap+exc", "ang", "sad"),
    "four_without_excited": ("neu", "hap", "ang", "sad"),
    "five": ("neu", "hap", "exc", "ang", "sad"),
    "six": ("neu", "hap", "exc", "ang", "sad", "fru"),
    "seven": ("neu", "hap", "exc", "ang", "sad", "fru", "sur"),
    "five_with_frustration": ("neu", "hap+exc", "ang", "sad", "fru"),
    "four_with_frustration_no_neutral": ("hap+exc", "ang", "sad", "fru"),
    "three_no_neutral": ("hap+exc", "ang", "sad"),
}
SUBSETS = {"all": ("impro", "script"), "impro": ("impro",), "script": ("script",)}
SESSIONS = (1, 2, 3, 4, 5)
EXPECTED_FORMAT = {"channels": 1, "sample_width": 2, "sample_rate": 16000, "compression": "NONE"}


@dataclasses.dataclass(frozen=True)
class IEMOCAPSettings:
    label_set: Literal["four_merged", "four_without_excited", "five", "six", "seven", "five_with_frustration",
                       "four_with_frustration_no_neutral", "three_no_neutral"]
    subset: Literal["all", "impro", "script"]
    split: Literal[iemocap_splits.SPLITS]     # the fold schemes of affect.data.iemocap_splits


# Parsing -------------------------------------------------------------------------------------

def _problem(problems: dict, key: str, text: str) -> None:
    problems.setdefault(key, []).append(text)


def parse_label_file(path: Path, problems: dict) -> list:
    """Records of one summary label file, in file order. Same rules as Phase 0."""
    dialog_file = path.stem
    records = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("["):
                m = HEADER_RE.match(line)
                if not m:
                    _problem(problems, "unparsed_header_lines", f"{dialog_file}: {line[:120]}")
                    cur = None
                    continue
                start, end, uid, lab, v, a, d = m.groups()
                rec = {
                    "utt_id": uid, "label": lab, "v": float(v), "a": float(a), "d": float(d),
                    "start": float(start), "end": float(end),
                    "header_dur": round(float(end) - float(start), 4),
                    "summary_file": dialog_file, "cat_votes_E": [], "n_cat_E": 0, "n_dim_E": 0,
                    "n_self_cat": 0, "n_self_dim": 0, "dim_E": [],
                }
                um = UTT_RE.match(uid)
                if not um:
                    _problem(problems, "utt_id_unparsed", uid)
                    cur = None
                    continue
                ss, marker, middle, spk, turn = um.groups()
                kind = "impro" if middle.startswith("impro") else ("script" if middle.startswith("script") else "other")
                rec.update({
                    "session": int(ss), "marker_gender": marker, "dialog": f"Ses{ss}{marker}_{middle}",
                    "speaker_gender": spk, "speaker": f"Ses{ss}_{spk}", "turn": turn, "kind": kind,
                    "scenario": middle,
                })
                if rec["dialog"] != dialog_file:
                    _problem(problems, "id_prefix_not_matching_file", f"{uid} in {dialog_file}")
                records.append(rec)
                cur = rec
            elif cur is not None and line.strip():
                em = EVAL_RE.match(line)
                if not em:
                    _problem(problems, "unparsed_evaluator_lines", f"{dialog_file}: {line[:120]}")
                    continue
                kind, who, idx, content = em.groups()
                body = content.split("\t")[0]
                if kind == "C":
                    cats = [c.strip() for c in body.split(";") if c.strip()]
                    if who == "E":
                        cur["n_cat_E"] += 1
                        cur["cat_votes_E"].append(cats)
                    else:
                        cur["n_self_cat"] += 1
                else:
                    dm = DIM_RE.search(body)
                    if who == "E":
                        cur["n_dim_E"] += 1
                        if dm:
                            cur["dim_E"].append({"eval": f"E{idx}", "val": int(dm.group(1)),
                                                 "act": int(dm.group(2)), "dom": int(dm.group(3))})
                        else:
                            _problem(problems, "unparsed_dim_lines", f"{cur['utt_id']}: {line[:120]}")
                    else:
                        cur["n_self_dim"] += 1
    for rec in records:
        votes_all, votes_first = collections.Counter(), collections.Counter()
        for cats in rec["cat_votes_E"]:
            codes = [CAT_MAP.get(c, "UNMAPPED:" + c) for c in cats]
            for c in set(codes):
                votes_all[c] += 1
            if codes:
                votes_first[codes[0]] += 1
        rec["votes_all"] = dict(sorted(votes_all.items()))
        rec["votes_first"] = dict(sorted(votes_first.items()))
        del rec["cat_votes_E"]
    return records


def parse_transcript_file(path: Path, problems: dict) -> dict:
    out = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            m = TRANS_RE.match(line)
            if not m:
                _problem(problems, "transcript_lines_unparsed", f"{path.name}: {line[:120]}")
                continue
            out[m.group(1)] = m.group(4)
    return out


def _label_files(root: Path, layout: str, session: int) -> list:
    if layout == "release":
        return sorted((root / f"Session{session}" / "dialog" / "EmoEvaluation").glob("*.txt"))
    if layout == "assembled":
        return sorted((root / "labels" / f"Session{session}").glob("*.txt"))
    raise ValueError(layout)


def _transcript_files(root: Path, layout: str, session: int) -> list:
    if layout == "release":
        return sorted((root / f"Session{session}" / "dialog" / "transcriptions").glob("*.txt"))
    if layout == "assembled":
        return sorted((root / "transcripts" / f"Session{session}").glob("*.txt"))
    raise ValueError(layout)


def load_records(root: Path, layout: str) -> tuple:
    """All labeled utterances under root, sorted by utt_id, plus the parse problems found.

    layout "release": the IEMOCAP_full_release folder structure.
    layout "assembled": data/iemocap/assembled/.
    Each record also gets `transcript` (text or None) and `has_transcript`.
    """
    problems: dict = {}
    by_id: dict = {}
    transcripts: dict = {}
    n_label_files = n_transcript_files = 0
    for session in SESSIONS:
        for path in _label_files(root, layout, session):
            n_label_files += 1
            for rec in parse_label_file(path, problems):
                if rec["utt_id"] in by_id:
                    _problem(problems, "duplicate_label_ids", rec["utt_id"])
                    continue
                if rec["session"] != session:
                    _problem(problems, "session_number_mismatch", rec["utt_id"])
                by_id[rec["utt_id"]] = rec
        for path in _transcript_files(root, layout, session):
            n_transcript_files += 1
            transcripts.update(parse_transcript_file(path, problems))
    for uid, rec in by_id.items():
        rec["transcript"] = transcripts.get(uid)
        rec["has_transcript"] = uid in transcripts
    records = [by_id[k] for k in sorted(by_id)]
    counts = {"label_files": n_label_files, "transcript_files": n_transcript_files}
    return records, problems, counts


# Assembled layout -----------------------------------------------------------------------------

def audio_rel(rec: dict) -> str:
    return f"audio/Session{rec['session']}/{rec['dialog']}/{rec['utt_id']}.wav"


def labels_rel(session: int, dialog: str) -> str:
    return f"labels/Session{session}/{dialog}.txt"


def transcripts_rel(session: int, dialog: str) -> str:
    return f"transcripts/Session{session}/{dialog}.txt"


def release_wav_rel(rec: dict) -> str:
    return f"Session{rec['session']}/sentences/wav/{rec['dialog']}/{rec['utt_id']}.wav"


def fill_wav_rel(rec: dict) -> str:
    return f"{rec['dialog']}/{rec['utt_id']}.wav"


def read_assembly_record(root: Path) -> dict:
    """ASSEMBLY.json of an assembled copy. Refuses a copy whose checks did not pass."""
    path = root / "ASSEMBLY.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: the dataset is not assembled here. "
            "Run: python -m affect.cli assemble --dataset iemocap ... (see README.md)"
        )
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("checks", {}).get("ok") is not True:
        raise RuntimeError(f"{path}: the assembly checks did not pass; do not train on this copy")
    return record


# Checks ---------------------------------------------------------------------------------------

def _first(items: list, n: int = 20) -> dict:
    return {"count": len(items), "first": items[:n]}


def compare_with_reference(records: list, wav_info: dict, file_bytes: dict, filled: set, ref_rows: list) -> dict:
    """Field-by-field parity of parsed records with the Phase 0 index (plan 6.6).

    wav_info: utt_id -> WavInfo of the assembled file. file_bytes: utt_id -> its size.
    filled: utt_ids whose audio came from the collaborator's folder. Phase 0 could not read
    those (their files were empty), so the index has no audio fields for them; for every other
    row the audio fields must match exactly.
    """
    ref = {r["utt_id"]: r for r in ref_rows}
    ours = {r["utt_id"]: r for r in records}
    mismatch = collections.defaultdict(list)
    text_fields = ("session", "speaker", "speaker_gender", "marker_gender", "dialog", "kind", "scenario",
                   "turn", "label", "summary_file", "has_transcript")
    float_fields = ("v", "a", "d", "start", "end", "header_dur")
    int_fields = ("n_cat_E", "n_dim_E", "n_self_cat", "n_self_dim")
    json_fields = ("votes_all", "votes_first", "dim_E")
    ref_without_audio = set()
    for uid in sorted(set(ref) & set(ours)):
        a, b = ours[uid], ref[uid]
        for f in text_fields:
            if str(a[f]) != b[f]:
                mismatch[f].append(uid)
        for f in float_fields:
            if float(b[f]) != a[f]:
                mismatch[f].append(uid)
        for f in int_fields:
            if int(b[f]) != a[f]:
                mismatch[f].append(uid)
        for f in json_fields:
            if json.loads(b[f]) != a[f]:
                mismatch[f].append(uid)
        if b["size_bytes"] == "":
            ref_without_audio.add(uid)
            continue
        info = wav_info[uid]
        if int(b["sr"]) != info.sample_rate:
            mismatch["sr"].append(uid)
        if int(b["channels"]) != info.channels:
            mismatch["channels"].append(uid)
        if int(b["sampwidth"]) != info.sample_width:
            mismatch["sampwidth"].append(uid)
        if b["comptype"] != info.compression:
            mismatch["comptype"].append(uid)
        if int(b["nframes"]) != info.n_frames:
            mismatch["nframes"].append(uid)
        if float(b["wav_dur"]) != round(info.n_frames / info.sample_rate, 6):
            mismatch["wav_dur"].append(uid)
        if int(b["size_bytes"]) != file_bytes[uid]:
            mismatch["size_bytes"].append(uid)
    only_ref = sorted(set(ref) - set(ours))
    only_ours = sorted(set(ours) - set(ref))
    filled_not_blank = sorted(filled - ref_without_audio)
    blank_not_filled = sorted(ref_without_audio - filled)
    ok = not (mismatch or only_ref or only_ours or filled_not_blank or blank_not_filled)
    return {
        "ok": ok,
        "n_reference_rows": len(ref),
        "n_parsed_rows": len(ours),
        "n_compared": len(set(ref) & set(ours)),
        "n_reference_rows_without_audio": len(ref_without_audio),
        "ids_only_in_reference": _first(only_ref),
        "ids_only_parsed": _first(only_ours),
        "filled_but_reference_has_audio": _first(filled_not_blank),
        "reference_without_audio_but_not_filled": _first(blank_not_filled),
        "mismatch_by_field": {f: _first(ids) for f, ids in sorted(mismatch.items())},
    }


def class_counts(records: list, label_set: str) -> dict:
    """Counts by session and class, and by kind and class, for one label set."""
    mapping = LABEL_SETS[label_set]
    kept = [r for r in records if r["label"] in mapping]
    by_session = {f"S{s}": dict(collections.Counter(mapping[r["label"]] for r in kept if r["session"] == s))
                  for s in SESSIONS}
    by_kind = {k: dict(collections.Counter(mapping[r["label"]] for r in kept if r["kind"] == k))
               for k in ("impro", "script")}
    return {
        "total": len(kept),
        "by_class": dict(collections.Counter(mapping[r["label"]] for r in kept)),
        "by_session": by_session,
        "session_totals": {s: sum(v.values()) for s, v in by_session.items()},
        "by_kind": by_kind,
        "kind_totals": {k: sum(v.values()) for k, v in by_kind.items()},
    }


def _reference_as_records(ref_rows: list) -> list:
    return [{"utt_id": r["utt_id"], "label": r["label"], "session": int(r["session"]), "kind": r["kind"]}
            for r in ref_rows]


def read_reference(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_assembled_copy(root: Path, manifest_rows: list) -> dict:
    """Checks of an assembled copy against its own files and the Phase 0 index.

    Reads the label and transcript files, every wav header, and the reference index copied
    into the assembled folder. Returns a report whose `ok` is True only if every check passed.
    """
    records, problems, file_counts = load_records(root, "assembled")
    by_path = {r["path"]: r for r in manifest_rows}
    wav_info, file_bytes, filled = {}, {}, set()
    formats = collections.Counter()
    bad_format, missing_audio, dur_over = [], [], []
    max_abs_diff = {"filled": 0.0, "release": 0.0}
    for rec in records:
        rel = audio_rel(rec)
        row = by_path.get(rel)
        if row is None:
            missing_audio.append(rec["utt_id"])
            continue
        info = read_wav_info(root / rel)
        wav_info[rec["utt_id"]] = info
        file_bytes[rec["utt_id"]] = row["bytes"]
        is_fill = row["source"] == RAW_FILL
        if is_fill:
            filled.add(rec["utt_id"])
        formats[f"ch={info.channels} bytes={info.sample_width} sr={info.sample_rate} comp={info.compression}"] += 1
        actual = {"channels": info.channels, "sample_width": info.sample_width,
                  "sample_rate": info.sample_rate, "compression": info.compression}
        if actual != EXPECTED_FORMAT or info.n_frames <= 0:
            bad_format.append(rec["utt_id"])
        diff = abs(info.n_frames / info.sample_rate - (rec["end"] - rec["start"]))
        key = "filled" if is_fill else "release"
        max_abs_diff[key] = max(max_abs_diff[key], diff)
        if diff > 0.05:
            dur_over.append(f"{rec['utt_id']}: {diff:.4f}s")
    ref_rows = read_reference(root / REFERENCE_REL)
    parity = compare_with_reference(records, wav_info, file_bytes, filled, ref_rows)
    ours4 = class_counts(records, "four_merged")
    ref4 = class_counts(_reference_as_records(ref_rows), "four_merged")
    dialogs = sorted({(r["session"], r["dialog"]) for r in records})
    blocking_problems = {k: v for k, v in problems.items()
                         if k in ("unparsed_header_lines", "utt_id_unparsed", "duplicate_label_ids",
                                  "session_number_mismatch", "id_prefix_not_matching_file",
                                  "unparsed_evaluator_lines")}
    checks = {
        "every_labeled_utterance_has_audio": not missing_audio,
        "every_wav_is_mono_16bit_16khz_pcm_nonempty": not bad_format,
        "wav_duration_within_50ms_of_label_times": not dur_over,
        "parity_with_phase0_index": parity["ok"],
        "four_class_counts_equal_phase0_index": ours4 == ref4,
        "one_label_file_per_dialog": file_counts["label_files"] == len(dialogs),
        "one_transcript_file_per_dialog": file_counts["transcript_files"] == len(dialogs),
        "no_blocking_parse_problems": not blocking_problems,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "n_utterances": len(records),
        "n_dialogs": len(dialogs),
        "n_filled_from_collaborator": len(filled),
        "files": file_counts,
        "wav_formats": dict(formats),
        "max_abs_wav_minus_label_duration_s": max_abs_diff,
        "missing_audio": _first(missing_audio),
        "bad_format": _first(bad_format),
        "duration_over_50ms": _first(dur_over),
        "parse_problems": {k: _first(v, 5) for k, v in sorted(problems.items())},
        "four_class": ours4,
        "four_class_phase0_index": ref4,
        "all_labels": dict(collections.Counter(r["label"] for r in records).most_common()),
        "parity": parity,
    }


def plan_checks(records: list, release_wav_sizes: dict, fill_rows: list) -> dict:
    """Checks that need only names and sizes, made before anything is copied.

    release_wav_sizes: utt_id -> size of SessionN/sentences/wav/<dialog>/<utt_id>.wav in the release.
    fill_rows: manifest rows of the ingested collaborator folder (<dialog>/<utt_id>.wav).
    - Every sentence wav of the release belongs to a labeled utterance, and every labeled
      utterance has a sentence wav file (possibly empty).
    - Every empty release wav has a non-empty collaborator file.
    - Every collaborator file has a release counterpart.
    """
    label_ids = {r["utt_id"] for r in records}
    fill = {}
    for row in fill_rows:
        parts = row["path"].split("/")
        if len(parts) == 2 and parts[1].endswith(".wav"):
            fill[parts[1][:-4]] = row
    wavs_without_label = sorted(set(release_wav_sizes) - label_ids)
    labels_without_wav = sorted(label_ids - set(release_wav_sizes))
    empty = sorted(u for u, size in release_wav_sizes.items() if size == 0)
    empty_unfilled = [u for u in empty if u not in fill or fill[u]["bytes"] == 0]
    fill_without_release = sorted(set(fill) - set(release_wav_sizes))
    non_wav_in_fill = sorted(r["path"] for r in fill_rows if not r["path"].endswith(".wav"))
    checks = {
        "every_release_sentence_wav_is_labeled": not wavs_without_label,
        "every_labeled_utterance_has_a_release_wav_file": not labels_without_wav,
        "every_empty_release_wav_has_a_collaborator_copy": not empty_unfilled,
        "every_collaborator_file_has_a_release_counterpart": not fill_without_release,
        "collaborator_folder_holds_only_wav_files": not non_wav_in_fill,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "n_labeled_utterances": len(label_ids),
        "n_release_sentence_wavs": len(release_wav_sizes),
        "n_release_sentence_wavs_empty": len(empty),
        "n_collaborator_files": len(fill_rows),
        "wavs_without_label": _first(wavs_without_label),
        "labels_without_wav": _first(labels_without_wav),
        "empty_release_wav_without_collaborator_copy": _first(empty_unfilled),
        "collaborator_without_release_counterpart": _first(fill_without_release),
        "non_wav_files_in_collaborator_folder": _first(non_wav_in_fill),
    }


def after_copy_checks(rows: list, manifests: dict) -> dict:
    """Checks that need the hashes made while copying.

    - Every file copied from the collaborator's folder has the hash its ingest recorded.
    - Every collaborator file whose release counterpart was not empty is byte-identical to it
      (Phase 0 found 1,669 such files, all identical). This supports using his other 501.
    - If the release folder was ingested too, every file copied from the release has the hash
      that ingest recorded (the copy in data/iemocap/raw and this copy came from the same bytes).
    """
    fill = {r["path"]: r for r in manifests[RAW_FILL]["rows"]} if RAW_FILL in manifests else None
    release = {r["path"]: r for r in manifests[RAW_RELEASE]["rows"]} if RAW_RELEASE in manifests else None
    wav_re = re.compile(r"^Session\d/sentences/wav/([^/]+)/([^/]+\.wav)$")
    fill_mismatch, identical, differs, release_mismatch = [], 0, [], []
    for row in rows:
        if row["source"] == RAW_FILL and fill is not None:
            if fill.get(row["source_path"], {}).get("sha256") != row["sha256"]:
                fill_mismatch.append(row["path"])
        elif row["source"] == RAW_RELEASE:
            m = wav_re.match(row["source_path"])
            if m and fill is not None:
                twin = fill.get(f"{m.group(1)}/{m.group(2)}")
                if twin is not None:
                    if twin["sha256"] == row["sha256"]:
                        identical += 1
                    else:
                        differs.append(row["path"])
            if release is not None and release.get(row["source_path"], {}).get("sha256") != row["sha256"]:
                release_mismatch.append(row["path"])
    checks = {}
    if fill is not None:
        checks["fills_match_the_collaborator_manifest"] = not fill_mismatch
        checks["collaborator_copies_of_non_empty_release_files_are_identical"] = not differs
    if release is not None:
        checks["release_copies_match_the_release_manifest"] = not release_mismatch
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "manifests_available": sorted(manifests),
        "n_collaborator_files_identical_to_release": identical,
        "release_manifest_available": release is not None,
        "fill_mismatch": _first(fill_mismatch),
        "collaborator_differs_from_release": _first(differs),
        "release_mismatch": _first(release_mismatch),
    }


# The registered dataset -----------------------------------------------------------------------

@register_dataset("iemocap", settings=IEMOCAPSettings)
class IEMOCAP(Dataset):
    name = "iemocap"
    sample_rate = EXPECTED_FORMAT["sample_rate"]
    meta_files = META_FILES
    assembly_sources = (RAW_RELEASE, RAW_FILL, SOURCE_PHASE0)
    manifests_required = (RAW_FILL,)

    def assembly_listings(self, sources: dict) -> dict:
        """Folders the plan needs listed: each session's sentence wav folder of the release."""
        return {f"S{s}_sentence_wav": sources[RAW_RELEASE] / f"Session{s}" / "sentences" / "wav" for s in SESSIONS}

    def plan_assembly(self, sources: dict, listings: dict, manifests: dict) -> tuple:
        """The files of the assembled copy and where each comes from, plus the plan checks.

        Audio comes from the release unless the release file is empty; then from the
        collaborator's folder. Labels and transcripts come from the release, and the Phase 0
        index is copied in as the reference the checks compare against. Returns (items, report).
        """
        from affect.data.transfer import CopyItem

        release = sources[RAW_RELEASE]
        records, problems, counts = load_records(release, "release")
        blocking = {k: v for k, v in problems.items()
                    if k in ("unparsed_header_lines", "utt_id_unparsed", "duplicate_label_ids",
                             "session_number_mismatch", "id_prefix_not_matching_file", "unparsed_evaluator_lines")}
        wav_sizes, wav_paths = {}, {}
        for s in SESSIONS:
            for rel, size in listings[f"S{s}_sentence_wav"].items():
                parts = rel.split("/")
                if len(parts) == 2 and parts[1].endswith(".wav"):
                    uid = parts[1][:-4]
                    wav_sizes[uid] = size
                    wav_paths[uid] = f"Session{s}/sentences/wav/{rel}"
        fill_rows = manifests[RAW_FILL]["rows"]
        report = plan_checks(records, wav_sizes, fill_rows)
        report["checks"]["label_files_parse_without_blocking_problems"] = not blocking
        report["label_parse"] = {"counts": counts, "problems": {k: _first(v, 5) for k, v in sorted(problems.items())}}
        report["ok"] = all(report["checks"].values())
        if not report["ok"]:
            return [], report
        fill_size = {r["path"]: r["bytes"] for r in fill_rows}
        items = []
        for rec in records:
            uid = rec["utt_id"]
            if wav_paths[uid] != release_wav_rel(rec):
                raise RuntimeError(f"{uid}: wav at {wav_paths[uid]}, expected {release_wav_rel(rec)}")
            if wav_sizes[uid] > 0:
                items.append(CopyItem(audio_rel(rec), RAW_RELEASE, wav_paths[uid], wav_sizes[uid]))
            else:
                frel = fill_wav_rel(rec)
                items.append(CopyItem(audio_rel(rec), RAW_FILL, frel, fill_size[frel]))
        for session, dialog in sorted({(r["session"], r["dialog"]) for r in records}):
            for rel_out, rel_in in (
                (labels_rel(session, dialog), f"Session{session}/dialog/EmoEvaluation/{dialog}.txt"),
                (transcripts_rel(session, dialog), f"Session{session}/dialog/transcriptions/{dialog}.txt"),
            ):
                items.append(CopyItem(rel_out, RAW_RELEASE, rel_in, (release / rel_in).stat().st_size))
        reference = sources[SOURCE_PHASE0] / REFERENCE_NAME
        items.append(CopyItem(REFERENCE_REL, SOURCE_PHASE0, REFERENCE_NAME, reference.stat().st_size))
        return items, report

    def check_after_copy(self, rows: list, manifests: dict) -> dict:
        return after_copy_checks(rows, manifests)

    def check_assembled(self, root: Path, manifest_rows: list) -> dict:
        return check_assembled_copy(root, manifest_rows)

    def class_order(self, settings) -> tuple:
        return CLASS_ORDER[settings.label_set]

    def utterances(self, settings) -> list:
        root = paths.assembled_dir(self.name)
        read_assembly_record(root)
        records, problems, _ = load_records(root, "assembled")
        mapping = LABEL_SETS[settings.label_set]
        kinds = SUBSETS[settings.subset]
        out = []
        for rec in records:
            if rec["label"] not in mapping or rec["kind"] not in kinds:
                continue
            rel = audio_rel(rec)
            info = read_wav_info(root / rel)
            actual = {"channels": info.channels, "sample_width": info.sample_width,
                      "sample_rate": info.sample_rate, "compression": info.compression}
            if actual != EXPECTED_FORMAT or info.n_frames <= 0:
                raise ValueError(f"{rel}: unexpected audio format {actual}, {info.n_frames} frames")
            out.append(Utterance(
                utt_id=rec["utt_id"], dataset=self.name, speaker=rec["speaker"],
                fold_group=f"S{rec['session']}", dialog=rec["dialog"], kind=rec["kind"],
                raw_label=rec["label"], label=mapping[rec["label"]], audio_path=rel,
                n_frames=info.n_frames, sample_rate=info.sample_rate, transcript=rec["transcript"],
                extra={k: rec[k] for k in ("session", "scenario", "turn", "speaker_gender", "marker_gender",
                                           "v", "a", "d", "start", "end", "votes_all", "votes_first", "n_cat_E")},
            ))
        return out

    def soft_targets(self, settings, utterances) -> list:
        """Per utterance, the external evaluators' votes over the classes, as a distribution.

        An evaluator may list several categories; each listed category counts once (the same
        counting as the released label, which is the plurality of these votes). Votes for
        categories outside the label set (frustration, surprise, ...) are left out.
        """
        mapping = LABEL_SETS[settings.label_set]
        classes = CLASS_ORDER[settings.label_set]
        out = []
        for u in utterances:
            counts = [0.0] * len(classes)
            for raw, n in u.extra["votes_all"].items():
                if raw in mapping:
                    counts[classes.index(mapping[raw])] += n
            total = sum(counts)
            if total == 0:
                raise ValueError(f"{u.utt_id}: no votes in the label set")
            out.append([c / total for c in counts])
        return out

    def folds(self, settings, utterances) -> tuple:
        if settings.split == "leave_one_session_out":
            return leave_one_out_next_validation(tuple(f"S{s}" for s in SESSIONS))
        return iemocap_splits.build(settings.split, utterances)

    def fold_names(self, settings) -> tuple:
        return iemocap_splits.fold_names(settings.split)
