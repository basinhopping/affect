"""A tiny IEMOCAP-shaped dataset with a known answer, for the end-to-end tests.

The spec below is the single source of truth. It is written out as a release folder (label
files, transcripts, sentence wavs, one of them empty), as a collaborator folder holding all
Session 5 sentence wavs, and as the index Phase 0 would have produced, with every field
computed from the spec directly rather than by the parser under test.
"""
import csv
import json
import wave
from pathlib import Path

CATS = {"neu": "Neutral", "ang": "Anger", "sad": "Sadness", "exc": "Excited", "hap": "Happiness",
        "fru": "Frustration"}
FIELDS = ["utt_id", "session", "speaker", "speaker_gender", "marker_gender", "dialog", "kind", "scenario",
          "turn", "label", "v", "a", "d", "start", "end", "header_dur", "wav_rel", "wav_folder", "sr",
          "channels", "sampwidth", "comptype", "nframes", "wav_dur", "size_bytes", "n_cat_E", "n_dim_E",
          "n_self_cat", "n_self_dim", "votes_all", "votes_first", "dim_E", "has_transcript", "summary_file"]


def spec() -> list:
    """Utterances: (session, dialog, speaker letter, turn, label, evaluator categories, empty wav)."""
    rows = []
    for s in range(1, 6):
        dialog = f"Ses0{s}F_impro01"
        rows.append((s, dialog, "F", "000", "neu", [["neu"], ["neu"], ["neu", "fru"]], False))
        rows.append((s, dialog, "M", "000", "ang", [["ang"], ["ang"], ["fru"]], False))
        rows.append((s, dialog, "F", "001", "xxx", [["neu"], ["sad"], ["ang"]], False))
    rows.append((5, "Ses05M_script01_1", "F", "000", "sad", [["sad"], ["sad"], ["sad"]], False))
    rows.append((5, "Ses05M_script01_1", "M", "000", "exc", [["exc"], ["exc"], ["hap"]], True))
    return rows


def _times(i: int) -> tuple:
    start = 1.0 + 3.0 * i
    dur = 1.25 + 0.5 * i
    return start, start + dur, int(round(dur * 16000))


def _write_wav(path: Path, n_frames: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytes(((seed * 31 + k * 7) % 256) for k in range(2 * n_frames))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames)


def build(root: Path) -> dict:
    """Write release/, collaborator/ and phase0_index.csv under root. Returns their paths."""
    release, collab = root / "release", root / "collaborator"
    index_rows = []
    by_dialog = {}
    for i, (s, dialog, spk, turn, label, cats, empty) in enumerate(spec()):
        uid = f"{dialog}_{spk}{turn}"
        by_dialog.setdefault((s, dialog), []).append((i, uid, label, cats))
        start, end, n_frames = _times(i)
        wav_rel = f"Session{s}/sentences/wav/{dialog}/{uid}.wav"
        _write_wav(collab / dialog / f"{uid}.wav", n_frames, i) if s == 5 else None
        if empty:
            (release / wav_rel).parent.mkdir(parents=True, exist_ok=True)
            (release / wav_rel).write_bytes(b"")
        else:
            _write_wav(release / wav_rel, n_frames, i)
        votes_all, votes_first = {}, {}
        for c in cats:
            for code in set(c):
                votes_all[code] = votes_all.get(code, 0) + 1
            votes_first[c[0]] = votes_first.get(c[0], 0) + 1
        dims = [{"eval": f"E{k + 2}", "val": 2 + k, "act": 3, "dom": 2 + (k % 2)} for k in range(2)]
        middle = dialog.split("_", 1)[1]
        row = {
            "utt_id": uid, "session": s, "speaker": f"Ses0{s}_{spk}", "speaker_gender": spk,
            "marker_gender": dialog[5], "dialog": dialog, "kind": "impro" if middle.startswith("impro") else "script",
            "scenario": middle, "turn": turn, "label": label, "v": 2.5, "a": 3.0, "d": 2.5,
            "start": round(start, 4), "end": round(end, 4), "header_dur": round(end - start, 4),
            "n_cat_E": 3, "n_dim_E": 2, "n_self_cat": 1, "n_self_dim": 1,
            "votes_all": json.dumps(dict(sorted(votes_all.items()))),
            "votes_first": json.dumps(dict(sorted(votes_first.items()))),
            "dim_E": json.dumps(dims), "has_transcript": True, "summary_file": dialog,
        }
        if empty:
            row.update({k: "" for k in ("wav_rel", "wav_folder", "sr", "channels", "sampwidth", "comptype",
                                        "nframes", "wav_dur", "size_bytes")})
        else:
            row.update({"wav_rel": wav_rel, "wav_folder": dialog, "sr": 16000, "channels": 1, "sampwidth": 2,
                        "comptype": "NONE", "nframes": n_frames, "wav_dur": round(n_frames / 16000, 6),
                        "size_bytes": (release / wav_rel).stat().st_size})
        index_rows.append(row)
    for (s, dialog), utts in by_dialog.items():
        emo = release / f"Session{s}" / "dialog" / "EmoEvaluation" / f"{dialog}.txt"
        trans = release / f"Session{s}" / "dialog" / "transcriptions" / f"{dialog}.txt"
        emo.parent.mkdir(parents=True, exist_ok=True)
        trans.parent.mkdir(parents=True, exist_ok=True)
        (emo.parent / "Attribute").mkdir(exist_ok=True)
        lines = ["% [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]", ""]
        tlines = []
        for i, uid, label, cats in utts:
            start, end, _ = _times(i)
            lines.append(f"[{start:.4f} - {end:.4f}]\t{uid}\t{label}\t[2.5000, 3.0000, 2.5000]")
            for k, c in enumerate(cats):
                lines.append(f"C-E{k + 2}:\t" + "; ".join(CATS[x] for x in c) + ";\t()")
            lines.append(f"C-{uid[-4]}1:\tNeutral;\t(self)")
            for k in range(2):
                lines.append(f"A-E{k + 2}:\tval {2 + k}; act 3; dom  {2 + (k % 2)};\t()")
            lines.append(f"A-{uid[-4]}1:\tval 3; act 2; dom  1;\t()")
            lines.append("")
            tlines.append(f"{uid} [{start:08.4f}-{end:08.4f}]: words {i}.")
        tlines.append("M: an unlabeled line")
        emo.write_text("\n".join(lines) + "\n", encoding="utf-8")
        trans.write_text("\n".join(tlines) + "\n", encoding="utf-8")
    (release / "Session1" / "sentences" / "wav" / "Ses01F_impro01" / "stray.pk").write_bytes(b"pk")
    index = root / "phase0" / "iemocap_index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    with open(index, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)
    return {"release": release, "collaborator": collab, "index": index, "rows": index_rows}
