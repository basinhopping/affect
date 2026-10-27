"""EmoFilm (Parada-Cabaleiro et al. 2018), a test corpus for the IEMOCAP models (added 2026-10-20).

Dr. Luitel's third request of 2026-10-12: the best three-class models tested cold on EmoFilm's clips of the emotions
that IEMOCAP's three-class set shares. EmoFilm holds 1,115 clips cut from films, English originals and their Spanish
and Italian dubbings, each named <gender>_<emotion><scene><segment><language>.wav (f_gio001aen.wav: a woman,
happiness, scene 001, segment a, English). Its readme names five emotions, fear (ans), disgust (dis), happiness (gio),
anger (rab) and sadness (tri), and no neutral. The assembly (artifacts/analyses/night20_prep_20261020/
assemble_emofilm.py, run on the PC) keeps happiness, anger and sadness, 726 clips, resampled with ffmpeg to 16 kHz mono
16-bit PCM, and writes data/emofilm/assembled/ (never in git):

    wav/<file>.wav             the 726 clips
    MANIFEST.csv               one row per clip (COLUMNS below)
    SPEAKERS.csv, FILMS.csv    the speaker and film IDs' names, from the corpus legend
    ASSEMBLY.json              counts, the ffmpeg version and command, and the SHA-256 of the readme, the legend, the
                               script and MANIFEST.csv

Label set three_no_neutral maps happiness to hap+exc, anger to ang and sadness to sad: the class names and order of
IEMOCAP's three_no_neutral (affect.data.iemocap.CLASS_ORDER), so the outputs of a model trained on IEMOCAP's three
classes line up with these labels. The subset is the language: all, en, es or it. Each utterance's speaker is the
actor's ID (emofilm_<f|m>_<NNN>), its fold_group and dialog the film's ID (film_<NN>), its kind the language; extra
holds the language, gender, film, emotion, scene and segment. EmoFilm is only a test corpus here: it has no fold scheme,
and folds() refuses.
"""
import csv
import dataclasses
import json
from pathlib import Path
from typing import Literal

from affect import paths
from affect.data.audio import read_wav_info
from affect.data.base import Dataset, Utterance
from affect.hashing import sha256_file
from affect.registry import register_dataset

LABEL_SETS = {"three_no_neutral": {"gio": "hap+exc", "rab": "ang", "tri": "sad"}}
CLASS_ORDER = {"three_no_neutral": ("hap+exc", "ang", "sad")}
SUBSETS = {"all": ("en", "es", "it"), "en": ("en",), "es": ("es",), "it": ("it",)}
EXPECTED_FORMAT = {"channels": 1, "sample_width": 2, "sample_rate": 16000, "compression": "NONE"}
COLUMNS = ("utt_id", "audio_path", "speaker", "film", "language", "gender", "emotion_code", "emotion", "scene",
           "segment", "n_frames", "sample_rate", "sha256", "source_sha256")
META_FILES = frozenset({"MANIFEST.csv", "ASSEMBLY.json", "SPEAKERS.csv", "FILMS.csv"})


@dataclasses.dataclass(frozen=True)
class EmoFilmSettings:
    label_set: Literal["three_no_neutral"]       # happiness, anger, sadness as IEMOCAP's hap+exc, ang, sad
    subset: Literal["all", "en", "es", "it"]     # the language: all three, or the English, Spanish or Italian clips


def read_manifest(root: Path) -> list:
    """MANIFEST.csv's rows, after checking it against ASSEMBLY.json (its SHA-256 and the number of clips)."""
    record = json.loads((root / "ASSEMBLY.json").read_text(encoding="utf-8"))
    _, sha = sha256_file(root / "MANIFEST.csv")
    if record.get("manifest_sha256") != sha:
        raise ValueError(f"{root}/MANIFEST.csv has SHA-256 {sha}; ASSEMBLY.json says {record.get('manifest_sha256')}")
    with open(root / "MANIFEST.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"MANIFEST.csv columns {reader.fieldnames}, expected {list(COLUMNS)}")
        rows = list(reader)
    if len(rows) != record.get("kept") or len({r["utt_id"] for r in rows}) != len(rows):
        raise ValueError(f"MANIFEST.csv has {len(rows)} rows ({len({r['utt_id'] for r in rows})} distinct IDs); "
                         f"ASSEMBLY.json says {record.get('kept')} clips")
    return rows


@register_dataset("emofilm", settings=EmoFilmSettings)
class EmoFilm(Dataset):
    name = "emofilm"
    sample_rate = EXPECTED_FORMAT["sample_rate"]
    meta_files = META_FILES

    def class_order(self, settings) -> tuple:
        return CLASS_ORDER[settings.label_set]

    def utterances(self, settings) -> list:
        root = paths.assembled_dir(self.name)
        mapping = LABEL_SETS[settings.label_set]
        languages = SUBSETS[settings.subset]
        out = []
        for row in read_manifest(root):
            if row["emotion_code"] not in mapping or row["language"] not in languages:
                continue
            info = read_wav_info(root / row["audio_path"])
            actual = {"channels": info.channels, "sample_width": info.sample_width,
                      "sample_rate": info.sample_rate, "compression": info.compression}
            if actual != EXPECTED_FORMAT or info.n_frames <= 0 or info.n_frames != int(row["n_frames"]):
                raise ValueError(f"{row['audio_path']}: audio format {actual}, {info.n_frames} frames (the manifest "
                                 f"says {row['n_frames']})")
            out.append(Utterance(
                utt_id=row["utt_id"], dataset=self.name, speaker=row["speaker"], fold_group=row["film"],
                dialog=row["film"], kind=row["language"], raw_label=row["emotion_code"],
                label=mapping[row["emotion_code"]], audio_path=row["audio_path"], n_frames=info.n_frames,
                sample_rate=info.sample_rate, transcript=None,
                extra={k: row[k] for k in ("language", "gender", "film", "emotion", "scene", "segment")}))
        if not out:
            raise ValueError(f"no EmoFilm clip in label set {settings.label_set}, subset {settings.subset}")
        return sorted(out, key=lambda u: u.utt_id)

    def folds(self, settings, utterances) -> tuple:
        raise ValueError("EmoFilm is a test corpus here: it has no fold scheme (train on IEMOCAP, cold-test on it)")

    def fold_names(self, settings) -> tuple:
        raise ValueError("EmoFilm is a test corpus here: it has no fold scheme (train on IEMOCAP, cold-test on it)")
