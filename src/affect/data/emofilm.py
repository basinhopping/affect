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
holds the language, gender, film, emotion, scene and segment. EmoFilm was only a test corpus until 2026-11-10: with
split none (the form of the settings before then) it has no fold scheme, and folds() refuses.

Added 2026-11-10 (week 33; Dr. Luitel, 2026-11-09: train every model on EmoFilm and test it on IEMOCAP, cold and warm,
the same models): split film_5fold, EmoFilm as a training corpus. Its films are dealt into five film folds as the warm
start deals them (affect.classical.finetune.film_folds: the films sorted by name, the k-th in film fold k mod 5, named
F1 to F5), so a clip never shares its film (its scenes, actors and dubbings) with a clip of another fold. Fold test_Fk
tests film fold k, validates on the next (F5 on F1) and trains on the other three, as IEMOCAP's session folds do; a
refit trains on four. soft_targets gives each clip its one label (EmoFilm publishes no votes).

Added 2026-11-13 (RULE_week33_addendum5.txt): split film_5fold_strat. film_5fold's dealing leaves film fold F3 with no
sadness clip (only 18 of the 55 films have one), so its folds test_F2 (validating on F3) and test_F3 have a class with no
example and their recall is undefined. film_5fold_strat keeps the five folds' roles and deals the films so that every
fold holds all three classes: first the films with a sad clip, then those with an angry clip, then the rest, each run in
the order of the films' names and the count going on from one run to the next, the k-th film dealt to film fold k mod 5.
film_5fold stays as it was, for the records made with it.

Added 2026-11-14 (RULE_week33_addendum8.txt): split film_5fold_whole. The corpus's file list, so FILMS.csv, spells 13
films' titles in two to four ways (letter case, spaces, typing errors, a shortened title), so the 55 film IDs of the 726
clips are 37 films (the paper counts 43 films over all 1,115 clips), and the two splits above, which deal film IDs, put
11 films in more than one fold. film_5fold_whole deals whole films: SAME_FILM maps each ID that spells a film another
way to the film's first ID, as reviewed by hand against FILMS.csv, and EmoFilm.folds checks FILMS.csv against that
review before it deals: the SHA-256 of each reviewed title (TITLE_SHA256; the titles stay in FILMS.csv, never in git),
and no two IDs whose titles differ only in case or spaces outside the table. The dealing is film_5fold_strat's, over
the 37 films. Voices recur across films (34 of the 183 speakers speak in more than one, most of them Spanish or
Italian dubbing voices), so some voices still recur across folds. film_5fold and film_5fold_strat stay as they were,
for the records made with them.
"""
import csv
import dataclasses
import hashlib
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


SPLITS = ("none", "film_5fold", "film_5fold_strat", "film_5fold_whole")
FOLD_SPLITS = ("film_5fold", "film_5fold_strat", "film_5fold_whole")     # the splits with folds (EmoFilm trains)
FILM_GROUPS = ("F1", "F2", "F3", "F4", "F5")     # the five film folds of all three

# Added 2026-11-14 (RULE_week33_addendum8.txt): each film ID that spells a film another way -> the film's first ID
# (13 films, 18 IDs, one line a film), as reviewed by hand against FILMS.csv; and the SHA-256 (of the UTF-8 text) of the
# title FILMS.csv holds for each of the 31 IDs of those films, so that a different film list is refused without the
# titles in the repository (the assembly keeps the names in data/, never in git)
SAME_FILM = {
    "film_02": "film_01", "film_07": "film_01",
    "film_04": "film_03",
    "film_09": "film_08",
    "film_15": "film_14",
    "film_46": "film_16", "film_47": "film_16", "film_55": "film_16",
    "film_24": "film_19",
    "film_26": "film_25", "film_50": "film_25",
    "film_28": "film_27",
    "film_51": "film_35",
    "film_37": "film_36",
    "film_40": "film_38",
    "film_43": "film_42", "film_52": "film_42",
    "film_53": "film_44",
}
TITLE_SHA256 = {
    "film_01": "fe855491f46391ae0d629481e1060f9a56f490432556c65cb1601d9c8f7d7a65",
    "film_02": "fcf1a0326564e1029e8694255bff321af5aa22311f493776f907155babe53e5d",
    "film_07": "1bdd2a410299a8ee3ebd7694dc5b458680620de52386548635cd56660ba66770",
    "film_03": "24de04b22efc4f46d5e350806b935b6a80b32cce2f06bde6794aca150908740f",
    "film_04": "7e138f94dcc1ebc36a754eb3f85269938884db21de566d7757fa3db0213949bf",
    "film_08": "8bf1d9e4c7e6f807a75d0b6989b4de8edab60c0faad70a649ea24f448fed70bc",
    "film_09": "722fa72ab266ef81ac19ff1011a674e7b58edc75922c99e541069b75ab3f97f8",
    "film_14": "601214fb0699a0807a1caeb577c8a84a0f51db632813e69a911b3978b4c690e8",
    "film_15": "cf7a3578abf38aef0e2f49392b4dc5ff93005706bea51628427f3863f67c7ee6",
    "film_16": "bf5b41cd9202fc6fe79c4dc08d81dc1a8022c247ba3fe2d62b575698b89a2b25",
    "film_46": "abf7b183181ca19b6c33881d7164af20b109fe80973059783efb81b9d2a8b59e",
    "film_47": "1ed75cb894347cf97f24b81807c5794dde844bf7062eabc0d5ed2482432df352",
    "film_55": "e6cb660afc5274cb83f6901e3655554c4f186a9bfdbab9a627ab58f119772e0a",
    "film_19": "47c698613009927ff888a6ae5bd865b5955596ec64a3a34827a9441937283991",
    "film_24": "4a992ab6aa4e8e02990234592b2c3a0c35e7340bb95d031d8c62d771bc8e7f2b",
    "film_25": "45f5cdd2cc9d0d6e5f942b592fb9723c4fb45924123519d36ed3df3aaddb4e4d",
    "film_26": "267b9a12077eeb58bdafd0dc61cb826ada941ce2ff9b5b5de73686e068c1d967",
    "film_50": "caa88c48bd2ba8fb45c26f79f17ac6ac20459221661687d86359110b8b94ab2b",
    "film_27": "28d224c360a1ee25200c972c99a9a9823840f6a0d564bab1d0aa7fb11f76f13e",
    "film_28": "9504bb9093bfcd4ef59c2c5e3fb31bb466e97cd85cefa99ab12a93ddf24d0bbb",
    "film_35": "08730a746c960b3ab947e61f0436651c731ad532ffedad95f4a75515b39404a8",
    "film_51": "dfc1a1888f01d9476764b3d0e9efc89a46ac3a769c13d0ef27dd38c0b63c5de1",
    "film_36": "3b514241d217dbc8f6319ff3317ebfd2eff32122f2bb27de171a3c94e32a9fd5",
    "film_37": "41965bcc21dd585014f74a0c08b7ad6c35f93f12f1cca3ef7eb7925fe8bcf25f",
    "film_38": "197881254f632e5833c9bbad457b3797a2b7aa872b54efe4bef289e5c8e08811",
    "film_40": "3413f9948435444f2265a40aba7e88a2fdea75d55f7c0c072adb32702b85b578",
    "film_42": "8b4c9ec4f3c01ef1d1aa93fcd4c8501a21199a0f03470a54b2d4449d5d33be2e",
    "film_43": "3a2a6432f0b72e42f0464331bc0cef574c004b43ae1810e81aca61db7eae6eee",
    "film_52": "8decd29b3b453697f1cce7200ea8c6cd45dd03a876b75ee988c7eb3e0f29461d",
    "film_44": "70729b5d0c2d3416a5b045f7f22cb744c7124ee4a41874d3f758b0cf5aad4e6b",
    "film_53": "8703edb7e61d7e67d8d1a4a1a489d999c6ef7027aa4c9465c506c78026c8920f",
}


@dataclasses.dataclass(frozen=True)
class EmoFilmSettings:
    label_set: Literal["three_no_neutral"]       # happiness, anger, sadness as IEMOCAP's hap+exc, ang, sad
    subset: Literal["all", "en", "es", "it"]     # the language: all three, or the English, Spanish or Italian clips
    split: Literal[SPLITS]
    # none: no fold scheme (a test corpus, as before 2026-11-10); film_5fold (added 2026-11-10): five film folds, one
    # tests, the next validates, three train; film_5fold_strat (added 2026-11-13): the same, with every class in every
    # fold; film_5fold_whole (added 2026-11-14): film_5fold_strat over whole films (SAME_FILM)

    ADDED_FIELDS = (("2026-11-10", {"split": "none"}),)


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

    def soft_targets(self, settings, utterances) -> list:
        """Each clip's one label as a distribution (EmoFilm publishes no votes; added 2026-11-10)."""
        classes = CLASS_ORDER[settings.label_set]
        return [[1.0 if c == u.label else 0.0 for c in classes] for u in utterances]

    def folds(self, settings, utterances) -> tuple:
        if settings.split not in FOLD_SPLITS:
            raise ValueError("EmoFilm with split none is a test corpus: it has no fold scheme (train on IEMOCAP, "
                             "cold-test on it; or use split film_5fold_whole)")
        if settings.split == "film_5fold_whole":
            check_whole_films(read_films(paths.assembled_dir(self.name)))
            return film_5fold_whole(utterances)
        return film_5fold(utterances) if settings.split == "film_5fold" else film_5fold_strat(utterances)

    def fold_names(self, settings) -> tuple:
        if settings.split not in FOLD_SPLITS:
            raise ValueError("EmoFilm with split none is a test corpus: it has no fold scheme (train on IEMOCAP, "
                             "cold-test on it; or use split film_5fold_whole)")
        return tuple(f"test_{g}" for g in FILM_GROUPS)


def read_films(root: Path) -> dict:
    """FILMS.csv: film ID -> title, as the corpus legend gives it (added 2026-11-14)."""
    with open(root / "FILMS.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("film", "title"):
            raise ValueError(f"FILMS.csv columns {reader.fieldnames}, expected ['film', 'title']")
        return {row["film"]: row["title"] for row in reader}


def title_sha256(title: str) -> str:
    """The SHA-256 of a title's UTF-8 text (added 2026-11-14)."""
    return hashlib.sha256(title.encode("utf-8")).hexdigest()


def check_whole_films(titles: dict) -> None:
    """Refuse a film list that SAME_FILM was not reviewed against (added 2026-11-14, RULE_week33_addendum8.txt): every
    ID of TITLE_SHA256 must hold a title with that SHA-256, and no two films left apart may have titles that differ only
    in case or spaces."""
    wrong = [f"{f}: {titles.get(f)!r}" for f, h in TITLE_SHA256.items()
             if f not in titles or title_sha256(titles[f]) != h]
    if wrong:
        raise ValueError(f"FILMS.csv is not the film list SAME_FILM was reviewed against (titles not as reviewed: "
                         f"{wrong})")
    seen = {}
    for film in sorted(titles):
        key = " ".join(titles[film].split()).casefold()
        if key in seen and whole_film(seen[key]) != whole_film(film):
            raise ValueError(f"{seen[key]} and {film} spell the same title ({titles[film]!r}) but SAME_FILM keeps "
                             f"them apart")
        seen.setdefault(key, film)


def whole_film(film: str) -> str:
    """The film a film ID belongs to: its first ID (added 2026-11-14)."""
    return SAME_FILM.get(film, film)


def film_5fold(utterances: list) -> tuple:
    """The five film folds (added 2026-11-10): the films dealt as affect.classical.finetune.film_folds deals them (the
    k-th film by name in fold k mod 5, fold k named F(k + 1)); fold test_Fk tests Fk, validates on the next and trains on
    the other three."""
    from affect.classical.finetune import film_folds
    from affect.data.splits import next_group_folds

    fold_of = film_folds([u.extra["film"] for u in utterances], len(FILM_GROUPS))
    members = {g: frozenset(u.utt_id for u in utterances if FILM_GROUPS[fold_of[u.extra["film"]]] == g)
               for g in FILM_GROUPS}
    empty = [g for g in FILM_GROUPS if not members[g]]
    if empty:
        raise ValueError(f"film folds {empty} hold no clip")
    return next_group_folds(FILM_GROUPS, members, lambda g: f"test_{g}", lambda g: f"films {g}")


STRAT_FIRST = ("sad", "ang")     # film_5fold_strat deals the films with a clip of these classes first, in this order


def _film_id(u) -> str:
    return u.extra["film"]


def _whole_film_of(u) -> str:
    return whole_film(u.extra["film"])


def strat_film_order(utterances: list, film_of=_film_id) -> list:
    """The films in the order film_5fold_strat deals them (added 2026-11-13): first the films with a sad clip, then
    those with an angry clip (and none sad), then the rest; within each run, the films sorted by name. film_of (added
    2026-11-14): a clip's film, its film ID by default, its whole film for film_5fold_whole."""
    labels = {}
    for u in utterances:
        labels.setdefault(film_of(u), set()).add(u.label)
    order, dealt = [], set()
    for cls in STRAT_FIRST:
        run = sorted(f for f, have in labels.items() if cls in have and f not in dealt)
        order += run
        dealt |= set(run)
    return order + sorted(f for f in labels if f not in dealt)


def _strat_folds(utterances: list, film_of) -> tuple:
    from affect.data.splits import next_group_folds

    order = strat_film_order(utterances, film_of)
    if len(order) < len(FILM_GROUPS):
        raise ValueError(f"{len(order)} films cannot make {len(FILM_GROUPS)} film folds")
    fold_of = {film: FILM_GROUPS[k % len(FILM_GROUPS)] for k, film in enumerate(order)}
    members = {g: frozenset(u.utt_id for u in utterances if fold_of[film_of(u)] == g) for g in FILM_GROUPS}
    classes = sorted({u.label for u in utterances})
    missing = {g: sorted(set(classes) - {u.label for u in utterances if fold_of[film_of(u)] == g})
               for g in FILM_GROUPS}
    missing = {g: m for g, m in missing.items() if m}
    if missing:
        raise ValueError(f"film folds without a clip of every class: {missing}")
    return next_group_folds(FILM_GROUPS, members, lambda g: f"test_{g}", lambda g: f"films {g}")


def film_5fold_strat(utterances: list) -> tuple:
    """The five film folds of split film_5fold_strat (added 2026-11-13, RULE_week33_addendum5.txt): the k-th film of
    strat_film_order in film fold k mod 5 (named F(k + 1)), every clip of a film in its film's fold; fold test_Fk tests
    Fk, validates on the next and trains on the other three, as film_5fold. A fold without a clip of every class the
    clips hold is refused, so neither the validation side nor the test side of any fold lacks a class."""
    return _strat_folds(utterances, _film_id)


def film_5fold_whole(utterances: list) -> tuple:
    """The five film folds of split film_5fold_whole (added 2026-11-14, RULE_week33_addendum8.txt): film_5fold_strat's
    dealing over whole films, each film ID that spells a film another way counted as that film (SAME_FILM), so every
    clip of a film, whatever its ID's spelling, is in its film's fold. EmoFilm.folds checks FILMS.csv (check_whole_films)
    before it calls this."""
    return _strat_folds(utterances, _whole_film_of)
