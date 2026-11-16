"""EmoFilm as a test corpus (affect.data.emofilm, added 2026-10-20), on the stand-in assembly of
tests/synthetic_emofilm.py: the label set's mapping onto IEMOCAP's three classes in IEMOCAP's order, the language
subsets, the fields each utterance carries, the refusals (a manifest that is not the one ASSEMBLY.json names, a clip
in another format, a fold scheme), and the extraction command's data settings for it. Then the neural models'
external test (train.external_test, added 2026-10-20): its setting and job checks (no torch), and, where torch is
installed (the node), external_files on the stand-in clips with a small CNN."""
import argparse
import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from affect import catalog, paths
from affect.registry import DATASETS
from affect.settings import validate
from tests.support import REPO, needs_module
from tests.synthetic_emofilm import write_assembly, write_wav

catalog.load_all()


class TempEmoFilm:
    def __init__(self, **assembly):
        self.assembly = assembly         # write_assembly's options (added 2026-11-10: per_cell, films)

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src" / "affect").mkdir(parents=True)
        (root / "src" / "affect" / "__init__.py").write_text("# stand-in package\n")
        self.assembled = root / "data" / "emofilm" / "assembled"
        self.rows = write_assembly(self.assembled, **self.assembly)
        self.saved = paths.repo_root
        paths.repo_root = lambda: root
        return self

    def __exit__(self, *exc):
        paths.repo_root = self.saved
        self.tmp.cleanup()


def utterances(label_set="three_no_neutral", subset="all"):
    entry = DATASETS.get("emofilm")
    settings = validate(entry.settings_class, {"label_set": label_set, "subset": subset}, "data")
    return entry.target(), settings, entry.target().utterances(settings)


class TestEmoFilm(unittest.TestCase):
    def test_classes_and_fields(self):
        from affect.data.iemocap import CLASS_ORDER

        with TempEmoFilm() as data:
            ds, settings, utts = utterances()
            self.assertEqual(ds.class_order(settings), CLASS_ORDER["three_no_neutral"])
            self.assertEqual(len(utts), len(data.rows))
            self.assertEqual([u.utt_id for u in utts], sorted(r["utt_id"] for r in data.rows))
            mapping = {"gio": "hap+exc", "rab": "ang", "tri": "sad"}
            for u in utts:
                self.assertEqual(u.label, mapping[u.raw_label])
                self.assertEqual((u.kind, u.fold_group, u.dialog), (u.extra["language"], u.extra["film"],
                                                                     u.extra["film"]))
                self.assertEqual(u.extra["gender"], u.utt_id[0])
                self.assertEqual((u.sample_rate, u.dataset, u.transcript), (16000, "emofilm", None))
            _, _, english = utterances(subset="en")
            self.assertEqual({u.extra["language"] for u in english}, {"en"})
            self.assertEqual(len(english), len(data.rows) // 3)
            with self.assertRaises(ValueError):
                ds.folds(settings, utts)
            with self.assertRaises(ValueError):
                ds.fold_names(settings)
            with self.assertRaises(ValueError):
                validate(DATASETS.get("emofilm").settings_class, {"label_set": "four_merged", "subset": "all"},
                         "data")

    def test_refusals(self):
        with TempEmoFilm() as data:
            record = json.loads((data.assembled / "ASSEMBLY.json").read_text())
            (data.assembled / "ASSEMBLY.json").write_text(json.dumps({**record, "manifest_sha256": "0" * 64}))
            with self.assertRaises(ValueError):
                utterances()
        with TempEmoFilm() as data:
            first = data.rows[0]
            write_wav(data.assembled / first["audio_path"], np.zeros(first["n_frames"]), rate=44100)
            with self.assertRaises(ValueError):
                utterances()

    def test_film_folds(self):
        # added 2026-11-10: split film_5fold, EmoFilm as a training corpus
        from affect.classical.finetune import film_folds
        from affect.data.splits import assign, folds_hash

        with TempEmoFilm(per_cell=3, films=12) as data:
            entry = DATASETS.get("emofilm")
            settings = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all",
                                                       "split": "film_5fold"}, "data")
            ds = entry.target()
            utts = ds.utterances(settings)
            self.assertEqual(len(utts), len(data.rows))
            self.assertEqual(ds.fold_names(settings), ("test_F1", "test_F2", "test_F3", "test_F4", "test_F5"))
            folds = ds.folds(settings, utts)
            self.assertEqual(tuple(f.name for f in folds), ds.fold_names(settings))
            fold_of = film_folds([u.extra["film"] for u in utts], 5)
            film = {u.utt_id: u.extra["film"] for u in utts}
            tested = []
            for j, f in enumerate(folds):
                roles = assign(utts, f)
                films = {r: {film[i] for i in roles[r]} for r in ("train", "val", "test")}
                self.assertEqual({fold_of[x] for x in films["test"]}, {j})
                self.assertEqual({fold_of[x] for x in films["val"]}, {(j + 1) % 5})
                self.assertEqual({fold_of[x] for x in films["train"]}, set(range(5)) - {j, (j + 1) % 5})
                self.assertFalse(films["train"] & films["test"] or films["val"] & films["test"]
                                 or films["train"] & films["val"])
                tested += roles["test"]
            self.assertEqual(sorted(tested), sorted(u.utt_id for u in utts), "every clip is tested once")
            self.assertEqual(folds_hash(utts, folds), folds_hash(list(reversed(utts)), ds.folds(settings,
                                                                                               list(reversed(utts)))))
            soft = ds.soft_targets(settings, utts)
            order = ds.class_order(settings)
            for u, row in zip(utts, soft):
                self.assertEqual(row, [1.0 if c == u.label else 0.0 for c in order])
            # the form before 2026-11-10 is split none: no folds, as before
            old = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all"}, "data")
            self.assertEqual(old.split, "none")
            with self.assertRaisesRegex(ValueError, "no fold scheme"):
                ds.folds(old, utts)
        with TempEmoFilm() as data:      # four films cannot make five film folds
            _, _, utts = utterances()
            entry = DATASETS.get("emofilm")
            settings = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all",
                                                       "split": "film_5fold"}, "data")
            with self.assertRaises(ValueError):
                entry.target().folds(settings, utts)

    def test_film_folds_strat(self):
        # added 2026-11-13 (RULE_week33_addendum5.txt): split film_5fold_strat deals the films with a sad clip first,
        # then those with an angry clip, then the rest, so that every fold holds every class; film_5fold's dealing
        # left a fold without sadness on the real corpus
        from affect.data.emofilm import film_5fold, film_5fold_strat, strat_film_order
        from affect.data.splits import assign, folds_hash

        def clips(films: dict) -> list:
            return [types.SimpleNamespace(utt_id=f"{film}_{i}", label=label, extra={"film": film})
                    for film, labels in films.items() for i, label in enumerate(labels)]

        # 15 films; sad clips only in the films film_5fold puts in folds F1 to F4 (k mod 5 != 4)
        films = {f"film_{k + 1:02d}": ["hap+exc", "ang"] + (["sad"] if k % 5 != 4 else []) + ["hap+exc"] * (k % 3)
                 for k in range(15)}
        utts = clips(films)
        old = film_5fold(utts)
        without_sad = [f.name for f in old if not any(u.label == "sad" for u in utts if u.utt_id in f.test)]
        self.assertEqual(without_sad, ["test_F5"], "film_5fold leaves a fold without sadness here")
        order = strat_film_order(utts)
        sad_films = sorted(f for f, labels in films.items() if "sad" in labels)
        ang_films = sorted(f for f, labels in films.items() if "sad" not in labels and "ang" in labels)
        self.assertEqual(order[:len(sad_films)], sad_films)
        self.assertEqual(order[len(sad_films):len(sad_films) + len(ang_films)], ang_films)
        self.assertEqual(sorted(order), sorted(films))
        folds = film_5fold_strat(utts)
        self.assertEqual(tuple(f.name for f in folds), ("test_F1", "test_F2", "test_F3", "test_F4", "test_F5"))
        group = {film: k % 5 for k, film in enumerate(order)}
        tested = []
        for j, f in enumerate(folds):
            roles = assign(utts, f)
            by_role = {r: [u for u in utts if u.utt_id in set(roles[r])] for r in ("train", "val", "test")}
            for r in ("val", "test"):
                self.assertEqual({u.label for u in by_role[r]}, {"hap+exc", "ang", "sad"}, f"{f.name} {r}")
            self.assertEqual({group[u.extra["film"]] for u in by_role["test"]}, {j})
            self.assertEqual({group[u.extra["film"]] for u in by_role["val"]}, {(j + 1) % 5})
            films_in = {r: {u.extra["film"] for u in by_role[r]} for r in by_role}
            self.assertFalse(films_in["train"] & films_in["test"] or films_in["val"] & films_in["test"]
                             or films_in["train"] & films_in["val"])
            tested += roles["test"]
        self.assertEqual(sorted(tested), sorted(u.utt_id for u in utts), "every clip is tested once")
        self.assertEqual(folds_hash(utts, folds), folds_hash(list(reversed(utts)), film_5fold_strat(utts[::-1])))
        # a class in fewer films than folds cannot reach every fold: refused
        few = clips({f"film_{k:02d}": ["hap+exc", "ang"] + (["sad"] if k < 3 else []) for k in range(10)})
        with self.assertRaisesRegex(ValueError, "without a clip of every class"):
            film_5fold_strat(few)
        with self.assertRaisesRegex(ValueError, "cannot make 5 film folds"):
            film_5fold_strat(clips({f"film_{k}": ["hap+exc", "ang", "sad"] for k in range(4)}))
        # through the dataset: the same five fold names, the folds of film_5fold_strat
        with TempEmoFilm(per_cell=3, films=12):
            entry = DATASETS.get("emofilm")
            settings = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all",
                                                       "split": "film_5fold_strat"}, "data")
            ds = entry.target()
            utts = ds.utterances(settings)
            self.assertEqual(ds.fold_names(settings), ("test_F1", "test_F2", "test_F3", "test_F4", "test_F5"))
            self.assertEqual(folds_hash(utts, ds.folds(settings, utts)), folds_hash(utts, film_5fold_strat(utts)))

    def test_film_folds_whole(self):
        # added 2026-11-14 (RULE_week33_addendum8.txt): split film_5fold_whole deals whole films. The IDs that spell a
        # film another way (SAME_FILM) go with that film, so no film is on two sides of a fold; FILMS.csv is checked
        # against the reviewed titles first
        from affect.data import emofilm as E
        from affect.data.splits import assign, folds_hash

        def clips(films: dict) -> list:
            return [types.SimpleNamespace(utt_id=f"{film}_{i}", label=label, extra={"film": film})
                    for film, labels in films.items() for i, label in enumerate(labels)]

        def films_split(utts, folds) -> int:     # whole films on two sides of a fold, summed over the folds
            n = 0
            for f in folds:
                roles = assign(utts, f)
                side = {r: {E.whole_film(u.extra["film"]) for u in utts if u.utt_id in set(roles[r])}
                        for r in ("train", "val", "test")}
                n += len(side["test"] & (side["train"] | side["val"])) + len(side["val"] & side["train"])
            return n

        # the table: 18 IDs onto 13 films, each film's own ID mapped to nothing, every ID of the 13 with its title's
        # SHA-256 (the titles themselves stay in FILMS.csv, never in git)
        self.assertEqual((len(E.SAME_FILM), len(set(E.SAME_FILM.values()))), (18, 13))
        self.assertTrue(all(E.whole_film(v) == v for v in E.SAME_FILM.values()))
        self.assertEqual(set(E.TITLE_SHA256), set(E.SAME_FILM) | set(E.SAME_FILM.values()))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in E.TITLE_SHA256.values()))
        self.assertEqual(E.title_sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        # stand-in clips: every ID of the table, and 12 films of their own
        ids = sorted(E.TITLE_SHA256) + [f"film_{k}" for k in range(60, 72)]
        films = {f: ["hap+exc", "ang", "sad"] + ["hap+exc"] * (i % 3) for i, f in enumerate(ids)}
        utts = clips(films)
        whole, strat = E.film_5fold_whole(utts), E.film_5fold_strat(utts)
        self.assertEqual(films_split(utts, whole), 0)
        self.assertGreater(films_split(utts, strat), 0, "dealing the IDs splits films here")
        self.assertEqual(E.strat_film_order(utts, E._whole_film_of), sorted({E.whole_film(f) for f in ids}),
                         "every film has a sad clip: one run, by name")
        tested = []
        for f in whole:
            roles = assign(utts, f)
            for r in ("val", "test"):
                self.assertEqual({u.label for u in utts if u.utt_id in set(roles[r])}, {"hap+exc", "ang", "sad"})
            tested += roles["test"]
        self.assertEqual(sorted(tested), sorted(u.utt_id for u in utts), "every clip is tested once")
        self.assertEqual(folds_hash(utts, whole), folds_hash(utts[::-1], E.film_5fold_whole(utts[::-1])))
        # check_whole_films, with stand-in titles and their SHA-256 in place of the reviewed ones (test_films33 checks
        # the real FILMS.csv): the reviewed titles pass; a changed or missing title, or two titles that differ only in
        # case and spaces left apart, are refused
        titles = {f: f"Title {E.whole_film(f)} as {f}" for f in ids}
        reviewed = {f: E.title_sha256(titles[f]) for f in E.TITLE_SHA256}
        with mock.patch.object(E, "TITLE_SHA256", reviewed):
            E.check_whole_films(titles)
            with self.assertRaisesRegex(ValueError, "not the film list SAME_FILM was reviewed against"):
                E.check_whole_films(dict(titles, film_47="Title film_16"))
            with self.assertRaisesRegex(ValueError, "not the film list SAME_FILM was reviewed against"):
                E.check_whole_films({f: t for f, t in titles.items() if f != "film_02"})
            with self.assertRaisesRegex(ValueError, "spell the same title"):
                E.check_whole_films(dict(titles, film_70=" title  FILM_60 as FILM_60 "))
        with self.assertRaisesRegex(ValueError, "not the film list SAME_FILM was reviewed against"):
            E.check_whole_films(titles)              # the stand-in titles are not the reviewed ones
        # through the dataset: FILMS.csv checked, then film_5fold_whole's folds; the stand-in's films film_01 to
        # film_12 hold film_02 and film_07 (with film_01), film_04 (with film_03) and film_09 (with film_08)
        with TempEmoFilm(per_cell=3, films=12) as data, mock.patch.object(E, "TITLE_SHA256", reviewed):
            entry = DATASETS.get("emofilm")
            settings = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all",
                                                       "split": "film_5fold_whole"}, "data")
            ds = entry.target()
            found = ds.utterances(settings)
            with self.assertRaises(FileNotFoundError):
                ds.folds(settings, found)                 # no FILMS.csv, no folds
            rows = dict(titles, **{f"film_{k:02d}": f"Film {k}" for k in range(1, 13)
                                   if f"film_{k:02d}" not in E.TITLE_SHA256})
            (data.assembled / "FILMS.csv").write_text("film,title\n" + "".join(f"{f},{t}\n" for f, t in sorted(
                rows.items())), encoding="utf-8")
            self.assertEqual(ds.fold_names(settings), ("test_F1", "test_F2", "test_F3", "test_F4", "test_F5"))
            got = ds.folds(settings, found)
            self.assertEqual(folds_hash(found, got), folds_hash(found, E.film_5fold_whole(found)))
            self.assertEqual(films_split(found, got), 0)
            self.assertEqual({E.whole_film(u.extra["film"]) for u in found} & {"film_02", "film_07", "film_04",
                                                                                "film_09"}, set())
            (data.assembled / "FILMS.csv").write_text("film,title\n" + "".join(f"{f},{t}\n" for f, t in sorted(
                dict(rows, film_02="Another title").items())), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the film list SAME_FILM was reviewed against"):
                ds.folds(settings, found)

    def test_extraction_settings(self):
        from affect.cli import _all_utterances

        args = argparse.Namespace(dataset="emofilm", label_set="three_no_neutral", subset="all")
        self.assertEqual(_all_utterances(args), {"label_set": "three_no_neutral", "subset": "all"})
        args = argparse.Namespace(dataset="iemocap", label_set="seven", subset="all")
        self.assertEqual(_all_utterances(args), {"label_set": "seven", "subset": "all",
                                                 "split": "leave_one_session_out"})
        with self.assertRaises(SystemExit):
            _all_utterances(argparse.Namespace(dataset="msp", label_set="x", subset="all"))


def three_class_settings(**train) -> dict:
    """The CNN + Transformer finalist's three-class settings in the current form, with these train settings."""
    from affect.runs.spec import plain, validate_job

    sweep = json.loads((REPO / "sweeps" / "cnn_transformer_final_1_classes_no_neu3.json").read_text())
    settings = plain(validate_job(sweep["base"]))
    settings["train"].update(train)
    return settings


class TestExternalSettings(unittest.TestCase):
    def test_setting_and_job_checks(self):
        from affect.runs.spec import check_job, config_hash, validate_job

        on = three_class_settings(external_test="emofilm")
        check_job(validate_job(on), 16000)
        off = three_class_settings()
        self.assertEqual(off["train"]["external_test"], "none")
        self.assertNotEqual(config_hash(validate_job(on)), config_hash(validate_job(off)))
        late = json.loads(json.dumps(off))
        for field in ("external_test", "aux_va", "aux_va_weight", "aux_va_loss", "refit", "refit_epochs", "warm_start",
                      "warm_start_folds", "warm_start_epochs", "warm_start_lr"):   # the form before 2026-10-20
            del late["train"][field]
        self.assertEqual(config_hash(validate_job(late)), config_hash(validate_job(off)))
        four = json.loads(json.dumps(on))
        four["data"]["label_set"] = "four_merged"          # allowed since 2026-10-24: scored on the three classes
        check_job(validate_job(four), 16000)
        self.assertNotEqual(config_hash(validate_job(four)), config_hash(validate_job(on)))
        for other in ("five", "four_without_excited", "seven"):    # hap and exc apart: no hap+exc class
            wrong = json.loads(json.dumps(on))
            wrong["data"]["label_set"] = other
            with self.assertRaisesRegex(ValueError, "holds its classes"):
                check_job(validate_job(wrong), 16000)
        by_speaker = json.loads(json.dumps(on))
        by_speaker["features"]["normalization"] = "speaker"
        with self.assertRaisesRegex(ValueError, "other than speaker"):
            check_job(validate_job(by_speaker), 16000)
        with self.assertRaises(ValueError):
            validate_job(three_class_settings(external_test="msp_podcast"))


@needs_module("torch")
class TestExternalTorch(unittest.TestCase):
    def test_external_files(self):
        import csv
        import io

        import torch

        from affect.features.logmel import LogMel, LogMelSettings
        from affect.models.cnn import CNNSettings, build_cnn
        from affect.train import data as train_data
        from affect.train.loop import Normalizer, external_files

        with TempEmoFilm() as stand_in:
            train_data._EXTERNAL.clear()
            ext = train_data.prepare_external("emofilm", "three_no_neutral")
            self.assertIs(train_data.prepare_external("emofilm", "three_no_neutral"), ext)
            self.assertEqual((ext.class_names, ext.sample_rate, len(ext.utterances)),
                             (("hap+exc", "ang", "sad"), 16000, len(stand_in.rows)))
            torch.manual_seed(0)
            model = build_cnn(CNNSettings(channels=(8, 16), convs_per_block=1, kernel_size=3, pool_freq=2,
                                          pool_time=2, pooling="mean", classifier_hidden=None, dropout=0.1,
                                          norm="batch", norm_groups=None, norm_subbands=None, activation="relu"),
                              40, 3).eval()
            feats = LogMel(LogMelSettings(win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                                          log_offset=1e-6, normalization="utterance"), 16000, torch.device("cpu"))
            ts = types.SimpleNamespace(eval_batch_size=4, precision="fp32")
            metrics, text, seconds = external_files(ext, model, feats, Normalizer("utterance"), ts, None,
                                                    torch.device("cpu"))
            rows = list(csv.DictReader(io.StringIO(text)))
            self.assertEqual([r["utt_id"] for r in rows], [u.utt_id for u in ext.utterances])
            self.assertEqual(metrics["n"], len(rows))
            pred = np.array([ext.class_names.index(r["pred"]) for r in rows])
            recalls = [np.mean(pred[ext.labels == c] == c) for c in range(3)]
            self.assertAlmostEqual(metrics["ua"], float(np.mean(recalls)), places=12)
            self.assertEqual(sorted(metrics["by_language"]), ["en", "es", "it"])
            self.assertEqual(sorted(metrics["by_gender"]), ["f", "m"])
            self.assertEqual(metrics["dataset"], "emofilm")
            for r in rows:
                self.assertAlmostEqual(sum(float(r[f"prob_{c}"]) for c in ext.class_names), 1.0, places=5)
            same, same_text, _ = external_files(ext, model, feats, Normalizer("utterance"), ts, None,
                                                torch.device("cpu"), ext.class_names)
            self.assertEqual(same_text, text, "the corpus's own classes: the files of 2026-10-20")
            timeless = [json.dumps({k: v for k, v in m.items() if k != "seconds"}, sort_keys=True) for m in (same, metrics)]
            self.assertEqual(timeless[0], timeless[1])
            train_data._EXTERNAL.clear()

    def test_external_files_with_more_classes(self):
        """A four-class model (four_merged, added 2026-10-24) on EmoFilm's three classes."""
        import csv
        import io

        import torch

        from affect.features.logmel import LogMel, LogMelSettings
        from affect.models.cnn import CNNSettings, build_cnn
        from affect.train import data as train_data
        from affect.train.loop import Normalizer, external_files

        model_classes = ("neu", "hap+exc", "ang", "sad")
        with TempEmoFilm():
            train_data._EXTERNAL.clear()
            ext = train_data.prepare_external("emofilm", "three_no_neutral")
            torch.manual_seed(1)
            model = build_cnn(CNNSettings(channels=(8, 16), convs_per_block=1, kernel_size=3, pool_freq=2,
                                          pool_time=2, pooling="mean", classifier_hidden=None, dropout=0.1,
                                          norm="batch", norm_groups=None, norm_subbands=None, activation="relu"),
                              40, 4).eval()
            feats = LogMel(LogMelSettings(win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                                          log_offset=1e-6, normalization="utterance"), 16000, torch.device("cpu"))
            ts = types.SimpleNamespace(eval_batch_size=4, precision="fp32")
            metrics, text, _ = external_files(ext, model, feats, Normalizer("utterance"), ts, None,
                                              torch.device("cpu"), model_classes)
            rows = list(csv.DictReader(io.StringIO(text)))
            self.assertEqual(list(rows[0]), ["utt_id", "speaker", "film", "language", "gender", "label", "pred",
                                             "pred_all", "prob_neu", "prob_hap+exc", "prob_ang", "prob_sad"])
            self.assertEqual([r["utt_id"] for r in rows], [u.utt_id for u in ext.utterances])
            probs = np.array([[float(r[f"prob_{c}"]) for c in model_classes] for r in rows])
            self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))
            pred = probs[:, 1:].argmax(axis=1)          # the most probable of the corpus's three classes
            self.assertEqual([ext.class_names[k] for k in pred], [r["pred"] for r in rows])
            self.assertEqual([model_classes[k] for k in probs.argmax(axis=1)], [r["pred_all"] for r in rows])
            recalls = [np.mean(pred[ext.labels == c] == c) for c in range(3)]
            self.assertAlmostEqual(metrics["ua"], float(np.mean(recalls)), places=12)
            self.assertEqual(metrics["class_names"], list(ext.class_names))
            strict = probs.argmax(axis=1)
            strict_recalls = [np.mean(strict[ext.labels == c] == c + 1) for c in range(3)]
            self.assertAlmostEqual(metrics["strict"]["ua"], float(np.mean(strict_recalls)), places=12)
            self.assertAlmostEqual(metrics["strict"]["share_given_all"]["neu"], float(np.mean(strict == 0)), places=12)
            self.assertEqual(sorted(metrics["strict"]["by_language"]), ["en", "es", "it"])
            self.assertEqual(sorted(metrics["by_gender"]), ["f", "m"])
            self.assertIn("neu", metrics["scoring"])
            with self.assertRaises(ValueError):          # a model without the corpus's hap+exc
                external_files(ext, model, feats, Normalizer("utterance"), ts, None, torch.device("cpu"),
                               ("neu", "hap", "ang", "sad"))
            train_data._EXTERNAL.clear()


if __name__ == "__main__":
    unittest.main()
