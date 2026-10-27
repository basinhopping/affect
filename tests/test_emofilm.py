"""EmoFilm as a test corpus (affect.data.emofilm, added 2026-10-20), on the stand-in assembly of
tests/synthetic_emofilm.py: the label set's mapping onto IEMOCAP's three classes in IEMOCAP's order, the language
subsets, the fields each utterance carries, the refusals (a manifest that is not the one ASSEMBLY.json names, a clip
in another format, a fold scheme), and the extraction command's data settings for it. Then the neural models'
external test (train.external_test, added 2026-10-20): its setting and job checks (no torch), and, where torch is
installed (the node), external_files on the stand-in clips with a small CNN."""
import argparse
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from affect import catalog, paths
from affect.registry import DATASETS
from affect.settings import validate
from tests.support import REPO, needs_module
from tests.synthetic_emofilm import write_assembly, write_wav

catalog.load_all()


class TempEmoFilm:
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src" / "affect").mkdir(parents=True)
        (root / "src" / "affect" / "__init__.py").write_text("# stand-in package\n")
        self.assembled = root / "data" / "emofilm" / "assembled"
        self.rows = write_assembly(self.assembled)
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
        for field in ("external_test", "aux_va", "aux_va_weight", "aux_va_loss"):   # the form before 2026-10-20
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
