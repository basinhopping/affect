"""The refit on training plus validation (train.refit, E1) and the warm start on the films (train.warm_start, F2), both
added 2026-10-29.

Without torch: the settings and their refusals; the configurations without the new fields keep their hashes and the
new settings change them; the status alert for a validation UA above 0.80 leaves refit runs out (they trained on
their validation session).

With torch (on the node): warm_start_files on the stand-in EmoFilm (tests/synthetic_emofilm.py, 36 clips of 4 films)
with a small four-class CNN: every clip is scored once, in the film fold its film belongs to (whole films per fold, the
rule of affect.classical.finetune.film_folds); the reported UA, per-fold and per-language numbers equal those computed
from the CSV; each clip's choice is the most probable of the three shared classes; the model's weights are exactly
those it came with; a second call gives the same file byte for byte; adapted scores differ from the cold model's."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from affect import catalog
from tests.support import REPO, needs_module
from tests.test_emofilm import TempEmoFilm, three_class_settings

catalog.load_all()
WARM = {"warm_start": "emofilm", "warm_start_folds": 5, "warm_start_epochs": 10, "warm_start_lr": 1e-4}


def four_class_settings(**train) -> dict:
    """The tuned CNN's four-class EmoFilm settings (night 25's emofilm25_cnn_so4) in the current form."""
    from affect.runs.spec import plain, validate_job

    settings = plain(validate_job(json.loads((REPO / "sweeps" / "emofilm25_cnn_so4.json").read_text())["base"]))
    settings["train"].update(train)
    return settings


class TestSettings(unittest.TestCase):
    def test_refit(self):
        from affect.runs.spec import check_job, config_hash, validate_job

        base = four_class_settings(external_test="none")
        self.assertEqual((base["train"]["refit"], base["train"]["refit_epochs"]), ("none", None))
        on = four_class_settings(external_test="none", refit="train_and_val", refit_epochs=30)
        check_job(validate_job(on), 16000)
        self.assertNotEqual(config_hash(validate_job(on)), config_hash(validate_job(base)))
        other = four_class_settings(external_test="none", refit="train_and_val", refit_epochs=31)
        self.assertNotEqual(config_hash(validate_job(on)), config_hash(validate_job(other)))
        epochs = base["train"]["epochs"]
        for bad in ({"refit": "train_and_val"}, {"refit_epochs": 3}, {"refit": "train_and_val", "refit_epochs": 0},
                    {"refit": "train_and_val", "refit_epochs": epochs + 1},
                    {"refit": "train_and_val", "refit_epochs": 3, "eval_train": "every_epoch"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check_job(validate_job(four_class_settings(external_test="none", **bad)), 16000)
        with self.assertRaises(ValueError):
            validate_job(four_class_settings(refit="train_only"))

    def test_warm_start(self):
        from affect.runs.spec import check_job, config_hash, validate_job

        cold = four_class_settings()
        self.assertEqual(cold["train"]["external_test"], "emofilm")
        self.assertEqual(cold["train"]["warm_start"], "none")
        warm = four_class_settings(**WARM)
        check_job(validate_job(warm), 16000)
        self.assertNotEqual(config_hash(validate_job(warm)), config_hash(validate_job(cold)))
        for bad in ({"warm_start": "emofilm"}, {"warm_start_folds": 5}, {**WARM, "warm_start_folds": 2},
                    {**WARM, "warm_start_epochs": 0}, {**WARM, "warm_start_lr": 0.0},
                    {**WARM, "external_test": "none"},
                    {**WARM, "aux_va": "multitask", "aux_va_weight": 0.3, "aux_va_loss": "ccc"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check_job(validate_job(four_class_settings(**bad)), 16000)
        # since 2026-11-05 a refit may have the films too (its last weights scored cold, then adapted)
        refit_warm = four_class_settings(**WARM, refit="train_and_val", refit_epochs=5)
        check_job(validate_job(refit_warm), 16000)
        self.assertNotEqual(config_hash(validate_job(refit_warm)), config_hash(validate_job(warm)))

    def test_forms_without_the_fields_hash_as_before(self):
        from affect.runs.spec import config_hash, plain, validate_job

        new = four_class_settings()
        before = json.loads(json.dumps(new))
        for field in ("refit", "refit_epochs", "warm_start", "warm_start_folds", "warm_start_epochs", "warm_start_lr"):
            del before["train"][field]
        self.assertEqual(config_hash(validate_job(before)), config_hash(validate_job(new)))
        self.assertEqual(plain(validate_job(before)), new)
        sweep = json.loads((REPO / "sweeps" / "emofilm25_cnn_so4.json").read_text())
        from affect.sweep.design import expand
        jobs, _ = expand(sweep)
        self.assertEqual(config_hash(validate_job(new)), jobs[0]["config_hash"])

    def test_status_leaves_refit_runs_out_of_the_alert(self):
        from affect import paths
        from affect.sweep import status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = paths.repo_root
            paths.repo_root = lambda: root
            try:
                sweep = {"sweep_id": "s", "folds": ["test_S1"], "stages": [{"name": "base", "kind": "base"}]}
                (root / "artifacts" / "sweeps" / "s").mkdir(parents=True)
                (root / "artifacts" / "sweeps" / "s" / "sweep_file.json").write_text(json.dumps(sweep))
                for run, refit in (("a_test_S1_seed1", True), ("b_test_S1_seed1", False)):
                    folder = root / "artifacts" / "runs" / "s" / run
                    folder.mkdir(parents=True)
                    spec = {"run_id": run, "stage": "base", "label": "base", "config_hash": run[0] * 64,
                            "fold": "test_S1", "seed": 1, "settings": {"train": {"control": "none"}}}
                    metrics = {"best_val_ua": 0.95, "best_epoch": 3, "epochs_run": 3, "final": {"ua": 0.95}}
                    if refit:
                        metrics["refit"] = {"rows": "train_and_val", "epochs": 3}
                    (folder / "spec.json").write_text(json.dumps(spec))
                    (folder / "metrics_val.json").write_text(json.dumps(metrics))
                    (folder / "efficiency.json").write_text(json.dumps({"train_seconds": 1.0}))
                (root / "queue").mkdir()
                alerts = status.summarise("s")["alerts"]
            finally:
                paths.repo_root = saved
        self.assertEqual(len(alerts), 1, alerts)
        self.assertIn("b_test_S1_seed1", alerts[0])


@needs_module("torch")
class TestWarmStartTorch(unittest.TestCase):
    def _parts(self):
        import torch

        from affect.features.logmel import LogMel, LogMelSettings
        from affect.models.cnn import CNNSettings, build_cnn
        from affect.runs.spec import validate_job

        settings = three_class_settings(external_test="emofilm", warm_start="emofilm", warm_start_folds=3,
                                        warm_start_epochs=2, warm_start_lr=1e-3, batch_size=4, crop_s=0.1,
                                        eval_batch_size=4, precision="fp32", max_batches_per_epoch=None)
        settings["augment"].update({"freq_masks": 0, "freq_mask_width": 0, "time_masks": 0, "time_mask_width": 0,
                                    "mixup_alpha": 0.0})
        job = validate_job(settings)
        torch.manual_seed(0)
        model = build_cnn(CNNSettings(channels=(8, 16), convs_per_block=1, kernel_size=3, pool_freq=2, pool_time=2,
                                      pooling="mean", classifier_hidden=None, dropout=0.1, norm="batch",
                                      norm_groups=None, norm_subbands=None, activation="relu"), 40, 4).eval()
        feats = LogMel(LogMelSettings(win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                                      log_offset=1e-6, normalization="utterance"), 16000, torch.device("cpu"))
        return job, model, feats

    def test_warm_start_files(self):
        import torch

        from affect.classical.finetune import film_folds
        from affect.train import data as train_data
        from affect.train.loop import Normalizer, external_files, warm_start_files

        model_classes = ("neu", "hap+exc", "ang", "sad")
        with TempEmoFilm():
            train_data._EXTERNAL.clear()
            ext = train_data.prepare_external("emofilm", "three_no_neutral")
            job, model, feats = self._parts()
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            cold, cold_text, _ = external_files(ext, model, feats, Normalizer("utterance"), job.train, None,
                                                torch.device("cpu"), model_classes)
            metrics, text, seconds = warm_start_files(ext, model, job, feats, Normalizer("utterance"), None,
                                                      torch.device("cpu"), 7, model_classes, lambda m: None)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]), k)
            again, again_text, _ = warm_start_files(ext, model, job, feats, Normalizer("utterance"), None,
                                                    torch.device("cpu"), 7, model_classes, lambda m: None)
            self.assertEqual(again_text, text)
            train_data._EXTERNAL.clear()
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([r["utt_id"] for r in rows], [u.utt_id for u in ext.utterances])
        expected = film_folds([u.extra["film"] for u in ext.utterances], 3)
        for r in rows:
            self.assertEqual(int(r["film_fold"]), expected[r["film"]])
        films_in = {}
        for r in rows:
            films_in.setdefault(r["film"], set()).add(r["film_fold"])
        self.assertTrue(all(len(v) == 1 for v in films_in.values()))
        probs = np.array([[float(r[f"prob_{c}"]) for c in model_classes] for r in rows])
        shared = probs[:, 1:] / probs[:, 1:].sum(axis=1, keepdims=True)
        pred = shared.argmax(axis=1)
        self.assertEqual([r["pred"] for r in rows], [ext.class_names[k] for k in pred])
        self.assertEqual([r["pred_all"] for r in rows], [model_classes[k] for k in probs.argmax(axis=1)])
        recalls = [np.mean(pred[ext.labels == c] == c) for c in range(3)]
        self.assertAlmostEqual(metrics["ua"], float(np.mean(recalls)), places=12)
        self.assertEqual(metrics["n"], 36)
        self.assertEqual(sorted(metrics["by_film_fold"]), ["0", "1", "2"])
        self.assertEqual(sorted(metrics["by_language"]), ["en", "es", "it"])
        self.assertEqual(sum(metrics["warm_start"]["clips_per_fold"]), 36)
        self.assertEqual(len(metrics["warm_start"]["curves"]), 3 * 2)
        self.assertIn("strict", metrics)
        cold_rows = list(csv.DictReader(io.StringIO(cold_text)))
        cold_probs = np.array([[float(r[f"prob_{c}"]) for c in model_classes] for r in cold_rows])
        self.assertGreater(np.abs(cold_probs - probs).max(), 1e-4, "the adapted models score like the cold one")


if __name__ == "__main__":
    unittest.main()
