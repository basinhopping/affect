"""The valence and activation head (train.aux_va, added 2026-10-24): its settings and job checks, the ratings it
learns and the reports it writes (no torch); where torch is installed (the node), the concordance correlation and the
two losses against numpy references, the head inside Step (the loss is the emotion loss plus the weighted head loss,
the same dropout draws as a hand-built reference; with mixup and a gender head too; the refusals), predict with the
head (the class probabilities unchanged, each utterance's outputs those of the head on it alone, segments averaged),
and that the head is built after the model, which keeps the model's initial weights."""
import json
import types
import unittest

import numpy as np

from affect import catalog
from tests.support import REPO, needs_module

catalog.load_all()
ON = {"aux_va": "multitask", "aux_va_weight": 0.3, "aux_va_loss": "ccc"}


def screen_settings(**train) -> dict:
    """The CNN screen's base settings in the current form, with these train settings."""
    from affect.runs.spec import plain, validate_job

    settings = plain(validate_job(json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"]))
    settings["train"].update(train)
    return settings


def ccc(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


class TestSettings(unittest.TestCase):
    def test_settings_and_job_checks(self):
        from affect.runs.spec import check_job, config_hash, validate_job

        off = screen_settings()
        self.assertEqual((off["train"]["aux_va"], off["train"]["aux_va_weight"], off["train"]["aux_va_loss"]),
                         ("none", 0.0, "none"))
        on = screen_settings(**ON)
        check_job(validate_job(on), 16000)
        self.assertNotEqual(config_hash(validate_job(on)), config_hash(validate_job(off)))
        mse = screen_settings(**{**ON, "aux_va_loss": "mse"})
        check_job(validate_job(mse), 16000)
        self.assertNotEqual(config_hash(validate_job(on)), config_hash(validate_job(mse)))
        for bad in ({"aux_va_weight": 0.3}, {"aux_va_loss": "ccc"}, {"aux_va": "multitask"},
                    {"aux_va": "multitask", "aux_va_weight": 0.3}, {"aux_va": "multitask", "aux_va_loss": "mse"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check_job(validate_job(screen_settings(**bad)), 16000)
        with self.assertRaises(ValueError):
            validate_job(screen_settings(aux_va="regression"))

    def test_ratings(self):
        from affect.train.data import VA_SCALES, va_ratings

        self.assertEqual(VA_SCALES, ("v", "a"))
        utts = [types.SimpleNamespace(utt_id=f"u{k}", extra={"v": 1.0 + k, "a": 2.5, "d": 3.0}) for k in range(3)]
        self.assertTrue(np.array_equal(va_ratings(utts), np.array([[1.0, 2.5], [2.0, 2.5], [3.0, 2.5]])))
        for bad in ({"a": 2.0}, {"v": "3", "a": 2.0}, {"v": float("nan"), "a": 2.0}, {"v": 7.0, "a": 2.0},
                    {"v": True, "a": 2.0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                va_ratings([types.SimpleNamespace(utt_id="x", extra=bad)])

    def test_reports(self):
        from affect.train.loop import concordance_numpy, predictions_va_csv, va_report

        rng = np.random.default_rng(0)
        t = rng.standard_normal((40, 2))
        p = 0.6 * t + 0.5 * rng.standard_normal((40, 2))
        mean, sd = np.array([3.0, 2.8]), np.array([0.9, 0.7])
        self.assertTrue(np.allclose(concordance_numpy(p, t), [ccc(p[:, j], t[:, j]) for j in range(2)], atol=1e-12))
        self.assertTrue(np.allclose(concordance_numpy(p * sd + mean, t * sd + mean), concordance_numpy(p, t)))
        self.assertTrue(np.allclose(concordance_numpy(t, t), 1.0))
        report = va_report(t, p, (mean, sd))
        self.assertAlmostEqual(report["ccc_v"], ccc(p[:, 0], t[:, 0]), places=12)
        self.assertAlmostEqual(report["mean_ccc"], np.mean([ccc(p[:, j], t[:, j]) for j in range(2)]), places=12)
        self.assertAlmostEqual(report["rmse_a"], float(np.sqrt(np.mean(((p[:, 1] - t[:, 1]) * 0.7) ** 2))), places=12)
        self.assertAlmostEqual(report["pearson_v"], float(np.corrcoef(t[:, 0], p[:, 0])[0, 1]), places=12)
        self.assertEqual((report["n"], report["scales"], report["train_sd"]), (40, ["v", "a"], [0.9, 0.7]))
        prepared = types.SimpleNamespace(utterances=[types.SimpleNamespace(utt_id=f"u{k}", speaker="Ses01F",
                                                                           label="ang") for k in range(3)])
        lines = predictions_va_csv(prepared, np.arange(3), t[:3], p[:3], (mean, sd)).splitlines()
        self.assertEqual(lines[0], "utt_id,speaker,label,true_v,true_a,pred_v,pred_a")
        first = lines[1].split(",")
        self.assertAlmostEqual(float(first[3]), t[0, 0] * 0.9 + 3.0, places=4)
        self.assertAlmostEqual(float(first[6]), p[0, 1] * 0.7 + 2.8, places=6)


@needs_module("torch")
class TestTorch(unittest.TestCase):
    def test_concordance_and_losses(self):
        import torch

        from affect.train.loop import concordance, concordance_numpy, va_loss

        rng = np.random.default_rng(0)
        p = rng.standard_normal((50, 2))
        t = 0.5 * p + rng.standard_normal((50, 2)) + 0.3
        got = concordance(torch.from_numpy(p), torch.from_numpy(t)).numpy()
        self.assertTrue(np.allclose(got, concordance_numpy(p, t), atol=1e-12))
        self.assertTrue(np.allclose(got, [ccc(p[:, j], t[:, j]) for j in range(2)], atol=1e-12))
        pf, tf = torch.from_numpy(p).float(), torch.from_numpy(t).float()
        self.assertAlmostEqual(va_loss(pf, tf, "ccc").item(), 1 - float(np.mean(got)), places=5)
        self.assertAlmostEqual(va_loss(pf, tf, "mse").item(), float(np.mean((p - t) ** 2)), places=4)
        self.assertAlmostEqual(va_loss(tf, tf, "ccc").item(), 0.0, places=6)
        with self.assertRaises(ValueError):
            va_loss(pf, tf, "mae")

    def parts(self, **changes):
        import torch

        from affect.features.logmel import LogMel
        from affect.models.cnn import build_cnn
        from affect.runs.spec import validate_job
        from affect.train.loop import make_aux_head, make_va_head

        settings = screen_settings(precision="fp32", **ON)
        for key, value in changes.items():
            section, field = key.split(".")
            settings[section][field] = value
        job = validate_job(settings)
        device = torch.device("cpu")
        torch.manual_seed(3)
        model = build_cnn(job.model, job.features.n_mels, 4).train()
        aux = make_aux_head(job.train, model, device)
        head = make_va_head(job.train, model, device).train()
        return job, LogMel(job.features, 16000, device), model, aux, head, device

    def test_step_with_the_head(self):
        import torch

        from affect.augment.specaugment import spec_augment
        from affect.train.loop import Normalizer, Step, soft_cross_entropy, trainable_parameters, va_loss

        job, feats, model, _, head, device = self.parts()
        self.assertEqual((head.linear.in_features, head.linear.out_features), (model.embedding_dim, 2))
        rng = np.random.default_rng(1)
        waves = (rng.standard_normal((4, 16000)) * 0.1).astype(np.float32)
        lengths = np.array([16000, 12000, 16000, 8000])
        rows = np.arange(4)
        targets = np.eye(4, dtype=np.float32)
        va = rng.standard_normal((4, 2)).astype(np.float32)
        norm = Normalizer("utterance")
        with self.assertRaises(ValueError):          # the settings ask for the head
            Step(job, model, None, feats, norm, None, device)
        step = Step(job, model, None, feats, norm, None, device, None, head)
        with self.assertRaises(ValueError):          # the head needs the ratings
            step.loss(waves, lengths, rows, targets, None, torch.Generator().manual_seed(5), np.random.default_rng(9))
        torch.manual_seed(11)
        loss, emotion, gender, contrastive = step.loss(waves, lengths, rows, targets, None,
                                                       torch.Generator().manual_seed(5), np.random.default_rng(9),
                                                       None, va)
        self.assertIsNone(gender)
        self.assertIsNone(contrastive)
        self.assertAlmostEqual(loss.item(), emotion.item() + 0.3 * step.last_va_loss.item(), places=5)
        torch.manual_seed(11)                        # the same dropout draws, in the same order
        x = feats(torch.from_numpy(waves))
        frames = feats.frames(torch.from_numpy(lengths)).clamp_max(x.shape[-1])
        x = spec_augment(norm(x, frames, rows), frames, job.augment, torch.Generator().manual_seed(5))
        embedding = model.embed(x, frames)
        reference = (soft_cross_entropy(model.classify(embedding), torch.from_numpy(targets), None)
                     + 0.3 * va_loss(head(embedding), torch.from_numpy(va), "ccc"))
        self.assertAlmostEqual(loss.item(), reference.item(), places=5)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))
        self.assertEqual(len(trainable_parameters(model, None, None, head)), len(list(model.parameters())) + 2)

    def test_step_with_mixup_and_a_gender_head(self):
        import torch

        from affect.train.loop import Normalizer, Step, trainable_parameters

        job, feats, model, aux, head, device = self.parts(**{"augment.mixup_alpha": 0.4, "train.aux_gender": "multitask",
                                                             "train.aux_gender_weight": 0.5, "train.aux_va_loss": "mse"})
        rng = np.random.default_rng(2)
        waves = (rng.standard_normal((4, 16000)) * 0.1).astype(np.float32)
        step = Step(job, model, aux, feats, Normalizer("utterance"), None, device, None, head)
        loss, emotion, gender, _ = step.loss(
            waves, np.array([16000, 9000, 16000, 12000]), np.arange(4), np.eye(4, dtype=np.float32),
            np.eye(2, dtype=np.float32)[[0, 1, 0, 1]], torch.Generator().manual_seed(0), np.random.default_rng(2),
            None, rng.standard_normal((4, 2)).astype(np.float32))
        self.assertAlmostEqual(loss.item(), emotion.item() + 0.5 * gender.item() + 0.3 * step.last_va_loss.item(),
                               places=5)
        self.assertTrue(torch.isfinite(step.last_va_loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in trainable_parameters(model, aux, None, head) if p.requires_grad))

    def test_predict_with_the_head(self):
        import torch

        from affect.train.loop import Normalizer, predict

        _, feats, model, _, head, device = self.parts()
        model.eval()
        rng = np.random.default_rng(2)
        audio = [(rng.standard_normal(n) * 3000).astype(np.int16) for n in (16000, 9000, 30000)]
        rows = np.arange(3)
        norm = Normalizer("utterance")
        without = predict(model, feats, norm, audio, rows, 1, device, "fp32")
        with_head = predict(model, feats, norm, audio, rows, 1, device, "fp32", None, None, head)
        self.assertIsNone(without.va)
        self.assertTrue(np.allclose(without.probs, with_head.probs, atol=1e-6))
        self.assertEqual(with_head.va.shape, (3, 2))
        head.eval()
        for k in range(3):
            x = feats(torch.from_numpy(audio[k].astype(np.float32) / 32768.0)[None, :])
            frames = feats.frames(torch.tensor([audio[k].shape[0]])).clamp_max(x.shape[-1])
            with torch.no_grad():
                alone = head(model.embed(norm(x, frames), frames))[0].numpy()
            self.assertTrue(np.allclose(with_head.va[k], alone, atol=1e-5), k)
        segments = predict(model, feats, norm, audio, rows, 2, device, "fp32", (16000, 8000), None, head)
        self.assertEqual(segments.va.shape, (3, 2))
        self.assertTrue(np.isfinite(segments.va).all())

    def test_the_head_is_built_after_the_model(self):
        import torch

        from affect.models.cnn import build_cnn
        from affect.runs.spec import validate_job
        from affect.train.loop import make_aux_head, make_contrastive_head, make_va_head

        device = torch.device("cpu")
        weights = []
        for settings in (screen_settings(), screen_settings(**ON)):
            job = validate_job(settings)
            torch.manual_seed(7)
            model = build_cnn(job.model, job.features.n_mels, 4)
            self.assertIsNone(make_aux_head(job.train, model, device))
            self.assertIsNone(make_contrastive_head(job.train, model, device))
            head = make_va_head(job.train, model, device)
            self.assertEqual(head is None, settings["train"]["aux_va"] == "none")
            weights.append(model.state_dict())
        for name, value in weights[0].items():
            self.assertTrue(torch.equal(value, weights[1][name]), name)


if __name__ == "__main__":
    unittest.main()
