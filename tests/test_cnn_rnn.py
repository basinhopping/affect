"""The CNN + recurrent model (model cnn_rnn, added 2026-10-24): its checks and a job that uses it (no torch); where
torch is installed (the node), that packing keeps padding out of the recurrence in both directions (each row of a
padded batch gives what it gives alone, whatever the padding holds) for a GRU and an LSTM, one direction and two,
every pooling and the last state; the summary's width; the forward pass; and one training step through Step."""
import json
import unittest

import numpy as np

from affect import catalog
from tests.support import REPO, needs_module

catalog.load_all()
BASE = {"name": "cnn_rnn", "channels": [16, 32], "convs_per_block": 1, "kernel_size": 3, "pool_freq": 2,
        "pool_time": 2, "norm": "batch", "norm_groups": None, "norm_subbands": None, "activation": "relu",
        "freq_reduce": "mean", "rnn": "gru", "rnn_hidden": 24, "rnn_layers": 2, "bidirectional": True,
        "rnn_dropout": 0.1, "pooling": "attention", "classifier_hidden": None, "dropout": 0.2}


def settings(**model):
    from affect.registry import MODELS

    _, validated = MODELS.validate_section({**BASE, **model}, "model")
    return validated


def job_settings(**model) -> dict:
    from affect.runs.spec import plain, validate_job

    out = plain(validate_job(json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"]))
    out["model"] = {**BASE, **model}
    return out


class TestChecks(unittest.TestCase):
    def test_checks(self):
        from affect.models.cnn_rnn import check, rnn_width

        check(settings(), 40)
        check(settings(rnn_layers=1, rnn_dropout=0.0), 40)
        for bad in ({"rnn_hidden": 0}, {"rnn_layers": 0}, {"rnn_layers": 5}, {"rnn_dropout": 1.0},
                    {"rnn_layers": 1, "rnn_dropout": 0.1}, {"dropout": 1.0}, {"kernel_size": 4},
                    {"channels": [16, 32, 64, 128, 256, 512]}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check(settings(**bad), 40)
        self.assertEqual((rnn_width(settings()), rnn_width(settings(bidirectional=False))), (48, 24))

    def test_a_job_with_the_model(self):
        from affect.runs.spec import check_job, validate_job

        check_job(validate_job(job_settings()), 16000)
        with self.assertRaises(ValueError):
            check_job(validate_job(job_settings(rnn_layers=1)), 16000)     # dropout 0.1 between layers it lacks
        with self.assertRaises(ValueError):
            validate_job(job_settings(rnn="transformer"))


@needs_module("torch")
class TestTorch(unittest.TestCase):
    def build(self, n_mels=40, **model):
        import torch

        from affect.models.cnn_rnn import build_cnn_rnn

        torch.manual_seed(0)
        return build_cnn_rnn(settings(**model), n_mels, 4).eval()

    def test_padding_stays_out_of_the_recurrence(self):
        import torch

        cases = ({}, {"rnn": "lstm"}, {"bidirectional": False}, {"pooling": "last"},
                 {"rnn": "lstm", "pooling": "last"}, {"pooling": "last", "bidirectional": False},
                 {"pooling": "mean", "rnn_layers": 1, "rnn_dropout": 0.0}, {"pooling": "max"},
                 {"pooling": "meanmax"}, {"pooling": "stats"}, {"rnn": "lstm", "rnn_layers": 3})
        for case in cases:
            model = self.build(**case)
            rng = np.random.default_rng(1)
            h = torch.from_numpy(rng.standard_normal((3, 9, model.rnn.input_size)).astype(np.float32))
            keep = torch.tensor([9, 4, 1])
            with torch.no_grad():
                together = model.sequence(h, keep)
                self.assertEqual(tuple(together.shape), (3, model.embedding_dim), case)
                for k in range(3):
                    alone = model.sequence(h[k:k + 1, :int(keep[k])], keep[k:k + 1])
                    self.assertTrue(torch.allclose(together[k], alone[0], atol=1e-5), (case, k))
                noisy = h.clone()
                noisy[1, 4:] = 100.0
                noisy[2, 1:] = -50.0
                self.assertTrue(torch.allclose(model.sequence(noisy, keep), together, atol=1e-5), case)

    def test_widths_and_forward(self):
        import torch

        for case, width in (({}, 48), ({"bidirectional": False}, 24), ({"pooling": "stats"}, 96),
                            ({"pooling": "last"}, 48), ({"pooling": "last", "bidirectional": False, "rnn": "lstm"}, 24),
                            ({"freq_reduce": "flatten"}, 48), ({"classifier_hidden": 16}, 48)):
            model = self.build(**case)
            self.assertEqual(model.embedding_dim, width, case)
            torch.manual_seed(1)
            x = torch.randn(2, 40, 120)
            with torch.no_grad():
                logits = model(x, torch.tensor([120, 37]))
            self.assertEqual(tuple(logits.shape), (2, 4), case)
            self.assertTrue(bool(torch.isfinite(logits).all()), case)
        self.assertEqual(self.build(freq_reduce="flatten").rnn.input_size, 32 * 10)   # 40 bands halved twice
        self.assertEqual(self.build().rnn.input_size, 32)

    def test_a_training_step(self):
        import torch

        from affect.features.logmel import LogMel
        from affect.registry import MODELS
        from affect.runs.spec import validate_job
        from affect.train.loop import Normalizer, Step, trainable_parameters

        for case in ({}, {"rnn": "lstm", "pooling": "last"}):
            raw = job_settings(**case)
            raw["train"]["precision"] = "fp32"
            job = validate_job(raw)
            device = torch.device("cpu")
            feats = LogMel(job.features, 16000, device)
            torch.manual_seed(0)
            model = MODELS.get(job.model_name).target(job.model, job.features.n_mels, 4).train()
            optimizer = torch.optim.AdamW(trainable_parameters(model, None), lr=1e-3)
            rng = np.random.default_rng(2)
            waves = (rng.standard_normal((4, 16000)) * 0.1).astype(np.float32)
            loss, emotion, _, _ = Step(job, model, None, feats, Normalizer("utterance"), None, device).loss(
                waves, np.array([16000, 12000, 16000, 3000]), np.arange(4), np.eye(4, dtype=np.float32), None,
                torch.Generator().manual_seed(1), np.random.default_rng(3))
            self.assertIs(loss, emotion)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.assertTrue(bool(torch.isfinite(loss)), case)
            self.assertTrue(all(p.grad is not None for p in model.rnn.parameters()), case)


if __name__ == "__main__":
    unittest.main()
