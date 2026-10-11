"""The spiking layers and both CNN + LIF models (affect.models.lif, cnn_lif, snn): the neuron against
the numpy reference (affect.models.lif_reference) forward and backward, padding, the readout, the
activity penalty, the operation counts, and both models end to end. Needs torch, so it runs on the
node, where scripts/check.sh requires it. Added 2026-09-29."""
import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np

from tests.support import needs_module


def neuron_settings(**changes):
    """The fields the spiking layers read, without the rest of a model's settings."""
    base = dict(beta=0.875, learn_beta="none", threshold=1.0, reset="subtract", detach_reset=False,
                surrogate="atan", surrogate_slope=2.0, rate_weight=0.0, rate_target=0.1, lif_dropout=0.0,
                lif_norm="none", recurrent=False)
    base.update(changes)
    return SimpleNamespace(**base)


@needs_module("torch")
class TestNeuron(unittest.TestCase):
    def test_forward_matches_the_reference(self):
        import torch

        from affect.models import lif
        from affect.models import lif_reference as ref

        Neuron = lif.classes()[0]
        rng = np.random.default_rng(0)
        current = rng.normal(0.5, 0.8, size=(3, 20, 5))
        for reset in ("subtract", "zero"):
            for beta in (0.0, 0.5, 0.875):
                n = Neuron((5,), neuron_settings(beta=beta, reset=reset), "n", lif.Recorder())
                got = n(torch.tensor(current), torch.ones(3, 20, 1)).numpy()
                for b in range(3):
                    want, _ = ref.lif_forward(current[b], beta, 1.0, reset)
                    self.assertTrue(np.array_equal(got[b], want), (reset, beta, b))
                got32 = n(torch.tensor(current, dtype=torch.float32), torch.ones(3, 20, 1))
                self.assertEqual(got32.dtype, torch.float32)
                self.assertTrue(np.array_equal(got32.numpy(), got), "float32 gives the same spikes here")

    def test_recurrent_forward_matches_the_reference(self):
        import torch

        from affect.models import lif
        from affect.models import lif_reference as ref

        Neuron = lif.classes()[0]
        rng = np.random.default_rng(1)
        current = rng.normal(0.3, 0.6, size=(2, 15, 4))
        R = rng.normal(0.0, 0.8, size=(4, 4))
        rec = torch.nn.Linear(4, 4, bias=False).double()
        with torch.no_grad():
            rec.weight.copy_(torch.tensor(R))
        for reset in ("subtract", "zero"):
            n = Neuron((4,), neuron_settings(beta=0.5, reset=reset), "n", lif.Recorder())
            got = n(torch.tensor(current), torch.ones(2, 15, 1), recurrent=rec).detach().numpy()
            for b in range(2):
                want, _ = ref.lif_forward(current[b], 0.5, 1.0, reset, recurrent=R)
                self.assertTrue(np.array_equal(got[b], want), (reset, b))

    def test_gradients_match_the_reference(self):
        import torch

        from affect.models import lif
        from affect.models import lif_reference as ref

        Neuron = lif.classes()[0]
        rng = np.random.default_rng(2)
        current = rng.normal(0.6, 0.9, size=(12, 6))
        g = rng.normal(size=current.shape)
        for surrogate, slope in (("atan", 2.0), ("fast_sigmoid", 25.0), ("sigmoid", 5.0), ("boxcar", 1.0)):
            for reset in ("subtract", "zero"):
                for detach in (False, True):
                    s = neuron_settings(learn_beta="layer", reset=reset, detach_reset=detach, surrogate=surrogate,
                                        surrogate_slope=slope)
                    n = Neuron((6,), s, "n", lif.Recorder())
                    x = torch.tensor(current[None], requires_grad=True)
                    spikes = n(x, torch.ones(1, 12, 1))
                    (spikes[0] * torch.tensor(g)).sum().backward()
                    want_i, want_b = ref.lif_backward(current, 0.875, 1.0, reset, surrogate, slope, detach, g)
                    key = (surrogate, reset, detach)
                    self.assertLess(np.abs(x.grad[0].numpy() - want_i).max(), 1e-9, key)
                    self.assertLess(abs(n.beta_value.grad.item() - want_b), 1e-5 * max(1.0, abs(want_b)), key)

    def test_learned_leak_per_neuron_and_its_clamp(self):
        import torch

        from affect.models import lif

        Neuron = lif.classes()[0]
        n = Neuron((3, 2), neuron_settings(learn_beta="neuron", beta=0.5), "n", lif.Recorder())
        self.assertEqual(tuple(n.beta_value.shape), (3, 2))
        with torch.no_grad():
            n.beta_value[0, 0] = 1.7
            n.beta_value[1, 1] = -0.2
        self.assertEqual(n.beta()[0, 0].item(), 1.0)
        self.assertEqual(n.beta()[1, 1].item(), 0.0)
        out = n(torch.randn(2, 7, 3, 2), torch.ones(2, 7, 1, 1))
        self.assertEqual(tuple(out.shape), (2, 7, 3, 2))
        fixed = Neuron((4,), neuron_settings(learn_beta="none"), "n", lif.Recorder())
        self.assertEqual(list(fixed.parameters()), [], "a fixed leak is a buffer, not a parameter")

    def test_activity_penalty(self):
        import torch

        from affect.models import lif

        Neuron = lif.classes()[0]
        torch.manual_seed(0)
        n = Neuron((5,), neuron_settings(rate_weight=0.5, rate_target=0.2), "n", lif.Recorder())
        x = torch.rand(3, 10, 5) * 1.5
        real = torch.ones(3, 10, 1)
        real[1, 6:] = 0.0
        spikes = n(x, real)
        rates = (spikes * real).sum(dim=(0, 1)) / real.expand_as(spikes).sum(dim=(0, 1))
        self.assertAlmostEqual(n.penalty.item(), ((rates - 0.2) ** 2).mean().item(), places=6)
        n.eval()
        n(x, real)
        self.assertIsNone(n.penalty, "no penalty outside training")


@needs_module("torch")
class TestSpikingParts(unittest.TestCase):
    def test_pool_time(self):
        import torch

        from affect.models import lif

        h = torch.arange(1.0, 6.0).reshape(1, 5, 1)
        pooled, mask = lif.pool_time(h, torch.ones(1, 5, dtype=torch.bool), 2)
        self.assertEqual(pooled[0, :, 0].tolist(), [1.5, 3.5, 5.0])
        self.assertEqual(mask[0].tolist(), [True, True, True])
        pooled, mask = lif.pool_time(h, torch.tensor([[True, True, True, False, False]]), 2)
        self.assertEqual(pooled[0, :, 0].tolist(), [1.5, 3.0, 0.0])
        self.assertEqual(mask[0].tolist(), [True, True, False])
        same, m = lif.pool_time(h, torch.ones(1, 5, dtype=torch.bool), 1)
        self.assertIs(same, h)

    def test_readout_matches_the_reference(self):
        import torch

        from affect.models import lif
        from affect.models import lif_reference as ref

        rng = np.random.default_rng(3)
        spikes = (rng.random((2, 9, 4)) < 0.3).astype(np.float64)
        valid = [9, 5]
        mask = torch.arange(9)[None, :] < torch.tensor(valid)[:, None]
        for beta in (0.0, 0.5, 0.9):
            got = lif.readout(torch.tensor(spikes), mask, beta).numpy()
            for b in range(2):
                self.assertTrue(np.allclose(got[b], ref.readout_embedding(spikes[b], valid[b], beta), atol=1e-6))

    def test_step_batch_norm_uses_real_steps_only(self):
        import torch

        from affect.models import lif

        _, StepBatchNorm, _ = lif.classes()
        torch.manual_seed(0)
        x = torch.randn(2, 6, 3)
        mask = torch.arange(6)[None, :] < torch.tensor([6, 4])[:, None]
        a, b = StepBatchNorm(3), StepBatchNorm(3)
        out_a = a(x, mask)
        changed = x.clone()
        changed[1, 4:] = 50.0
        out_b = b(changed, mask)
        self.assertTrue(torch.allclose(out_a, out_b))
        self.assertTrue(torch.allclose(a.norm.running_mean, b.norm.running_mean))
        self.assertTrue((out_a[1, 4:] == 0).all())
        real = x[mask]
        self.assertTrue(torch.allclose(out_a[mask], (real - real.mean(0)) / torch.sqrt(real.var(0, unbiased=False) + 1e-5),
                                       atol=1e-5))

    def test_spiking_linear_ignores_padded_steps(self):
        import torch

        from affect.models import lif

        _, _, SpikingLinear = lif.classes()
        mask = torch.arange(7)[None, :] < torch.tensor([7, 4])[:, None]
        for norm in ("none", "batch", "layer"):
            for recurrent in (False, True):
                torch.manual_seed(1)
                layer = SpikingLinear(4, 6, neuron_settings(lif_norm=norm, recurrent=recurrent), "l", lif.Recorder(),
                                      real_input=True)
                x = torch.randn(2, 7, 4) * 2
                a = layer(x, mask)
                changed = x.clone()
                changed[1, 4:] = -30.0
                b = layer(changed, mask)
                self.assertTrue(torch.equal(a[:, :4], b[:, :4]), (norm, recurrent))
                self.assertTrue(torch.equal(a[0], b[0]), (norm, recurrent))
                self.assertTrue(torch.equal(lif.readout(a, mask, 0.8), lif.readout(b, mask, 0.8)))

    def test_time_dropout_keeps_one_mask_per_sequence(self):
        import torch

        from affect.models import lif

        torch.manual_seed(0)
        x = torch.ones(4, 9, 16)
        out = lif.time_dropout(x, 0.5, True)
        self.assertTrue(torch.equal(out, out[:, :1].expand_as(out)), "the same units dropped at every step")
        self.assertTrue(set(out.unique().tolist()) <= {0.0, 2.0})
        self.assertIs(lif.time_dropout(x, 0.5, False), x)

    def test_counts_and_report(self):
        import torch

        from affect.models import lif

        _, _, SpikingLinear = lif.classes()
        rec = lif.Recorder()
        rec.on = True
        first = SpikingLinear(3, 5, neuron_settings(), "lif1", rec, real_input=True)
        second = SpikingLinear(5, 2, neuron_settings(recurrent=True), "lif2", rec, real_input=False)
        mask = torch.arange(6)[None, :] < torch.tensor([6, 3])[:, None]
        rec.utterances += 2
        s1 = first(torch.randn(2, 6, 3) * 3, mask)
        s2 = second(s1, mask)
        m = mask[..., None].float()
        self.assertEqual(rec.sums["lif1.mac"], 9 * 3 * 5, "real steps times fan-in times fan-out")
        self.assertEqual(rec.sums["lif2.ac"], (s1 * m).sum().item() * 2 + (s2 * m).sum().item() * 2)
        self.assertEqual(rec.sums["lif1.neuron_steps"], 9 * 5)
        report = lif.report(rec)
        self.assertEqual(report["sequences"], 2)
        self.assertAlmostEqual(report["firing_rate"]["lif1"], (s1 * m).sum().item() / 45)
        self.assertAlmostEqual(report["mac_per_sequence"], 9 * 3 * 5 / 2)
        self.assertAlmostEqual(report["energy_uj_per_sequence_estimate"],
                               (report["mac_per_sequence"] * 4.6 + report["ac_per_sequence"] * 0.9) / 1e6)


@needs_module("torch")
class TestSpikingModels(unittest.TestCase):
    def neuron_fields(self, **changes):
        return dict(lif_dropout=0.1, beta=0.9, learn_beta="none", threshold=1.0, reset="subtract", detach_reset=True,
                    surrogate="atan", surrogate_slope=2.0, readout_beta=0.8, rate_weight=0.0, rate_target=0.1,
                    dropout=0.1, **changes)

    def hybrid(self, **changes):
        from affect.models.cnn_lif import CNNLIFSettings

        base = dict(channels=(8, 16), convs_per_block=1, kernel_size=3, pool_freq=2, pool_time=2, norm="batch",
                    norm_groups=None, norm_subbands=None, activation="relu", freq_reduce="mean", time_pool=1,
                    lif_hidden=(32,), recurrent=False, lif_norm="none")
        base.update(self.neuron_fields())
        base.update(changes)
        return CNNLIFSettings(**base)

    def spiking(self, **changes):
        from affect.models.snn import SNNSettings

        base = dict(encoding="frames", static_steps=None, channels=(4, 8), convs_per_block=1, kernel_size=3,
                    pool_freq=2, pool_time=2, conv_norm="batch", lif_hidden=(16,), recurrent=False, lif_norm="none")
        base.update(self.neuron_fields())
        base.update(changes)
        return SNNSettings(**base)

    def run_model(self, model, x, valid):
        import torch

        out = model(x, valid)
        self.assertEqual(tuple(out.shape), (x.shape[0], 4))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        return out

    def test_hybrid(self):
        import torch

        from affect.models.cnn_lif import build_cnn_lif

        x = torch.randn(3, 40, 80) * 2
        valid = torch.tensor([80, 41, 12])
        for changes in ({}, {"freq_reduce": "flatten"}, {"time_pool": 3}, {"lif_hidden": (32, 16, 8)},
                        {"recurrent": True}, {"lif_norm": "batch"}, {"lif_norm": "layer"}, {"learn_beta": "layer"},
                        {"learn_beta": "neuron", "lif_hidden": (32, 16)}, {"reset": "zero", "detach_reset": False},
                        {"surrogate": "boxcar", "surrogate_slope": 1.0}, {"readout_beta": 0.0},
                        {"rate_weight": 0.5}, {"norm": "group", "norm_groups": 4, "activation": "gelu"}):
            torch.manual_seed(0)
            model = build_cnn_lif(self.hybrid(**changes), 40, 4).train()
            self.run_model(model, x, valid)
            if changes.get("rate_weight"):
                self.assertIsNotNone(model.activity_penalty())
            else:
                self.assertIsNone(model.activity_penalty())
        model = build_cnn_lif(self.hybrid(), 40, 4)
        self.assertEqual(model.frames_after_blocks(valid).tolist(), [20, 10, 3])
        self.assertEqual(model.embedding_dim, 32)

    def test_hybrid_readout_ignores_padding_past_the_front_end(self):
        import torch

        from affect.models.cnn_lif import build_cnn_lif

        torch.manual_seed(0)
        model = build_cnn_lif(self.hybrid(recurrent=True, lif_norm="batch", time_pool=2), 40, 4).eval()
        x = torch.randn(2, 40, 64)
        valid = torch.tensor([64, 24])
        with torch.no_grad():
            a = model(x, valid)
            changed = x.clone()
            changed[1, :, 40:] = 25.0          # frames the front end's receptive field cannot reach
            b = model(changed, valid)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_fully_spiking_both_encodings(self):
        import torch

        from affect.models.snn import build_snn

        x = torch.randn(3, 16, 24) * 2
        valid = torch.tensor([24, 13, 5])
        for changes in ({}, {"lif_hidden": ()}, {"lif_hidden": (16, 8), "recurrent": True, "lif_norm": "batch"},
                        {"conv_norm": "none"}, {"learn_beta": "neuron"}, {"convs_per_block": 2},
                        {"encoding": "repeat", "static_steps": 3}, {"encoding": "repeat", "static_steps": 2,
                                                                     "lif_hidden": ()},
                        {"encoding": "repeat", "static_steps": 2, "learn_beta": "neuron", "rate_weight": 0.3}):
            torch.manual_seed(0)
            model = build_snn(self.spiking(**changes), 16, 4).train()
            self.run_model(model, x, valid)

    def test_recording_through_a_model(self):
        import torch

        from affect.models.cnn_lif import build_cnn_lif
        from affect.models.snn import build_snn

        x = torch.randn(2, 16, 24) * 2
        valid = torch.tensor([24, 10])
        for model in (build_cnn_lif(self.hybrid(lif_hidden=(8, 8)), 16, 4), build_snn(self.spiking(), 16, 4),
                      build_snn(self.spiking(encoding="repeat", static_steps=3), 16, 4)):
            model.eval()
            model.record(True)
            with torch.no_grad():
                model(x, valid)
                model(x, valid)
            model.record(False)
            report = model.spike_report()
            self.assertEqual(report["sequences"], 4)
            self.assertTrue(report["firing_rate"])
            self.assertTrue(all(0.0 <= r <= 1.0 for r in report["firing_rate"].values()))
            self.assertGreater(report["mac_per_sequence"], 0)
            with torch.no_grad():
                model(x, valid)
            self.assertEqual(model.spike_report()["sequences"], 4, "nothing is added while recording is off")

    def test_step_adds_the_activity_penalty(self):
        import torch

        from affect import catalog
        from affect.features.logmel import LogMel
        from affect.runs.spec import plain, validate_job
        from affect.train.loop import Normalizer, Step
        from tests.support import REPO

        catalog.load_all()
        settings = plain(validate_job(json.loads((REPO / "sweeps" / "cnn_transformer_screen_1.json").read_text())["base"]))
        settings["model"] = {"name": "cnn_lif", **{k: (list(v) if isinstance(v, tuple) else v)
                                                   for k, v in vars(self.hybrid(rate_weight=0.3)).items()}}
        settings["train"]["precision"] = "fp32"
        job = validate_job(settings)
        device = torch.device("cpu")
        feats = LogMel(job.features, 16000, device)
        from affect.registry import MODELS

        torch.manual_seed(0)
        model = MODELS.get("cnn_lif").target(job.model, job.features.n_mels, 4).train()
        waves = (np.random.default_rng(1).standard_normal((4, 16000)) * 0.1).astype(np.float32)
        loss, emotion, gender = Step(job, model, None, feats, Normalizer("utterance"), None, device).loss(
            waves, np.array([16000, 9000, 16000, 12000]), np.arange(4), np.eye(4, dtype=np.float32),
            None, torch.Generator().manual_seed(0), np.random.default_rng(2))
        self.assertIsNone(gender)
        self.assertAlmostEqual(loss.item(), emotion.item() + model.activity_penalty().item(), places=5)
        loss.backward()

    def test_a_tiny_hybrid_learns(self):
        import torch

        from affect.models.cnn_lif import build_cnn_lif
        from affect.train.loop import soft_cross_entropy

        torch.manual_seed(0)
        model = build_cnn_lif(self.hybrid(lif_dropout=0.0, dropout=0.0, learn_beta="layer"), 40, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        x = torch.randn(16, 40, 64)
        y = torch.arange(16) % 4
        x[y == 1, :10] += 3.0
        x[y == 2, 20:] -= 3.0
        x[y == 3, :, :32] += 2.0
        valid = torch.full((16,), 64)
        target = torch.nn.functional.one_hot(y, 4).float()
        first = None
        for _ in range(150):
            loss = soft_cross_entropy(model(x, valid), target, None)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.6 * first)


if __name__ == "__main__":
    unittest.main()
