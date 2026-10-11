"""The LIF neuron's plain reference (affect.models.lif_reference) and the spiking models' settings.
Needs numpy only, so it runs everywhere; tests/test_lif_torch.py checks the torch layers against
this reference on the node. Added 2026-09-29."""
import copy
import json
import unittest

import numpy as np

from affect.models import lif_reference as ref

SMOOTH_SLOPE = 3.0


def smooth_fire(x):
    """A smooth stand-in for the spike, whose derivative is exactly the "sigmoid" surrogate."""
    return 1.0 / (1.0 + np.exp(-SMOOTH_SLOPE * x))


def forward_with_fixed_reset(current, beta, threshold, reset, fixed):
    """The forward pass with the reset term's spikes taken from `fixed` instead of the run: finite
    differences of this give the gradient with the reset detached."""
    u = np.zeros(current.shape[1:])
    out = np.zeros_like(current)
    for t in range(current.shape[0]):
        r = fixed[t - 1] if t > 0 else np.zeros(current.shape[1:])
        u = beta * u + current[t] - threshold * r if reset == "subtract" else beta * (1.0 - r) * u + current[t]
        out[t] = smooth_fire(u - threshold)
    return out


class TestReference(unittest.TestCase):
    def test_hand_worked_steps(self):
        current = np.full((5, 1), 0.6)
        s, u = ref.lif_forward(current, 0.5, 1.0, "subtract")
        self.assertTrue(np.allclose(u[:, 0], [0.6, 0.9, 1.05, 0.125, 0.6625]))
        self.assertEqual(s[:, 0].tolist(), [0.0, 0.0, 1.0, 0.0, 0.0])
        s, u = ref.lif_forward(current, 0.5, 1.0, "zero")
        self.assertTrue(np.allclose(u[:, 0], [0.6, 0.9, 1.05, 0.6, 0.9]))
        self.assertEqual(s[:, 0].tolist(), [0.0, 0.0, 1.0, 0.0, 0.0])
        s, _ = ref.lif_forward(np.full((3, 1), 1.0), 0.0, 1.0, "subtract")
        self.assertEqual(s[:, 0].tolist(), [1.0, 0.0, 1.0], "at the threshold a neuron fires (>=)")

    def test_recurrent_weight_adds_the_previous_spikes(self):
        current = np.array([[1.2, 0.0], [0.0, 0.0], [0.0, 0.0]])
        rec = np.array([[0.0, 0.0], [1.5, 0.0]])          # neuron 0's spike drives neuron 1 next step
        s, u = ref.lif_forward(current, 0.0, 1.0, "subtract", recurrent=rec)
        self.assertEqual(s.tolist(), [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def test_surrogates(self):
        x = np.array([-2.0, -0.3, 0.0, 0.3, 2.0])
        for kind, slope, at_zero in (("atan", 2.0, 1.0), ("fast_sigmoid", 25.0, 1.0), ("sigmoid", 25.0, 6.25),
                                     ("boxcar", 1.0, 0.5)):
            d = ref.surrogate_derivative(x, kind, slope)
            self.assertAlmostEqual(float(d[2]), at_zero, msg=kind)
            self.assertTrue(np.allclose(d, d[::-1]), f"{kind} is symmetric")
            self.assertTrue((d >= 0).all())
        self.assertEqual(ref.surrogate_derivative(np.array([0.49, 0.51]), "boxcar", 1.0).tolist(), [0.5, 0.0])
        grid, dx = np.linspace(-5, 5, 200001, retstep=True)
        self.assertAlmostEqual(float(ref.surrogate_derivative(grid, "boxcar", 2.0).sum() * dx), 0.5, places=3)
        self.assertAlmostEqual(float(ref.surrogate_derivative(grid, "sigmoid", 4.0).sum() * dx), 1.0, places=3)
        with self.assertRaises(ValueError):
            ref.surrogate_derivative(x, "tanh", 1.0)

    def test_backward_matches_finite_differences_with_a_smooth_spike(self):
        rng = np.random.default_rng(1)
        current = rng.normal(0.5, 0.7, size=(9, 4))
        g = rng.normal(size=current.shape)
        for reset in ("subtract", "zero"):
            for beta in (0.0, 0.7, 0.93):
                def loss(cur, b=beta):
                    return float(np.sum(g * ref.lif_forward(cur, b, 1.0, reset, fire=smooth_fire)[0]))

                gi, gb = ref.lif_backward(current, beta, 1.0, reset, "sigmoid", SMOOTH_SLOPE, False, g, fire=smooth_fire)
                eps = 1e-6
                num = np.zeros_like(current)
                for idx in np.ndindex(current.shape):
                    d = np.zeros_like(current)
                    d[idx] = eps
                    num[idx] = (loss(current + d) - loss(current - d)) / (2 * eps)
                self.assertLess(np.abs(gi - num).max(), 1e-7, (reset, beta))
                if beta > 0:
                    nb = (loss(current, beta + eps) - loss(current, beta - eps)) / (2 * eps)
                    self.assertLess(abs(gb - nb), 1e-7, (reset, beta))

    def test_detached_reset_matches_finite_differences_with_the_reset_frozen(self):
        rng = np.random.default_rng(2)
        current = rng.normal(0.6, 0.8, size=(8, 3))
        g = rng.normal(size=current.shape)
        for reset in ("subtract", "zero"):
            fixed = ref.lif_forward(current, 0.8, 1.0, reset, fire=smooth_fire)[0]
            gi, _ = ref.lif_backward(current, 0.8, 1.0, reset, "sigmoid", SMOOTH_SLOPE, True, g, fire=smooth_fire)
            eps = 1e-6
            num = np.zeros_like(current)
            for idx in np.ndindex(current.shape):
                d = np.zeros_like(current)
                d[idx] = eps
                up = np.sum(g * forward_with_fixed_reset(current + d, 0.8, 1.0, reset, fixed))
                down = np.sum(g * forward_with_fixed_reset(current - d, 0.8, 1.0, reset, fixed))
                num[idx] = (up - down) / (2 * eps)
            self.assertLess(np.abs(gi - num).max(), 1e-7, reset)
            full, _ = ref.lif_backward(current, 0.8, 1.0, reset, "sigmoid", SMOOTH_SLOPE, False, g, fire=smooth_fire)
            self.assertGreater(np.abs(full - gi).max(), 1e-3, "detaching the reset changes the gradient")

    def test_readout(self):
        spikes = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertTrue(np.allclose(ref.readout_embedding(spikes, 4, 0.0), [0.75, 0.75]), "spike count per step")
        self.assertTrue(np.allclose(ref.readout_embedding(spikes, 2, 0.0), [1.0, 0.5]), "only the real steps")
        # beta 0.5, neuron 0: Y = 1, 1.5, 0.75, 1.375 -> mean 1.15625, times 0.5
        self.assertAlmostEqual(float(ref.readout_embedding(spikes, 4, 0.5)[0]), 0.578125)
        steady = np.ones((400, 1))
        self.assertAlmostEqual(float(ref.readout_embedding(steady, 400, 0.9)[0]), 1.0, delta=0.03,
                               msg="a steady rate reads as about that rate whatever the leak")


class TestSpikingSettings(unittest.TestCase):
    def setUp(self):
        from affect import catalog
        from tests.support import REPO

        catalog.load_all()
        self.base = json.loads((REPO / "sweeps" / "cnn_transformer_screen_1.json").read_text())["base"]
        self.neuron = {"lif_dropout": 0.1, "beta": 0.9, "learn_beta": "layer", "threshold": 1.0, "reset": "subtract",
                       "detach_reset": True, "surrogate": "atan", "surrogate_slope": 2.0, "readout_beta": 0.8,
                       "rate_weight": 0.0, "rate_target": 0.1, "dropout": 0.2}

    def hybrid(self, **changes):
        model = {"name": "cnn_lif", "channels": [32, 64, 128], "convs_per_block": 1, "kernel_size": 3, "pool_freq": 2,
                 "pool_time": 2, "norm": "batch", "norm_groups": None, "norm_subbands": None, "activation": "relu",
                 "freq_reduce": "mean", "time_pool": 1, "lif_hidden": [128], "recurrent": False, "lif_norm": "none",
                 **self.neuron, **changes}
        s = copy.deepcopy(self.base)
        s["model"] = model
        return s

    def spiking(self, **changes):
        model = {"name": "snn", "encoding": "frames", "static_steps": None, "channels": [16, 32], "convs_per_block": 1,
                 "kernel_size": 3, "pool_freq": 2, "pool_time": 2, "conv_norm": "batch", "lif_hidden": [64],
                 "recurrent": True, "lif_norm": "batch", **self.neuron, **changes}
        s = copy.deepcopy(self.base)
        s["model"] = model
        return s

    def test_valid_settings_pass_the_job_checks(self):
        from affect.runs.spec import check_job, config_hash, validate_job

        hashes = set()
        for settings in (self.hybrid(), self.hybrid(freq_reduce="flatten", time_pool=4, lif_hidden=[256, 128, 64]),
                         self.spiking(), self.spiking(encoding="repeat", static_steps=8, lif_hidden=[])):
            job = validate_job(settings)
            check_job(job, 16000)
            hashes.add(config_hash(job))
        self.assertEqual(len(hashes), 4)

    def test_bad_settings_are_refused(self):
        from affect.runs.spec import check_job, validate_job
        from affect.settings import SettingsError

        bad = [self.hybrid(beta=1.0), self.hybrid(lif_hidden=[]), self.hybrid(lif_hidden=[8, 8, 8, 8]),
               self.hybrid(threshold=0.0), self.hybrid(time_pool=0), self.hybrid(readout_beta=1.0),
               self.hybrid(surrogate_slope=0.0), self.hybrid(rate_target=1.5), self.hybrid(lif_dropout=1.0),
               self.spiking(static_steps=4), self.spiking(encoding="repeat", static_steps=None),
               self.spiking(lif_hidden=[8, 8, 8]), self.spiking(channels=[8] * 7)]
        for settings in bad:
            with self.assertRaises(ValueError, msg=str(settings["model"])):
                check_job(validate_job(settings), 16000)
        for settings in (self.hybrid(surrogate="tanh"), self.hybrid(learn_beta="yes"), self.spiking(conv_norm="group")):
            with self.assertRaises(SettingsError):
                validate_job(settings)
        missing = self.hybrid()
        del missing["model"]["readout_beta"]
        with self.assertRaises(SettingsError):
            validate_job(missing)

    def test_a_sweep_over_both_models_expands(self):
        from affect.sweep.design import expand

        sweep = {"sweep_id": "lif_probe", "description": "", "folds": ["test_S1", "test_S2"],
                 "base": self.hybrid(),
                 "stages": [{"name": "base", "kind": "base", "priority": 1, "seeds": [1, 2]},
                            {"name": "levels", "kind": "one_factor_at_a_time", "priority": 1, "seeds": [1],
                             "factors": {"model.reset": ["zero"], "model.recurrent": [True],
                                         "model.surrogate": ["fast_sigmoid", "sigmoid", "boxcar"]}},
                            {"name": "space", "kind": "random", "priority": 1, "seeds": [1], "n_configs": 5,
                             "sampling_seed": 7, "space": {"model.beta": [0.8, 0.9, 0.95],
                                                           "model.lif_hidden": [[64], [128, 64]]}}]}
        jobs, summary = expand(sweep)
        self.assertEqual(summary["configs"], 1 + 5 + 5)
        self.assertEqual(len(jobs), 2 * 2 + 5 * 2 + 5 * 2)


if __name__ == "__main__":
    unittest.main()
