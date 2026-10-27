"""The supervised contrastive loss (train.contrastive, added 2026-10-15): its settings and the job check
that refuses it with mixup (no torch), a reference written as a loop from the paper's equation (Khosla
et al. 2020, equation 2), and, where torch is installed (the node; scripts/check.sh requires it), the
loss, its gradient, the projection head and the training step against that reference."""
import dataclasses
import json
import math
import unittest

import numpy as np

from affect import catalog
from tests.support import REPO, needs_module

catalog.load_all()
SUPCON = {"contrastive": "supcon", "contrastive_weight": 0.5, "contrastive_temperature": 0.1, "contrastive_dim": 16}


def reference_loss(z, labels, temperature):
    """L_out of Khosla et al. (2020), equation 2, with one view: for each anchor i, A(i) is every other
    row and P(i) the rows of A(i) with i's label; anchors with an empty P(i) are left out, and the result
    is the mean over the other anchors (0 when there is none). Rows are L2-normalised first; float64."""
    z = np.asarray(z, dtype=np.float64)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    n = z.shape[0]
    losses = []
    for i in range(n):
        others = [a for a in range(n) if a != i]
        positives = [p for p in others if labels[p] == labels[i]]
        if not positives:
            continue
        denominator = sum(math.exp(z[i] @ z[a] / temperature) for a in others)
        losses.append(-sum(math.log(math.exp(z[i] @ z[p] / temperature) / denominator) for p in positives)
                      / len(positives))
    return float(np.mean(losses)) if losses else 0.0


def screen_settings(**train) -> dict:
    """cnn_screen_1's base in the current form, with these train settings changed."""
    from affect.runs.spec import plain, validate_job

    settings = plain(validate_job(json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"]))
    settings["train"].update(train)
    return settings


class TestReference(unittest.TestCase):
    def test_hand_worked_batches(self):
        # anchors 0 and 1 have each other as their positive and row 2 at -1; row 2 has no positive
        self.assertAlmostEqual(reference_loss([[1, 0], [2, 0], [-3, 0]], [0, 0, 1], 0.5),
                               math.log(1 + math.exp(-4)), places=12)
        # one label, orthogonal rows: every similarity is 0, so each anchor's loss is log 2 at any temperature
        for t in (0.07, 1.0):
            self.assertAlmostEqual(reference_loss(np.eye(3), [0, 0, 0], t), math.log(2), places=12)
        self.assertEqual(reference_loss(np.random.default_rng(0).standard_normal((3, 4)), [0, 1, 2], 0.1), 0.0)

    def test_scale_of_the_rows_does_not_matter(self):
        rng = np.random.default_rng(1)
        z, labels = rng.standard_normal((8, 5)), [0, 1, 0, 1, 2, 2, 0, 3]
        self.assertAlmostEqual(reference_loss(z, labels, 0.1),
                               reference_loss(z * rng.uniform(0.1, 10, size=(8, 1)), labels, 0.1), places=10)


class TestSettings(unittest.TestCase):
    def test_check_rules(self):
        from affect.runs.spec import validate_job
        from affect.train.settings import check

        none = validate_job(screen_settings()).train
        check(none)
        supcon = validate_job(screen_settings(**SUPCON)).train
        check(supcon)
        for bad in ({"contrastive_weight": 0.1}, {"contrastive_temperature": 0.1}, {"contrastive_dim": 8}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check(dataclasses.replace(none, **bad))
        for bad in ({"contrastive_weight": 0.0}, {"contrastive_weight": -1.0}, {"contrastive_temperature": None},
                    {"contrastive_temperature": 0.0}, {"contrastive_dim": None}, {"contrastive_dim": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check(dataclasses.replace(supcon, **bad))

    def test_check_job_refuses_mixup_with_a_contrastive_loss(self):
        from affect.runs.spec import check_job, validate_job

        settings = screen_settings(**SUPCON)
        check_job(validate_job(settings), 16000)
        settings["augment"]["mixup_alpha"] = 0.2
        with self.assertRaisesRegex(ValueError, "mixup"):
            check_job(validate_job(settings), 16000)
        settings["train"].update({"contrastive": "none", "contrastive_weight": 0.0, "contrastive_temperature": None,
                                  "contrastive_dim": None})
        check_job(validate_job(settings), 16000)

    def test_settings_without_the_fields_hash_as_before(self):
        from affect.runs.spec import config_hash, validate_job

        new = screen_settings()
        late = json.loads(json.dumps(new))
        # the form of 2026-10-03 to 10-14 (external_test: 10-20; the valence and activation fields: 10-24)
        for field in list(SUPCON) + ["external_test", "aux_va", "aux_va_weight", "aux_va_loss"]:
            del late["train"][field]
        self.assertEqual(config_hash(validate_job(late)), config_hash(validate_job(new)))
        self.assertNotEqual(config_hash(validate_job(screen_settings(**SUPCON))), config_hash(validate_job(new)))


@needs_module("torch")
class TestTorch(unittest.TestCase):
    LABELS = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 0, 1, 4, 2, 0, 1]     # labels 3 and 4 have one row each

    def test_loss_matches_the_reference(self):
        import torch

        from affect.train.loop import supervised_contrastive_loss

        rng = np.random.default_rng(2)
        labels = torch.tensor(self.LABELS)
        for t in (0.07, 0.1, 0.3):
            for scale in (0.01, 1.0, 30.0):
                z = rng.standard_normal((16, 8)) * scale
                ref = reference_loss(z, self.LABELS, t)
                got64 = supervised_contrastive_loss(torch.from_numpy(z), labels, t)
                got32 = supervised_contrastive_loss(torch.from_numpy(z).float(), labels, t)
                self.assertEqual(got64.dtype, torch.float64)
                self.assertEqual(got32.dtype, torch.float32)
                self.assertAlmostEqual(got64.item(), ref, places=10)
                self.assertLess(abs(got32.item() - ref), 1e-4 * max(1.0, abs(ref)))

    def test_bf16_projections_are_scored_in_float32(self):
        import torch

        from affect.train.loop import supervised_contrastive_loss

        z = torch.from_numpy(np.random.default_rng(3).standard_normal((16, 8))).to(torch.bfloat16)
        got = supervised_contrastive_loss(z, torch.tensor(self.LABELS), 0.1)
        self.assertEqual(got.dtype, torch.float32)
        self.assertAlmostEqual(got.item(), reference_loss(z.float().numpy(), self.LABELS, 0.1), places=4)

    def test_gradient_matches_finite_differences(self):
        import torch

        from affect.train.loop import supervised_contrastive_loss

        z = torch.from_numpy(np.random.default_rng(4).standard_normal((10, 4))).requires_grad_(True)
        labels = torch.tensor([0, 0, 1, 1, 1, 2, 0, 3, 2, 1])
        for t in (0.07, 0.3):
            self.assertTrue(torch.autograd.gradcheck(lambda x, t=t: supervised_contrastive_loss(x, labels, t), (z,)))

    def test_no_positive_pair_gives_zero_and_zero_gradients(self):
        import torch

        from affect.train.loop import supervised_contrastive_loss

        z = torch.randn(5, 3, requires_grad=True)
        loss = supervised_contrastive_loss(z, torch.arange(5), 0.1)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))

    def test_free_vectors_learn_to_cluster_by_label(self):
        # Adam on 12 free vectors of 3 labels in 4 dimensions; in float64 numpy (the container, 2026-10-15)
        # the same run reaches 1.2233, the loss of each label collapsed to a point and the three points
        # 120 degrees apart, from every one of 10 starts
        import torch

        from affect.train.loop import supervised_contrastive_loss

        labels = torch.arange(3).repeat_interleave(4)
        u = torch.nn.Parameter(torch.from_numpy(np.random.default_rng(0).standard_normal((12, 4))).float())
        opt = torch.optim.Adam([u], lr=0.05)
        first = None
        for _ in range(300):
            loss = supervised_contrastive_loss(u, labels, 0.5)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        last = supervised_contrastive_loss(u, labels, 0.5).item()
        self.assertGreater(first, 2.0)
        self.assertLess(last, math.log(3 + 8 * math.exp(-3)) + 0.01)
        z = torch.nn.functional.normalize(u.detach(), dim=1)
        nearest = (z @ z.T).fill_diagonal_(-2.0).argmax(dim=1)
        self.assertTrue(torch.equal(labels[nearest], labels))

    def test_projection_head(self):
        import torch

        from affect.models.heads import build_projection_head

        head = build_projection_head(12, 5)
        self.assertEqual(head(torch.randn(3, 12)).shape, (3, 5))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 12 * 12 + 12 + 12 * 5 + 5)
        for width, dim in ((12, 0), (0, 5)):
            with self.assertRaises(ValueError):
                build_projection_head(width, dim)

    def test_no_head_and_no_draws_without_the_option(self):
        import torch

        from affect.models.cnn import build_cnn
        from affect.runs.spec import validate_job
        from affect.train.loop import make_aux_head, make_contrastive_head, trainable_parameters

        job = validate_job(screen_settings())
        torch.manual_seed(0)
        model = build_cnn(job.model, job.features.n_mels, 4)
        self.assertIsNone(make_aux_head(job.train, model, torch.device("cpu")))
        self.assertIsNone(make_contrastive_head(job.train, model, torch.device("cpu")))
        after = torch.rand(3)
        torch.manual_seed(0)
        build_cnn(job.model, job.features.n_mels, 4)
        self.assertTrue(torch.equal(after, torch.rand(3)), "the same random stream as without the option")
        params = trainable_parameters(model, None, None)
        self.assertEqual(len(params), len(list(model.parameters())))
        self.assertTrue(all(a is b for a, b in zip(params, model.parameters(), strict=True)))

    def test_step_with_a_contrastive_head(self):
        import torch

        from affect.augment.specaugment import spec_augment
        from affect.features.logmel import LogMel
        from affect.models.cnn import build_cnn
        from affect.runs.spec import validate_job
        from affect.train.loop import (Normalizer, Step, make_contrastive_head, supervised_contrastive_loss,
                                       trainable_parameters)

        settings = screen_settings(**SUPCON, precision="fp32")
        job = validate_job(settings)
        device = torch.device("cpu")
        feats = LogMel(job.features, 16000, device)
        norm = Normalizer("utterance")
        torch.manual_seed(3)
        model = build_cnn(job.model, job.features.n_mels, 4).train()
        head = make_contrastive_head(job.train, model, device)
        self.assertEqual((head.hidden.in_features, head.out.out_features), (model.embedding_dim, 16))
        waves = (np.random.default_rng(1).standard_normal((4, 16000)) * 0.1).astype(np.float32)
        lengths, rows, labels = np.array([16000, 9000, 16000, 12000]), np.arange(4), np.array([0, 1, 0, 1])
        targets = np.eye(4, dtype=np.float32)[labels]
        step = Step(job, model, None, feats, norm, None, device, head)
        loss, emotion, gender, contrastive = step.loss(waves, lengths, rows, targets, None,
                                                       torch.Generator().manual_seed(5), np.random.default_rng(9),
                                                       labels)
        self.assertIsNone(gender)
        self.assertAlmostEqual(loss.item(), emotion.item() + 0.5 * contrastive.item(), places=5)
        loss.backward()
        params = trainable_parameters(model, None, head)
        self.assertEqual(len(params), len(list(model.parameters())) + 4)
        self.assertTrue(all(p.grad is not None for p in params if p.requires_grad))
        self.assertGreater(head.out.weight.grad.abs().sum().item(), 0.0)
        # the same loss from the parts: features, masks, the model's summary and the projection head (after
        # the backward pass, since a second forward pass in training mode updates the batch-norm statistics)
        x = feats(torch.from_numpy(waves))
        frames = feats.frames(torch.from_numpy(lengths)).clamp_max(x.shape[-1])
        x = spec_augment(norm(x, frames, rows), frames, job.augment, torch.Generator().manual_seed(5))
        with torch.no_grad():
            projections = head(model.embed(x, frames))
        self.assertAlmostEqual(contrastive.item(),
                               supervised_contrastive_loss(projections, torch.from_numpy(labels), 0.1).item(), places=5)
        self.assertAlmostEqual(contrastive.item(), reference_loss(projections.numpy(), labels, 0.1), places=4)
        with self.assertRaisesRegex(ValueError, "label"):
            step.loss(waves, lengths, rows, targets, None, torch.Generator().manual_seed(5), np.random.default_rng(9))
        with self.assertRaisesRegex(ValueError, "projection head"):
            Step(job, model, None, feats, norm, None, device)
        none_job = validate_job(screen_settings(precision="fp32"))
        with self.assertRaisesRegex(ValueError, "projection head"):
            Step(none_job, model, None, feats, norm, None, device, head)
        settings["augment"]["mixup_alpha"] = 0.2
        with self.assertRaisesRegex(ValueError, "mixup"):
            Step(validate_job(settings), model, None, feats, norm, None, device, head)


if __name__ == "__main__":
    unittest.main()
