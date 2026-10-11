"""Features, augmentation, model and loss, against known answers and against librosa.
Needs torch (and librosa for two tests); runs on the node, where scripts/check.sh requires both."""
import math
import unittest

import numpy as np

from tests.support import needs_module


@needs_module("torch")
@needs_module("librosa")
class TestAgainstLibrosa(unittest.TestCase):
    def test_filterbank_matches_librosa(self):
        import librosa

        from affect.features.logmel import mel_filterbank

        for n_fft, n_mels, f_max in ((512, 64, 8000.0), (1024, 128, 4000.0), (512, 40, 8000.0)):
            ours = mel_filterbank(16000, n_fft, n_mels, 0.0, f_max)
            ref = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=f_max, htk=False, norm="slaney")
            self.assertEqual(ours.shape, ref.shape)
            self.assertLess(np.abs(ours - ref).max(), 1e-6 * max(1.0, np.abs(ref).max()))

    def test_logmel_matches_librosa(self):
        import librosa
        import torch

        from affect.features.logmel import LogMel, LogMelSettings

        rng = np.random.default_rng(0)
        t = np.arange(16000 * 2) / 16000
        y = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(t.shape)).astype(np.float32)
        for win_ms, hop_ms, n_fft, n_mels in ((25.0, 10.0, 512, 64), (50.0, 12.5, 1024, 80)):
            s = LogMelSettings(win_ms=win_ms, hop_ms=hop_ms, n_fft=n_fft, n_mels=n_mels, f_min=0.0, f_max=8000.0,
                               log_offset=1e-6, normalization="none")
            ours = LogMel(s, 16000, torch.device("cpu"))(torch.from_numpy(y)[None, :])[0].numpy()
            win, hop = int(round(win_ms * 16)), int(round(hop_ms * 16))
            mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=n_fft, hop_length=hop, win_length=win,
                                                 window="hann", center=True, pad_mode="constant", power=2.0,
                                                 n_mels=n_mels, fmin=0.0, fmax=8000.0, htk=False, norm="slaney")
            ref = np.log(mel + 1e-6)
            self.assertEqual(ours.shape, ref.shape)
            self.assertLess(np.abs(ours - ref).max(), 1e-2)
            self.assertEqual(ours.shape[1], 1 + y.shape[0] // hop)


@needs_module("torch")
class TestModelAndTraining(unittest.TestCase):
    def cnn(self, pooling, hidden):
        from affect.models.cnn import CNNSettings, build_cnn

        s = CNNSettings(channels=(8, 16, 32), convs_per_block=1, kernel_size=3, pool_freq=2, pool_time=2,
                        pooling=pooling, classifier_hidden=hidden, dropout=0.1, norm="batch", norm_groups=None,
                        norm_subbands=None, activation="relu")
        return build_cnn(s, 40, 4)

    def test_every_pooling_gives_finite_logits(self):
        import torch

        x = torch.randn(3, 40, 101)
        valid = torch.tensor([101, 60, 9])
        for pooling in ("mean", "max", "meanmax", "stats", "attention"):
            for hidden in (None, 16):
                model = self.cnn(pooling, hidden)
                out = model(x, valid)
                self.assertEqual(tuple(out.shape), (3, 4))
                self.assertTrue(torch.isfinite(out).all(), pooling)
                out.sum().backward()

    def test_frames_after_blocks_matches_the_network(self):
        import torch

        model = self.cnn("mean", None).eval()
        x = torch.randn(1, 40, 101)
        h = model.blocks(x.unsqueeze(1))
        self.assertEqual(int(model.frames_after_blocks(torch.tensor([101]))[0]), h.shape[-1])

    def test_masked_pool_ignores_padding(self):
        import torch

        from affect.models.cnn import masked_pool

        h = torch.randn(2, 5, 10)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, :4] = True
        mask[1, :] = True
        padded = h.clone()
        padded[0, :, 4:] = 1e6
        for kind in ("mean", "max", "meanmax", "stats"):
            a = masked_pool(h, mask, kind, None)
            b = masked_pool(padded, mask, kind, None)
            self.assertTrue(torch.allclose(a, b), kind)
        self.assertTrue(torch.allclose(masked_pool(h, mask, "mean", None)[0], h[0, :, :4].mean(dim=-1)))
        self.assertTrue(torch.allclose(masked_pool(h, mask, "max", None)[0], h[0, :, :4].amax(dim=-1)))

    def test_spec_augment(self):
        import torch

        from affect.augment.settings import AugmentSettings
        from affect.augment.specaugment import spec_augment

        x = torch.ones(4, 40, 100)
        valid = torch.tensor([100, 50, 20, 100])
        off = AugmentSettings(freq_masks=0, freq_mask_width=0, time_masks=0, time_mask_width=0,
                              speed_factors=(1.0,), mixup_alpha=0.0)
        self.assertIs(spec_augment(x, valid, off, torch.Generator().manual_seed(0)), x)
        s = AugmentSettings(freq_masks=2, freq_mask_width=5, time_masks=2, time_mask_width=10,
                            speed_factors=(1.0,), mixup_alpha=0.0)
        out = spec_augment(x, valid, s, torch.Generator().manual_seed(0))
        zero_bands = (out == 0).all(dim=2).sum(dim=1)
        self.assertTrue((zero_bands <= 10).all())
        zero_frames = (out == 0).all(dim=1)
        for b in range(4):
            idx = torch.nonzero(zero_frames[b]).flatten()
            if idx.numel():
                self.assertLess(int(idx.max()), int(valid[b]))

    def test_utterance_normalisation(self):
        import torch

        from affect.train.loop import Normalizer

        x = torch.randn(2, 8, 50) * 3 + 7
        frames = torch.tensor([50, 30])
        out = Normalizer("utterance")(x, frames)
        self.assertLess(out[1, :, :30].mean(dim=-1).abs().max().item(), 1e-4)
        self.assertLess((out[1, :, :30].std(dim=-1, unbiased=False) - 1).abs().max().item(), 1e-2)
        self.assertEqual(out[1, :, 30:].abs().max().item(), 0.0)

    def test_speaker_normalisation(self):
        import torch

        from affect.train.loop import Normalizer

        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 8, 50, generator=g)
        x[:2] = x[:2] * 2 + 5          # speaker 0
        x[2:] = x[2:] * 0.5 - 3        # speaker 1
        frames = torch.tensor([50, 50, 50, 50])
        speaker_of = np.array([0, 0, 1, 1])
        mean = torch.stack([x[:2].mean(dim=(0, 2)), x[2:].mean(dim=(0, 2))])
        std = torch.stack([x[:2].std(dim=(0, 2), unbiased=False), x[2:].std(dim=(0, 2), unbiased=False)])
        norm = Normalizer("speaker", mean, std, speaker_of)
        out = norm(x, frames, np.array([0, 1, 2, 3]))
        for part in (slice(0, 2), slice(2, 4)):
            self.assertLess(out[part].mean(dim=(0, 2)).abs().max().item(), 1e-5)
            self.assertLess((out[part].std(dim=(0, 2), unbiased=False) - 1).abs().max().item(), 1e-4)
        # the rows pick each example's speaker, whatever the order of the batch
        self.assertTrue(torch.allclose(norm(x[[3, 0]], frames[:2], np.array([3, 0])), out[[3, 0]]))
        # padding is zeroed as in the other modes
        short = norm(x, torch.tensor([50, 20, 50, 10]), np.array([0, 1, 2, 3]))
        self.assertEqual(short[1, :, 20:].abs().max().item(), 0.0)
        with self.assertRaises(ValueError):
            norm(x, frames)
        with self.assertRaises(ValueError):
            Normalizer("speaker", mean, std)

    def test_speaker_stats_match_a_direct_computation(self):
        import torch

        from affect.features.logmel import LogMel, LogMelSettings
        from affect.train.loop import speaker_stats

        rng = np.random.default_rng(0)
        audio = [rng.integers(-3000, 3000, size=n).astype(np.int16) for n in (16000, 12000, 20000, 8000, 16000)]
        speaker_of = np.array([0, 1, 0, 1, 1])
        s = LogMelSettings(win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                           log_offset=1e-6, normalization="speaker")
        feats = LogMel(s, 16000, torch.device("cpu"))
        mean, std = speaker_stats(feats, audio, np.arange(5), speaker_of, 2, 2, torch.device("cpu"))
        self.assertEqual(tuple(mean.shape), (2, 40))
        for spk in (0, 1):
            per_utt = [feats(torch.from_numpy(audio[r].astype(np.float32) / 32768.0)[None, :])[0]
                       for r in np.flatnonzero(speaker_of == spk)]
            frames = torch.cat(per_utt, dim=1).double()
            self.assertTrue(torch.allclose(mean[spk].double(), frames.mean(dim=1), atol=1e-4))
            self.assertTrue(torch.allclose(std[spk].double(), frames.std(dim=1, unbiased=False), atol=1e-3))
        with self.assertRaises(ValueError):          # a speaker without rows
            speaker_stats(feats, audio, np.array([0, 2]), speaker_of, 2, 2, torch.device("cpu"))

    def test_soft_cross_entropy_equals_cross_entropy_for_one_hot(self):
        import torch

        from affect.train.loop import soft_cross_entropy

        logits = torch.randn(6, 4)
        y = torch.tensor([0, 1, 2, 3, 1, 0])
        ours = soft_cross_entropy(logits, torch.nn.functional.one_hot(y, 4).float(), None)
        self.assertAlmostEqual(ours.item(), torch.nn.functional.cross_entropy(logits, y).item(), places=5)

    def test_learning_rate_schedule(self):
        from affect.train.loop import lr_factor

        self.assertAlmostEqual(lr_factor(0, 10, 100, "cosine"), 0.1)
        self.assertAlmostEqual(lr_factor(9, 10, 100, "cosine"), 1.0)
        self.assertAlmostEqual(lr_factor(10, 10, 100, "cosine"), 1.0)
        self.assertLess(lr_factor(99, 10, 100, "cosine"), 0.01)
        self.assertEqual(lr_factor(50, 10, 100, "constant"), 1.0)
        self.assertTrue(math.isclose(lr_factor(55, 10, 100, "cosine"), 0.5, abs_tol=1e-9))

    def test_cnn_layers_match_the_cnn_before_the_new_settings(self):
        """norm batch + relu (the values of every configuration written before 2026-09-26) must build the
        very network the CNN built before: same layers, same order, same initial weights, same output."""
        import torch
        from torch import nn

        from affect.models.cnn import CNNSettings, build_cnn, masked_pool

        for pooling, hidden in (("mean", None), ("stats", 16), ("attention", None), ("meanmax", 8)):
            s = CNNSettings(channels=(8, 16), convs_per_block=2, kernel_size=3, pool_freq=2, pool_time=2,
                            pooling=pooling, classifier_hidden=hidden, dropout=0.1, norm="batch", norm_groups=None,
                            norm_subbands=None, activation="relu")
            torch.manual_seed(7)
            new = build_cnn(s, 40, 4)
            torch.manual_seed(7)
            layers, c_in = [], 1                   # the construction code of the CNN before 2026-09-26
            for c_out in s.channels:
                for _ in range(s.convs_per_block):
                    layers += [nn.Conv2d(c_in, c_out, s.kernel_size, padding=s.kernel_size // 2, bias=False),
                               nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
                    c_in = c_out
                layers.append(nn.MaxPool2d((s.pool_freq, s.pool_time)))
            blocks = nn.Sequential(*layers)
            width = s.channels[-1]
            attention = nn.Linear(width, 1) if pooling == "attention" else None
            if pooling in ("meanmax", "stats"):
                width *= 2
            head = [nn.Dropout(s.dropout)]
            if hidden is not None:
                head += [nn.Linear(width, hidden), nn.ReLU(inplace=True), nn.Dropout(s.dropout)]
                width = hidden
            head.append(nn.Linear(width, 4))
            head = nn.Sequential(*head)
            self.assertEqual([type(m) for m in new.blocks], [type(m) for m in blocks])
            self.assertEqual([type(m) for m in new.head], [type(m) for m in head])
            old_params = list(blocks.parameters()) + (list(attention.parameters()) if attention else []) + list(head.parameters())
            new_params = list(new.parameters())
            self.assertEqual(len(old_params), len(new_params))
            for a, b in zip(old_params, new_params):
                self.assertTrue(torch.equal(a, b), pooling)
            new.eval()
            blocks.eval()
            head.eval()
            x = torch.randn(3, 40, 64)
            valid = torch.tensor([64, 40, 9])
            with torch.no_grad():
                h = blocks(x.unsqueeze(1)).float().mean(dim=2)
                keep = new.frames_after_blocks(valid).clamp_max(h.shape[-1])
                mask = torch.arange(h.shape[-1])[None, :] < keep[:, None]
                reference = head(masked_pool(h, mask, pooling, attention))
                self.assertTrue(torch.equal(new(x, valid), reference), pooling)
                self.assertTrue(torch.equal(new.classify(new.embed(x, valid)), reference))
            self.assertEqual(new.embedding_dim, 2 * 16 if pooling in ("meanmax", "stats") else 16)

    def test_every_normalisation_and_activation(self):
        import torch
        from torch import nn

        from affect.models.cnn import CNNSettings, build_cnn, check

        cases = [("batch", None, None), ("subspectral", None, 2), ("subspectral", None, 4), ("group", 8, None),
                 ("group", 1, None), ("none", None, None)]
        x = torch.randn(4, 40, 50)
        valid = torch.tensor([50, 30, 20, 50])
        for norm, groups, subbands in cases:
            for activation in ("relu", "leaky_relu", "elu", "gelu", "silu"):
                s = CNNSettings(channels=(8, 16, 16), convs_per_block=1, kernel_size=3, pool_freq=2, pool_time=2,
                                pooling="stats", classifier_hidden=8, dropout=0.1, norm=norm, norm_groups=groups,
                                norm_subbands=subbands, activation=activation)
                model = build_cnn(s, 40, 4)
                out = model(x, valid)
                self.assertEqual(tuple(out.shape), (4, 4))
                self.assertTrue(torch.isfinite(out).all(), (norm, activation))
                out.sum().backward()
                convs = [m for m in model.blocks if isinstance(m, nn.Conv2d)]
                self.assertTrue(all((c.bias is not None) == (norm == "none") for c in convs), norm)
                norms = {"batch": nn.BatchNorm2d, "group": nn.GroupNorm}.get(norm)
                if norms is not None:
                    self.assertEqual(sum(isinstance(m, norms) for m in model.blocks), 3)
        bad = dict(channels=(8, 12), convs_per_block=1, kernel_size=3, pool_freq=2, pool_time=2, pooling="mean",
                   classifier_hidden=None, dropout=0.1, activation="relu")
        for norm, groups, subbands in (("group", 8, None), ("group", None, None), ("subspectral", None, 1),
                                       ("subspectral", None, 32), ("batch", 4, None), ("none", None, 2)):
            with self.assertRaises(ValueError, msg=str((norm, groups, subbands))):
                check(CNNSettings(**bad, norm=norm, norm_groups=groups, norm_subbands=subbands), 40)

    def test_subspectral_norm_normalises_each_band_apart(self):
        import torch

        from affect.models.cnn import _subspectral_class

        norm = _subspectral_class()(3, 2).train()
        x = torch.randn(8, 3, 10, 20)
        x[:, :, :5] = x[:, :, :5] * 4 + 9          # the two halves of the frequency axis differ
        out = norm(x)
        for band in (slice(0, 5), slice(5, 10)):
            part = out[:, :, band]
            self.assertLess(part.mean(dim=(0, 2, 3)).abs().max().item(), 1e-4)
            self.assertLess((part.var(dim=(0, 2, 3), unbiased=False) - 1).abs().max().item(), 1e-2)
        odd = _subspectral_class()(3, 4)(torch.randn(4, 3, 10, 6))
        self.assertEqual(tuple(odd.shape), (4, 3, 10, 6), "bands of 3, 3, 2, 2 rows")

    def test_cnn_transformer(self):
        import torch

        from affect.models.cnn_transformer import CNNTransformerSettings, build_cnn_transformer, check, sinusoidal

        base = dict(channels=(8, 16), convs_per_block=1, kernel_size=3, pool_freq=2, pool_time=2, norm="batch",
                    norm_groups=None, norm_subbands=None, activation="relu", freq_reduce="mean", d_model=32, n_heads=4,
                    n_layers=2, ff_dim=64, transformer_dropout=0.1, transformer_activation="gelu", norm_first=True,
                    positional="sinusoidal", pooling="mean", classifier_hidden=None, dropout=0.1)
        x = torch.randn(3, 40, 80)
        valid = torch.tensor([80, 41, 12])
        for changes in ({}, {"freq_reduce": "flatten"}, {"norm_first": False}, {"positional": "none"},
                        {"pooling": "attention"}, {"pooling": "stats", "classifier_hidden": 16},
                        {"norm": "group", "norm_groups": 4, "activation": "gelu", "transformer_activation": "relu"}):
            s = CNNTransformerSettings(**{**base, **changes})
            model = build_cnn_transformer(s, 40, 4)
            out = model(x, valid)
            self.assertEqual(tuple(out.shape), (3, 4), changes)
            self.assertTrue(torch.isfinite(out).all(), changes)
            out.sum().backward()
            self.assertTrue(torch.isfinite(model.project.weight.grad).all())
        # the key padding mask must hide padded steps from the real ones (True marks padding)
        model = build_cnn_transformer(CNNTransformerSettings(**base), 40, 4).eval()
        h = torch.randn(2, 10, 32)
        keep = torch.tensor([10, 6])
        pad = torch.arange(10)[None, :] >= keep[:, None]
        with torch.no_grad():
            a = model.encoder(h, src_key_padding_mask=pad)
            changed = h.clone()
            changed[1, 6:] = 100.0
            b = model.encoder(changed, src_key_padding_mask=pad)
        self.assertTrue(torch.allclose(a[1, :6], b[1, :6], atol=1e-5))
        self.assertTrue(torch.allclose(a[0], b[0], atol=1e-6))
        code = sinusoidal(5, 8, torch.device("cpu"), torch.float32)
        self.assertTrue(torch.allclose(code[:, 0], torch.sin(torch.arange(5.0))))
        self.assertTrue(torch.allclose(code[:, 1], torch.cos(torch.arange(5.0))))
        for bad in ({"n_heads": 5}, {"d_model": 33, "n_heads": 3, "positional": "sinusoidal"}, {"n_layers": 0},
                    {"transformer_dropout": 1.0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check(CNNTransformerSettings(**{**base, **bad}), 40)

    def test_gender_head_multitask_and_adversarial(self):
        import torch

        from affect.models.heads import build_aux_head

        torch.manual_seed(0)
        plain = build_aux_head("multitask", 6, 2, 0.0)
        reverse = build_aux_head("adversarial", 6, 2, 0.0)
        reverse.load_state_dict(plain.state_dict())
        e1 = torch.randn(4, 6, requires_grad=True)
        e2 = e1.detach().clone().requires_grad_(True)
        out1, out2 = plain(e1), reverse(e2)
        self.assertTrue(torch.equal(out1, out2), "the reversal changes nothing going forward")
        out1.pow(2).sum().backward()
        out2.pow(2).sum().backward()
        self.assertTrue(torch.allclose(e1.grad, -e2.grad), "and negates the gradient going back")
        self.assertTrue(torch.allclose(plain.linear.weight.grad, reverse.linear.weight.grad),
                        "the head itself still learns to predict gender")
        with self.assertRaises(ValueError):
            build_aux_head("none", 6, 2, 0.0)

    def test_mixup_mix(self):
        import torch

        from affect.augment import mixup

        x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        frames = torch.tensor([4, 2])
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights, partner = np.array([0.75, 1.0], dtype=np.float32), np.array([1, 0])
        mixed, mixed_frames, (mt,) = mixup.mix(x, frames, [t], weights, partner)
        self.assertTrue(torch.allclose(mixed[0], 0.75 * x[0] + 0.25 * x[1]))
        self.assertTrue(torch.allclose(mixed[1], x[1]))
        self.assertEqual(mixed_frames.tolist(), [4, 4])
        self.assertTrue(torch.allclose(mt[0], torch.tensor([0.75, 0.25])))
        self.assertTrue(torch.allclose(mt[1], torch.tensor([0.0, 1.0])))

    def test_predict_averages_segment_probabilities(self):
        import torch

        from affect.features.logmel import LogMel, LogMelSettings
        from affect.train.data import segment_starts
        from affect.train.loop import Normalizer, predict

        torch.manual_seed(0)
        model = self.cnn("mean", None).eval()
        feats = LogMel(LogMelSettings(win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                                      log_offset=1e-6, normalization="utterance"), 16000, torch.device("cpu"))
        norm = Normalizer("utterance")
        rng = np.random.default_rng(0)
        audio = [rng.integers(-3000, 3000, size=n).astype(np.int16) for n in (40000, 12000, 25000)]
        rows = np.array([0, 1, 2])
        segment = (16000, 8000)
        got = predict(model, feats, norm, audio, rows, 1, torch.device("cpu"), "fp32", segment)
        whole = predict(model, feats, norm, audio, rows, 1, torch.device("cpu"), "fp32")
        self.assertIsNone(got.aux_probs)
        for k, row in enumerate(rows):
            probs = []
            for start in segment_starts(audio[row].shape[0], *segment):
                piece = audio[row][start:start + segment[0]].astype(np.float32) / 32768.0
                with torch.no_grad():
                    x = feats(torch.from_numpy(piece)[None, :])
                    frames = feats.frames(torch.tensor([piece.shape[0]])).clamp_max(x.shape[-1])
                    probs.append(torch.softmax(model(norm(x, frames), frames), dim=-1)[0].numpy())
            self.assertTrue(np.allclose(got.probs[k], np.mean(probs, axis=0), atol=1e-5), k)
        self.assertTrue(np.allclose(got.probs[1], whole.probs[1], atol=1e-5), "a short utterance is one segment")
        self.assertTrue(np.allclose(got.probs.sum(axis=1), 1.0))

    def test_step_matches_the_training_step_before_the_new_settings(self):
        """With no speed perturbation, mixup or gender head, Step.loss computes exactly what the
        training loop computed before 2026-09-26."""
        import json

        import torch

        from affect import catalog
        from affect.augment.specaugment import spec_augment
        from affect.features.logmel import LogMel
        from affect.models.cnn import build_cnn
        from affect.runs.spec import validate_job
        from affect.train.loop import Normalizer, Step, soft_cross_entropy
        from tests.support import REPO

        catalog.load_all()
        sweep = json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())
        settings = sweep["base"]
        settings["augment"].update({"freq_masks": 2, "freq_mask_width": 8, "time_masks": 2, "time_mask_width": 20})
        settings["train"]["precision"] = "fp32"      # the reference below runs without autocast
        job = validate_job(settings)
        device = torch.device("cpu")
        feats = LogMel(job.features, 16000, device)
        rng = np.random.default_rng(1)
        waves = (rng.standard_normal((4, 16000)) * 0.1).astype(np.float32)
        lengths = np.array([16000, 12000, 16000, 8000])
        rows = np.arange(4)
        targets = np.eye(4, dtype=np.float32)[[0, 1, 2, 3]]
        weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
        norm = Normalizer("utterance")
        torch.manual_seed(3)
        model = build_cnn(job.model, job.features.n_mels, 4).train()
        torch.manual_seed(11)
        loss, emotion, gender = Step(job, model, None, feats, norm, weights, device).loss(
            waves, lengths, rows, targets, None, torch.Generator().manual_seed(5), np.random.default_rng(9))
        self.assertIsNone(gender)
        self.assertIs(loss, emotion)
        torch.manual_seed(11)
        x = feats(torch.from_numpy(waves))
        frames = feats.frames(torch.from_numpy(lengths)).clamp_max(x.shape[-1])
        x = spec_augment(norm(x, frames, rows), frames, job.augment, torch.Generator().manual_seed(5))
        reference = soft_cross_entropy(model(x, frames), torch.from_numpy(targets), weights)
        self.assertEqual(loss.item(), reference.item())

    def test_step_with_mixup_and_a_gender_head(self):
        import json

        import torch

        from affect import catalog
        from affect.features.logmel import LogMel
        from affect.models.cnn import build_cnn
        from affect.runs.spec import validate_job
        from affect.train.loop import Normalizer, Step, make_aux_head, trainable_parameters
        from tests.support import REPO

        catalog.load_all()
        settings = json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"]
        from affect.runs.spec import plain

        settings = plain(validate_job(settings))
        settings["augment"]["mixup_alpha"] = 0.4
        settings["train"]["precision"] = "fp32"
        for kind in ("multitask", "adversarial"):
            settings["train"].update({"aux_gender": kind, "aux_gender_weight": 0.5})
            job = validate_job(settings)
            device = torch.device("cpu")
            feats = LogMel(job.features, 16000, device)
            model = build_cnn(job.model, job.features.n_mels, 4).train()
            aux = make_aux_head(job.train, model, device)
            self.assertEqual(aux.linear.in_features, model.embedding_dim)
            waves = (np.random.default_rng(1).standard_normal((4, 16000)) * 0.1).astype(np.float32)
            loss, emotion, gender = Step(job, model, aux, feats, Normalizer("utterance"), None, device).loss(
                waves, np.array([16000, 9000, 16000, 12000]), np.arange(4), np.eye(4, dtype=np.float32),
                np.eye(2, dtype=np.float32)[[0, 1, 0, 1]], torch.Generator().manual_seed(0), np.random.default_rng(2))
            self.assertAlmostEqual(loss.item(), emotion.item() + 0.5 * gender.item(), places=5)
            loss.backward()
            self.assertTrue(all(p.grad is not None for p in trainable_parameters(model, aux) if p.requires_grad))

    def test_a_tiny_model_learns(self):
        import torch

        from affect.train.loop import soft_cross_entropy

        torch.manual_seed(0)
        model = self.cnn("mean", None)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        x = torch.randn(16, 40, 64)
        y = torch.arange(16) % 4
        x[y == 1, :10] += 3.0
        x[y == 2, 20:] -= 3.0
        x[y == 3, :, :32] += 2.0
        valid = torch.full((16,), 64)
        target = torch.nn.functional.one_hot(y, 4).float()
        first = None
        for _ in range(60):
            loss = soft_cross_entropy(model(x, valid), target, None)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.5 * first)


if __name__ == "__main__":
    unittest.main()
