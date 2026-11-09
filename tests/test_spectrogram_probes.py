"""The spectrogram encoders of the classical pipeline (added 2026-10-29, workstream C): Whisper large-v3's encoder and
the Audio Spectrogram Transformer, probed through their layers' mean vectors (affect.classical.pretrained).

Without torch: the pieces an utterance is cut into (every sample once, each piece within the window, the fewest
pieces, lengths within one sample of each other); the feature-set names and the settings that accept them; the frames
features keep the six waveform encoders and Whisper's encoder (added 2026-11-01), and extract_frames refuses AST.

With torch and transformers (on the node, in ~/venvs/affect_pretrained): tiny Whisper and AST models with random
weights, saved as snapshots with their feature extractors; load_encoder sends them to the spectrogram path, and each
utterance's vector equals, layer by layer, the mean taken by hand from the model's own outputs over exactly the
outputs that cover the utterance's samples (Whisper: the first ceil(samples / 320) encoder frames; AST: the patch
tokens of every frequency row whose time span starts within the Kaldi frames), for a short utterance and for one
longer than the window (cut into pieces whose outputs are pooled together); the padding's outputs are left out (the
vector differs from the mean over every output); mean_sd gives the mean and the population SD side by side. Whisper's
frames (load_whisper_frame_encoder, added 2026-11-01) are exactly those outputs, piece after piece, in float16, for
each requested layer: their count is the sum over pieces of ceil(samples / 320), and their mean is the probe's mean
vector of the same layer.

Added 2026-11-02 (week 32): w2v-BERT 2.0 and AST's frames. Without torch: w2v_bert_2's names, its 25 layers of 1024,
every encoder in the frames features (AST at 10 frames a second, the others at 50), extract_frames refusing an unknown
encoder and a layer out of range. With torch and transformers: a tiny w2v-BERT 2.0 with random weights and its feature
extractor, saved as a snapshot; its vector equals, layer by layer, the mean over the frames the extractor's attention
mask marks real, which number half the filterbank frames rounded down (so an odd count's padded frame is left out, and
the vector then differs from the mean over every output); its frames (load_frame_encoder) are exactly those outputs and
their mean is the probe's vector. AST's frames (load_ast_frame_encoder): one per time column of the probe's tokens,
the mean of its frequency rows, piece after piece (count: the sum over pieces of ceil(Kaldi frames / 10)); their mean is
the probe's vector; it refuses a Whisper snapshot and a layer out of range."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from affect import catalog
from tests.support import needs_module

catalog.load_all()


class TestPieces(unittest.TestCase):
    def test_pieces(self):
        from affect.classical.pretrained import pieces

        for n in (1, 399, 400, 401, 16239, 16240, 16241, 32479, 480000, 480001, 1000003):
            for window in (400, 16240, 480000):
                got = pieces(n, window)
                k = -(-n // window)
                self.assertEqual(len(got), k, (n, window))
                self.assertEqual(got[0][0], 0)
                self.assertEqual(got[-1][1], n)
                for (a, b), (c, _) in zip(got, got[1:]):
                    self.assertEqual(b, c)
                lengths = [b - a for a, b in got]
                self.assertTrue(all(1 <= m <= window for m in lengths), (n, window, lengths[:3]))
                self.assertLessEqual(max(lengths) - min(lengths), 1)
        for bad in ((0, 10), (10, 0), (-1, 5)):
            with self.assertRaises(ValueError):
                pieces(*bad)

    def test_names_and_settings(self):
        from affect.classical.pretrained import REPOS, SPECTROGRAM_TYPES, feature_sets
        from affect.classical.settings import (PRETRAINED_AVERAGE_SETS, PRETRAINED_MODELS, SPECTROGRAM_MODELS,
                                               FunctionalSettings)
        from affect.features.frames import ENCODERS, FramesSettings
        from affect.settings import validate

        self.assertEqual(SPECTROGRAM_MODELS, ("whisper_large_v3", "ast_audioset", "w2v_bert_2", "whisper_small",
                                              "bud_e_whisper", "whisper_medium"))
        self.assertEqual(PRETRAINED_MODELS["whisper_large_v3"], (33, 1280))
        self.assertEqual(PRETRAINED_MODELS["whisper_small"], (13, 768))          # added 2026-11-05
        self.assertEqual(REPOS["whisper_small"], "openai/whisper-small")
        self.assertEqual(feature_sets("whisper_small"), [f"whisper_small_l{k}" for k in range(13)])
        self.assertEqual(PRETRAINED_MODELS["bud_e_whisper"], (13, 768))          # added 2026-11-07
        self.assertEqual(REPOS["bud_e_whisper"], "laion/BUD-E-Whisper")
        self.assertEqual(feature_sets("bud_e_whisper")[-1], "bud_e_whisper_l12")
        self.assertEqual(PRETRAINED_MODELS["whisper_medium"], (25, 1024))
        self.assertEqual(REPOS["whisper_medium"], "openai/whisper-medium")
        self.assertEqual(feature_sets("whisper_medium"), [f"whisper_medium_l{k}" for k in range(25)])
        self.assertEqual(PRETRAINED_MODELS["ast_audioset"], (13, 768))
        self.assertEqual(PRETRAINED_MODELS["w2v_bert_2"], (25, 1024))
        self.assertEqual(REPOS["whisper_large_v3"], "openai/whisper-large-v3")
        self.assertEqual(REPOS["ast_audioset"], "MIT/ast-finetuned-audioset-10-10-0.4593")
        self.assertEqual(REPOS["w2v_bert_2"], "facebook/w2v-bert-2.0")
        self.assertEqual(feature_sets("w2v_bert_2")[0], "w2v_bert_2_l0")
        self.assertEqual(feature_sets("w2v_bert_2")[-1], "w2v_bert_2_l24")
        self.assertEqual(sorted(SPECTROGRAM_TYPES.values()), ["ast", "whisper"])
        self.assertEqual(feature_sets("whisper_large_v3")[-1], "whisper_large_v3_l32")
        self.assertEqual(len(feature_sets("ast_audioset", "mean_sd")), 13)
        self.assertIn("ast_audioset_lavg", PRETRAINED_AVERAGE_SETS)
        for name in ("whisper_large_v3_l0", "whisper_large_v3_l32", "ast_audioset_l12", "ast_audioset_l7_msd",
                     "whisper_large_v3_lavg", "w2v_bert_2_l0", "w2v_bert_2_l24", "w2v_bert_2_lavg", "whisper_small_l0",
                     "whisper_small_l12", "bud_e_whisper_l12", "whisper_medium_l24"):
            validate(FunctionalSettings, {"feature_set": name, "normalization": "speaker"}, "functionals")
        for name in ("whisper_large_v3_l33", "ast_audioset_l13", "w2v_bert_2_l25", "whisper_small_l13",
                     "bud_e_whisper_l13", "whisper_medium_l25"):
            with self.assertRaises(ValueError):
                validate(FunctionalSettings, {"feature_set": name, "normalization": "speaker"}, "functionals")
        # every encoder gives frames: the six waveform encoders, Whisper's since 2026-11-01, AST's (10 a second) and
        # w2v-BERT 2.0's since 2026-11-02
        from affect.features import frames as frame_features

        self.assertEqual(ENCODERS, ("wavlm_base_plus", "hubert_base", "wav2vec2_base", "wavlm_large", "hubert_large",
                                    "wav2vec2_large", "whisper_large_v3", "ast_audioset", "w2v_bert_2", "whisper_small",
                                    "bud_e_whisper", "whisper_medium"))
        self.assertEqual({m: frame_features.FRAME_RATES[m] for m in ENCODERS},
                         {**{m: 50 for m in ENCODERS}, "ast_audioset": 10})
        for model, layer, width in (("whisper_large_v3", 31, 1280), ("ast_audioset", 8, 768), ("w2v_bert_2", 24, 1024),
                                    ("whisper_small", 12, 768), ("bud_e_whisper", 12, 768), ("whisper_medium", 24, 1024)):
            s = validate(FramesSettings, {"model": model, "layer": layer, "normalization": "global"}, "features")
            frame_features.check(s)
            self.assertEqual(frame_features.width(s), width)
            self.assertEqual(frame_features.feature_set(s), f"{model}_l{layer}_frames")
        for model, layer in (("ast_audioset", 13), ("w2v_bert_2", 25), ("whisper_small", 13), ("bud_e_whisper", 13),
                             ("whisper_medium", 25)):
            s = validate(FramesSettings, {"model": model, "layer": layer, "normalization": "global"}, "features")
            with self.assertRaises(ValueError):
                frame_features.check(s)
        with self.assertRaises(ValueError):
            validate(FramesSettings, {"model": "w2v_bert", "layer": 3, "normalization": "global"}, "features")

    def test_extract_frames_refuses(self):
        from affect.classical.pretrained import extract_frames

        for model, layers in (("whisper", (3,)), ("ast_audioset", (13,)), ("w2v_bert_2", (25,)), ("w2v_bert_2", (3, 3))):
            with self.assertRaises(ValueError):
                extract_frames("iemocap", {"label_set": "four_merged", "subset": "all"}, model, layers,
                               Path("/nonexistent"), "cpu", print)


class TestInputsAndSnapshots(unittest.TestCase):
    """Without torch (added 2026-11-02): w2v-BERT 2.0's inputs must come with an attention mask that marks a prefix of
    the frames; a spectrogram encoder's snapshot must hold that model (w2v-BERT 2.0 has the Large waveform encoders'
    layers and width, so nothing else would catch a WavLM Large snapshot given for it)."""

    def test_inputs(self):
        from affect.classical.pretrained import _model_inputs

        features = np.zeros((1, 5, 160), dtype=np.float32)
        with self.assertRaises(ValueError):
            _model_inputs({"input_features": features}, "cpu")
        for bad in ([[1, 1, 0, 1, 0]], [[0, 0, 0, 0, 0]], [[1, 1, 1, 1]]):
            with self.assertRaises(ValueError):
                _model_inputs({"input_features": features, "attention_mask": np.array(bad)}, "cpu")

    def test_snapshot_types(self):
        from affect.classical.pretrained import check_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "config.json").write_text('{"model_type": "wavlm"}')
            with self.assertRaises(ValueError):
                check_snapshot("w2v_bert_2", folder)
            check_snapshot("wavlm_large", folder)            # the waveform encoders are not checked
            (folder / "config.json").write_text('{"model_type": "wav2vec2-bert"}')
            check_snapshot("w2v_bert_2", folder)
            with self.assertRaises(ValueError):
                check_snapshot("ast_audioset", folder)


def _tiny_whisper(folder: Path):
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

    # the default vocabulary keeps the default special tokens inside it (a small vocabulary would put the padding token
    # outside the decoder's embedding); the weights are saved in float16, as whisper-large-v3's are, so the cast to
    # float32 is tested too
    config = WhisperConfig(num_mel_bins=80, encoder_layers=2, encoder_attention_heads=2, decoder_layers=1,
                           decoder_attention_heads=2, d_model=32, encoder_ffn_dim=64, decoder_ffn_dim=64,
                           max_source_positions=100, max_target_positions=16)
    WhisperModel(config).eval().half().save_pretrained(str(folder))
    WhisperFeatureExtractor(feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=2,
                            n_fft=400).save_pretrained(str(folder))
    return 3, 32, 32000            # hidden states, width, window samples (2 s)


def _tiny_ast(folder: Path):
    from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

    config = ASTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       patch_size=16, frequency_stride=10, time_stride=10, max_length=100, num_mel_bins=128)
    ASTModel(config).eval().save_pretrained(str(folder))
    ASTFeatureExtractor(feature_size=1, sampling_rate=16000, num_mel_bins=128, max_length=100,
                        do_normalize=True).save_pretrained(str(folder))
    return 3, 32, 400 + 160 * 99    # hidden states, width, window samples (16,240)


def _tiny_w2v_bert(folder: Path):
    """A tiny w2v-BERT 2.0 (added 2026-11-02): its feature extractor as facebook/w2v-bert-2.0's (80 Kaldi filterbank
    bands, frames stacked in pairs) and a Conformer of 2 layers of 32 with random weights, saved in float16 so that the
    cast to float32 is tested too; no layer dropping or masking (they act only in training anyway)."""
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel

    config = Wav2Vec2BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                feature_projection_input_dim=160, conv_depthwise_kernel_size=3,
                                position_embeddings_type="relative_key", left_max_position_embeddings=8,
                                right_max_position_embeddings=2, layerdrop=0.0, mask_time_prob=0.0, add_adapter=False)
    Wav2Vec2BertModel(config).eval().half().save_pretrained(str(folder))
    SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=16000, padding_value=0.0,
                                stride=2).save_pretrained(str(folder))
    return 3, 32                    # hidden states, width


def _samples(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    wave = 0.3 * np.sin(2 * np.pi * (180 + 40 * seed) * t) + 0.05 * rng.normal(size=n)
    return np.clip(wave * 32768.0, -32768, 32767).astype(np.int16)


@needs_module("torch")
@needs_module("transformers")
class TestTinyEncoders(unittest.TestCase):
    def _by_hand_whisper(self, folder, samples):
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel

        from affect.classical.pretrained import pieces

        extractor = WhisperFeatureExtractor.from_pretrained(str(folder))
        encoder = WhisperModel.from_pretrained(str(folder)).float().get_encoder().eval()
        wave = samples.astype(np.float32) / 32768.0
        parts, everything = None, None
        for a, b in pieces(len(wave), int(extractor.n_samples)):
            x = extractor(wave[a:b], sampling_rate=16000, return_tensors="pt")["input_features"]
            with torch.inference_mode():
                hs = encoder(x, output_hidden_states=True).hidden_states
            real = min(100, -(-(b - a) // 320))
            kept = [h[0, :real].double().numpy() for h in hs]
            full = [h[0].double().numpy() for h in hs]
            parts = kept if parts is None else [np.concatenate([p, k]) for p, k in zip(parts, kept)]
            everything = full if everything is None else [np.concatenate([p, k]) for p, k in zip(everything, full)]
        return parts, everything

    def _by_hand_ast(self, folder, samples):
        import torch
        from transformers import ASTFeatureExtractor, ASTModel

        from affect.classical.pretrained import pieces

        extractor = ASTFeatureExtractor.from_pretrained(str(folder))
        model = ASTModel.from_pretrained(str(folder)).eval()
        wave = samples.astype(np.float32) / 32768.0
        f_dim, t_dim = (128 - 16) // 10 + 1, (100 - 16) // 10 + 1
        parts, everything = None, None
        for a, b in pieces(len(wave), 400 + 160 * 99):
            x = extractor(wave[a:b], sampling_rate=16000, return_tensors="pt")["input_values"]
            with torch.inference_mode():
                hs = model(x, output_hidden_states=True).hidden_states
            self.assertEqual(hs[0].shape[1], 2 + f_dim * t_dim)
            frames = 1 + ((b - a) - 400) // 160
            real_t = min(t_dim, -(-frames // 10))
            keep = [2 + f * t_dim + t for f in range(f_dim) for t in range(real_t)]
            kept = [h[0].double().numpy()[keep] for h in hs]
            full = [h[0].double().numpy() for h in hs]
            parts = kept if parts is None else [np.concatenate([p, k]) for p, k in zip(parts, kept)]
            everything = full if everything is None else [np.concatenate([p, k]) for p, k in zip(everything, full)]
        return parts, everything

    def _check(self, make, by_hand, lengths):
        from affect.classical.pretrained import load_encoder

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            n_layers, width, window = make(folder)
            encode, info = load_encoder(folder, "cpu", "mean")
            self.assertEqual((info["layers"], info["hidden_size"]), (n_layers, width))
            self.assertEqual(info["feature_extractor"]["sampling_rate"], 16000)
            self.assertEqual(info["feature_extractor"]["window_samples"], window)
            for i, n in enumerate(lengths):
                samples = _samples(n, i)
                got = encode(samples)
                self.assertEqual(got.shape, (n_layers, width))
                self.assertEqual(got.dtype, np.float32)
                kept, everything = by_hand(folder, samples)
                want = np.stack([k.mean(axis=0) for k in kept])
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
                if n < window:     # one piece: the padding's outputs are not in the mean
                    every = np.stack([e.mean(axis=0) for e in everything])
                    self.assertGreater(np.abs(every - want).max(), 1e-4)
            encode_sd, _ = load_encoder(folder, "cpu", "mean_sd")
            samples = _samples(lengths[0], 0)
            got = encode_sd(samples)
            kept, _ = by_hand(folder, samples)
            want = np.stack([np.concatenate([k.mean(axis=0), k.std(axis=0)]) for k in kept])
            self.assertEqual(got.shape, (n_layers, 2 * width))
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

    def test_whisper(self):
        # 0.7 s (one piece), 1.9 s (one piece, near the 2 s window), 4.5 s (three pieces of 1.5 s)
        self._check(_tiny_whisper, self._by_hand_whisper, (11200, 30400, 72000))

    def test_whisper_frames(self):
        from affect.classical.pretrained import load_encoder, load_whisper_frame_encoder, pieces

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            n_layers, width, window = _tiny_whisper(folder)
            encode, info = load_whisper_frame_encoder(folder, "cpu", (2, 1))
            self.assertEqual((info["layers"], info["hidden_size"], info["frames_of_layers"]), (n_layers, width, [2, 1]))
            self.assertEqual(info["feature_extractor"]["sampling_rate"], 16000)
            probe, _ = load_encoder(folder, "cpu", "mean")
            for i, n in enumerate((11200, 30400, 72000)):     # one piece, one piece near the window, three pieces
                samples = _samples(n, i)
                got = encode(samples)
                self.assertEqual(len(got), 2)
                kept, _ = self._by_hand_whisper(folder, samples)
                frames = sum(min(100, -(-(b - a) // 320)) for a, b in pieces(n, window))
                for j, k in enumerate((2, 1)):
                    self.assertEqual(got[j].dtype, np.float16)
                    self.assertEqual(got[j].shape, (frames, width))
                    np.testing.assert_allclose(got[j].astype(np.float64), kept[k], rtol=2e-3, atol=2e-3)
                    np.testing.assert_allclose(got[j].astype(np.float64).mean(axis=0), probe(samples)[k],
                                               rtol=2e-3, atol=2e-3)
            with self.assertRaises(ValueError):
                load_whisper_frame_encoder(folder, "cpu", (3,))

    def test_whisper_frames_refuses_ast(self):
        from affect.classical.pretrained import load_whisper_frame_encoder

        with tempfile.TemporaryDirectory() as tmp:
            _tiny_ast(Path(tmp))
            with self.assertRaises(ValueError):
                load_whisper_frame_encoder(Path(tmp), "cpu", (1,))

    def test_ast(self):
        # 0.5 s (one piece), 1.0 s (one piece), 2.5 s (three pieces)
        self._check(_tiny_ast, self._by_hand_ast, (8000, 16000, 40000))

    def test_ast_token_order(self):
        """Independently of the index formula: the patch tokens of the embedding output, less their position
        embeddings, are the patch projection of the spectrogram laid out as [frequency, time] and read row by row, so
        token 2 + f x t_dim + t holds the patch at frequency row f and time column t."""
        import torch
        from transformers import ASTFeatureExtractor, ASTModel

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _tiny_ast(folder)
            extractor = ASTFeatureExtractor.from_pretrained(str(folder))
            model = ASTModel.from_pretrained(str(folder)).float().eval()
            x = extractor(_samples(8000, 3).astype(np.float32) / 32768.0, sampling_rate=16000,
                          return_tensors="pt")["input_values"]
            with torch.inference_mode():
                tokens = model(x, output_hidden_states=True).hidden_states[0][0]
                grid = model.embeddings.patch_embeddings.projection(x.unsqueeze(1).transpose(2, 3))[0]   # [H, f, t]
                positions = model.embeddings.position_embeddings[0].detach()   # a view of a parameter keeps requires_grad even in inference mode
            f_dim, t_dim = grid.shape[1], grid.shape[2]
            self.assertEqual((f_dim, t_dim), ((128 - 16) // 10 + 1, (100 - 16) // 10 + 1))
            self.assertEqual(tokens.shape[0], 2 + f_dim * t_dim)
            for f, t in ((0, 0), (0, 1), (1, 0), (3, 4), (f_dim - 1, t_dim - 1)):
                np.testing.assert_allclose((tokens[2 + f * t_dim + t] - positions[2 + f * t_dim + t]).numpy(),
                                           grid[:, f, t].numpy(), rtol=1e-4, atol=1e-5)

    def _by_hand_w2v_bert(self, folder, samples):
        """(the real frames of every layer, every frame of every layer, the filterbank's frame count) of one utterance,
        straight from the extractor and the model."""
        import torch
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

        extractor = SeamlessM4TFeatureExtractor.from_pretrained(str(folder))
        model = Wav2Vec2BertModel.from_pretrained(str(folder)).float().eval()
        inputs = extractor(samples.astype(np.float32) / 32768.0, sampling_rate=16000, return_tensors="pt")
        mask = inputs["attention_mask"][0].numpy()
        with torch.inference_mode():
            hs = model(inputs["input_features"], attention_mask=inputs["attention_mask"],
                       output_hidden_states=True).hidden_states
        real = int(mask.sum())
        self.assertTrue((mask[:real] == 1).all() and (mask[real:] == 0).all(), mask)
        fbank = 1 + (len(samples) - 400) // 160
        self.assertEqual(real, fbank // 2, (len(samples), fbank, mask.shape))
        self.assertEqual(hs[0].shape[1], mask.shape[0])
        return [h[0, :real].double().numpy() for h in hs], [h[0].double().numpy() for h in hs], fbank

    def test_w2v_bert(self):
        from affect.classical.pretrained import load_encoder

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            n_layers, width = _tiny_w2v_bert(folder)
            encode, info = load_encoder(folder, "cpu", "mean")
            self.assertEqual((info["layers"], info["hidden_size"]), (n_layers, width))
            self.assertEqual(info["feature_extractor"]["sampling_rate"], 16000)
            self.assertIn(info["checkpoint_dtype_as_loaded"], ("torch.float16", "torch.float32"))   # by version
            seen_odd = False
            # 0.71 s (69 filterbank frames, odd: one padded), 0.70 s (68, even), 3.01 s (299, odd)
            for i, n in enumerate((11360, 11200, 48160)):
                samples = _samples(n, i)
                got = encode(samples)
                self.assertEqual(got.shape, (n_layers, width))
                self.assertEqual(got.dtype, np.float32)
                kept, everything, fbank = self._by_hand_w2v_bert(folder, samples)
                want = np.stack([k.mean(axis=0) for k in kept])
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
                if fbank % 2:      # the padded pair's output is not in the mean
                    seen_odd = True
                    every = np.stack([e.mean(axis=0) for e in everything])
                    self.assertGreater(np.abs(every - want).max(), 1e-5)
                else:
                    self.assertEqual(everything[0].shape[0], kept[0].shape[0])
            self.assertTrue(seen_odd)
            encode_sd, _ = load_encoder(folder, "cpu", "mean_sd")
            samples = _samples(11360, 0)
            kept, _, _ = self._by_hand_w2v_bert(folder, samples)
            want = np.stack([np.concatenate([k.mean(axis=0), k.std(axis=0)]) for k in kept])
            np.testing.assert_allclose(encode_sd(samples), want, rtol=1e-4, atol=1e-5)

    def test_w2v_bert_frames(self):
        from affect.classical.pretrained import load_encoder, load_frame_encoder

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            n_layers, width = _tiny_w2v_bert(folder)
            encode, info = load_frame_encoder(folder, "cpu", (2, 0))
            self.assertEqual((info["layers"], info["hidden_size"], info["frames_of_layers"]), (n_layers, width, [2, 0]))
            probe, _ = load_encoder(folder, "cpu", "mean")
            for i, n in enumerate((11360, 11200, 48160)):
                samples = _samples(n, i)
                got = encode(samples)
                kept, _, fbank = self._by_hand_w2v_bert(folder, samples)
                self.assertEqual(len(got), 2)
                for j, k in enumerate((2, 0)):
                    self.assertEqual(got[j].dtype, np.float16)
                    self.assertEqual(got[j].shape, (fbank // 2, width))
                    np.testing.assert_allclose(got[j].astype(np.float64), kept[k], rtol=2e-3, atol=2e-3)
                    np.testing.assert_allclose(got[j].astype(np.float64).mean(axis=0), probe(samples)[k],
                                               rtol=2e-3, atol=2e-3)
            with self.assertRaises(ValueError):
                load_frame_encoder(folder, "cpu", (3,))

    def test_ast_frames(self):
        from affect.classical.pretrained import load_ast_frame_encoder, load_encoder, pieces

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            n_layers, width, window = _tiny_ast(folder)
            encode, info = load_ast_frame_encoder(folder, "cpu", (2, 1))
            self.assertEqual((info["layers"], info["hidden_size"], info["frames_of_layers"]), (n_layers, width, [2, 1]))
            self.assertEqual(info["frame_rate"], 10.0)
            probe, _ = load_encoder(folder, "cpu", "mean")
            f_dim, t_dim = (128 - 16) // 10 + 1, (100 - 16) // 10 + 1
            for i, n in enumerate((8000, 16000, 40000)):     # one piece, one piece, three pieces
                samples = _samples(n, i)
                got = encode(samples)
                kept, _ = self._by_hand_ast(folder, samples)
                columns = [min(t_dim, -(-(1 + ((b - a) - 400) // 160) // 10)) for a, b in pieces(n, window)]
                want = {}
                for k in (2, 1):                 # the kept tokens, piece by piece: f_dim rows of that piece's columns
                    rows, start = [], 0
                    for c in columns:
                        block = kept[k][start:start + f_dim * c].reshape(f_dim, c, width)
                        rows.append(block.mean(axis=0))
                        start += f_dim * c
                    self.assertEqual(start, kept[k].shape[0])
                    want[k] = np.concatenate(rows)
                for j, k in enumerate((2, 1)):
                    self.assertEqual(got[j].dtype, np.float16)
                    self.assertEqual(got[j].shape, (sum(columns), width))
                    np.testing.assert_allclose(got[j].astype(np.float64), want[k], rtol=2e-3, atol=2e-3)
                    np.testing.assert_allclose(got[j].astype(np.float64).mean(axis=0), probe(samples)[k],
                                               rtol=2e-3, atol=2e-3)
            with self.assertRaises(ValueError):
                load_ast_frame_encoder(folder, "cpu", (3,))
        with tempfile.TemporaryDirectory() as tmp:
            _tiny_whisper(Path(tmp))
            with self.assertRaises(ValueError):
                load_ast_frame_encoder(Path(tmp), "cpu", (1,))

    def test_refuses_short_piece(self):
        from affect.classical.pretrained import load_encoder

        with tempfile.TemporaryDirectory() as tmp:
            _tiny_ast(Path(tmp))
            encode, _ = load_encoder(Path(tmp), "cpu", "mean")
            with self.assertRaises(ValueError):
                encode(_samples(300, 0))


if __name__ == "__main__":
    unittest.main()
