"""A pretrained encoder's frames as the input of trained heads (features pretrained_frames, models frames_*, added
2026-10-26). Without torch: the extraction's store (a stand-in encoder on the stand-in EmoFilm clips), the store's
reader and its refusals, the feature settings and the job checks, the crops and evaluation batches of frames (values
kept, padding zero, the same random draws as for samples), and job_source. With torch (the node): every head's output
for a padded row equals its output alone, the summary widths, the feature callable, a training step through Step and
predict over frames."""
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from affect import catalog, paths
from tests.support import REPO, needs_module
from tests.synthetic_emofilm import write_assembly

catalog.load_all()
WIDTH = 1024


class TempRepo:
    """A throwaway repository root holding the stand-in EmoFilm assembly (tests/synthetic_emofilm.py)."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src" / "affect").mkdir(parents=True)
        (root / "src" / "affect" / "__init__.py").write_text("# stand-in package\n")
        self.rows = write_assembly(root / "data" / "emofilm" / "assembled")
        self.snapshot = root / "snapshot_0123abcd"
        self.snapshot.mkdir()
        (self.snapshot / "model.safetensors").write_bytes(b"stand-in weights")
        self.saved = paths.repo_root
        paths.repo_root = lambda: root
        from affect.features import frames
        frames._STORES.clear()
        return self

    def __exit__(self, *exc):
        from affect.features import frames
        frames._STORES.clear()
        paths.repo_root = self.saved
        self.tmp.cleanup()


def fake_encoder(layers):
    """A stand-in for load_frame_encoder: frames = samples // 320, each frame's values a function of the utterance's
    first sample, the frame and the layer, so every stored value can be checked."""
    def encode(samples):
        n = max(1, samples.shape[0] // 320)
        base = float(samples[0]) / 32768.0
        return [(base + k + np.arange(n)[:, None] * 0.001 + np.zeros((n, WIDTH))).astype(np.float16) for k in layers]
    info = {"feature_extractor": {"sampling_rate": 16000}, "layers": 25, "hidden_size": WIDTH, "stand_in": True}
    return encode, info


def frames_settings(layer=20, normalization="none"):
    from affect.features.frames import FramesSettings

    return FramesSettings(model="wavlm_large", layer=layer, normalization=normalization)


POOL = {"name": "frames_pool", "time_pool": 1, "input_dropout": 0.1, "proj_dim": 64, "activation": "gelu",
        "pooling": "attention", "classifier_hidden": None, "dropout": 0.2}
TRANSFORMER = {"name": "frames_transformer", "time_pool": 2, "input_dropout": 0.1, "d_model": 32, "n_heads": 4,
               "n_layers": 2, "ff_dim": 64, "transformer_dropout": 0.1, "transformer_activation": "gelu",
               "norm_first": True, "positional": "sinusoidal", "pooling": "stats", "classifier_hidden": 16,
               "activation": "relu", "dropout": 0.2}
LIF = {"name": "frames_lif", "time_pool": 2, "input_dropout": 0.0, "proj_dim": 64, "activation": "relu",
       "lif_hidden": [48, 32], "recurrent": True, "lif_norm": "layer", "lif_dropout": 0.1, "beta": 0.9,
       "learn_beta": "layer", "threshold": 1.0, "reset": "subtract", "detach_reset": False, "surrogate": "atan",
       "surrogate_slope": 2.0, "readout_beta": 0.5, "rate_weight": 0.5, "rate_target": 0.1, "dropout": 0.2}
RNN = {"name": "frames_rnn", "time_pool": 1, "input_dropout": 0.1, "proj_dim": None, "activation": "relu",
       "rnn": "gru", "rnn_hidden": 24, "rnn_layers": 2, "bidirectional": True, "rnn_dropout": 0.1,
       "pooling": "attention", "classifier_hidden": None, "dropout": 0.2}


def job_settings(model: dict, normalization="none", **train) -> dict:
    """The CNN screen's base settings with these frames features and head."""
    from affect.runs.spec import plain, validate_job

    out = plain(validate_job(json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"]))
    out["features"] = {"name": "pretrained_frames", "model": "wavlm_large", "layer": 20, "normalization": normalization}
    out["model"] = dict(model)
    out["augment"].update({"freq_masks": 0, "freq_mask_width": 0})
    out["train"].update(train)
    return out


class TestStore(unittest.TestCase):
    def test_extract_and_read(self):
        from affect.classical.pretrained import extract_frames
        from affect.features import frames
        from affect.registry import DATASETS
        from affect.settings import validate

        with TempRepo() as repo:
            written = extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "wavlm_large",
                                     (20, 12), repo.snapshot, "cpu", lambda m: None, encoder=fake_encoder((20, 12)))
            self.assertEqual([Path(w).name for w in written], ["wavlm_large_l20_frames", "wavlm_large_l12_frames"])
            record = json.loads((Path(written[0]) / "record.json").read_text())
            self.assertEqual((record["layer"], record["width"], record["dtype"], record["frame_rate"]),
                             (20, WIDTH, "float16", 50))
            entry = DATASETS.get("emofilm")
            utts = entry.target().utterances(validate(entry.settings_class, {"label_set": "three_no_neutral",
                                                                              "subset": "all"}, "data"))
            self.assertEqual(record["n_utterances"], len(utts))
            got = frames.source("emofilm", frames_settings(20), list(reversed(utts)))
            encode, _ = fake_encoder((20,))
            from affect.train.data import read_int16
            root = paths.assembled_dir("emofilm")
            for u, a in zip(reversed(utts), got):
                expected = encode(read_int16(root / u.audio_path, u.n_frames))[0]
                self.assertEqual(a.dtype, np.float16)
                self.assertTrue(np.array_equal(a, expected), u.utt_id)
            self.assertEqual(sum(a.shape[0] for a in got), record["total_frames"])
            l12 = frames.source("emofilm", frames_settings(12), utts[:2])
            self.assertTrue(np.allclose(l12[0].astype(np.float32) - got[-1].astype(np.float32), -8, atol=0.05))
            with self.assertRaises(FileExistsError):
                extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "wavlm_large", (20,),
                               repo.snapshot, "cpu", lambda m: None, encoder=fake_encoder((20,)))
            with self.assertRaises(FileNotFoundError):
                frames.source("emofilm", frames_settings(3), utts)
            with self.assertRaises(ValueError):
                frames.source("emofilm", frames_settings(20), [types.SimpleNamespace(utt_id="not_there")])
            with self.assertRaises(ValueError):
                extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "wavlm_large", (25,),
                               repo.snapshot, "cpu", lambda m: None, encoder=fake_encoder((25,)))

    def test_whisper_store(self):
        """Whisper large-v3's frames (added 2026-11-01): written and read as a waveform encoder's are, with its width
        (1280) and frame rate (50). An encoder whose width is not the table's is refused before anything is written
        (here Whisper's width for AST)."""
        from affect.classical.pretrained import extract_frames
        from affect.features import frames
        from affect.features.frames import FramesSettings
        from affect.registry import DATASETS
        from affect.settings import validate

        def whisper_encoder(layers):
            def encode(samples):
                n = max(1, -(-samples.shape[0] // 320))
                return [(float(samples[0]) / 32768.0 + k + np.zeros((n, 1280))).astype(np.float16) for k in layers]
            return encode, {"feature_extractor": {"sampling_rate": 16000}, "layers": 33, "hidden_size": 1280}

        with TempRepo() as repo:
            written = extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "whisper_large_v3",
                                     (31,), repo.snapshot, "cpu", lambda m: None, encoder=whisper_encoder((31,)))
            self.assertEqual([Path(w).name for w in written], ["whisper_large_v3_l31_frames"])
            record = json.loads((Path(written[0]) / "record.json").read_text())
            self.assertEqual((record["model"], record["layer"], record["width"], record["frame_rate"]),
                             ("whisper_large_v3", 31, 1280, 50))
            entry = DATASETS.get("emofilm")
            utts = entry.target().utterances(validate(entry.settings_class, {"label_set": "three_no_neutral",
                                                                              "subset": "all"}, "data"))
            settings = FramesSettings(model="whisper_large_v3", layer=31, normalization="none")
            self.assertEqual(frames.width(settings), 1280)
            got = frames.source("emofilm", settings, utts[:3])
            self.assertEqual([a.shape[1] for a in got], [1280, 1280, 1280])
            with self.assertRaises(ValueError):
                extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "ast_audioset", (3,),
                               repo.snapshot, "cpu", lambda m: None, encoder=whisper_encoder((3,)))

    def test_ast_and_w2v_bert_stores(self):
        """AST's frames (added 2026-11-02): width 768, 10 frames a second, written and read like the others; an encoder
        that says another frame rate is refused. w2v-BERT 2.0's: width 1024, 50 a second."""
        from affect.classical.pretrained import extract_frames
        from affect.features import frames
        from affect.features.frames import FramesSettings
        from affect.registry import DATASETS
        from affect.settings import validate

        def stand_in(layers, width, n_layers, per_second, said_rate=None):
            def encode(samples):
                n = max(1, int(samples.shape[0] * per_second // 16000))
                return [(float(samples[0]) / 32768.0 + k + np.zeros((n, width))).astype(np.float16) for k in layers]
            info = {"feature_extractor": {"sampling_rate": 16000}, "layers": n_layers, "hidden_size": width}
            if said_rate is not None:
                info["frame_rate"] = said_rate
            return encode, info

        data = {"label_set": "three_no_neutral", "subset": "all"}
        with TempRepo() as repo:
            with self.assertRaises(ValueError):
                extract_frames("emofilm", data, "ast_audioset", (8,), repo.snapshot, "cpu", lambda m: None,
                               encoder=stand_in((8,), 768, 13, 10, said_rate=50.0))
            written = extract_frames("emofilm", data, "ast_audioset", (8,), repo.snapshot, "cpu", lambda m: None,
                                     encoder=stand_in((8,), 768, 13, 10, said_rate=10.0))
            record = json.loads((Path(written[0]) / "record.json").read_text())
            self.assertEqual((record["model"], record["layer"], record["width"], record["frame_rate"]),
                             ("ast_audioset", 8, 768, 10))
            written = extract_frames("emofilm", data, "w2v_bert_2", (12,), repo.snapshot, "cpu", lambda m: None,
                                     encoder=stand_in((12,), 1024, 25, 50))
            record = json.loads((Path(written[0]) / "record.json").read_text())
            self.assertEqual((record["model"], record["layer"], record["width"], record["frame_rate"]),
                             ("w2v_bert_2", 12, 1024, 50))
            entry = DATASETS.get("emofilm")
            utts = entry.target().utterances(validate(entry.settings_class, data, "data"))
            for model, layer, width in (("ast_audioset", 8, 768), ("w2v_bert_2", 12, 1024)):
                settings = FramesSettings(model=model, layer=layer, normalization="none")
                got = frames.source("emofilm", settings, utts[:3])
                self.assertEqual([a.shape[1] for a in got], [width] * 3)
                self.assertEqual(frames.frame_rate(settings), 10 if model == "ast_audioset" else 50)

    def test_a_store_that_disagrees_is_refused(self):
        from affect.classical.pretrained import extract_frames
        from affect.features import frames

        with TempRepo() as repo:
            written = extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "wavlm_large",
                                     (20,), repo.snapshot, "cpu", lambda m: None, encoder=fake_encoder((20,)))
            offsets = np.load(Path(written[0]) / "offsets.npy")
            np.save(Path(written[0]) / "offsets.npy", offsets[:-1])
            with self.assertRaises(ValueError):
                frames.load_store("emofilm", frames_settings(20))


class TestChecks(unittest.TestCase):
    def test_feature_settings(self):
        from affect.features import frames

        frames.check(frames_settings(0))
        frames.check(frames_settings(24))
        with self.assertRaises(ValueError):
            frames.check(frames_settings(25))
        self.assertEqual((frames.width(frames_settings()), frames.frame_rate(frames_settings()),
                          frames.feature_set(frames_settings())), (WIDTH, 50, "wavlm_large_l20_frames"))

    def test_jobs(self):
        from affect.runs.spec import check_job, input_width, validate_job

        for model in (POOL, TRANSFORMER, LIF, RNN):
            job = validate_job(job_settings(model))
            check_job(job, 16000)
            self.assertEqual(input_width(job), WIDTH)
        refused = [job_settings(POOL), job_settings(POOL, normalization="speaker"), job_settings(POOL),
                   job_settings({**POOL, "proj_dim": 0}), job_settings({**TRANSFORMER, "n_heads": 5}),
                   job_settings({**RNN, "rnn_layers": 1}), job_settings({**LIF, "lif_hidden": []})]
        refused[0]["augment"]["speed_factors"] = [0.9, 1.0, 1.1]
        # the external test reads the corpus's frames since 2026-11-05, but never with speaker normalization
        refused[1]["data"]["label_set"] = "three_no_neutral"
        refused[1]["train"]["external_test"] = "emofilm"
        refused[2]["model"] = json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"]["model"]
        for normalization in ("utterance", "global", "none"):
            films = job_settings(POOL, normalization=normalization, external_test="emofilm", warm_start="emofilm",
                                 warm_start_folds=5, warm_start_epochs=10, warm_start_lr=5e-5)
            self.assertEqual(films["data"]["label_set"], "four_merged")     # scored on EmoFilm's three classes
            check_job(validate_job(films), 16000)
            films["train"].update({"refit": "train_and_val", "refit_epochs": 3})
            check_job(validate_job(films), 16000)
        for k, settings in enumerate(refused):
            with self.assertRaises(ValueError, msg=str(k)):
                check_job(validate_job(settings), 16000)
        logmel_with_head = json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"]
        logmel_with_head["model"] = dict(POOL)
        with self.assertRaises(ValueError):
            check_job(validate_job(logmel_with_head), 16000)


class TestBatches(unittest.TestCase):
    def test_crops_and_evaluation_batches_of_frames(self):
        from affect.train.data import crop_arrays, eval_batches, segment_batches

        rng = np.random.default_rng(0)
        frames = [rng.standard_normal((n, 6)).astype(np.float16) for n in (30, 12, 50)]
        samples = [rng.integers(-3000, 3000, size=n).astype(np.int16) for n in (30, 12, 50)]
        waves, valid = crop_arrays(frames, 20, np.random.default_rng(5))
        waves_s, valid_s = crop_arrays(samples, 20, np.random.default_rng(5))
        self.assertEqual(waves.shape, (3, 20, 6))
        self.assertEqual(waves.dtype, np.float32)
        self.assertTrue(np.array_equal(valid, valid_s), "the same crops are drawn for the same lengths")
        draws = np.random.default_rng(5)
        start0 = int(draws.integers(0, 30 - 20 + 1))
        self.assertTrue(np.array_equal(waves[0], frames[0][start0:start0 + 20].astype(np.float32)), "values kept")
        self.assertTrue(np.array_equal(waves[1, :12], frames[1].astype(np.float32)))
        self.assertTrue((waves[1, 12:] == 0).all())
        self.assertTrue(np.allclose(waves_s[1, :12], samples[1] / 32768.0), "samples still scaled")
        rows = np.arange(3)
        batches = list(eval_batches(frames, rows, 2))
        self.assertEqual([b[0].tolist() for b in batches], [[2, 0], [1]])
        positions, w, lengths = batches[0]
        self.assertEqual(w.shape, (2, 50, 6))
        self.assertTrue(np.array_equal(w[1, :30], frames[0].astype(np.float32)))
        self.assertTrue((w[1, 30:] == 0).all())
        segs = list(segment_batches(frames, rows, 4, 20, 10))
        total = sum(len(p) for p, _, _ in segs)
        self.assertEqual(total, 2 + 1 + 4)      # 30 frames: starts 0, 10; 12: one; 50: 0, 10, 20, 30
        for p, w, lengths in segs:
            self.assertEqual(w.shape[2], 6)

    def test_job_source(self):
        from affect.runs.spec import validate_job
        from affect.train.data import job_source

        with TempRepo():
            from affect.classical.pretrained import extract_frames
            from affect.registry import DATASETS
            from affect.settings import validate

            extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "wavlm_large", (20,),
                           Path(paths.repo_root()) / "snapshot_0123abcd", "cpu", lambda m: None,
                           encoder=fake_encoder((20,)))
            entry = DATASETS.get("emofilm")
            utts = entry.target().utterances(validate(entry.settings_class, {"label_set": "three_no_neutral",
                                                                              "subset": "all"}, "data"))
            job = validate_job(job_settings(POOL))
            prepared = types.SimpleNamespace(dataset="emofilm", utterances=utts, audio=["samples"], sample_rate=16000)
            source, rate = job_source(job, prepared)
            self.assertEqual((len(source), rate, source[0].shape[1]), (len(utts), 50, WIDTH))
            logmel = validate_job(json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())["base"])
            self.assertEqual(job_source(logmel, prepared), (["samples"], 16000))

    def test_external_source(self):
        """The external test of a frames job reads the corpus's frames of the job's encoder and layer at the frame rate
        (added 2026-11-05); a log-mel job's corpus is the same object at the sample rate."""
        from affect.features import frames
        from affect.runs.spec import validate_job
        from affect.train import data as train_data

        with TempRepo() as repo:
            from affect.classical.pretrained import extract_frames

            extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "wavlm_large", (20,),
                           repo.snapshot, "cpu", lambda m: None, encoder=fake_encoder((20,)))
            train_data._EXTERNAL.clear()
            try:
                ext = train_data.prepare_external("emofilm", "three_no_neutral")
                job = validate_job(job_settings(POOL, external_test="emofilm"))
                read, rate = train_data.external_source(job, ext)
                self.assertEqual(rate, 50)
                expected = frames.source("emofilm", job.features, ext.utterances)
                self.assertEqual(len(read.audio), len(ext.utterances))
                for a, b in zip(read.audio, expected):
                    self.assertTrue(np.array_equal(a, b))
                    self.assertEqual((a.dtype, a.shape[1]), (np.float16, WIDTH))
                self.assertTrue(all(a.dtype == np.int16 for a in ext.audio), "the corpus itself keeps its samples")
                self.assertTrue(np.array_equal(read.labels, ext.labels))
                self.assertEqual((read.class_names, read.sample_rate, read.manifest_sha256, read.utterances),
                                 (ext.class_names, ext.sample_rate, ext.manifest_sha256, ext.utterances))
                from affect.runs.spec import plain
                logmel_settings = plain(validate_job(json.loads((REPO / "sweeps" / "cnn_screen_1.json").read_text())
                                                     ["base"]))
                logmel_settings["train"]["external_test"] = "emofilm"
                logmel_settings["features"]["normalization"] = "utterance"
                same, rate = train_data.external_source(validate_job(logmel_settings), ext)
                self.assertIs(same, ext)
                self.assertEqual(rate, 16000)
                other_layer_settings = job_settings(POOL, external_test="emofilm")
                other_layer_settings["features"]["layer"] = 12
                with self.assertRaises(FileNotFoundError):        # no frames of layer 12 were extracted
                    train_data.external_source(validate_job(other_layer_settings), ext)
            finally:
                train_data._EXTERNAL.clear()


@needs_module("torch")
class TestHeads(unittest.TestCase):
    def build(self, model: dict):
        import torch

        from affect.registry import MODELS

        entry, settings = MODELS.validate_section(model, "model")
        torch.manual_seed(0)
        return entry.target(settings, WIDTH, 4).eval()

    def test_padding_changes_nothing(self):
        import torch

        rng = np.random.default_rng(1)
        x = torch.from_numpy(rng.standard_normal((3, WIDTH, 23)).astype(np.float32))
        valid = torch.tensor([23, 9, 2])
        for model in (POOL, {**POOL, "pooling": "stats", "proj_dim": None}, TRANSFORMER,
                      {**TRANSFORMER, "positional": "none", "norm_first": False, "pooling": "mean"}, LIF,
                      {**LIF, "lif_norm": "batch", "recurrent": False, "time_pool": 1}, RNN,
                      {**RNN, "rnn": "lstm", "pooling": "last", "bidirectional": False}):
            net = self.build(model)
            padded = x.clone()
            for k, n in enumerate(valid.tolist()):
                padded[k, :, n:] = 7.0                   # whatever the padding holds
            with torch.no_grad():
                together = net(padded, valid)
                self.assertEqual(tuple(together.shape), (3, 4), model["name"])
                for k, n in enumerate(valid.tolist()):
                    alone = net(x[k:k + 1, :, :n], valid[k:k + 1])
                    self.assertTrue(torch.allclose(together[k], alone[0], atol=1e-4), (model["name"], k))
            self.assertEqual(net.embed(x, valid).shape[1], net.embedding_dim, model["name"])

    def test_widths(self):
        self.assertEqual(self.build(POOL).embedding_dim, 64)
        self.assertEqual(self.build({**POOL, "proj_dim": None, "pooling": "meanmax"}).embedding_dim, 2 * WIDTH)
        self.assertEqual(self.build(TRANSFORMER).embedding_dim, 64)
        self.assertEqual(self.build(LIF).embedding_dim, 32)
        self.assertEqual(self.build(RNN).embedding_dim, 48)

    def test_feature_callable_step_and_predict(self):
        import torch

        from affect.features.frames import FrameFeatures
        from affect.runs.spec import validate_job
        from affect.train.loop import Normalizer, Step, predict

        rng = np.random.default_rng(2)
        source = [rng.standard_normal((n, WIDTH)).astype(np.float16) for n in (40, 25, 60, 33)]
        for model in (POOL, TRANSFORMER, LIF, RNN):
            job = validate_job(job_settings(model, normalization="utterance", precision="fp32"))
            feats = FrameFeatures(job.features, 16000, torch.device("cpu"))
            self.assertEqual(feats.n_mels, WIDTH)
            self.assertEqual(tuple(feats(torch.zeros(2, 5, WIDTH)).shape), (2, WIDTH, 5))
            net = self.build(model).train()
            from affect.train.data import training_waves
            waves, lengths = training_waves(source, np.arange(4), 30, (1.0,), np.random.default_rng(3),
                                            np.random.default_rng(4))
            step = Step(job, net, None, feats, Normalizer("utterance"), None, torch.device("cpu"))
            loss, emotion, _, _ = step.loss(waves, lengths, np.arange(4), np.eye(4, dtype=np.float32), None,
                                            torch.Generator().manual_seed(1), np.random.default_rng(5))
            self.assertTrue(bool(torch.isfinite(loss)), model["name"])
            loss.backward()
            net.eval()
            together = predict(net, feats, Normalizer("utterance"), source, np.arange(4), 3, torch.device("cpu"),
                               "fp32")
            alone = predict(net, feats, Normalizer("utterance"), source, np.arange(4), 1, torch.device("cpu"), "fp32")
            self.assertEqual(together.probs.shape, (4, 4))
            self.assertTrue(np.allclose(together.probs, alone.probs, atol=1e-4), model["name"])
            segments = predict(net, feats, Normalizer("utterance"), source, np.arange(4), 4, torch.device("cpu"),
                               "fp32", (20, 10))
            self.assertTrue(np.allclose(segments.probs.sum(axis=1), 1.0), model["name"])

    def test_films_on_frames(self):
        """The external test and the warm start of a head on the stand-in EmoFilm's frames (added 2026-11-05): every
        clip scored once, cold and adapted, each adapted clip in its film's fold; the head's weights come back
        unchanged; a second warm start gives the same file byte for byte; the crops count frames (crop_s at 50 a
        second); adapted scores differ from the cold ones."""
        import csv
        import io

        import torch

        from affect.classical.finetune import film_folds
        from affect.features.frames import FrameFeatures
        from affect.runs.spec import validate_job
        from affect.train import data as train_data
        from affect.train.loop import Normalizer, external_files, warm_start_files

        model_classes = ("neu", "hap+exc", "ang", "sad")
        with TempRepo() as repo:
            from affect.classical.pretrained import extract_frames

            extract_frames("emofilm", {"label_set": "three_no_neutral", "subset": "all"}, "wavlm_large", (20,),
                           repo.snapshot, "cpu", lambda m: None, encoder=fake_encoder((20,)))
            train_data._EXTERNAL.clear()
            try:
                ext = train_data.prepare_external("emofilm", "three_no_neutral")
                settings = job_settings(POOL, normalization="none", external_test="emofilm",
                                        warm_start="emofilm", warm_start_folds=3, warm_start_epochs=2,
                                        warm_start_lr=1e-3, batch_size=4, crop_s=0.06, eval_batch_size=4,
                                        precision="fp32", refit="train_and_val", refit_epochs=2)
                settings["augment"].update({"time_masks": 0, "time_mask_width": 0, "mixup_alpha": 0.0})
                job = validate_job(settings)
                read, rate = train_data.external_source(job, ext)
                self.assertEqual(rate, 50)
                feats = FrameFeatures(job.features, 16000, torch.device("cpu"))
                net = self.build(POOL)
                before = {k: v.detach().clone() for k, v in net.state_dict().items()}
                cold, cold_text, _ = external_files(read, net, feats, Normalizer("none"), job.train, None,
                                                    torch.device("cpu"), model_classes)
                metrics, text, _ = warm_start_files(read, net, job, feats, Normalizer("none"), None,
                                                    torch.device("cpu"), 7, model_classes, lambda m: None, rate=rate)
                for k, v in net.state_dict().items():
                    self.assertTrue(torch.equal(v, before[k]), k)
                again = warm_start_files(read, net, job, feats, Normalizer("none"), None, torch.device("cpu"), 7,
                                         model_classes, lambda m: None, rate=rate)[1]
                self.assertEqual(again, text)
            finally:
                train_data._EXTERNAL.clear()
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([r["utt_id"] for r in rows], [u.utt_id for u in ext.utterances])
        expected = film_folds([u.extra["film"] for u in ext.utterances], 3)
        self.assertTrue(all(int(r["film_fold"]) == expected[r["film"]] for r in rows))
        self.assertEqual((metrics["n"], cold["n"]), (36, 36))
        self.assertEqual(metrics["warm_start"]["crop_s"], 0.06)
        self.assertIn("strict", metrics)
        cold_rows = list(csv.DictReader(io.StringIO(cold_text)))
        probs = np.array([[float(r[f"prob_{c}"]) for c in model_classes] for r in rows])
        cold_probs = np.array([[float(r[f"prob_{c}"]) for c in model_classes] for r in cold_rows])
        self.assertGreater(np.abs(cold_probs - probs).max(), 1e-4, "the adapted heads score like the cold one")


if __name__ == "__main__":
    unittest.main()
