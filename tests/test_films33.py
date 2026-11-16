"""Week 33's films the other way (added 2026-11-10): models trained on EmoFilm's film folds, tested on IEMOCAP.

Without torch: the settings (external_test iemocap and warm_start iemocap need data emofilm in its three classes with
split film_5fold, a normalization other than speaker, and ten folds for the warm start; the old values still work as
before), and the configuration hashes of the ten week-32 records the week's chains start from are what the node gave
them (artifacts/sweeps/<id>/jobs.csv), so the new code changes no existing configuration.
With torch (the node): iemocap_external_files and iemocap_warm_files on a stand-in IEMOCAP corpus of ten speakers:
every utterance is scored once, each speaker's by a model adapted on the other nine; the reported numbers equal those
computed from the CSV; the model's weights come back unchanged; a second call gives the same file byte for byte; the
adapted scores differ from the cold ones. A whole job (train_job) refitted on the stand-in EmoFilm's film folds with
both tests writes its test fold's clips and the four IEMOCAP files, and repeats exactly.
With the assembled corpora (the node): IEMOCAP as a test corpus holds its 3,823 utterances of the three classes, ten
speakers; EmoFilm's film folds hold its 726 clips, 55 films, each film in one fold.
"""
import csv
import io
import json
import unittest

import numpy as np

from affect import catalog
from tests.support import REPO, needs_assembled, needs_module
from tests.test_emofilm import TempEmoFilm

catalog.load_all()

RECORDS = ("heads32w_pool_spk", "heads32w_rnn_spk", "heads32w_transformer_spk", "heads32w_lif_spk", "spk32_base",
           "spk32_cnn", "spk32_transformer", "spk32_threeblock", "spk32_lif", "spk32_crnn")
CLASSES = ("hap+exc", "ang", "sad")


def film_settings(**train) -> dict:
    """The tuned CNN's record (sweeps/spk32_cnn.json) on EmoFilm's film folds, refitted, with both tests on IEMOCAP,
    small enough for a CPU: these train settings on top."""
    from affect.runs.spec import plain, validate_job

    settings = plain(validate_job(json.loads((REPO / "sweeps" / "spk32_cnn.json").read_text())["base"]))
    settings["data"] = {"name": "emofilm", "label_set": "three_no_neutral", "subset": "all", "split": "film_5fold"}
    settings["train"].update({"refit": "train_and_val", "refit_epochs": 2, "external_test": "iemocap",
                              "warm_start": "iemocap", "warm_start_folds": 10, "warm_start_epochs": 2,
                              "warm_start_lr": 3e-5, **train})
    return settings


class TestSettings(unittest.TestCase):
    def test_checks(self):
        from affect.runs.spec import check_job, validate_job

        check_job(validate_job(film_settings()), 16000)
        cold_only = film_settings(warm_start="none", warm_start_folds=None, warm_start_epochs=None,
                                  warm_start_lr=None)
        check_job(validate_job(cold_only), 16000)
        on_iemocap = film_settings()
        on_iemocap["data"] = {"name": "iemocap", "label_set": "three_no_neutral", "subset": "all",
                              "split": "leave_one_speaker_out"}
        with self.assertRaisesRegex(ValueError, "data emofilm"):
            check_job(validate_job(on_iemocap), 16000)
        no_folds = film_settings()
        no_folds["data"]["split"] = "none"
        with self.assertRaisesRegex(ValueError, "film_5fold"):
            check_job(validate_job(no_folds), 16000)
        with self.assertRaisesRegex(ValueError, "ten speakers"):
            check_job(validate_job(film_settings(warm_start_folds=5)), 16000)
        with self.assertRaisesRegex(ValueError, "needs external_test"):
            check_job(validate_job(film_settings(external_test="emofilm")), 16000)
        films_on_films = film_settings(external_test="emofilm", warm_start="none", warm_start_folds=None,
                                       warm_start_epochs=None, warm_start_lr=None)
        with self.assertRaisesRegex(ValueError, "IEMOCAP label set"):
            check_job(validate_job(films_on_films), 16000)
        by_speaker = film_settings()
        by_speaker["features"]["normalization"] = "speaker"
        with self.assertRaises(ValueError):
            check_job(validate_job(by_speaker), 16000)

    def test_the_records_keep_their_hashes(self):
        from affect.sweep.design import expand

        folder = REPO / "artifacts" / "sweeps"
        present = [r for r in RECORDS if (folder / r / "jobs.csv").is_file()]
        if not present:
            self.skipTest("the records' artifacts/sweeps folders are not here")
        for record in present:
            jobs, _ = expand(json.loads((folder / record / "sweep_file.json").read_text()))
            with open(folder / record / "jobs.csv", newline="") as handle:
                queued = list(csv.DictReader(handle))
            self.assertEqual([j["run_id"] for j in jobs], [q["run_id"] for q in queued], record)
            self.assertEqual({j["config_hash"] for j in jobs}, {q["config_hash"] for q in queued}, record)


def stand_in_iemocap(per_speaker: int = 6):
    """IEMOCAP as a test corpus, ten speakers of per_speaker utterances, each class and kind on every speaker; the
    samples carry the class in their level."""
    from affect.data.base import Utterance
    from affect.train.data import External

    rng = np.random.default_rng(5)
    level = {"hap+exc": 3000, "ang": 6000, "sad": 1000}
    utterances, audio = [], []
    for s in range(1, 6):
        for g in ("F", "M"):
            for k in range(per_speaker):
                label = CLASSES[k % 3]
                n = 3200 + 160 * (k % 4)
                dialog = f"Ses0{s}{g}_impro0{1 + k % 2}"
                utterances.append(Utterance(
                    utt_id=f"{dialog}_{g}{k:03d}", dataset="iemocap", speaker=f"Ses0{s}_{g}", fold_group=f"S{s}",
                    dialog=dialog, kind="impro" if k % 2 == 0 else "script", raw_label=label, label=label,
                    audio_path="x.wav", n_frames=n, sample_rate=16000, transcript=None, extra={"speaker_gender": g}))
                audio.append(rng.normal(0, level[label], n).clip(-32768, 32767).astype(np.int16))
    utterances, audio = zip(*sorted(zip(utterances, audio), key=lambda pair: pair[0].utt_id))
    return External(dataset="iemocap", utterances=list(utterances), audio=list(audio),
                    labels=np.array([CLASSES.index(u.label) for u in utterances], dtype=np.int64),
                    class_names=CLASSES, manifest_sha256="0" * 64, sample_rate=16000)


@needs_module("torch")
class TestTorch(unittest.TestCase):
    def parts(self):
        import torch

        from affect.features.logmel import LogMel, LogMelSettings
        from affect.models.cnn import CNNSettings, build_cnn
        from affect.runs.spec import validate_job

        settings = film_settings(batch_size=4, crop_s=0.1, eval_batch_size=4, precision="fp32",
                                 max_batches_per_epoch=None, warm_start_lr=1e-3)
        settings["augment"].update({"freq_masks": 0, "freq_mask_width": 0, "time_masks": 0, "time_mask_width": 0,
                                    "mixup_alpha": 0.0})
        job = validate_job(settings)
        torch.manual_seed(0)
        model = build_cnn(CNNSettings(channels=(8, 16), convs_per_block=1, kernel_size=3, pool_freq=2, pool_time=2,
                                      pooling="mean", classifier_hidden=None, dropout=0.1, norm="batch",
                                      norm_groups=None, norm_subbands=None, activation="relu"), 40, 3).eval()
        feats = LogMel(LogMelSettings(win_ms=25.0, hop_ms=10.0, n_fft=512, n_mels=40, f_min=0.0, f_max=8000.0,
                                      log_offset=1e-6, normalization="utterance"), 16000, torch.device("cpu"))
        return job, model, feats

    def test_cold_and_warm(self):
        import torch

        from affect.train.loop import Normalizer, iemocap_external_files, iemocap_warm_files

        ext = stand_in_iemocap()
        job, model, feats = self.parts()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        cpu = torch.device("cpu")
        cold, cold_text, _ = iemocap_external_files(ext, model, feats, Normalizer("utterance"), job.train, None, cpu)
        metrics, text, _ = iemocap_warm_files(ext, model, job, feats, Normalizer("utterance"), None, cpu, 7,
                                              lambda m: None)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        _, again, _ = iemocap_warm_files(ext, model, job, feats, Normalizer("utterance"), None, cpu, 7,
                                         lambda m: None)
        self.assertEqual(again, text)
        for numbers, table in ((cold, cold_text), (metrics, text)):
            rows = list(csv.DictReader(io.StringIO(table)))
            self.assertEqual([r["utt_id"] for r in rows], [u.utt_id for u in ext.utterances])
            self.assertEqual([r["speaker"] for r in rows], [u.speaker for u in ext.utterances])
            self.assertEqual([r["kind"] for r in rows], [u.kind for u in ext.utterances])
            probs = np.array([[float(r[f"prob_{c}"]) for c in CLASSES] for r in rows])
            pred = probs.argmax(axis=1)
            self.assertEqual([r["pred"] for r in rows], [CLASSES[k] for k in pred])
            recalls = [np.mean(pred[ext.labels == c] == c) for c in range(3)]
            self.assertAlmostEqual(numbers["ua"], float(np.mean(recalls)), places=5)
            self.assertEqual(numbers["n"], 60)
            self.assertEqual(sorted(numbers["by_speaker"]), sorted({u.speaker for u in ext.utterances}))
            self.assertEqual(sorted(numbers["by_kind"]), ["impro", "script"])
            self.assertEqual(sorted(numbers["by_gender"]), ["F", "M"])
            self.assertEqual(numbers["dataset"], "iemocap")
        self.assertEqual(metrics["warm_start"]["utterances_per_speaker"], [6] * 10)
        self.assertEqual(len(metrics["warm_start"]["curves"]), 10 * 2)
        self.assertEqual(metrics["warm_start"]["speakers"], sorted({u.speaker for u in ext.utterances}))
        cold_probs = np.array([[float(r[f"prob_{c}"]) for c in CLASSES]
                               for r in csv.DictReader(io.StringIO(cold_text))])
        warm_probs = np.array([[float(r[f"prob_{c}"]) for c in CLASSES] for r in csv.DictReader(io.StringIO(text))])
        self.assertGreater(np.abs(cold_probs - warm_probs).max(), 1e-4, "the adapted models score like the cold one")

    def test_a_whole_job(self):
        import torch

        from affect.runs.spec import config_hash, plain, run_id, validate_job
        from affect.train import data as train_data
        from affect.train.loop import train_job

        settings = film_settings(batch_size=4, crop_s=0.1, eval_batch_size=8, precision="fp32",
                                 max_batches_per_epoch=None, epochs=3, warmup_epochs=0, early_stopping_patience=1,
                                 refit_epochs=2, warm_start_epochs=1, warm_start_lr=1e-3)
        settings["model"].update({"channels": [8, 16]})
        settings["augment"].update({"freq_masks": 0, "freq_mask_width": 0, "time_masks": 0, "time_mask_width": 0,
                                    "mixup_alpha": 0.0})
        job = validate_job(settings)
        spec = {"sweep_id": "test_films33", "stage": "refit", "label": "base", "config_hash": config_hash(job),
                "run_id": run_id(job, "test_F1", 3), "fold": "test_F1", "seed": 3, "settings": plain(job)}
        outputs = []
        with TempEmoFilm(per_cell=3, films=12):
            train_data._PREPARED.clear()
            train_data._EXTERNAL.clear()
            train_data._EXTERNAL[("iemocap", "three_no_neutral")] = stand_in_iemocap()
            try:
                for _ in range(2):
                    outputs.append(train_job(json.loads(json.dumps(spec)), torch.device("cpu"), lambda m: None))
                prepared = train_data.prepare("emofilm", job.data)
            finally:
                train_data._PREPARED.clear()
                train_data._EXTERNAL.clear()
        files = outputs[0]
        for name in ("metrics_iemocap.json", "predictions_iemocap.csv", "metrics_iemocap_warm.json",
                     "predictions_iemocap_warm.csv", "predictions_test.csv", "metrics_val.json"):
            self.assertIn(name, files)
        fold = next(f for f in prepared.folds if f.name == "test_F1")
        tested = [r["utt_id"] for r in csv.DictReader(io.StringIO(files["predictions_test.csv"]))]
        self.assertEqual(sorted(tested), sorted(fold.test))
        self.assertEqual(files["metrics_val.json"]["refit"]["epochs"], 2)
        self.assertEqual(files["metrics_iemocap.json"]["n"], 60)
        self.assertEqual(files["metrics_iemocap_warm.json"]["n"], 60)
        self.assertEqual(files["provenance.json"]["external_test"]["dataset"], "iemocap")
        for name in ("predictions_test.csv", "predictions_iemocap.csv", "predictions_iemocap_warm.csv"):
            self.assertEqual(outputs[1][name], files[name], f"{name} repeats")


class TestRealCorpora(unittest.TestCase):
    @needs_assembled("iemocap")
    def test_iemocap_as_a_test_corpus(self):
        from affect.train import data as train_data

        train_data._EXTERNAL.clear()
        ext = train_data.prepare_external("iemocap", "three_no_neutral")
        try:
            self.assertEqual(len(ext.utterances), 3823)
            self.assertEqual(ext.class_names, CLASSES)
            self.assertEqual(len({u.speaker for u in ext.utterances}), 10)
            self.assertEqual(np.bincount(ext.labels).tolist(), [1636, 1103, 1084])
        finally:
            train_data._EXTERNAL.clear()

    @needs_assembled("iemocap")
    def test_improvised_speaker_folds(self):
        from affect.data.splits import assign
        from affect.registry import DATASETS
        from affect.settings import validate

        entry = DATASETS.get("iemocap")
        settings = validate(entry.settings_class, {"label_set": "four_merged", "subset": "all",
                                                   "split": "leave_one_speaker_out_train_impro"}, "data")
        utterances = entry.target().utterances(settings)
        by_id = {u.utt_id: u for u in utterances}
        plain = {f.name: f for f in entry.target().folds(validate(
            entry.settings_class, {"label_set": "four_merged", "subset": "all", "split": "leave_one_speaker_out"},
            "data"), utterances)}
        lines = []
        for f in entry.target().folds(settings, utterances):
            roles = assign(utterances, f)
            self.assertEqual({by_id[i].kind for i in roles["train"] + roles["val"]}, {"impro"})
            self.assertEqual(sorted(roles["test"]), sorted(plain[f.name].test))
            angry = sum(by_id[i].label == "ang" for i in roles["val"])
            lines.append(f"{f.name}: train {len(roles['train'])}, val {len(roles['val'])} ({angry} angry), "
                         f"test {len(roles['test'])}")
        self.assertEqual(sum(by_id[i].kind == "impro" for i in by_id), 2943)
        print("improvised-only speaker folds:\n  " + "\n  ".join(lines))

    @needs_assembled("emofilm")
    def test_emofilm_film_folds(self):
        from affect.data.splits import assign
        from affect.registry import DATASETS
        from affect.settings import validate

        entry = DATASETS.get("emofilm")
        settings = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all",
                                                   "split": "film_5fold"}, "data")
        utterances = entry.target().utterances(settings)
        folds = entry.target().folds(settings, utterances)
        self.assertEqual(len(utterances), 726)
        self.assertEqual(len({u.extra["film"] for u in utterances}), 55)
        tested = []
        for f in folds:
            roles = assign(utterances, f)
            tested += roles["test"]
            films = {r: {u.extra["film"] for u in utterances if u.utt_id in set(roles[r])} for r in roles}
            self.assertFalse(films["test"] & (films["train"] | films["val"]))
        self.assertEqual(sorted(tested), sorted(u.utt_id for u in utterances))


    @needs_assembled("emofilm")
    def test_emofilm_strat_film_folds(self):
        # added 2026-11-13 (RULE_week33_addendum5.txt): on the real corpus, film_5fold's film fold F3 holds no sadness
        # clip; film_5fold_strat's five folds each hold all three classes on both their validation and test sides
        from affect.data.emofilm import film_5fold
        from affect.data.splits import assign
        from affect.registry import DATASETS
        from affect.settings import validate

        entry = DATASETS.get("emofilm")
        settings = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all",
                                                   "split": "film_5fold_strat"}, "data")
        utterances = entry.target().utterances(settings)
        self.assertEqual((len(utterances), len({u.extra["film"] for u in utterances})), (726, 55))
        old = {f.name: f for f in film_5fold(utterances)}
        self.assertFalse(any(u.label == "sad" for u in utterances if u.utt_id in old["test_F3"].test),
                         "film_5fold's F3 holds no sadness clip")
        by_id = {u.utt_id: u for u in utterances}
        counts, tested = {}, []
        for f in entry.target().folds(settings, utterances):
            roles = assign(utterances, f)
            tested += roles["test"]
            films = {r: {by_id[i].extra["film"] for i in roles[r]} for r in roles}
            self.assertFalse(films["test"] & (films["train"] | films["val"]) or films["train"] & films["val"])
            for r in ("val", "test"):
                self.assertEqual({by_id[i].label for i in roles[r]}, set(CLASSES), f"{f.name} {r}")
            counts[f.name] = tuple(sum(by_id[i].label == c for i in roles["test"]) for c in CLASSES)
        self.assertEqual(sorted(tested), sorted(by_id))
        self.assertEqual(counts, {"test_F1": (30, 58, 89), "test_F2": (36, 39, 40), "test_F3": (42, 45, 64),
                                  "test_F4": (76, 59, 24), "test_F5": (56, 31, 37)})
        print("film_5fold_strat folds (hap+exc, ang, sad):\n  " + "\n  ".join(f"{k[5:]}: {v}" for k, v in counts.items()))

    @needs_assembled("emofilm")
    def test_emofilm_whole_film_folds(self):
        # added 2026-11-14 (RULE_week33_addendum8.txt): on the real corpus, FILMS.csv is the film list SAME_FILM was
        # reviewed against; the 55 film IDs are 37 films; film_5fold_strat puts 11 films in more than one fold,
        # film_5fold_whole none, and its five folds each hold all three classes on both sides, with the counts of
        # RULE_week33_addendum8.txt part 2
        from affect import paths
        from affect.data.emofilm import check_whole_films, film_5fold_strat, read_films, whole_film
        from affect.data.splits import assign
        from affect.registry import DATASETS
        from affect.settings import validate

        titles = read_films(paths.assembled_dir("emofilm"))
        check_whole_films(titles)
        entry = DATASETS.get("emofilm")
        settings = validate(entry.settings_class, {"label_set": "three_no_neutral", "subset": "all",
                                                   "split": "film_5fold_whole"}, "data")
        utterances = entry.target().utterances(settings)
        self.assertEqual((len(utterances), len({u.extra["film"] for u in utterances}),
                          len({whole_film(u.extra["film"]) for u in utterances})), (726, 55, 37))
        fold_of = {}
        for f in film_5fold_strat(utterances):
            for i in f.test:
                fold_of[i] = f.name
        split = {w for w in {whole_film(u.extra["film"]) for u in utterances}
                 if len({fold_of[u.utt_id] for u in utterances if whole_film(u.extra["film"]) == w}) > 1}
        self.assertEqual(len(split), 11, "film_5fold_strat puts 11 films in more than one fold")
        by_id = {u.utt_id: u for u in utterances}
        counts, tested = {}, []
        for f in entry.target().folds(settings, utterances):
            roles = assign(utterances, f)
            tested += roles["test"]
            films = {r: {whole_film(by_id[i].extra["film"]) for i in roles[r]} for r in roles}
            self.assertFalse(films["test"] & (films["train"] | films["val"]) or films["train"] & films["val"])
            for r in ("val", "test"):
                self.assertEqual({by_id[i].label for i in roles[r]}, set(CLASSES), f"{f.name} {r}")
            counts[f.name] = tuple(sum(by_id[i].label == c for i in roles["test"]) for c in CLASSES)
        self.assertEqual(sorted(tested), sorted(by_id))
        self.assertEqual(counts, {"test_F1": (73, 16, 61), "test_F2": (51, 58, 59), "test_F3": (57, 32, 33),
                                  "test_F4": (38, 34, 37), "test_F5": (21, 92, 64)})
        print("film_5fold_whole folds (hap+exc, ang, sad):\n  " + "\n  ".join(f"{k[5:]}: {v}" for k, v in counts.items()))


if __name__ == "__main__":
    unittest.main()
