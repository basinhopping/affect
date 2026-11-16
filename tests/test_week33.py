"""affect.sweep.week33, week 33's automatic stages (added 2026-11-09), in a throwaway root with stand-in records and
stand-in run folders (spec.json, metrics_val.json and predictions_val.csv, no test file). The first pass writes one
validation sweep per model and setup (ten models in the two speaker-specific setups, the six networks in the two pooled
ones), each the record with only its data settings and the refit switched off, seeds 11 to 15 (speaker folds, 50 jobs)
or 6 to 10 (pooled, 25 jobs); nothing more is written while a validation sweep lacks a run or has a failed job; then the
refit is the same settings with only train.refit, train.refit_epochs and train.eval_train changed, for the median of the
validation runs' best epochs rounded half up; a second pass writes nothing new; queued sweeps and marked smoke failures
are not listed to queue, and the rest are listed highest priority first; a chain whose record is missing is reported as
an error and the others go on; a sweep whose run IDs exist already is refused.

Addendum 1 (added 2026-11-10): the first pass also writes the validation sweeps of the three setups it adds (impro4 and
film for the ten models, wmel4 for the six networks), each the record with only its setup's settings changed (wmel4: the
features whole and the frequency masks' width, scaled to 128 bands); the film refit adds the tests on IEMOCAP, cold and
adapted speaker by speaker, at a tenth of the record's peak rate; the four setups of RULE_week33.txt are written as
before.
Addendum 5 (added 2026-11-13): the film chain's validation sweep is w33_<tag>_film2v on split film_5fold_strat, its
refit keeps w33_<tag>_film; the superseded w33_<tag>_filmv sweeps, queued with failed jobs, are left as they are: never
written again, never listed to queue, never the source of a refit, and listed by status as superseded.
Addendum 8 (added 2026-11-14): the film chain's validation sweep is w33_<tag>_film3v on split film_5fold_whole; the
w33_<tag>_film2v sweeps of addendum 5, written on the node and never queued, are superseded too: left as they are,
never listed to queue (although written and not queued), never the source of a refit, and listed by status."""
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from affect import catalog, paths
from affect.data.iemocap_splits import SPEAKER_OUT, UTTERANCE_5, fold_names
from affect.sweep import week33 as W
from tests.support import REPO

catalog.load_all()

HEAD_SOURCES = {"wpool": "pool", "wrnn": "rnn", "wtransformer": "transformer", "wlif": "lif"}
MAIN = ("spk4", "spk3", "pooled4", "pooled3")          # RULE_week33.txt part 2
ADDED = ("impro4", "film", "wmel4")                    # RULE_week33_addendum1.txt


def record_sweep(tag: str) -> dict:
    """A stand-in for the queued week-32 speaker-specific refit of one model: the networks' own sweep files
    (sweeps/spk32_<net>.json), and for the heads their week-31 records (sweeps/heads31w_<head>_so4.json) moved to
    speaker folds and refitted, as week 32 made heads32w_<head>_spk."""
    source = W.MODELS[tag][1]
    if tag in HEAD_SOURCES:
        sweep = json.loads((REPO / "sweeps" / f"heads31w_{HEAD_SOURCES[tag]}_so4.json").read_text())
        base = sweep["base"]
        base["data"]["split"] = SPEAKER_OUT
        base["train"].update({"refit": "train_and_val", "refit_epochs": 9, "eval_train": "none"})
        return {"sweep_id": source, "description": "stand-in", "folds": list(fold_names(SPEAKER_OUT)), "base": base,
                "stages": [{"name": "refit", "kind": "base", "priority": 192, "seeds": [6, 7, 8, 9, 10]}]}
    return json.loads((REPO / "sweeps" / f"{source}.json").read_text())


class Root:
    """A throwaway repository root (affect.paths.repo_root) with the ten records queued, as enqueue writes them."""

    def __init__(self, skip=()):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        for d in ("artifacts/sweeps", "artifacts/runs", "queue/failed"):
            (self.path / d).mkdir(parents=True)
        self.saved = paths.repo_root
        paths.repo_root = lambda: self.path
        W._EXISTING = None
        for tag in W.MODELS:
            if tag not in skip:
                self.enqueue(record_sweep(tag))

    def close(self):
        paths.repo_root = self.saved
        W._EXISTING = None
        self.tmp.cleanup()

    def enqueue(self, sweep: dict) -> list:
        jobs, _ = W.expand(sweep)
        folder = self.path / "artifacts" / "sweeps" / sweep["sweep_id"]
        folder.mkdir(parents=True)
        (folder / "sweep_file.json").write_text(json.dumps(sweep))
        with open(folder / "jobs.csv", "w", newline="") as handle:
            w = csv.writer(handle, lineterminator="\n")
            w.writerow(["seq", "run_id", "stage", "label", "priority", "fold", "seed", "config_hash"])
            for seq, j in enumerate(jobs):
                w.writerow([seq, j["run_id"], j["stage"], j["label"], j["priority"], j["fold"], j["seed"],
                            j["config_hash"]])
        return jobs

    def enqueue_file(self, sweep_id: str) -> list:
        return self.enqueue(json.loads((W.out_dir() / "sweeps" / f"{sweep_id}.json").read_text()))

    def finish(self, jobs: list, epoch_of=lambda j: 12, skip=()):
        for j in jobs:
            if j["run_id"] in skip:
                continue
            d = self.path / "artifacts" / "runs" / j["sweep_id"] / j["run_id"]
            d.mkdir(parents=True)
            (d / "spec.json").write_text(json.dumps({k: j[k] for k in ("sweep_id", "stage", "label", "config_hash",
                                                                         "run_id", "fold", "seed", "settings")}))
            ua = 0.5 + 0.01 * (int(j["seed"]) % 5)
            (d / "metrics_val.json").write_text(json.dumps({"best_epoch": epoch_of(j), "best_val_ua": ua,
                                                            "final": {"ua": ua}}))
            (d / "predictions_val.csv").write_text("utt_id,label,pred\n")


def settings_of(sweep: dict) -> dict:
    return W.plain(W.validate_job(sweep["base"]))


class TestWeek33(unittest.TestCase):
    def setUp(self):
        self.root = Root()

    def tearDown(self):
        self.root.close()

    def advance(self) -> dict:
        return {(tag, setup): W.advance_chain(tag, setup) for tag, setup in W.chains()}

    def test_the_first_pass_writes_every_validation_sweep_and_no_refit(self):
        lines = self.advance()
        self.assertEqual(len(lines), 10 * 2 + 6 * 2 + 10 + 10 + 6)
        written = sorted(p.stem for p in (W.out_dir() / "sweeps").glob("*.json"))
        self.assertEqual(len(written), 58)
        self.assertTrue(all(name.endswith("v") for name in written))
        self.assertEqual(tuple(W.SETUPS), MAIN + ADDED)
        for tag, setup in W.chains():
            if setup not in MAIN:
                continue
            v_id, r_id = W.ids(tag, setup)
            self.assertFalse(W.written(r_id))
            sweep = json.loads((W.out_dir() / "sweeps" / f"{v_id}.json").read_text())
            dotted, _, seeds, _, p_val, _ = W.SETUPS[setup]
            jobs, summary = W.expand(sweep)
            n_folds = 10 if setup.startswith("spk") else 5
            self.assertEqual((len(jobs), summary["configs"]), (n_folds * 5, 1))
            self.assertEqual({j["priority"] for j in jobs}, {p_val})
            self.assertEqual(sorted({int(j["seed"]) for j in jobs}), seeds)
            record = W.record_settings(tag)
            ran = settings_of(sweep)
            self.assertLessEqual(set(W.changed(record, ran)), set(dotted))
            self.assertEqual(ran["train"]["refit"], "none")
            self.assertIsNone(ran["train"]["refit_epochs"])
            self.assertEqual(ran["data"]["split"], SPEAKER_OUT if setup.startswith("spk") else UTTERANCE_5)
            self.assertEqual(ran["data"]["label_set"], "three_no_neutral" if setup.endswith("3") else "four_merged")
            # the record itself: the queued speaker-specific refit with only the refit switched off
            source = settings_of(json.loads((self.root.path / "artifacts" / "sweeps" / W.MODELS[tag][1] /
                                             "sweep_file.json").read_text()))
            self.assertEqual(W.changed(source, record), ["train.refit", "train.refit_epochs"])
        self.assertNotIn(("wpool", "pooled4"), W.chains())
        self.assertIn(("cnn", "pooled3"), W.chains())

    def test_the_first_pass_of_the_added_setups(self):
        self.advance()
        for tag, setup in W.chains():
            if setup in MAIN:
                continue
            v_id, r_id = W.ids(tag, setup)
            self.assertFalse(W.written(r_id))
            sweep = json.loads((W.out_dir() / "sweeps" / f"{v_id}.json").read_text())
            dotted, _, seeds, models, p_val, _ = W.SETUPS[setup]
            self.assertIn(tag, models)
            jobs, summary = W.expand(sweep)
            n_folds = 5 if setup == "film" else 10
            self.assertEqual((len(jobs), summary["configs"]), (n_folds * 5, 1))
            self.assertEqual({j["priority"] for j in jobs}, {p_val})
            self.assertEqual(sorted({int(j["seed"]) for j in jobs}), [11, 12, 13, 14, 15])
            self.assertIn("RULE_week33_addendum1.txt", sweep["description"])
            record = W.record_settings(tag)
            ran = settings_of(sweep)
            self.assertEqual((ran["train"]["refit"], ran["train"]["refit_epochs"]), ("none", None))
            self.assertEqual(ran["train"]["external_test"], "none")
            if setup == "impro4":
                self.assertEqual(W.changed(record, ran), sorted(k for k in dotted if W.changed(
                    record, W.with_overrides(record, {k: dotted[k]}))))
                self.assertEqual((ran["data"]["split"], ran["data"]["label_set"]),
                                 ("leave_one_speaker_out_train_impro", "four_merged"))
                self.assertEqual(sweep["folds"], list(fold_names(SPEAKER_OUT)))
            elif setup == "film":
                self.assertEqual(ran["data"], {"name": "emofilm", "label_set": "three_no_neutral", "subset": "all",
                                               "split": "film_5fold_whole"})      # addendum 8 (2026-11-14)
                self.assertEqual(v_id, f"w33_{tag}_film3v")
                self.assertIn("RULE_week33_addendum8.txt", sweep["description"])
                self.assertEqual(sweep["folds"], ["test_F1", "test_F2", "test_F3", "test_F4", "test_F5"])
                self.assertLessEqual(set(W.changed(record, ran)), set(dotted))
                self.assertEqual({k for k in W.changed(record, ran)} - {"data.label_set", "data.split", "data.name"},
                                 set())
            else:
                self.assertIn(tag, W.NETWORKS)
                self.assertEqual(ran["features"], {"name": "whisper_logmel", "n_mels": 128,
                                                   "normalization": record["features"]["normalization"]})
                width, bands = record["augment"]["freq_mask_width"], record["features"]["n_mels"]
                self.assertEqual(ran["augment"]["freq_mask_width"], int(width * 128 / bands + 0.5))
                rest_before = {k: v for k, v in record.items() if k != "features"}
                rest_after = {k: v for k, v in ran.items() if k != "features"}
                self.assertLessEqual(set(W.changed(rest_before, rest_after)),
                                     {"data.split", "data.label_set", "augment.freq_mask_width"})
                self.assertEqual(ran["data"]["split"], SPEAKER_OUT)
        widths = {tag: settings_of(json.loads((W.out_dir() / "sweeps" / f"{W.ids(tag, 'wmel4')[0]}.json")
                                              .read_text()))["augment"]["freq_mask_width"] for tag in W.NETWORKS}
        self.assertEqual(widths, {"base": 0, "cnn": 26, "transformer": 32, "threeblock": 38, "lif": 16, "crnn": 32})
        self.assertNotIn(("wpool", "wmel4"), W.chains())

    def test_the_film_refit_tests_on_iemocap(self):
        self.advance()
        for tag in ("wtransformer", "lif"):
            v_id, r_id = W.ids(tag, "film")
            jobs = self.root.enqueue_file(v_id)
            self.assertEqual(len(jobs), 25)
            self.root.finish(jobs, epoch_of=lambda j: 4 + jobs.index(j) % 3)
            W.advance_chain(tag, "film")
            refit = json.loads((W.out_dir() / "sweeps" / f"{r_id}.json").read_text())
            before = settings_of(json.loads((W.out_dir() / "sweeps" / f"{v_id}.json").read_text()))
            after = settings_of(refit)
            self.assertEqual(W.changed(before, after), sorted(
                ["train.refit", "train.refit_epochs", "train.external_test", "train.warm_start",
                 "train.warm_start_folds", "train.warm_start_epochs", "train.warm_start_lr"]))
            self.assertEqual((after["train"]["refit"], after["train"]["refit_epochs"]), ("train_and_val", 5))
            self.assertEqual((after["train"]["external_test"], after["train"]["warm_start"]), ("iemocap", "iemocap"))
            self.assertEqual((after["train"]["warm_start_folds"], after["train"]["warm_start_epochs"]), (10, 10))
            self.assertAlmostEqual(after["train"]["warm_start_lr"] * 10, before["train"]["lr"], places=15)
            self.assertEqual(after["train"]["warm_start_lr"], float(f"{before['train']['lr'] / 10:.6g}"))
            self.assertEqual(after["data"]["name"], "emofilm")
            r_jobs, _ = W.expand(refit)
            self.assertEqual((len(r_jobs), {j["priority"] for j in r_jobs}), (25, {179}))
            self.assertIn("train.warm_start iemocap", refit["description"])
            decision = json.loads((W.out_dir() / "decisions" / f"{r_id}.json").read_text())
            self.assertEqual(decision["film_tests"]["train.warm_start_lr"], after["train"]["warm_start_lr"])

    def test_the_superseded_film_sweeps_stay_as_they_are(self):
        # addendum 5: what the node holds, addendum 1's film validation sweeps (split film_5fold) queued with folds
        # test_F2 and test_F3 failed, is left alone; addendum 8: so are addendum 5's w33_<tag>_film2v (split
        # film_5fold_strat), written on the node by its first pass and never queued; the chain goes on through
        # w33_<tag>_film3v
        tag = "wrnn"
        old_id, mid_id = f"w33_{tag}_filmv", f"w33_{tag}_film2v"
        self.assertIn(old_id, W.SUPERSEDED)
        self.assertIn(mid_id, W.SUPERSEDED)
        self.assertEqual(len(W.SUPERSEDED), 20)
        old = json.loads(json.dumps(record_sweep(tag)))
        old_base = W.with_overrides(W.record_settings(tag), {"data.name": "emofilm", "data.label_set": "three_no_neutral",
                                                             "data.subset": "all", "data.split": "film_5fold"})
        old = {"sweep_id": old_id, "description": "addendum 1's film validation sweep (stand-in)",
               "folds": ["test_F1", "test_F2", "test_F3", "test_F4", "test_F5"], "base": W.plain(W.validate_job(old_base)),
               "stages": [{"name": "validation", "kind": "base", "priority": 178, "seeds": [11, 12, 13, 14, 15]}]}
        (W.out_dir() / "sweeps").mkdir(parents=True, exist_ok=True)
        (W.out_dir() / "sweeps" / f"{old_id}.json").write_text(json.dumps(old, indent=1) + "\n")
        old_jobs = self.root.enqueue(old)
        self.root.finish([j for j in old_jobs if j["fold"] not in ("test_F2", "test_F3")])
        for j in old_jobs:
            if j["fold"] in ("test_F2", "test_F3"):
                (self.root.path / "queue" / "failed" / f"821_000005__{old_id}__{j['run_id']}.json").write_text("{}")
        mid_base = W.with_overrides(old_base, {"data.split": "film_5fold_strat"})
        mid = dict(old, sweep_id=mid_id, description="addendum 5's film validation sweep (stand-in)",
                   base=W.plain(W.validate_job(mid_base)))
        (W.out_dir() / "sweeps" / f"{mid_id}.json").write_text(json.dumps(mid, indent=1) + "\n")
        before = (W.out_dir() / "sweeps" / f"{old_id}.json").read_bytes()
        before_mid = (W.out_dir() / "sweeps" / f"{mid_id}.json").read_bytes()
        self.advance()
        v_id, r_id = W.ids(tag, "film")
        self.assertEqual((v_id, r_id), (f"w33_{tag}_film3v", f"w33_{tag}_film"))
        self.assertTrue(W.written(v_id))
        self.assertEqual((W.out_dir() / "sweeps" / f"{old_id}.json").read_bytes(), before)
        self.assertEqual((W.out_dir() / "sweeps" / f"{mid_id}.json").read_bytes(), before_mid)
        queue_list = W.to_enqueue()
        self.assertNotIn(W.out_dir() / "sweeps" / f"{old_id}.json", queue_list)
        self.assertNotIn(W.out_dir() / "sweeps" / f"{mid_id}.json", queue_list)      # written, never queued
        self.assertIn(W.out_dir() / "sweeps" / f"{v_id}.json", queue_list)
        self.assertFalse(any(p.stem.endswith("_film2v") or p.stem.endswith("_filmv") for p in queue_list))
        new_jobs = self.root.enqueue_file(v_id)
        self.assertFalse({j["run_id"] for j in new_jobs} & {j["run_id"] for j in old_jobs})
        self.root.finish(new_jobs, epoch_of=lambda j: 7)
        self.assertIn("refit written", W.advance_chain(tag, "film"))
        decision = json.loads((W.out_dir() / "decisions" / f"{r_id}.json").read_text())
        self.assertEqual((decision["from"], decision["rule"], decision["refit_epochs"]),
                         (v_id, "RULE_week33_addendum8.txt", 7))
        out = io.StringIO()
        with redirect_stdout(out):
            W.cmd_status(None)
        text = out.getvalue()
        self.assertIn(f"{tag}_filmv 15/25", text)
        self.assertIn("superseded (RULE_week33_addendum5.txt", text)
        self.assertIn(f"superseded (RULE_week33_addendum8.txt; kept, never read): {tag}_film2v written", text)
        self.assertIn("film3v 25/25", text)

    def test_the_refit_waits_for_every_run_and_for_no_failed_job(self):
        self.advance()
        v_id, r_id = W.ids("cnn", "spk4")
        jobs = self.root.enqueue_file(v_id)
        self.root.finish(jobs, skip={jobs[0]["run_id"]})
        self.assertIn("validation 49/50", W.advance_chain("cnn", "spk4"))
        self.assertFalse(W.written(r_id))
        self.root.finish(jobs[:1])
        (self.root.path / "queue" / "failed" / f"800_000001__{v_id}__{jobs[3]['run_id']}.json").write_text("{}")
        self.assertIn("WAITING", W.advance_chain("cnn", "spk4"))
        self.assertFalse(W.written(r_id))
        (self.root.path / "queue" / "failed" / f"800_000001__{v_id}__{jobs[3]['run_id']}.json").unlink()
        self.assertIn("refit written", W.advance_chain("cnn", "spk4"))
        self.assertTrue(W.written(r_id))

    def test_the_refit_takes_the_median_epoch_rounded_half_up(self):
        self.advance()
        v_id, r_id = W.ids("wlif", "spk3")
        jobs = self.root.enqueue_file(v_id)
        # 50 runs: 25 at epoch 7 and 25 at epoch 8, so the median is 7.5, rounded up to 8
        self.root.finish(jobs, epoch_of=lambda j: 7 if jobs.index(j) % 2 == 0 else 8)
        W.advance_chain("wlif", "spk3")
        refit = json.loads((W.out_dir() / "sweeps" / f"{r_id}.json").read_text())
        before, after = settings_of(json.loads((W.out_dir() / "sweeps" / f"{v_id}.json").read_text())), \
            settings_of(refit)
        self.assertLessEqual(set(W.changed(before, after)), {"train.refit", "train.refit_epochs", "train.eval_train"})
        self.assertEqual(after["train"]["refit"], "train_and_val")
        self.assertEqual(after["train"]["refit_epochs"], 8)
        self.assertEqual(after["train"]["eval_train"], "none")
        r_jobs, summary = W.expand(refit)
        self.assertEqual((len(r_jobs), summary["configs"]), (50, 1))
        self.assertEqual({j["priority"] for j in r_jobs}, {W.SETUPS["spk3"][5]})
        self.assertEqual(sorted({int(j["seed"]) for j in r_jobs}), W.SEEDS_SPEAKER)
        decision = json.loads((W.out_dir() / "decisions" / f"{r_id}.json").read_text())
        self.assertEqual((decision["refit_epochs"], len(decision["best_epochs"])), (8, 50))
        # the pooled setups: 25 runs, seeds 6 to 10, the median of an odd count
        v_id, r_id = W.ids("base", "pooled4")
        jobs = self.root.enqueue_file(v_id)
        self.root.finish(jobs, epoch_of=lambda j: 3 + jobs.index(j) % 5)
        W.advance_chain("base", "pooled4")
        refit = json.loads((W.out_dir() / "sweeps" / f"{r_id}.json").read_text())
        self.assertEqual(settings_of(refit)["train"]["refit_epochs"], 5)
        self.assertEqual(len(W.expand(refit)[0]), 25)

    def test_a_second_pass_writes_nothing_new_and_the_queue_list(self):
        self.advance()
        before = {p.name: p.read_text() for p in (W.out_dir() / "sweeps").glob("*.json")}
        self.advance()
        after = {p.name: p.read_text() for p in (W.out_dir() / "sweeps").glob("*.json")}
        self.assertEqual(before, after)
        listed = [p.stem for p in W.to_enqueue()]
        self.assertEqual(len(listed), 58)
        priorities = [max(st["priority"] for st in json.loads((W.out_dir() / "sweeps" / f"{s}.json").read_text())
                          ["stages"]) for s in listed]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
        self.root.enqueue_file(listed[0])
        (W.out_dir() / "smoke_failed").mkdir()
        (W.out_dir() / "smoke_failed" / f"{listed[1]}.txt").write_text("smoke test failed")
        self.assertEqual([p.stem for p in W.to_enqueue()], listed[2:])

    def test_a_missing_record_is_an_error_for_its_chains_only(self):
        self.root.close()
        self.root = Root(skip=("crnn",))
        out = io.StringIO()
        with redirect_stdout(out):
            W.cmd_advance(None)
        text = out.getvalue()
        self.assertEqual(text.count("ERROR"), 7)                   # crnn's four setups and three added ones
        self.assertIn("w33_crnn: spk4: ERROR", text)
        self.assertEqual(text.count("ENQUEUE "), 51)
        self.assertFalse(any(W.written(W.ids("crnn", s)[0]) for s in W.SETUPS))

    def test_run_ids_that_exist_are_refused(self):
        sweep = {"sweep_id": "w33_cnn_spk4v", "description": "x", "folds": list(fold_names(SPEAKER_OUT)),
                 "base": W.with_overrides(W.record_settings("cnn"), {}),
                 "stages": [{"name": "validation", "kind": "base", "priority": 188, "seeds": [11, 12, 13, 14, 15]}]}
        jobs, _ = W.expand(sweep)
        (self.root.path / "artifacts" / "runs" / "other" / jobs[0]["run_id"]).mkdir(parents=True)
        W._EXISTING = None
        with self.assertRaises(SystemExit):
            W.write_sweep(sweep, 50, {})


if __name__ == "__main__":
    unittest.main()
