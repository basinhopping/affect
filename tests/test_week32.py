"""affect.sweep.week32, week 32's automatic stages and the new encoders' sweeps (added 2026-11-02), in a throwaway root
with stand-in run folders (spec.json, metrics_val.json and predictions_val.csv, no test file). The confirmation takes the
two configurations with the highest mean validation macro accuracy, a tie to the one listed first in jobs.csv, and
nothing while the screen lacks a run or has a failed job; the record is the better of the two over seeds 6 to 10, a tie
to rank 1, its refit epochs the median of its runs' best_epoch; the record's refit, speaker folds, script splits and
setups differ from it only where they should; a setup's refit takes that setup's own median; a second pass writes
nothing new; queued sweeps and marked smoke failures are not listed again. AST's screens are Whisper's with layer 8 and
the time masks divided by 5; w2v-BERT 2.0's layers are chosen on validation (a tie to the lower layer), and its probe
refits and protocol checks differ from the chosen probe only where they should. The frames check passes a store whose
means match the probe's, and fails one that does not.

Added 2026-11-05 (RULE_week32_addendum7.txt): a chain's films wait for its refit to be queued and for EmoFilm's checked
frames, then are the refit with the external test and the warm start (a tenth of its rate) and nothing else; the
networks' films are their refits' the same way, each once that refit is queued; Whisper Small's layers are chosen like
w2v-BERT 2.0's, its screens are Whisper large-v3's at priority 163 with only the encoder and layer changed, and its
probe refits and checks are w2v-BERT 2.0's with its own names; AST's and w2v-BERT 2.0's sweep files are the bytes the
node wrote on 2026-11-02 and 2026-11-03; the frames check of a test corpus checks every clip and records a pass.

Added 2026-11-07 (RULE_week32_addendum9.txt part N): BUD-E-Whisper (letter b, priority 162) and Whisper medium (letter
m, priority 161) are done as Whisper Small is, their sweeps citing addendum 9, and Whisper Small's still citing
addendum 7."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from affect import catalog, paths
from affect.sweep import week32 as W
from tests.support import REPO

catalog.load_all()


class Root:
    """A throwaway repository root (affect.paths.repo_root) holding the sweep files the module reads, the sweeps'
    records as enqueue writes them, and stand-in finished runs."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        for d in ("artifacts/sweeps", "artifacts/runs", "queue/failed", "sweeps"):
            (self.path / d).mkdir(parents=True)
        for f in (REPO / "sweeps").glob("*.json"):
            if f.name.startswith(("heads31w_", "heads27_", "heads32w_", "classical_pretrained_w2v_bert_2_",
                                  "classical_pretrained_whisper_small_", "classical_pretrained_bud_e_whisper_",
                                  "classical_pretrained_whisper_medium_", "refit30_", "refit32_")):
                shutil.copy(f, self.path / "sweeps" / f.name)
        self.saved = paths.repo_root
        paths.repo_root = lambda: self.path
        W._EXISTING = None

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

    def finish(self, jobs: list, ua_of, epoch_of=lambda j: 12, skip=()):
        for j in jobs:
            if j["run_id"] in skip:
                continue
            d = self.path / "artifacts" / "runs" / j["sweep_id"] / j["run_id"]
            d.mkdir(parents=True)
            (d / "spec.json").write_text(json.dumps({k: j[k] for k in ("sweep_id", "stage", "label", "config_hash",
                                                                         "run_id", "fold", "seed", "settings")}))
            ua = ua_of(j)
            (d / "metrics_val.json").write_text(json.dumps({"best_epoch": epoch_of(j), "best_val_ua": ua,
                                                            "final": {"ua": ua}}))
            (d / "predictions_val.csv").write_text("utt_id,label,pred\n")


def screen_of(head: str = "pool") -> dict:
    return json.loads((REPO / "sweeps" / f"heads32w_{head}_screen.json").read_text())


class TestChain(unittest.TestCase):
    def setUp(self):
        self.root = Root()

    def tearDown(self):
        self.root.close()

    def advance(self) -> dict:
        lines = {}
        for letter in W.ENCODERS:
            for head in W.HEADS:
                lines[f"heads32{letter}_{head}"] = W.advance_chain(letter, head)
        return lines

    def screen(self, ua_of):
        jobs = self.root.enqueue(screen_of())
        configs = list(dict.fromkeys(j["config_hash"] for j in jobs))
        self.root.finish(jobs, lambda j: ua_of(configs.index(j["config_hash"]), j))
        return jobs, configs

    def test_nothing_before_the_screen_is_whole_or_with_a_failed_job(self):
        jobs = self.root.enqueue(screen_of())
        self.root.finish(jobs, lambda j: 0.5, skip={jobs[7]["run_id"]})
        self.assertEqual(self.advance()["heads32w_pool"], "screen 464 of 465")
        (self.root.path / "queue" / "failed" / f"815_000007__heads32w_pool_screen__{jobs[7]['run_id']}.w.json").write_text("{}")
        self.assertTrue(self.advance()["heads32w_pool"].startswith("WAITING: screen has 1 failed"))
        self.assertFalse((self.root.path / "artifacts" / "advance_week32" / "sweeps").exists())
        self.assertEqual(self.advance()["heads32w_rnn"], "screen not queued")

    def test_confirmation_takes_the_top_two_and_a_tie_goes_to_the_first_listed(self):
        # configuration 20 is best; 9 and 4 tie for second, 4 is listed first
        quality = {20: 0.80, 9: 0.75, 4: 0.75}
        jobs, configs = self.screen(lambda i, j: quality.get(i, 0.60 + i / 1000) + (0.01 if j["seed"] == 2 else 0.0)
                                    - (0.01 if j["seed"] == 3 else 0.0))
        self.advance()
        path = W.written("heads32w_pool_confirm")
        self.assertIsNotNone(path)
        sweep = json.loads(path.read_text())
        cj, summary = W.expand(sweep)
        self.assertEqual((len(cj), summary["configs"]), (50, 2))
        self.assertEqual({j["seed"] for j in cj}, {6, 7, 8, 9, 10})
        self.assertEqual({j["priority"] for j in cj}, {190})
        self.assertEqual([j["config_hash"] for j in cj if j["stage"] == "rank_1"][0], configs[20])
        self.assertEqual([j["config_hash"] for j in cj if j["stage"] == "rank_2"][0], configs[4])
        decision = json.loads((self.root.path / "artifacts/advance_week32/decisions/heads32w_pool_confirm.json").read_text())
        self.assertEqual(decision["chosen"], [configs[20], configs[4]])
        self.assertEqual(len(decision["ranking"]), 31)
        self.assertAlmostEqual(decision["ranking"][0]["ua"], 0.80)
        self.assertEqual(W.to_enqueue(), [path])
        before = path.read_text()
        self.advance()
        self.assertEqual(path.read_text(), before, "a second pass keeps the file")
        (self.root.path / "artifacts/advance_week32/smoke_failed").mkdir(parents=True)
        (self.root.path / "artifacts/advance_week32/smoke_failed/heads32w_pool_confirm.txt").write_text("x")
        self.assertEqual(W.to_enqueue(), [], "a marked smoke failure is not listed again")

    def test_record_and_its_stages(self):
        jobs, configs = self.screen(lambda i, j: 0.80 if i == 3 else (0.79 if i == 17 else 0.6))
        self.advance()
        confirm = json.loads(W.written("heads32w_pool_confirm").read_text())
        cj = self.root.enqueue(confirm)
        self.assertEqual(W.to_enqueue(), [], "a queued sweep is not listed")
        self.assertEqual(self.advance()["heads32w_pool"], "confirmation 0 of 50")
        # rank 2 (configuration 17) wins on seeds 6 to 10; its best epochs: 1 to 25 over its 25 runs, median 13
        epochs = iter(range(1, 26))
        self.root.finish(cj, lambda j: 0.70 if j["stage"] == "rank_1" else 0.72,
                         epoch_of=lambda j: next(epochs) if j["stage"] == "rank_2" else 30)
        self.advance()
        record = configs[17]
        settings = next(j["settings"] for j in jobs if j["config_hash"] == record)
        expect = {"refit": (25, {"train.refit", "train.refit_epochs"}, 195),
                  "spk": (50, {"data.split", "train.refit", "train.refit_epochs"}, 192),
                  "script": (75, {"data.split", "train.refit", "train.refit_epochs"}, 192),
                  "oscript": (75, {"data.split", "train.refit", "train.refit_epochs"}, 192),
                  "pooled4": (25, {"data.split"}, 191), "so3": (25, {"data.label_set"}, 191),
                  "pooled3": (25, {"data.split", "data.label_set"}, 191)}
        for key, (n, fields, priority) in expect.items():
            sweep = json.loads(W.written(f"heads32w_pool_{key}").read_text())
            sj, summary = W.expand(sweep)
            self.assertEqual((len(sj), summary["configs"]), (n, 1), key)
            self.assertEqual({j["priority"] for j in sj}, {priority}, key)
            self.assertEqual({j["seed"] for j in sj}, {6, 7, 8, 9, 10}, key)
            self.assertEqual(set(W.changed(settings, sweep["base"])) - {"train.eval_train"}, fields, key)
            if "train.refit" in fields:
                self.assertEqual(sweep["base"]["train"]["refit_epochs"], 13, key)
                self.assertEqual(sweep["base"]["train"]["refit"], "train_and_val", key)
        self.assertEqual(json.loads(W.written("heads32w_pool_spk").read_text())["base"]["data"]["split"],
                         "leave_one_speaker_out")
        self.assertEqual(len(W.to_enqueue()), 7)
        self.assertEqual(json.loads(W.to_enqueue()[0].read_text())["sweep_id"], "heads32w_pool_refit",
                         "the highest priority first")
        # a tie goes to rank 1
        self.assertEqual(W.record_of(W.chain_ids("w", "pool"))["config_hash"], record)
        # the setup's own refit, once its runs exist: median of 20 x 7 and 5 x 9 -> 7
        pooled = json.loads(W.written("heads32w_pool_pooled4").read_text())
        pj = self.root.enqueue(pooled)
        k = iter(range(25))
        self.root.finish(pj, lambda j: 0.7, epoch_of=lambda j: 7 if next(k) < 20 else 9)
        self.advance()
        refit = json.loads(W.written("heads32w_pool_pooled4_refit").read_text())
        self.assertEqual(refit["base"]["train"]["refit_epochs"], 7)
        self.assertEqual(refit["base"]["data"]["split"], "utterance_5fold")
        self.assertEqual(set(W.changed(pooled["base"], refit["base"])) - {"train.eval_train"},
                         {"train.refit", "train.refit_epochs"})
        self.assertIsNone(W.written("heads32w_pool_so3_refit"), "so3's runs do not exist yet")

    def test_a_tie_in_the_confirmation_goes_to_rank_1(self):
        jobs, configs = self.screen(lambda i, j: 0.80 if i == 3 else (0.79 if i == 17 else 0.6))
        self.advance()
        cj = self.root.enqueue(json.loads(W.written("heads32w_pool_confirm").read_text()))
        self.root.finish(cj, lambda j: 0.70)
        self.assertEqual(W.record_of(W.chain_ids("w", "pool"))["config_hash"], configs[3])


class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.root = Root()

    def tearDown(self):
        self.root.close()

    def test_ast_screens(self):
        for head in W.HEADS:
            sweep = W.screen_sweep("ast_audioset", head, W.AST_LAYER)
            jobs, summary = W.expand(sweep)
            self.assertEqual((len(jobs), summary["configs"]), (465, 31))
            self.assertEqual({j["priority"] for j in jobs}, {165})
            whisper = screen_of(head)
            self.assertEqual(sweep["base"]["augment"]["time_mask_width"] * 5, whisper["base"]["augment"]["time_mask_width"])
            self.assertEqual(sweep["base"]["features"]["model"], "ast_audioset")
            self.assertEqual(sweep["base"]["features"]["layer"], 8)
        plan = json.loads((REPO / "artifacts/analyses/week32_prep_20261102/plan_week32b.json").read_text()) \
            if (REPO / "artifacts/analyses/week32_prep_20261102/plan_week32b.json").is_file() else None
        if plan is not None:     # the bytes the node writes are those checked in the container
            import hashlib
            for head in W.HEADS:
                text = json.dumps(W.screen_sweep("ast_audioset", head, 8), indent=1) + "\n"
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                                 plan["node_writes"][f"heads32a_{head}_screen"]["sha256"])

    def probe_runs(self, best: dict):
        """Stand-in runs of w2v-BERT 2.0's four probe sweeps; best[setup] = (layer, ua) with every other layer 0.6, and
        layer 20 tying layer best in so3."""
        for setup in W.PROBE_SETUPS:
            sweep = json.loads((REPO / "sweeps" / f"classical_pretrained_w2v_bert_2_{setup}_1.json").read_text())
            jobs = self.root.enqueue(sweep)
            layer, ua = best[setup]

            def ua_of(j, layer=layer, ua=ua, setup=setup):
                k = int(j["settings"]["functionals"]["feature_set"].rsplit("_l", 1)[1])
                return ua if k == layer or (setup == "so3" and k == 20) else 0.6
            self.root.finish(jobs, ua_of)

    def test_the_encoders_screens_are_not_the_stages_to_queue(self):
        """enc32.sh queues an encoder's screens once its frames are checked; advance must never list them (review of
        2026-11-02): they go to encoders/, and a *_screen file in sweeps/ is skipped too."""
        W.cmd_encoder_sweeps(type("A", (), {"model": "ast_audioset"})())
        self.assertEqual(sorted(p.name for p in (self.root.path / "artifacts/advance_week32/encoders").glob("*.json")),
                         [f"heads32a_{h}_screen.json" for h in ("lif", "pool", "rnn", "transformer")])
        self.assertEqual(W.to_enqueue(), [])
        stray = self.root.path / "artifacts/advance_week32/sweeps"
        stray.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.root.path / "artifacts/advance_week32/encoders/heads32a_pool_screen.json", stray)
        self.assertEqual(W.to_enqueue(), [])

    def test_a_failing_chain_does_not_stop_the_others(self):
        jobs = self.root.enqueue(screen_of("pool"))
        self.root.finish(jobs, lambda j: 0.6)
        bad = self.root.path / "artifacts/runs/heads32w_pool_screen" / jobs[0]["run_id"] / "metrics_val.json"
        bad.write_text("{not json")
        rnn = self.root.enqueue(screen_of("rnn"))
        self.root.finish(rnn, lambda j: 0.7 if j["label"].startswith("random 1:") else 0.6)
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            W.cmd_advance(None)
        text = out.getvalue()
        self.assertIn("heads32w_pool: ERROR", text)
        self.assertIn("ENQUEUE ", text, "the rnn head's confirmation is still listed")
        self.assertIsNotNone(W.written("heads32w_rnn_confirm"))

    def test_w2v_bert_layers_and_sweeps(self):
        self.probe_runs({"so4": (17, 0.74), "pooled4": (16, 0.77), "so3": (21, 0.86), "pooled3": (18, 0.88)})
        self.assertEqual(W.cmd_choose_layers(type("A", (), {"model": "w2v_bert_2"})()), 0)
        layers = json.loads((self.root.path / "artifacts/advance_week32/layers_w2v_bert_2.json").read_text())
        self.assertEqual({s: v["layer"] for s, v in layers["setups"].items()},
                         {"so4": 17, "pooled4": 16, "so3": 20, "pooled3": 18}, "so3: a tie to the lower layer")
        self.assertEqual(W.heads_layer("w2v_bert_2"), 17)
        out = W.probe_sweeps("w2v_bert_2")
        self.assertEqual([s["sweep_id"] for s, _ in out],
                         ["classical_refit32_probe32_w2v_bert_2_so4", "classical_refit32_probe32_w2v_bert_2_pooled4",
                          "classical_refit32_probe32_w2v_bert_2_so3", "classical_refit32_probe32_w2v_bert_2_pooled3",
                          "classical_spk32_w2v", "classical_script32_w2v", "classical_oscript32_w2v",
                          "classical_gnorm32_w2v"])
        self.assertEqual([n for _, n in out], [5, 5, 5, 5, 10, 15, 15, 5])
        for sweep, n in out:
            jobs, summary = W.expand(sweep)
            self.assertEqual((len(jobs), summary["configs"]), (n, 1), sweep["sweep_id"])
            self.assertEqual(sweep["base"]["protocol"]["refit"], "train_and_val")
        self.assertEqual(out[2][0]["base"]["functionals"]["feature_set"], "w2v_bert_2_l20")
        self.assertEqual(out[4][0]["base"]["functionals"]["feature_set"], "w2v_bert_2_l17")
        self.assertEqual(out[7][0]["base"]["functionals"]["normalization"], "global")
        screen = W.screen_sweep("w2v_bert_2", "lif", 17)
        jobs, _ = W.expand(screen)
        self.assertEqual({j["settings"]["features"]["layer"] for j in jobs}, {17})
        self.assertEqual({j["priority"] for j in jobs}, {164})

    def test_layers_wait_for_every_run(self):
        sweep = json.loads((REPO / "sweeps" / "classical_pretrained_w2v_bert_2_so4_1.json").read_text())
        jobs = self.root.enqueue(sweep)
        self.root.finish(jobs[:-1], lambda j: 0.6)
        self.assertEqual(W.cmd_choose_layers(type("A", (), {"model": "w2v_bert_2"})()), 1)
        self.assertFalse((self.root.path / "artifacts/advance_week32/layers_w2v_bert_2.json").exists())

    def test_check_frames(self):
        base = self.root.path / "artifacts/features/iemocap"
        rng = np.random.default_rng(0)
        ids = [f"u{i}" for i in range(120)]
        counts = rng.integers(3, 30, size=len(ids))
        frames = rng.normal(scale=3.0, size=(int(counts.sum()), 8)).astype(np.float16)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        means = np.stack([frames[offsets[i]:offsets[i + 1]].astype(np.float64).mean(0) for i in range(len(ids))])
        (base / "m_l3_frames").mkdir(parents=True)
        (base / "m_l3").mkdir(parents=True)
        np.save(base / "m_l3_frames/frames.npy", frames)
        np.save(base / "m_l3_frames/offsets.npy", offsets)
        (base / "m_l3_frames/utt_ids.txt").write_text("\n".join(ids) + "\n")
        (base / "m_l3_frames/record.json").write_text(json.dumps({"width": 8, "total_frames": int(counts.sum()),
                                                                   "frame_rate": 50}))
        order = list(reversed(range(len(ids))))
        np.save(base / "m_l3/values.npy", means[order].astype(np.float32))
        (base / "m_l3/utt_ids.txt").write_text("\n".join(ids[i] for i in order) + "\n")
        args = type("A", (), {"model": "m", "layer": 3, "dataset": "iemocap"})()
        marker = W.frames_checked_path("iemocap", "m", 3)
        np.save(base / "m_l3/values.npy", (means[order] + 0.5).astype(np.float32))
        self.assertEqual(W.cmd_check_frames(args), 1)
        self.assertFalse(marker.exists(), "a failed check records nothing")
        np.save(base / "m_l3/values.npy", means[order].astype(np.float32))
        self.assertEqual(W.cmd_check_frames(args), 0)
        record = json.loads(marker.read_text())
        self.assertEqual((record["dataset"], record["model"], record["layer"], record["checked"], record["every"]),
                         ("iemocap", "m", 3, 3, 50))
        # a test corpus's frames: every clip is checked (added 2026-11-05)
        film = self.root.path / "artifacts/features/emofilm"
        shutil.copytree(base / "m_l3_frames", film / "m_l3_frames")
        shutil.copytree(base / "m_l3", film / "m_l3")
        args = type("A", (), {"model": "m", "layer": 3, "dataset": "emofilm"})()
        self.assertEqual(W.cmd_check_frames(args), 0)
        record = json.loads(W.frames_checked_path("emofilm", "m", 3).read_text())
        self.assertEqual((record["checked"], record["every"], record["utterances"]), (120, 1, 120))


class TestFilms(unittest.TestCase):
    """The films of the final records (added 2026-11-05, RULE_week32_addendum7.txt part F)."""

    def setUp(self):
        self.root = Root()

    def tearDown(self):
        self.root.close()

    def to_refit(self):
        """A pool chain on Whisper up to its stage-two sweeps (written, not queued)."""
        jobs = self.root.enqueue(screen_of())
        configs = list(dict.fromkeys(j["config_hash"] for j in jobs))
        self.root.finish(jobs, lambda j: 0.80 if j["config_hash"] == configs[3] else 0.6)
        W.advance_chain("w", "pool")
        cj = self.root.enqueue(json.loads(W.written("heads32w_pool_confirm").read_text()))
        self.root.finish(cj, lambda j: 0.70, epoch_of=lambda j: 9)
        W.advance_chain("w", "pool")

    def test_a_chains_films(self):
        self.to_refit()
        self.assertTrue(W.advance_chain("w", "pool").endswith("film -"), "the refit is not queued yet")
        refit = json.loads(W.written("heads32w_pool_refit").read_text())
        self.root.enqueue(refit)
        line = W.advance_chain("w", "pool")
        self.assertTrue(line.endswith("film waits for EmoFilm's checked frames (whisper_large_v3 layer 31)"), line)
        self.assertIsNone(W.written("heads32w_pool_film"))
        marker = W.frames_checked_path("emofilm", "whisper_large_v3", 31)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"dataset": "emofilm", "checked": 726}))
        self.assertTrue(W.advance_chain("w", "pool").endswith("film written"))
        path = W.written("heads32w_pool_film")
        sweep = json.loads(path.read_text())
        jobs, summary = W.expand(sweep)
        rjobs, _ = W.expand(refit)
        self.assertEqual((len(jobs), summary["configs"]), (25, 1))
        self.assertEqual({j["priority"] for j in jobs}, {194})
        self.assertEqual([(j["fold"], j["seed"]) for j in jobs], [(j["fold"], j["seed"]) for j in rjobs])
        self.assertEqual(W.changed(refit["base"], sweep["base"]),
                         ["train.external_test", "train.warm_start", "train.warm_start_epochs", "train.warm_start_folds",
                          "train.warm_start_lr"])
        train = sweep["base"]["train"]
        self.assertEqual((train["external_test"], train["warm_start"], train["warm_start_folds"],
                          train["warm_start_epochs"], train["refit"]), ("emofilm", "emofilm", 5, 10, "train_and_val"))
        self.assertAlmostEqual(train["warm_start_lr"] * 10, refit["base"]["train"]["lr"], places=15)
        decision = json.loads((self.root.path / "artifacts/advance_week32/decisions/heads32w_pool_film.json").read_text())
        self.assertEqual((decision["from"], decision["emofilm_frames"]["checked"]), ("heads32w_pool_refit", 726))
        self.assertIn(path, W.to_enqueue())
        before = path.read_text()
        W.advance_chain("w", "pool")
        self.assertEqual(path.read_text(), before, "a second pass keeps the file")
        self.root.enqueue(sweep)
        self.assertTrue(W.advance_chain("w", "pool").endswith("film 0/25"))
        self.assertNotIn(path, W.to_enqueue())

    def test_the_films_start_from_a_refit(self):
        settings = screen_of()["base"]
        with self.assertRaises(SystemExit):
            W.film_settings(settings)                        # not a refit
        films, lr = W.film_settings(W.refit_of(settings, 12))
        with self.assertRaises(SystemExit):
            W.film_settings(films)                           # already has the films

    def test_the_networks_films(self):
        import contextlib
        import io

        tags = [t for t, _ in W.NETWORK_FILMS]
        self.assertEqual(tags, ["base", "cnn", "transformer", "threeblock", "lif", "crnn"])
        lines = W.advance_network_films()
        self.assertTrue(all("waits for" in line for line in lines), lines)
        refits = {}
        for tag, refit_id in W.NETWORK_FILMS:
            refits[tag] = json.loads((self.root.path / "sweeps" / f"{refit_id}.json").read_text())
            self.root.enqueue(refits[tag])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            W.cmd_advance(None)
        self.assertIn("network films: film32_base_so4 written; film32_cnn_so4 written", out.getvalue())
        listed = [Path(line[len("ENQUEUE "):]).stem for line in out.getvalue().splitlines()
                  if line.startswith("ENQUEUE ")]
        self.assertEqual(sorted(listed), sorted(f"film32_{t}_so4" for t in tags))
        for tag, refit_id in W.NETWORK_FILMS:
            sweep = json.loads(W.written(f"film32_{tag}_so4").read_text())
            jobs, summary = W.expand(sweep)
            rjobs, _ = W.expand(refits[tag])
            self.assertEqual((len(jobs), summary["configs"]), (25, 1), tag)
            self.assertEqual({j["priority"] for j in jobs}, {168}, tag)
            self.assertEqual([(j["fold"], j["seed"]) for j in jobs], [(j["fold"], j["seed"]) for j in rjobs], tag)
            self.assertEqual(W.changed(refits[tag]["base"], sweep["base"]),
                             ["train.external_test", "train.warm_start", "train.warm_start_epochs",
                              "train.warm_start_folds", "train.warm_start_lr"], tag)
            self.assertAlmostEqual(sweep["base"]["train"]["warm_start_lr"] * 10, refits[tag]["base"]["train"]["lr"],
                                   places=15)
        self.assertEqual({j["seed"] for j in W.expand(json.loads(W.written("film32_base_so4").read_text()))[0]},
                         {1, 2, 3, 4, 5}, "the base CNN's record has seeds 1 to 5")
        plan = REPO / "artifacts/analyses/week32_prep_20261102/plan_week32c.json"
        if plan.is_file():           # the bytes the node writes are those checked in the container
            import hashlib
            expected = json.loads(plan.read_text())["node_writes"]
            for tag, _ in W.NETWORK_FILMS:
                text = W.written(f"film32_{tag}_so4").read_text()
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), expected[f"film32_{tag}_so4"]["sha256"])


class TestWhisperSmall(unittest.TestCase):
    """Whisper Small as a fourth encoder (added 2026-11-05, RULE_week32_addendum7.txt part S)."""

    def setUp(self):
        self.root = Root()

    def tearDown(self):
        self.root.close()

    def test_the_fourth_chain(self):
        self.assertEqual(W.ENCODERS["s"], "whisper_small")
        self.assertEqual(W.SCREEN_PRIORITY["whisper_small"], 163)
        self.assertEqual(W.advance_chain("s", "pool"), "screen not queued")

    def test_layers_screens_and_probe_sweeps(self):
        best = {"so4": (9, 0.71), "pooled4": (8, 0.75), "so3": (10, 0.84), "pooled3": (7, 0.86)}
        for setup in W.PROBE_SETUPS:
            sweep = json.loads((REPO / "sweeps" / f"classical_pretrained_whisper_small_{setup}_1.json").read_text())
            jobs = self.root.enqueue(sweep)
            self.assertEqual(len(jobs), 65)
            layer, ua = best[setup]
            self.root.finish(jobs, lambda j, layer=layer, ua=ua:
                             ua if j["settings"]["functionals"]["feature_set"] == f"whisper_small_l{layer}" else 0.6)
        self.assertEqual(W.cmd_choose_layers(type("A", (), {"model": "whisper_small"})()), 0)
        layers = json.loads((self.root.path / "artifacts/advance_week32/layers_whisper_small.json").read_text())
        self.assertEqual({k: v["layer"] for k, v in layers["setups"].items()},
                         {"so4": 9, "pooled4": 8, "so3": 10, "pooled3": 7})
        self.assertEqual(layers["rule"], "RULE_week32_addendum7.txt part S")
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            W.cmd_encoder_sweeps(type("A", (), {"model": "whisper_small"})())
        written = sorted(Path(line.split(" ", 1)[1]).stem for line in out.getvalue().splitlines()
                         if line.startswith(("SCREEN ", "CLASSICAL ")))
        self.assertEqual(written, sorted([f"heads32s_{h}_screen" for h in W.HEADS]
                                         + [f"classical_refit32_probe32_whisper_small_{s}" for s in W.PROBE_SETUPS]
                                         + ["classical_spk32_wsmall", "classical_script32_wsmall",
                                            "classical_oscript32_wsmall", "classical_gnorm32_wsmall"]))
        for head in W.HEADS:
            sweep = json.loads((self.root.path / f"artifacts/advance_week32/encoders/heads32s_{head}_screen.json")
                               .read_text())
            jobs, summary = W.expand(sweep)
            wjobs, _ = W.expand(screen_of(head))
            self.assertEqual((len(jobs), summary["configs"]), (465, 31))
            self.assertEqual({j["priority"] for j in jobs}, {163})
            self.assertEqual({j["settings"]["features"]["layer"] for j in jobs}, {9})
            for j, w in zip(jobs, wjobs):
                self.assertEqual(W.changed(w["settings"], j["settings"]), ["features.layer", "features.model"])
            self.assertIn("RULE_week32_addendum7.txt", sweep["description"])
        decision = json.loads((self.root.path / "artifacts/advance_week32/decisions/classical_spk32_wsmall.json")
                              .read_text())
        self.assertEqual(decision["rule"], "RULE_week32_addendum7.txt part S")
        spk = json.loads((self.root.path / "artifacts/advance_week32/encoders/classical_spk32_wsmall.json").read_text())
        self.assertEqual(spk["base"]["functionals"]["feature_set"], "whisper_small_l9")
        self.assertEqual(W.to_enqueue(), [], "the encoders' sweeps are never the stages to queue")

    def test_ast_and_w2v_sweeps_keep_the_bytes_the_node_wrote(self):
        """The node's own files of 2026-11-02 and 2026-11-03 (the pull of 2026-11-05), when they are in the container:
        the text changes for Whisper Small leave AST's and w2v-BERT 2.0's files as they were."""
        node = REPO / "artifacts/advance_week32"
        if not (node / "encoders" / "heads32v_pool_screen.json").is_file():
            self.skipTest("the node's files are not in this tree")
        layer = json.loads((node / "layers_w2v_bert_2.json").read_text())["setups"]["so4"]["layer"]
        for head in W.HEADS:
            for model, k in (("ast_audioset", 8), ("w2v_bert_2", layer)):
                text = json.dumps(W.screen_sweep(model, head, k), indent=1) + "\n"
                self.assertEqual(text, (node / "encoders" / f"heads32{W.LETTER[model]}_{head}_screen.json").read_text(),
                                 (model, head))
        target = self.root.path / "artifacts/advance_week32"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(node / "layers_w2v_bert_2.json", target)
        for sweep, _ in W.probe_sweeps("w2v_bert_2"):
            self.assertEqual(json.dumps(sweep, indent=1) + "\n",
                             (node / "encoders" / f"{sweep['sweep_id']}.json").read_text(), sweep["sweep_id"])


class TestBudeAndMedium(unittest.TestCase):
    """BUD-E-Whisper and Whisper medium as the fifth and sixth encoders (added 2026-11-07, RULE_week32_addendum9.txt
    part N), each done as Whisper Small is."""
    CASES = {"bud_e_whisper": ("b", 162, 13, "bude", 7), "whisper_medium": ("m", 161, 25, "wmed", 18)}

    def setUp(self):
        self.root = Root()

    def tearDown(self):
        self.root.close()

    def test_the_fifth_and_sixth_chains(self):
        for model, (letter, priority, _, _, _) in self.CASES.items():
            self.assertEqual(W.ENCODERS[letter], model)
            self.assertEqual(W.SCREEN_PRIORITY[model], priority)
            self.assertEqual(W.advance_chain(letter, "pool"), "screen not queued")
        self.assertEqual(W.ENCODERS["s"], "whisper_small")
        self.assertEqual(W.SCREEN_PRIORITY["whisper_small"], 163)

    def test_layers_screens_and_probe_sweeps(self):
        import contextlib
        import io
        for model, (letter, priority, n_layers, short, layer) in self.CASES.items():
            for setup in W.PROBE_SETUPS:
                sweep = json.loads((REPO / "sweeps" / f"classical_pretrained_{model}_{setup}_1.json").read_text())
                self.assertIn("RULE_week32_addendum9.txt part N", sweep["description"])
                jobs = self.root.enqueue(sweep)
                self.assertEqual(len(jobs), 5 * n_layers)
                best = layer + W.PROBE_SETUPS.index(setup)
                self.root.finish(jobs, lambda j, best=best: 0.8 if j["settings"]["functionals"]["feature_set"]
                                 == f"{model}_l{best}" else 0.6)
            self.assertEqual(W.cmd_choose_layers(type("A", (), {"model": model})()), 0)
            layers = json.loads((self.root.path / f"artifacts/advance_week32/layers_{model}.json").read_text())
            self.assertEqual({k: v["layer"] for k, v in layers["setups"].items()},
                             {"so4": layer, "pooled4": layer + 1, "so3": layer + 2, "pooled3": layer + 3})
            self.assertEqual(layers["rule"], "RULE_week32_addendum9.txt part N")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                W.cmd_encoder_sweeps(type("A", (), {"model": model})())
            written = sorted(Path(line.split(" ", 1)[1]).stem for line in out.getvalue().splitlines()
                             if line.startswith(("SCREEN ", "CLASSICAL ")))
            self.assertEqual(written, sorted([f"heads32{letter}_{h}_screen" for h in W.HEADS]
                                             + [f"classical_refit32_probe32_{model}_{s}" for s in W.PROBE_SETUPS]
                                             + [f"classical_{p}32_{short}" for p in ("spk", "script", "oscript",
                                                                                      "gnorm")]))
            for head in W.HEADS:
                sweep = json.loads((self.root.path / f"artifacts/advance_week32/encoders/heads32{letter}_{head}"
                                    f"_screen.json").read_text())
                jobs, summary = W.expand(sweep)
                wjobs, _ = W.expand(screen_of(head))
                self.assertEqual((len(jobs), summary["configs"]), (465, 31))
                self.assertEqual({j["priority"] for j in jobs}, {priority})
                self.assertEqual({j["settings"]["features"]["layer"] for j in jobs}, {layer})
                self.assertEqual({j["settings"]["features"]["model"] for j in jobs}, {model})
                for j, w in zip(jobs, wjobs):
                    self.assertEqual(W.changed(w["settings"], j["settings"]), ["features.layer", "features.model"])
                self.assertIn("RULE_week32_addendum9.txt part N", sweep["description"])
                self.assertIn(W.RULE9, sweep["description"])
            decision = json.loads((self.root.path / f"artifacts/advance_week32/decisions/classical_spk32_{short}.json")
                                  .read_text())
            self.assertEqual(decision["rule"], "RULE_week32_addendum9.txt part N")
            spk = json.loads((self.root.path / f"artifacts/advance_week32/encoders/classical_spk32_{short}.json")
                             .read_text())
            self.assertEqual(spk["base"]["functionals"]["feature_set"], f"{model}_l{layer}")
            self.assertIn(W.RULE9, spk["description"])
        self.assertEqual(W.to_enqueue(), [], "the encoders' sweeps are never the stages to queue")

    def test_whisper_smalls_sweeps_still_cite_addendum_7(self):
        sweep = W.screen_sweep("whisper_small", "pool", 9)
        self.assertIn("RULE_week32_addendum7.txt part S", sweep["description"])
        self.assertNotIn("addendum9", sweep["description"])
        self.assertEqual(W.SWEEP_RULES["whisper_small"], ("RULE_week32_addendum7.txt part S", W.RULE7))
        ast = W.screen_sweep("ast_audioset", "pool", 8)
        self.assertIn("RULE_week32.txt part 6", ast["description"])


if __name__ == "__main__":
    unittest.main()
