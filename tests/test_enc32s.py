"""scripts/enc32s.sh (added 2026-11-05, RULE_week32_addendum7.txt) in a throwaway root, with stand-ins for the
extraction, the probe night, the classical sweeps, the smoke test and enqueue, nvidia-smi and affect.sweep.week32,
and the week's window always open: EmoFilm's probe means and frames of the three encoders' heads are made and checked
first (Whisper large-v3 layer 31, AST layer 8, w2v-BERT 2.0 its chosen layer), then Whisper Small's probes, layers,
probe refits and checks (eight CPU sweeps), its IEMOCAP frames and their check, its EmoFilm frames and their check, and
its four screens are smoke-tested, queued and ordered; a second run makes nothing again; without a Whisper Small
snapshot the films' steps are done and it stops with the reason. Also common.sh's gpu_allowed_at: weekday nights from
NIGHT_START (18 by default, 21 as enc32s.sh sets it), weekend days GPUs 2 and 3."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tests.support import REPO

LAYERS = {"whisper_large_v3": 33, "ast_audioset": 13, "w2v_bert_2": 25, "whisper_small": 13}
FAKE_PRETRAINED = r"""#!/usr/bin/env bash
# stand-in for scripts/pretrained.sh: writes the record files the real command would write
echo "pretrained $*" >> "$FAKE_DIR/calls"
cmd="$1"; shift
while [ "$#" -gt 0 ]; do case "$1" in --dataset) d="$2"; shift 2 ;; --model) m="$2"; shift 2 ;;
    --layers) k="$2"; shift 2 ;; *) shift ;; esac; done
F="$AFFECT_ROOT/artifacts/features/$d"
case "$cmd" in
  extract-pretrained) n=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$LAYERS_JSON" "$m")
      for ((i = 0; i < n; i++)); do mkdir -p "$F/${m}_l$i"; echo '{}' > "$F/${m}_l$i/record.json"; done ;;
  extract-frames) mkdir -p "$F/${m}_l${k}_frames"; echo '{}' > "$F/${m}_l${k}_frames/record.json" ;;
esac
"""
FAKE_NIGHT = r"""#!/usr/bin/env bash
echo "night $*" >> "$FAKE_DIR/calls"
mkdir -p "$AFFECT_ROOT/logs"; echo "2026-11-07T05:00:00Z ALL DONE: 4 sweeps, 0 with a failure" > "$AFFECT_ROOT/logs/pretrained_night_20261107T050000Z.txt"
"""
FAKE_CLASSICAL = r"""#!/usr/bin/env bash
echo "classical $*" >> "$FAKE_DIR/calls"
"""
FAKE_SMOKE = r"""#!/usr/bin/env bash
for f in "$@"; do s=$(basename "$f" .json)
  if [ -d "$AFFECT_ROOT/artifacts/sweeps/$s" ]; then echo "already enqueued: $s"; continue; fi
  echo "smoke $SMOKE_GPU $s" >> "$FAKE_DIR/calls"; mkdir -p "$AFFECT_ROOT/artifacts/sweeps/$s"
  touch "$AFFECT_ROOT/artifacts/sweeps/$s/jobs.csv"; echo "SMOKE OK $s"; done
echo "ALL DONE"
"""
FAKE_ORDER = r"""#!/usr/bin/env bash
echo "order $*" >> "$FAKE_DIR/calls"
"""
FAKE_SMI = r"""#!/usr/bin/env bash
case "$*" in *index,uuid*) for g in 0 1 2 3; do echo "$g, GPU-$g"; done ;; esac
"""
FAKE_PYTHON = r"""#!/usr/bin/env bash
# stand-in for ~/venvs/affect/bin/python: -c runs, affect.sweep.week32 is simulated
if [ "$1" = "-c" ]; then exec python3 "$@"; fi
[ "$1 $2" = "-m affect.sweep.week32" ] || { echo "unexpected python $*"; exit 2; }
shift 2; echo "week32 $*" >> "$FAKE_DIR/calls"
OUT="$AFFECT_ROOT/artifacts/advance_week32"
cmd="$1"; shift; d=iemocap
while [ "$#" -gt 0 ]; do case "$1" in --dataset) d="$2"; shift 2 ;; --model) m="$2"; shift 2 ;;
    --layer) k="$2"; shift 2 ;; *) shift ;; esac; done
case "$cmd" in
  check-frames) mkdir -p "$OUT/frames_checked"; echo '{}' > "$OUT/frames_checked/${d}_${m}_l${k}.json"
      echo "FRAMES OK: $m layer $k ($d)" ;;
  choose-layers) python3 -c "import json,sys; json.dump({'setups': {s: {'layer': l, 'validation_ua': 0.7} for s, l in (('so4', 9), ('pooled4', 8), ('so3', 10), ('pooled3', 7))}}, open(sys.argv[1], 'w'))" "$OUT/layers_whisper_small.json"
      echo "chosen" ;;
  encoder-sweeps) mkdir -p "$OUT/encoders"
      for h in pool rnn transformer lif; do f="$OUT/encoders/heads32s_${h}_screen.json"
          echo "{\"sweep_id\": \"heads32s_${h}_screen\"}" > "$f"; echo "SCREEN $f"; done
      for s in classical_refit32_probe32_whisper_small_so4 classical_refit32_probe32_whisper_small_pooled4 \
               classical_refit32_probe32_whisper_small_so3 classical_refit32_probe32_whisper_small_pooled3 \
               classical_spk32_wsmall classical_script32_wsmall classical_oscript32_wsmall classical_gnorm32_wsmall; do
          echo "{\"sweep_id\": \"$s\"}" > "$OUT/encoders/$s.json"; echo "CLASSICAL $OUT/encoders/$s.json"; done ;;
  *) echo "unexpected week32 $cmd"; exit 2 ;;
esac
"""


@unittest.skipIf(shutil.which("bash") is None, "needs bash")
class TestEnc32s(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.home, self.root, self.fake = base / "home", base / "home" / "affect", base / "fake"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "queue" / "pending").mkdir(parents=True)
        (self.root / "artifacts" / "advance_week32").mkdir(parents=True)
        self.fake.mkdir()
        shutil.copy(REPO / "scripts" / "enc32s.sh", self.root / "scripts" / "enc32s.sh")
        common = (REPO / "scripts" / "common.sh").read_text()
        # the week's window always open here; the real gpu_allowed_at is tested below
        (self.root / "scripts" / "common.sh").write_text(common + "\ngpu_allowed_at() { return 0; }\n")
        for name, text in (("pretrained.sh", FAKE_PRETRAINED), ("pretrained_night.sh", FAKE_NIGHT),
                           ("classical.sh", FAKE_CLASSICAL), ("smoke_enqueue_files.sh", FAKE_SMOKE),
                           ("order_sweeps.sh", FAKE_ORDER)):
            (self.root / "scripts" / name).write_text(text)
        venv = self.home / "venvs" / "affect" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text(FAKE_PYTHON)
        (venv / "python").chmod(0o755)
        bin_dir = self.fake / "bin"
        bin_dir.mkdir()
        (bin_dir / "nvidia-smi").write_text(FAKE_SMI)
        (bin_dir / "nvidia-smi").chmod(0o755)
        self.path = f"{bin_dir}:{os.environ['PATH']}"
        (self.root / "artifacts/advance_week32/layers_w2v_bert_2.json").write_text(
            json.dumps({"setups": {"so4": {"layer": 12, "validation_ua": 0.7233}}}))

    def tearDown(self):
        self.tmp.cleanup()

    def snapshots(self, *repos):
        for repo in repos:
            (self.home / "cache/huggingface" / f"models--{repo}" / "snapshots" / "0123abcd").mkdir(parents=True)

    def run_script(self) -> subprocess.CompletedProcess:
        env = dict(os.environ, HOME=str(self.home), PATH=self.path, FAKE_DIR=str(self.fake), ENC_GPU="3",
                   SLEEP_S="0", AFFECT_ROOT=str(self.root), LAYERS_JSON=json.dumps(LAYERS))
        env.pop("NIGHT_START", None)
        return subprocess.run(["bash", str(self.root / "scripts" / "enc32s.sh")], env=env, capture_output=True,
                              text=True, timeout=120)

    def calls(self) -> list:
        path = self.fake / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def test_everything_once_then_nothing_again(self):
        self.snapshots("openai--whisper-large-v3", "MIT--ast-finetuned-audioset-10-10-0.4593", "facebook--w2v-bert-2.0",
                       "openai--whisper-small")
        done = self.run_script()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        log = next((self.root / "logs").glob("enc32s_*.txt")).read_text()
        self.assertIn("weekday nights from 21:00", log)
        self.assertIn("ALL DONE: 0 CPU sweeps ended with a failure", log)
        calls = self.calls()
        extractions = [c for c in calls if c.startswith("pretrained ")]
        self.assertEqual([(c.split()[1], c.split()[3], c.split()[9]) for c in extractions], [
            ("extract-pretrained", "emofilm", "whisper_large_v3"), ("extract-frames", "emofilm", "whisper_large_v3"),
            ("extract-pretrained", "emofilm", "ast_audioset"), ("extract-frames", "emofilm", "ast_audioset"),
            ("extract-pretrained", "emofilm", "w2v_bert_2"), ("extract-frames", "emofilm", "w2v_bert_2"),
            ("extract-frames", "iemocap", "whisper_small"),
            ("extract-pretrained", "emofilm", "whisper_small"), ("extract-frames", "emofilm", "whisper_small")])
        self.assertIn("--layers 31", extractions[1])
        self.assertIn("--layers 8", extractions[3])
        self.assertIn("--layers 12", extractions[5])
        self.assertIn("--layers 9", extractions[6])
        self.assertIn("--label-set three_no_neutral --subset all", extractions[0])
        self.assertIn("--label-set four_merged --subset all", extractions[6])
        night = [c for c in calls if c.startswith("night ")]
        self.assertEqual(len(night), 1)
        self.assertIn("--models whisper_small --sweeps classical_pretrained_whisper_small_so4_1", night[0])
        self.assertEqual(len([c for c in calls if c.startswith("classical ")]), 8)
        self.assertEqual(sorted(c.split()[2] for c in calls if c.startswith("smoke ")),
                         sorted(f"heads32s_{h}_screen" for h in ("pool", "rnn", "transformer", "lif")))
        self.assertTrue(all(c.split()[1] == "3" for c in calls if c.startswith("smoke ")))
        self.assertEqual(len([c for c in calls if c.startswith("order ")]), 1)
        checked = sorted(p.stem for p in (self.root / "artifacts/advance_week32/frames_checked").glob("*.json"))
        self.assertEqual(checked, ["emofilm_ast_audioset_l8", "emofilm_w2v_bert_2_l12", "emofilm_whisper_large_v3_l31",
                                   "emofilm_whisper_small_l9", "iemocap_whisper_small_l9"])
        # the films' frames come before anything of Whisper Small
        first_small = next(i for i, c in enumerate(calls) if "whisper_small" in c)
        last_film = max(i for i, c in enumerate(calls) if "emofilm" in c and "whisper_small" not in c)
        self.assertLess(last_film, first_small)
        # a second run makes nothing again
        (self.fake / "calls").unlink()
        again = self.run_script()
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        calls = self.calls()
        self.assertEqual([c for c in calls if c.startswith(("pretrained ", "night ", "smoke "))], [])
        self.assertEqual(len([c for c in calls if c.startswith("classical ")]), 8, "the CPU sweeps resume (skip)")

    def test_without_whisper_small_the_films_steps_are_done(self):
        self.snapshots("openai--whisper-large-v3", "MIT--ast-finetuned-audioset-10-10-0.4593", "facebook--w2v-bert-2.0")
        done = self.run_script()
        self.assertEqual(done.returncode, 1)
        log = next((self.root / "logs").glob("enc32s_*.txt")).read_text()
        self.assertIn("STOPPED: no Whisper Small snapshot", log)
        checked = sorted(p.stem for p in (self.root / "artifacts/advance_week32/frames_checked").glob("*.json"))
        self.assertEqual(checked, ["emofilm_ast_audioset_l8", "emofilm_w2v_bert_2_l12", "emofilm_whisper_large_v3_l31"])

    def test_a_missing_encoder_snapshot_stops_first(self):
        self.snapshots("openai--whisper-large-v3", "openai--whisper-small")
        done = self.run_script()
        self.assertEqual(done.returncode, 1)
        self.assertEqual(self.calls(), [])


PACIFIC = ZoneInfo("America/Los_Angeles")


def at(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=PACIFIC).timestamp())


@unittest.skipIf(shutil.which("bash") is None, "needs bash")
class TestWindows(unittest.TestCase):
    def allowed(self, when: str, gpu: int, night_start=None) -> bool:
        env = {k: v for k, v in os.environ.items() if k != "NIGHT_START"}
        if night_start is not None:
            env["NIGHT_START"] = str(night_start)
        code = f'source "{REPO}/scripts/common.sh"; gpu_allowed_at {at(when)} {gpu}'
        return subprocess.run(["bash", "-c", code], env=env, capture_output=True).returncode == 0

    def test_weekday_nights_start_at_night_start(self):
        self.assertTrue(self.allowed("2026-11-06 19:00", 3), "18:00 by default")
        self.assertFalse(self.allowed("2026-11-06 19:00", 3, 21), "Friday 19:00 is still day from 21:00")
        self.assertFalse(self.allowed("2026-11-06 20:59", 3, 21))
        self.assertTrue(self.allowed("2026-11-06 21:00", 0, 21))
        self.assertTrue(self.allowed("2026-11-05 07:59", 1, 21))
        self.assertFalse(self.allowed("2026-11-05 08:00", 3, 21))
        self.assertTrue(self.allowed("2026-11-07 10:00", 3, 21), "Saturday by day: GPUs 2 and 3")
        self.assertFalse(self.allowed("2026-11-07 10:00", 1, 21))
        self.assertTrue(self.allowed("2026-11-07 22:30", 1, 21), "Saturday night: every GPU")


if __name__ == "__main__":
    unittest.main()
