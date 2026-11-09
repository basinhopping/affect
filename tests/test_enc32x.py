"""scripts/enc32x.sh (added 2026-11-07, RULE_week32_addendum9.txt part N) in a throwaway root, with stand-ins for the
extraction, the probe night, the classical sweeps, the smoke test and enqueue, nvidia-smi and affect.sweep.week32, and
the week's window always open: BUD-E-Whisper's and Whisper medium's mean vectors in one probe night (their eight probe
sweeps together), their layers chosen, then for each in the order given its eight CPU sweeps, its IEMOCAP frames and
their check, EmoFilm's mean vectors and frames and their check, and its four screens smoke-tested, queued and ordered;
a second run makes nothing again; a missing snapshot, an unknown model or too little free space stops it before
anything runs, with the reason."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.support import REPO
from tests.test_enc32s import FAKE_CLASSICAL, FAKE_NIGHT, FAKE_ORDER, FAKE_PRETRAINED, FAKE_SMI, FAKE_SMOKE

LAYERS = {"bud_e_whisper": 13, "whisper_medium": 25}
CHOSEN = {"bud_e_whisper": 7, "whisper_medium": 18}
LETTER = {"bud_e_whisper": "b", "whisper_medium": "m"}
SHORT = {"bud_e_whisper": "bude", "whisper_medium": "wmed"}
FAKE_PYTHON = r"""#!/usr/bin/env bash
# stand-in for ~/venvs/affect/bin/python: -c runs, affect.sweep.week32 is simulated
if [ "$1" = "-c" ]; then exec python3 "$@"; fi
[ "$1 $2" = "-m affect.sweep.week32" ] || { echo "unexpected python $*"; exit 2; }
shift 2; echo "week32 $*" >> "$FAKE_DIR/calls"
OUT="$AFFECT_ROOT/artifacts/advance_week32"
cmd="$1"; shift; d=iemocap
while [ "$#" -gt 0 ]; do case "$1" in --dataset) d="$2"; shift 2 ;; --model) m="$2"; shift 2 ;;
    --layer) k="$2"; shift 2 ;; *) shift ;; esac; done
case "$m" in bud_e_whisper) x=b; short=bude; l=7 ;; whisper_medium) x=m; short=wmed; l=18 ;; *) x=?; short=?; l=0 ;; esac
case "$cmd" in
  check-frames) mkdir -p "$OUT/frames_checked"; echo '{}' > "$OUT/frames_checked/${d}_${m}_l${k}.json"
      echo "FRAMES OK: $m layer $k ($d)" ;;
  choose-layers) python3 -c "import json,sys; l=int(sys.argv[2]); json.dump({'setups': {s: {'layer': l + i, 'validation_ua': 0.7} for i, s in enumerate(('so4', 'pooled4', 'so3', 'pooled3'))}}, open(sys.argv[1], 'w'))" "$OUT/layers_$m.json" "$l"
      echo "chosen" ;;
  encoder-sweeps) mkdir -p "$OUT/encoders"
      for h in pool rnn transformer lif; do f="$OUT/encoders/heads32${x}_${h}_screen.json"
          echo "{\"sweep_id\": \"heads32${x}_${h}_screen\"}" > "$f"; echo "SCREEN $f"; done
      for s in classical_refit32_probe32_${m}_so4 classical_refit32_probe32_${m}_pooled4 \
               classical_refit32_probe32_${m}_so3 classical_refit32_probe32_${m}_pooled3 \
               classical_spk32_$short classical_script32_$short classical_oscript32_$short classical_gnorm32_$short; do
          echo "{\"sweep_id\": \"$s\"}" > "$OUT/encoders/$s.json"; echo "CLASSICAL $OUT/encoders/$s.json"; done ;;
  *) echo "unexpected week32 $cmd"; exit 2 ;;
esac
"""


@unittest.skipIf(shutil.which("bash") is None, "needs bash")
class TestEnc32x(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.home, self.root, self.fake = base / "home", base / "home" / "affect", base / "fake"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "queue" / "pending").mkdir(parents=True)
        (self.root / "artifacts" / "advance_week32").mkdir(parents=True)
        self.fake.mkdir()
        shutil.copy(REPO / "scripts" / "enc32x.sh", self.root / "scripts" / "enc32x.sh")
        common = (REPO / "scripts" / "common.sh").read_text()
        # the week's window always open here; the real gpu_allowed_at is tested in test_enc32s
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

    def tearDown(self):
        self.tmp.cleanup()

    def snapshots(self, *repos):
        for repo in repos:
            (self.home / "cache/huggingface" / f"models--{repo}" / "snapshots" / "0123abcd").mkdir(parents=True)

    def run_script(self, *models, **extra) -> subprocess.CompletedProcess:
        env = dict(os.environ, HOME=str(self.home), PATH=self.path, FAKE_DIR=str(self.fake), ENC_GPU="3",
                   SLEEP_S="0", AFFECT_ROOT=str(self.root), LAYERS_JSON=json.dumps(LAYERS), **extra)
        env.pop("NIGHT_START", None)
        if "MIN_FREE_GB" not in extra:
            env.pop("MIN_FREE_GB", None)
        return subprocess.run(["bash", str(self.root / "scripts" / "enc32x.sh"), *models], env=env,
                              capture_output=True, text=True, timeout=120)

    def calls(self) -> list:
        path = self.fake / "calls"
        return path.read_text().splitlines() if path.exists() else []

    def log(self) -> str:
        return next((self.root / "logs").glob("enc32x_*.txt")).read_text()

    def test_both_encoders_once_then_nothing_again(self):
        self.snapshots("laion--BUD-E-Whisper", "openai--whisper-medium")
        done = self.run_script("bud_e_whisper", "whisper_medium")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        log = self.log()
        self.assertIn("weekday nights from 21:00", log)
        self.assertIn("ALL DONE: 0 CPU sweeps ended with a failure", log)
        calls = self.calls()
        night = [c for c in calls if c.startswith("night ")]
        self.assertEqual(len(night), 1, "one probe night for both")
        self.assertIn("--label-set four_merged --models bud_e_whisper whisper_medium --sweeps "
                      "classical_pretrained_bud_e_whisper_so4_1 classical_pretrained_bud_e_whisper_pooled4_1 "
                      "classical_pretrained_bud_e_whisper_so3_1 classical_pretrained_bud_e_whisper_pooled3_1 "
                      "classical_pretrained_whisper_medium_so4_1 classical_pretrained_whisper_medium_pooled4_1 "
                      "classical_pretrained_whisper_medium_so3_1 classical_pretrained_whisper_medium_pooled3_1", night[0])
        extractions = [c for c in calls if c.startswith("pretrained ")]
        self.assertEqual([(c.split()[1], c.split()[3], c.split()[9]) for c in extractions], [
            ("extract-frames", "iemocap", "bud_e_whisper"),
            ("extract-pretrained", "emofilm", "bud_e_whisper"), ("extract-frames", "emofilm", "bud_e_whisper"),
            ("extract-frames", "iemocap", "whisper_medium"),
            ("extract-pretrained", "emofilm", "whisper_medium"), ("extract-frames", "emofilm", "whisper_medium")])
        self.assertIn("--layers 7", extractions[0])
        self.assertIn("--layers 7", extractions[2])
        self.assertIn("--layers 18", extractions[3])
        self.assertIn("--layers 18", extractions[5])
        self.assertIn("--label-set four_merged --subset all", extractions[0])
        self.assertIn("--label-set three_no_neutral --subset all", extractions[1])
        self.assertEqual(sorted(c.split("/")[-1] for c in calls if c.startswith("classical ")),
                         sorted(f"{s}.json --max-consecutive-failures 3" for m in LAYERS for s in (
                             [f"classical_refit32_probe32_{m}_{x}" for x in ("so4", "pooled4", "so3", "pooled3")]
                             + [f"classical_{p}32_{SHORT[m]}" for p in ("spk", "script", "oscript", "gnorm")])))
        smoked = [c.split()[2] for c in calls if c.startswith("smoke ")]
        self.assertEqual(smoked, [f"heads32{LETTER[m]}_{h}_screen" for m in LAYERS
                                  for h in ("pool", "rnn", "transformer", "lif")])
        self.assertTrue(all(c.split()[1] == "3" for c in calls if c.startswith("smoke ")))
        self.assertEqual(len([c for c in calls if c.startswith("order ")]), 2)
        checked = sorted(p.stem for p in (self.root / "artifacts/advance_week32/frames_checked").glob("*.json"))
        self.assertEqual(checked, ["emofilm_bud_e_whisper_l7", "emofilm_whisper_medium_l18",
                                   "iemocap_bud_e_whisper_l7", "iemocap_whisper_medium_l18"])
        # BUD-E-Whisper's screens are queued before anything of Whisper medium's frames
        first_medium_frames = next(i for i, c in enumerate(calls)
                                   if c.startswith("pretrained extract-frames") and "whisper_medium" in c)
        last_bude_smoke = max(i for i, c in enumerate(calls) if c.startswith("smoke ") and "heads32b_" in c)
        self.assertLess(last_bude_smoke, first_medium_frames)
        # a second run makes nothing again
        (self.fake / "calls").unlink()
        again = self.run_script("bud_e_whisper", "whisper_medium")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        calls = self.calls()
        self.assertEqual([c for c in calls if c.startswith(("pretrained ", "night ", "smoke "))], [])
        self.assertEqual(len([c for c in calls if c.startswith("classical ")]), 16, "the CPU sweeps resume (skip)")

    def test_a_missing_snapshot_stops_before_anything(self):
        self.snapshots("laion--BUD-E-Whisper")
        done = self.run_script("bud_e_whisper", "whisper_medium")
        self.assertEqual(done.returncode, 1)
        self.assertIn("STOPPED: not exactly one snapshot of whisper_medium", self.log())
        self.assertEqual(self.calls(), [])

    def test_an_unknown_model_stops_before_anything(self):
        self.snapshots("laion--BUD-E-Whisper", "openai--whisper-small")
        done = self.run_script("bud_e_whisper", "whisper_small")
        self.assertEqual(done.returncode, 1)
        self.assertIn("STOPPED: whisper_small is not one of enc32x's models", self.log())
        self.assertEqual(self.calls(), [])

    def test_too_little_space_stops_before_the_mean_vectors(self):
        self.snapshots("laion--BUD-E-Whisper", "openai--whisper-medium")
        done = self.run_script("bud_e_whisper", "whisper_medium", MIN_FREE_GB="1000000")
        self.assertEqual(done.returncode, 1)
        self.assertIn("1000000 needed before the mean vectors of bud_e_whisper whisper_medium (the workers stop below 10)",
                      self.log())
        self.assertEqual(self.calls(), [])

    def test_no_model_is_a_usage_error(self):
        done = self.run_script()
        self.assertEqual(done.returncode, 1)
        self.assertIn("usage: enc32x.sh", done.stdout)


if __name__ == "__main__":
    unittest.main()
