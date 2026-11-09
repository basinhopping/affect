"""scripts/gpu_week.sh in a throwaway root with stand-ins for tmux and nvidia-smi and a simulated clock (added
2026-11-02): which GPUs get workers in each window of the week (weekday nights every free GPU, weekday days none,
weekend days GPUs 2 and 3, weekend nights every free GPU), stop files at the edges, no start without pending jobs or
while queue/STOP exists or while another user is on a GPU (our own processes' memory not counted), the end at WEEK_END,
and the hook: its tmux session started only when its file and the enable file exist and queue/STOP does not, every
third loop, not while it is still running, and only when a GPU may run now and 30 minutes later, with SMOKE_GPU the
highest-numbered such GPU of ours. Added after the review of 2026-11-02: stop files one look before the window closes
(no job starts after 08:00), a GPU held after its workers stopped for jobs failed in a row (until its HELD file is
removed), and jobs a killed worker left in queue/running requeued while no gpu session runs. Added later that night
(Harrison: GPU 2 for another project): a process counts as ours only when it runs affect.cli, so another project of
ours keeps a GPU busy; a reserved GPU (queue/RESERVED_gpu<g>) gets no workers, sends ours off, and is released after
an hour with nothing else on it, or at once when its file is removed."""
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tests.support import REPO

PACIFIC = ZoneInfo("America/Los_Angeles")
FAKE_TMUX = r"""#!/usr/bin/env bash
S="$FAKE_DIR/sessions"; mkdir -p "$S"
case "$1" in
  has-session) name="${3#=}"; [ -e "$S/$name" ] ;;
  new) shift; while [ "$#" -gt 0 ]; do case "$1" in -d) shift ;; -s) name="$2"; shift 2 ;; *) cmd="$1"; shift ;; esac; done
       touch "$S/$name"; echo "$name|$cmd" >> "$FAKE_DIR/started" ;;
  ls) for f in "$S"/*; do [ -e "$f" ] && echo "$(basename "$f"): 1 windows"; done; true ;;
  *) exit 1 ;;
esac
"""
FAKE_SMI = r"""#!/usr/bin/env bash
case "$*" in
  *index,uuid*) for g in 0 1 2 3; do echo "$g, GPU-$g"; done ;;
  *compute-apps*) for g in ${FAKE_BUSY:-}; do echo "GPU-$g, 999999, 5000"; done
                  for x in ${FAKE_OURS:-}; do IFS=: read -r g p m <<< "$x"; echo "GPU-$g, $p, $m"; done ;;
  *memory.used*) for g in 0 1 2 3; do m=0; for x in ${FAKE_MEM:-}; do IFS=: read -r gg mm <<< "$x"
                 [ "$gg" = "$g" ] && m=$mm; done; echo "$g, $m"; done ;;
esac
"""


def at(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=PACIFIC).timestamp())


@unittest.skipIf(shutil.which("bash") is None, "needs bash")
class TestGpuWeek(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "affect"
        self.fake = Path(self.tmp.name) / "fake"
        self.fake.mkdir()
        (self.root / "scripts").mkdir(parents=True)
        shutil.copy(REPO / "scripts" / "gpu_week.sh", self.root / "scripts" / "gpu_week.sh")
        for state in ("pending", "running", "done", "failed"):
            (self.root / "queue" / state).mkdir(parents=True)
        for name, text in (("tmux", FAKE_TMUX), ("nvidia-smi", FAKE_SMI)):
            (self.fake / name).write_text(text)
            (self.fake / name).chmod(0o755)
        self.end = at("2026-11-09 08:00")

    def tearDown(self):
        self.tmp.cleanup()

    def pend(self, n: int):
        for k in range(n):
            (self.root / "queue" / "pending" / f"185_{k:06d}__s__r{k}.json").write_text("{}")

    def run_at(self, *times: str, busy: str = "", **extra) -> str:
        env = dict(os.environ, AFFECT_ROOT=str(self.root), TMUX_CMD=str(self.fake / "tmux"),
                   NVIDIA_SMI_CMD=str(self.fake / "nvidia-smi"), FAKE_DIR=str(self.fake), FAKE_BUSY=busy,
                   WEEK_END=str(self.end), TEST_TIMES=" ".join(str(at(t)) for t in times), **extra)
        r = subprocess.run(["bash", str(self.root / "scripts" / "gpu_week.sh")], capture_output=True, text=True,
                           env=env, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        logs = sorted((self.root / "logs").glob("gpu_week_*.txt"))
        return logs[-1].read_text()

    def sessions(self) -> set:
        d = self.fake / "sessions"
        return {p.name for p in d.iterdir()} if d.exists() else set()

    def stops(self) -> set:
        return {p.name for p in (self.root / "queue").glob("STOP*")}

    def end_sessions(self, *names):
        for n in names:
            (self.fake / "sessions" / n).unlink()

    def test_weekday_night_starts_every_free_gpu_and_the_morning_stops_them(self):
        self.pend(10)
        log = self.run_at("2026-11-02 21:00")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})
        self.assertIn("window: night", log)
        started = (self.fake / "started").read_text()
        self.assertIn("worker.sh 2 3", started)
        self.run_at("2026-11-03 08:05")
        self.assertEqual(self.stops(), {"STOP_gpu0", "STOP_gpu1", "STOP_gpu2", "STOP_gpu3"})
        self.end_sessions("gpu0", "gpu1", "gpu2", "gpu3")
        self.run_at("2026-11-03 12:00", "2026-11-03 17:55")
        self.assertEqual(self.sessions(), set(), "nothing starts on a weekday by day")
        self.run_at("2026-11-03 18:05")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})
        self.assertEqual(self.stops(), set(), "a GPU's stop file goes when it starts again")

    def test_another_users_gpu_is_left_alone_until_it_is_free(self):
        self.pend(10)
        log = self.run_at("2026-11-02 21:00", busy="0 1")
        self.assertEqual(self.sessions(), {"gpu2", "gpu3"})
        self.assertIn("GPU 0: NOT started: processes on it that are not our runs: 1", log)
        self.run_at("2026-11-02 23:00", busy="")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})

    def test_no_pending_job_no_start_then_new_jobs_start_the_workers(self):
        log = self.run_at("2026-11-04 02:00")
        self.assertEqual(self.sessions(), set())
        self.assertIn("no pending job", log)
        self.pend(3)
        self.run_at("2026-11-04 02:05")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})

    def test_weekend_days_use_gpus_2_and_3_and_nights_all_four(self):
        self.pend(10)
        self.run_at("2026-11-06 23:00")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})
        log = self.run_at("2026-11-07 10:00")
        self.assertEqual(self.stops(), {"STOP_gpu0", "STOP_gpu1"})
        self.assertIn("window: weekend day (GPUs that may run: 2 3)", log)
        self.end_sessions("gpu0", "gpu1", "gpu2")
        self.run_at("2026-11-07 10:05")
        self.assertEqual(self.sessions(), {"gpu2", "gpu3"}, "GPU 2 restarts by day, GPUs 0 and 1 do not")
        self.run_at("2026-11-07 22:05")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})
        self.assertEqual(self.stops(), set())

    def test_friday_evening_and_monday_morning_are_nights(self):
        self.pend(10)
        self.run_at("2026-11-06 12:00")
        self.assertEqual(self.sessions(), set())
        self.run_at("2026-11-06 18:01")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})
        self.end_sessions("gpu0", "gpu1", "gpu2", "gpu3")
        self.run_at("2026-11-09 03:00")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})

    def test_queue_stop_starts_nothing(self):
        self.pend(10)
        (self.root / "queue" / "STOP").touch()
        log = self.run_at("2026-11-02 21:00")
        self.assertEqual(self.sessions(), set())
        self.assertIn("queue/STOP exists: no GPU is started", log)

    def test_the_end_writes_stop_and_exits(self):
        self.pend(10)
        (self.root / "queue" / "STOP_gpu0").touch()
        log = self.run_at("2026-11-09 08:00", "2026-11-09 08:05")
        self.assertEqual(self.stops(), {"STOP"})
        self.assertIn("the week's run is over", log)
        self.assertNotIn("test times used up", log, "it exits at the end, before using the second time")

    def test_the_hook_starts_its_session_only_with_its_file_and_the_enable_file(self):
        hook = self.root / "scripts" / "advance_week.sh"
        hook.write_text("echo hook ran\n")
        three = ("2026-11-03 21:00", "2026-11-03 21:05", "2026-11-03 21:10")
        self.run_at(*three, HOOK_EVERY="3")
        self.assertNotIn("advance32", self.sessions(), "no enable file: no hook")
        enable = self.root / "artifacts" / "advance_week32" / "ENABLED"
        enable.parent.mkdir(parents=True)
        enable.touch()
        log = self.run_at(*three, HOOK_EVERY="3")
        self.assertIn("advance32", self.sessions())
        started = [line for line in (self.fake / "started").read_text().splitlines() if line.startswith("advance32|")]
        self.assertEqual(started, [f"advance32|SMOKE_GPU=3 bash {hook}"], "once, on the third loop, smoke tests on GPU 3")
        self.assertIn("hook: advance32 started", log)
        log = self.run_at(*three, HOOK_EVERY="3")
        self.assertIn("hook: advance32 is still running", log)
        self.end_sessions("advance32")
        self.pend(10)
        self.run_at("2026-11-03 21:20", busy="3")
        self.end_sessions("gpu0", "gpu1", "gpu2")
        self.run_at("2026-11-03 21:25", "2026-11-03 21:30", "2026-11-03 21:35", busy="3", HOOK_EVERY="3")
        started = [line for line in (self.fake / "started").read_text().splitlines() if line.startswith("advance32|")]
        self.assertEqual(started[-1], f"advance32|SMOKE_GPU=2 bash {hook}", "GPU 3 is another user's: GPU 2")

    def test_the_hook_waits_for_a_window_that_lasts(self):
        hook = self.root / "scripts" / "advance_week.sh"
        hook.write_text("echo hook ran\n")
        enable = self.root / "artifacts" / "advance_week32" / "ENABLED"
        enable.parent.mkdir(parents=True)
        enable.touch()
        log = self.run_at("2026-11-03 12:00", "2026-11-03 12:05", "2026-11-03 12:10", HOOK_EVERY="3")
        self.assertNotIn("advance32", self.sessions(), "weekday day: no GPU, no smoke test")
        self.assertIn("hook: not started", log)
        log = self.run_at("2026-11-04 07:35", "2026-11-04 07:40", "2026-11-04 07:45", HOOK_EVERY="3")
        self.assertNotIn("advance32", self.sessions(), "08:00 is less than 30 minutes away")
        self.run_at("2026-11-07 07:35", "2026-11-07 07:40", "2026-11-07 07:45", HOOK_EVERY="3")
        started = [line for line in (self.fake / "started").read_text().splitlines() if line.startswith("advance32|")]
        self.assertEqual(started, [f"advance32|SMOKE_GPU=3 bash {hook}"], "Saturday 08:00 keeps GPUs 2 and 3")

    def test_stop_files_come_one_look_before_the_window_closes(self):
        self.pend(10)
        self.run_at("2026-11-04 07:50")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu2", "gpu3"})
        self.assertEqual(self.stops(), set(), "07:55 is still night")
        self.run_at("2026-11-04 07:57")
        self.assertEqual(self.stops(), {"STOP_gpu0", "STOP_gpu1", "STOP_gpu2", "STOP_gpu3"}, "08:02 is day: stop now")
        self.end_sessions("gpu0", "gpu1", "gpu2", "gpu3")
        self.run_at("2026-11-04 07:58")
        self.assertEqual(self.sessions(), set(), "nothing starts that would run past 08:00")
        self.run_at("2026-11-07 07:57")             # Saturday: GPUs 2 and 3 go on by day, 0 and 1 stop
        self.assertEqual(self.sessions(), {"gpu2", "gpu3"})
        log = self.run_at("2026-11-09 07:56")
        self.assertIn("the week's run is over", log, "the end comes at the last look before it")

    def test_a_gpu_whose_workers_failed_in_a_row_is_held(self):
        self.pend(10)
        self.run_at("2026-11-03 21:00")
        self.end_sessions("gpu2")
        worker_log = self.root / "logs" / "worker_gpu2_1_20261104T050000Z.log"
        worker_log.write_text("... worker gpu2_1: 5 jobs failed in a row, stopping so the rest of the queue stays pending\n")
        os.utime(worker_log, (at("2026-11-03 21:30"), at("2026-11-03 21:30")))
        log = self.run_at("2026-11-03 21:20", "2026-11-03 21:40")
        self.assertNotIn("gpu2", self.sessions())
        self.assertTrue((self.root / "queue" / "HELD_gpu2").exists())
        self.assertIn("GPU 2: HELD", log)
        self.run_at("2026-11-03 21:45")
        self.assertNotIn("gpu2", self.sessions(), "held until a person removes the file")
        (self.root / "queue" / "HELD_gpu2").unlink()
        self.run_at("2026-11-03 21:50")
        self.assertIn("gpu2", self.sessions(), "the old log does not hold it again once the file is gone")

    def test_the_hook_waits_while_queue_stop_exists(self):
        hook = self.root / "scripts" / "advance_week.sh"
        hook.write_text("echo hook ran\n")
        enable = self.root / "artifacts" / "advance_week32" / "ENABLED"
        enable.parent.mkdir(parents=True)
        enable.touch()
        (self.root / "queue" / "STOP").touch()
        log = self.run_at("2026-11-03 21:00", "2026-11-03 21:05", "2026-11-03 21:10", HOOK_EVERY="3")
        self.assertNotIn("advance32", self.sessions())
        self.assertIn("hook: not started (queue/STOP exists)", log)

    def test_jobs_left_in_running_go_back_when_no_worker_runs(self):
        name = "815_000007__s__r7.json"
        left = self.root / "queue" / "running" / "815_000007__s__r7.gpu3_2.json"
        left.write_text("{}")
        os.utime(left, (time.time() - 700, time.time() - 700))
        fresh = self.root / "queue" / "running" / "815_000008__s__r8.gpu3_1.json"
        fresh.write_text("{}")
        log = self.run_at("2026-11-03 12:00")
        self.assertTrue((self.root / "queue" / "pending" / name).exists())
        self.assertFalse(left.exists())
        self.assertTrue(fresh.exists(), "a job taken less than 10 minutes ago stays")
        self.assertIn("requeued 815_000007__s__r7.gpu3_2.json", log)

    def spawn(self, argv0: str) -> subprocess.Popen:
        """A process of ours whose command line starts with argv0 (a stand-in for a worker or another project)."""
        proc = subprocess.Popen(["bash", "-c", f'exec -a "{argv0}" sleep 120'])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        time.sleep(0.2)
        return proc

    def test_our_own_runs_memory_does_not_make_a_gpu_busy(self):
        self.pend(10)
        worker = self.spawn("python -m affect.cli worker --worker-id gpu3_1")
        self.run_at("2026-11-03 21:00", FAKE_MEM="3:6000", FAKE_OURS=f"3:{worker.pid}:5500")
        self.assertIn("gpu3", self.sessions(), "6,000 MiB in use, 5,500 of them our run's")
        self.end_sessions("gpu3")
        log = self.run_at("2026-11-03 21:05", FAKE_MEM="3:9000", FAKE_OURS=f"3:{worker.pid}:5500")
        self.assertNotIn("gpu3", self.sessions())
        self.assertIn("memory in use beyond ours: 3500", log)

    def test_another_project_of_ours_keeps_a_gpu_busy(self):
        """Harrison, 2026-11-02: GPU 2 runs another project of his, under the same user: it counts as busy."""
        self.pend(10)
        other = self.spawn("python train_other_project.py")
        log = self.run_at("2026-11-03 21:00", FAKE_MEM="2:3000", FAKE_OURS=f"2:{other.pid}:3000")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu3"})
        self.assertIn("GPU 2: NOT started: processes on it that are not our runs: 1", log)

    def test_a_reserved_gpu_waits_for_a_quiet_hour(self):
        self.pend(10)
        reserved = self.root / "queue" / "RESERVED_gpu2"
        reserved.write_text("GPU 2 reserved for another project\n")
        log = self.run_at("2026-11-03 21:00", "2026-11-03 21:30", "2026-11-03 21:59")
        self.assertEqual(self.sessions(), {"gpu0", "gpu1", "gpu3"})
        self.assertTrue(reserved.exists())
        self.assertIn("GPU 2: reserved", log)
        self.end_sessions("gpu0", "gpu1", "gpu3")
        log = self.run_at("2026-11-03 22:00", "2026-11-03 22:30", "2026-11-03 23:00", busy="2")
        self.assertNotIn("gpu2", self.sessions(), "in use all along: kept")
        self.assertTrue(reserved.exists())
        log = self.run_at("2026-11-03 23:00", "2026-11-03 23:30", "2026-11-04 00:00")
        self.assertIn("GPU 2: reservation released", log)
        self.assertFalse(reserved.exists())
        self.assertIn("gpu2", self.sessions(), "an hour with nothing else on it: used")

    def test_reserving_a_gpu_sends_our_workers_off_and_removing_the_file_frees_it(self):
        self.pend(10)
        self.run_at("2026-11-03 21:00")
        reserved = self.root / "queue" / "RESERVED_gpu2"
        reserved.write_text("reserved\n")
        self.run_at("2026-11-03 21:05")
        self.assertIn("STOP_gpu2", self.stops())
        self.end_sessions("gpu2")
        self.run_at("2026-11-03 21:10")
        self.assertNotIn("gpu2", self.sessions())
        reserved.unlink()
        self.run_at("2026-11-03 21:15")
        self.assertIn("gpu2", self.sessions())
        self.assertNotIn("STOP_gpu2", self.stops())

if __name__ == "__main__":
    unittest.main()
