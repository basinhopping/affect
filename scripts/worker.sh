#!/usr/bin/env bash
# Workers for one GPU. Start one tmux session per GPU, on the node:
#     tmux new -d -s gpu0 "bash ~/affect/scripts/worker.sh 0 3"
#     tmux new -d -s gpu1 "bash ~/affect/scripts/worker.sh 1 3"
#     tmux new -d -s gpu2 "bash ~/affect/scripts/worker.sh 2 3"
#     tmux new -d -s gpu3 "bash ~/affect/scripts/worker.sh 3 3"
# The first argument is the GPU (0 to 3); CUDA_VISIBLE_DEVICES is set to it, so each worker sees
# only that card. The second is how many worker processes share the card. On night 1 (3 per card)
# the few runs that ended up sharing a card with fewer others ran proportionally faster, so 3 already
# keeps an H100 busy with the CNN; more would mostly slow each job. Each process takes jobs from
# ~/affect/queue until it is empty, writes
# logs/worker_gpu<g>_<n>_<utc>.log, and prints "JOB DONE" or "JOB FAILED" for every job. The tmux
# session closes when all its processes have finished: that is the normal end.
# A process also stops after 5 failed jobs in a row, so a problem that fails every job cannot
# empty the queue into queue/failed/; after the fix, affect.sh requeue-failed --sweep <id> puts
# them back.
# To stop early: touch ~/affect/queue/STOP (each process finishes its current job, then exits).
# To stop one GPU's workers only: touch ~/affect/queue/STOP_gpu<g> (added 2026-10-03; remove the
# file before starting that GPU's workers again). scripts/gpu_schedule.sh uses both.
set -euo pipefail
source "$(dirname "$0")/common.sh"
GPU="${1:?usage: worker.sh <gpu 0-3> <processes on this gpu>}"
PROCESSES="${2:?usage: worker.sh <gpu 0-3> <processes on this gpu>}"
export CUDA_VISIBLE_DEVICES="$GPU"
# Several processes share the CPU; the 32-thread cap in common.sh is per process.
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$AFFECT_ROOT"
STAMP=$(stamp)
for n in $(seq 1 "$PROCESSES"); do
    LOG="$AFFECT_ROOT/logs/worker_gpu${GPU}_${n}_${STAMP}.log"
    affect_python -m affect.cli worker --worker-id "gpu${GPU}_${n}" --expected-gpus 1 --min-free-gb 10 \
        --max-consecutive-failures 5 2>&1 | tee -a "$LOG" &
    sleep 5
done
wait
echo "all workers on GPU $GPU finished at $(date -u +%FT%TZ)"
