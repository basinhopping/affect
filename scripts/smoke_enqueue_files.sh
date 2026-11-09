#!/usr/bin/env bash
# Smoke-test, then enqueue, sweep files given by path (added 2026-10-18), for sweeps that arrive while workers run.
# unpack.sh refuses to install a pushed code archive while gpu sessions run, so the new sweep files are taken out of
# that archive into a folder of their own (with this script), and the tree installed in ~/affect, the workers' code,
# smoke-tests and enqueues them: nothing in ~/affect changes but artifacts/, logs/ and queue/. Run on the node,
# usually in its own tmux session:
#     SMOKE_GPU=2 bash <folder>/scripts/smoke_enqueue_files.sh <folder>/sweeps/<id>.json [...]
# The same loop as smoke_enqueue.sh, which takes sweep IDs and reads ~/affect/sweeps: each file (an absolute path,
# named <sweep_id>.json) is smoke-tested (affect.sh smoke) on the GPU that SMOKE_GPU names (0 unless set) and enqueued
# only if the test ends with SMOKE OK; a sweep already enqueued (artifacts/sweeps/<id> exists) is skipped; one that
# fails is reported as NOT ENQUEUED with its smoke log, and the next file follows. Jobs run in priority order, so the
# order given only orders sweeps of equal priority. Ends with ALL DONE and the number of pending jobs.
# Since 2026-11-02 the verdict comes from the smoke test's own exit status and output, kept in
# logs/smoke_<sweep id>_<utc>.txt (the log named in NOT ENQUEUED), not from the newest logs/smoke_*.log: two smoke
# tests may now run at once (week 32's automatic stages beside a night's enqueue), and the newest log could be the
# other one's.
set -uo pipefail
ROOT="${AFFECT_ROOT:-$HOME/affect}"
cd "$ROOT" || exit 1
mkdir -p logs
SMOKE_GPU="${SMOKE_GPU:-0}"
case "$SMOKE_GPU" in 0|1|2|3) ;; *) echo "SMOKE_GPU must be 0, 1, 2 or 3, not $SMOKE_GPU"; exit 1 ;; esac
for f in "$@"; do
    case "$f" in /*) ;; *) echo "NOT ENQUEUED: $f, not an absolute path"; continue ;; esac
    if [ ! -f "$f" ]; then echo "NOT ENQUEUED: $f, no such file"; continue; fi
    s=$(basename "$f" .json)
    if ! grep -q "\"sweep_id\": \"$s\"" "$f"; then echo "NOT ENQUEUED: $f, its sweep_id is not its name"; continue; fi
    if [ -d "artifacts/sweeps/$s" ]; then echo "already enqueued: $s"; continue; fi
    L="logs/smoke_${s}_$(date -u +%Y%m%dT%H%M%S%NZ).txt"
    CUDA_VISIBLE_DEVICES="$SMOKE_GPU" bash scripts/affect.sh smoke --sweep-file "$f" --expected-gpus 1 > "$L" 2>&1
    RC=$?
    grep -E "^sweep |FAILED|^timing|^SMOKE" "$L"
    if [ "$RC" = 0 ] && grep -q "^SMOKE OK" "$L"; then
        bash scripts/affect.sh enqueue --sweep-file "$f" 2>&1 | grep -E '^ "(sweep_id|configs|jobs)"|Error'
    else
        echo "NOT ENQUEUED: $s, see $L"
    fi
done
echo "ALL DONE $(date -u +%FT%TZ), pending $(ls queue/pending | wc -l)"
