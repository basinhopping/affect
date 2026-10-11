#!/usr/bin/env bash
# Smoke-test, then enqueue, sweeps in the order given (added 2026-10-03; the loop of night 8's enqueue line, as a
# script). Run on the node, usually in its own tmux session:
#     bash ~/affect/scripts/smoke_enqueue.sh <sweep id> [<sweep id> ...]
# Each sweep's file is sweeps/<id>.json. A sweep already enqueued (artifacts/sweeps/<id> exists) is skipped. Each
# other sweep is smoke-tested (affect.sh smoke) on GPU 0, or on the GPU that SMOKE_GPU names (added 2026-10-10, for
# weekend days when only GPUs 2 and 3 are used), and enqueued only if the test ends with SMOKE OK; one that
# fails is reported as NOT ENQUEUED with its smoke log, and the next sweep follows. Jobs run in priority order,
# so the order given only orders sweeps of equal priority. Ends with ALL DONE and the number of pending jobs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
SMOKE_GPU="${SMOKE_GPU:-0}"
case "$SMOKE_GPU" in 0|1|2|3) ;; *) echo "SMOKE_GPU must be 0, 1, 2 or 3, not $SMOKE_GPU"; exit 1 ;; esac
for s in "$@"; do
    if [ -d "artifacts/sweeps/$s" ]; then echo "already enqueued: $s"; continue; fi
    if [ ! -f "sweeps/$s.json" ]; then echo "NOT ENQUEUED: $s, no file sweeps/$s.json"; continue; fi
    CUDA_VISIBLE_DEVICES="$SMOKE_GPU" bash scripts/affect.sh smoke --sweep-file "sweeps/$s.json" --expected-gpus 1 \
        | grep -E "^sweep |FAILED|^timing|^SMOKE"
    RC=${PIPESTATUS[0]}
    L=$(ls -t logs/smoke_*.log 2>/dev/null | head -1)
    if [ "$RC" = 0 ] && [ -n "$L" ] && grep -q "^SMOKE OK" "$L"; then
        bash scripts/affect.sh enqueue --sweep-file "sweeps/$s.json" 2>&1 | grep -E '^ "(sweep_id|configs|jobs)"|Error'
    else
        echo "NOT ENQUEUED: $s, see $L"
    fi
done
echo "ALL DONE $(date -u +%FT%TZ), pending $(ls queue/pending | wc -l)"
