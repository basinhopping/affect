#!/usr/bin/env bash
# One pass of week 32's automatic stages (added 2026-11-02; RULE_week32.txt part 2): affect.sweep.week32 advance writes
# every stage sweep that is due, from validation files only, into artifacts/advance_week32/sweeps/; then each sweep file
# written there and not yet queued is smoke-tested on GPU $SMOKE_GPU and queued (scripts/smoke_enqueue_files.sh), the
# highest priority first, one at a time, each only while that GPU may run now and 15 minutes later (common.sh
# gpu_allowed_at) and queue/STOP does not exist; the rest wait for the next pass. A sweep whose smoke test fails is
# marked (artifacts/advance_week32/smoke_failed/<id>.txt, with its smoke log) and not tried again until the mark is
# removed. scripts/gpu_week.sh starts this in its own tmux session
# (advance32) every 15 minutes while a GPU may run and artifacts/advance_week32/ENABLED exists; by hand:
#     tmux new -d -s advance32 "SMOKE_GPU=3 bash ~/affect/scripts/advance_week.sh"
# It ends by itself. Decisions: artifacts/advance_week32/log.txt and decisions/; this pass: logs/advance_week_<utc>.txt.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$AFFECT_ROOT" || exit 1
SMOKE_GPU="${SMOKE_GPU:?SMOKE_GPU (0 to 3) must be set}"
case "$SMOKE_GPU" in 0|1|2|3) ;; *) echo "SMOKE_GPU must be 0, 1, 2 or 3, not $SMOKE_GPU"; exit 1 ;; esac
OUT="$AFFECT_ROOT/artifacts/advance_week32"
mkdir -p "$OUT/smoke_failed"
LOG="$AFFECT_ROOT/logs/advance_week_$(stamp).txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if ! affect_python -m affect.sweep.week32 advance > "$LOG.advance" 2>&1; then
    say "affect.sweep.week32 advance FAILED: nothing queued; see $LOG.advance"
    tail -5 "$LOG.advance" | tee -a "$LOG"
    exit 1
fi
cat "$LOG.advance" >> "$LOG"
mapfile -t FILES < <(grep '^ENQUEUE ' "$LOG.advance" | cut -c9-)
rm -f "$LOG.advance"
if [ "${#FILES[@]}" -eq 0 ]; then say "nothing to queue"; exit 0; fi
say "${#FILES[@]} sweep files to smoke-test on GPU $SMOKE_GPU and queue: $(for f in "${FILES[@]}"; do basename "$f" .json; done | tr '\n' ' ')"
for f in "${FILES[@]}"; do      # one at a time, each only while the GPU may run now and 15 minutes later
    now=$(date +%s)
    if [ -e queue/STOP ]; then say "queue/STOP exists: the rest wait"; break; fi
    if ! gpu_allowed_at "$now" "$SMOKE_GPU" || ! gpu_allowed_at $((now + 900)) "$SMOKE_GPU"; then
        say "GPU $SMOKE_GPU may not run now or in 15 minutes: the rest wait for the next pass"; break
    fi
    SMOKE_GPU="$SMOKE_GPU" bash "$AFFECT_ROOT/scripts/smoke_enqueue_files.sh" "$f" 2>&1 | tee -a "$LOG" | \
    while IFS= read -r line; do
        if [[ "$line" =~ ^NOT\ ENQUEUED:\ ([A-Za-z0-9_-]+),\ see\ (.*)$ ]]; then
            echo "smoke test failed $(date -u +%FT%TZ); log ${BASH_REMATCH[2]}" > "$OUT/smoke_failed/${BASH_REMATCH[1]}.txt"
        fi
    done
done
say "pass done: pending $(ls queue/pending | wc -l); smoke failures marked $(ls "$OUT/smoke_failed" | wc -l)"
