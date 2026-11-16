#!/usr/bin/env bash
# One pass of week 33's automatic stages (added 2026-11-09; RULE_week33.txt part 2), made from advance_week.sh (week
# 32's, unchanged): affect.sweep.week33 advance writes every sweep that is due, from validation files only, into
# artifacts/advance_week33/sweeps/; then each sweep file
# written there and not yet queued is smoke-tested on GPU $SMOKE_GPU and queued (scripts/smoke_enqueue_files.sh), the
# highest priority first, one at a time, each only while that GPU may run now and 15 minutes later (common.sh
# gpu_allowed_at) and queue/STOP does not exist; the rest wait for the next pass. A sweep whose smoke test fails is
# marked (artifacts/advance_week33/smoke_failed/<id>.txt, with its smoke log) and not tried again until the mark is
# removed. scripts/gpu_week.sh starts this in its own tmux session (advance33, with HOOK_FILE, ENABLE_FILE and
# HOOK_SESSION set to this script, artifacts/advance_week33/ENABLED and advance33) every 15 minutes while a GPU may run
# and artifacts/advance_week33/ENABLED exists; by hand:
#     tmux new -d -s advance33 "SMOKE_GPU=3 bash ~/affect/scripts/advance_week33.sh"
# It ends by itself. Decisions: artifacts/advance_week33/log.txt and decisions/; this pass: logs/advance_week33_<utc>.txt.
# The hours (added 2026-11-14): the week's since RULE_week33_addendum6.txt, every GPU from 20:00 to 10:00 and weekends
# nights only, as line 13 starts the week script; its tmux session does not get the week script's settings, and
# gpu_allowed_at's defaults are week 32's, which let GPUs 2 and 3 run on weekend days.
set -uo pipefail
source "$(dirname "$0")/common.sh"
export NIGHT_START=20 NIGHT_END=10 WEEKEND=nights ALL_FROM="" DAY_GPUS="2 3"
cd "$AFFECT_ROOT" || exit 1
SMOKE_GPU="${SMOKE_GPU:?SMOKE_GPU (0 to 3) must be set}"
case "$SMOKE_GPU" in 0|1|2|3) ;; *) echo "SMOKE_GPU must be 0, 1, 2 or 3, not $SMOKE_GPU"; exit 1 ;; esac
OUT="$AFFECT_ROOT/artifacts/advance_week33"
mkdir -p "$OUT/smoke_failed"
LOG="$AFFECT_ROOT/logs/advance_week33_$(stamp).txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if ! affect_python -m affect.sweep.week33 advance > "$LOG.advance" 2>&1; then
    say "affect.sweep.week33 advance FAILED: nothing queued; see $LOG.advance"
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
