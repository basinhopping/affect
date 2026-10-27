#!/usr/bin/env bash
# Install new code under running workers (added 2026-10-24): stop every worker and wait until none is left, install a
# code archive into ~/affect, check its tree digest, start the workers again, then smoke-test and enqueue sweeps while
# they run. Run on the node in its own tmux session, from another tree (~/affect_next holds the same archive's code),
# since ~/affect's files are replaced underneath it:
#     tmux new -d -s swap "bash ~/affect_next/scripts/swap_and_restart.sh <archive in ~/affect_incoming> <digest> \
#         [<sweep id> ...]"
# 1. queue/STOP: each worker finishes the job it is on, then exits. Waits until tmux lists no gpu session and no
#    classical process (affect.cli classical) runs, at most WAIT_MINUTES (default 90); on a timeout it removes
#    queue/STOP, gives every GPU left without workers its workers again (the old code), installs nothing and exits.
# 2. unpack.sh installs the archive; the tree digest of ~/affect (the README's recipe) must be <digest>. If it is not,
#    the archive installed before (the last line of CODE_VERSIONS.log) is installed again, and nothing is enqueued.
# 3. requeue-running, the STOP files removed, and gpu_schedule.sh's night step now: 3 workers on every GPU no other
#    user is on. The end step of a schedule already running (its own tmux session) still stops them at its time.
# 4. smoke_enqueue.sh with the sweeps given, on SMOKE_GPU (default 2), while the workers run.
# The first use, 2026-10-24, failed: unpack.sh then deleted the old code, and on the node's NFS home a file still
# held open by the night's schedule could not be deleted; unpack.sh moves old code aside since 2026-10-26.
# Log: logs/swap_<utc>.txt in ~/affect. For tests, AFFECT_HOME replaces $HOME and TMUX_CMD and PGREP_CMD replace tmux
# and pgrep; SWAP_POLL_SECONDS (default 20) is the wait between checks.
set -uo pipefail
BASE="${AFFECT_HOME:-$HOME}"
ROOT="$BASE/affect"
IN="$BASE/affect_incoming"
TMUX_CMD="${TMUX_CMD:-tmux}"
PGREP_CMD="${PGREP_CMD:-pgrep}"
POLL="${SWAP_POLL_SECONDS:-20}"
WAIT_MINUTES="${WAIT_MINUTES:-90}"
SMOKE_GPU="${SMOKE_GPU:-2}"
ARCHIVE="${1:?usage: swap_and_restart.sh <archive> <digest> [<sweep id> ...]}"
DIGEST="${2:?usage: swap_and_restart.sh <archive> <digest> [<sweep id> ...]}"
shift 2
SWEEPS=("$@")
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/swap_$(date -u +%Y%m%dT%H%M%SZ).txt"
say() { echo "$(date -u +%FT%TZ) ($(TZ=America/Los_Angeles date '+%a %H:%M %Z')) $*" | tee -a "$LOG"; }
tree_digest() {
    (cd "$1" && find . \( -path ./data -o -path ./artifacts -o -path ./queue -o -path ./logs -o -path ./CODE_VERSIONS.log \
        -o -name __pycache__ \) -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | cut -c1-16,65- \
        | sha256sum | cut -c1-64)
}
running() {   # how many gpu sessions and classical processes are still running
    local sessions classical
    sessions=$("$TMUX_CMD" ls 2>/dev/null | grep -c '^gpu')
    classical=$("$PGREP_CMD" -fc 'affect.cli classical' 2>/dev/null)
    echo $(( ${sessions:-0} + ${classical:-0} ))
}
restart() {
    (cd "$ROOT" && bash scripts/affect.sh requeue-running --all-workers-stopped) >> "$LOG" 2>&1
    rm -f "$ROOT/queue/STOP" "$ROOT/queue/STOP_gpu0" "$ROOT/queue/STOP_gpu1" "$ROOT/queue/STOP_gpu2" "$ROOT/queue/STOP_gpu3"
    (cd "$ROOT" && SCHEDULE="$(date +%s):night" bash scripts/gpu_schedule.sh) >> "$LOG" 2>&1
    say "workers started: $("$TMUX_CMD" ls 2>/dev/null | grep -c '^gpu') gpu sessions; pending $(ls "$ROOT/queue/pending" | wc -l)"
}

if [ ! -f "$IN/$ARCHIVE" ] || [ ! -f "$IN/$ARCHIVE.sha256" ]; then say "no $IN/$ARCHIVE or its .sha256: stop"; exit 1; fi
PREVIOUS=$(tail -1 "$ROOT/CODE_VERSIONS.log" | cut -f3)
BEFORE=$(tree_digest "$ROOT")
say "swap: $ROOT holds $PREVIOUS (tree $BEFORE); installing $ARCHIVE (expected tree $DIGEST)"
touch "$ROOT/queue/STOP"
say "queue/STOP written; waiting for every worker and classical process to end (at most $WAIT_MINUTES minutes)"
waited=0
while [ "$(running)" != 0 ]; do
    if [ "$waited" -ge $((WAIT_MINUTES * 60)) ]; then
        say "TIMEOUT: $(running) still running after $WAIT_MINUTES minutes; nothing installed"
        restart
        exit 1
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
done
say "every worker has stopped (after ${waited}s); jobs in queue/running: $(ls "$ROOT/queue/running" | wc -l)"
ok=0
if bash "$IN/unpack.sh" "$ARCHIVE" >> "$LOG" 2>&1; then
    NOW=$(tree_digest "$ROOT")
    if [ "$NOW" = "$DIGEST" ]; then ok=1; say "TREE OK $NOW"; else say "TREE DIFFERS $NOW (expected $DIGEST)"; fi
else
    say "unpack.sh FAILED for $ARCHIVE (see this log)"
fi
if [ "$ok" != 1 ]; then
    if bash "$IN/unpack.sh" "$PREVIOUS" >> "$LOG" 2>&1 && [ "$(tree_digest "$ROOT")" = "$BEFORE" ]; then
        say "the code before ($PREVIOUS) installed again: tree $BEFORE"
    else
        say "REINSTALL FAILED: $ROOT is tree $(tree_digest "$ROOT"); the workers stay stopped (queue/STOP)"
        exit 1
    fi
fi
restart
if [ "$ok" = 1 ] && [ "${#SWEEPS[@]}" -gt 0 ]; then
    say "smoke tests and enqueueing of ${#SWEEPS[@]} sweeps on GPU $SMOKE_GPU"
    (cd "$ROOT" && SMOKE_GPU="$SMOKE_GPU" bash scripts/smoke_enqueue.sh "${SWEEPS[@]}") 2>&1 | tee -a "$LOG"
fi
say "swap finished: $([ "$ok" = 1 ] && echo "new code" || echo "OLD CODE, nothing enqueued"), pending $(ls "$ROOT/queue/pending" | wc -l)"
