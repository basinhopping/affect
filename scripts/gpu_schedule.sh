#!/usr/bin/env bash
# The weekend schedule, first used 2026-10-03 to 10-05: all four GPUs at night, only GPUs 2 and 3 (nvidia-smi's
# numbering) by day, so the shared node keeps two GPUs free for other users in the daytime. Run on the
# node in its own tmux session (not named gpu..., which unpack.sh and the start line read as workers), after
# the workers have been started:
#     tmux new -d -s weekend "bash ~/affect/scripts/gpu_schedule.sh"
# or, to start the workers too (a night step now, then the steps below):
#     tmux new -d -s weekend "bash ~/affect/scripts/gpu_schedule.sh --start-now"
# Times are Pacific. Each step runs once, at its time:
#     Sat 2026-10-03 08:00  day:   GPUs 0 and 1 stop (queue/STOP_gpu0 and STOP_gpu1: their workers finish
#                                  the job they are on, then exit); GPUs 2 and 3 keep running
#     Sat 2026-10-03 22:00  night: each GPU without workers gets 3 again, if no other user's process is on it
#     Sun 2026-10-04 08:00  day
#     Sun 2026-10-04 22:00  night
#     Mon 2026-10-05 08:00  end:   every worker stops (queue/STOP; the per-GPU files are removed), and this
#                                  script exits. Weekdays are nights only again, started by hand.
# A fourth kind of step, day_start (added 2026-10-10 for weekend days on which nothing runs yet), writes
# STOP_gpu0 and STOP_gpu1 and gives GPUs 2 and 3 their workers, each only if no other user is on it, as night does
# for all four; a schedule can begin with it ("<now>:day_start <evening>:night ...").
# A step whose time has passed is skipped when the next step's time has passed too, so a late start applies
# only the current state. Nothing else is touched: no other tmux session, no queue file but these stop files.
# It never starts a GPU that another user is on (an idle MPS server does not count; more than 2,000 MiB in use
# does), or any GPU while queue/STOP exists. To end it early:
#     tmux kill-session -t weekend
# Log: logs/gpu_schedule_<utc>.txt. For tests, SCHEDULE replaces the steps ("<epoch>:<day|night|end> ..."),
# and TMUX_CMD and NVIDIA_SMI_CMD replace tmux and nvidia-smi.
set -uo pipefail
ROOT="${AFFECT_ROOT:-$HOME/affect}"
TMUX_CMD="${TMUX_CMD:-tmux}"
NVIDIA_SMI_CMD="${NVIDIA_SMI_CMD:-nvidia-smi}"
Q="$ROOT/queue"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/gpu_schedule_$(date -u +%Y%m%dT%H%M%SZ).txt"
log() { echo "$(date -u +%FT%TZ) ($(TZ=America/Los_Angeles date '+%a %H:%M %Z')) $*" | tee -a "$LOG"; }
at() { TZ=America/Los_Angeles date -d "$1" +%s; }

if [ -z "${SCHEDULE:-}" ]; then
    SCHEDULE="$(at '2026-10-03 08:00'):day $(at '2026-10-03 22:00'):night $(at '2026-10-04 08:00'):day"
    SCHEDULE="$SCHEDULE $(at '2026-10-04 22:00'):night $(at '2026-10-05 08:00'):end"
fi
if [ "${1:-}" = "--start-now" ]; then SCHEDULE="$(date +%s):night $SCHEDULE"; fi
read -r -a STEPS <<< "$SCHEDULE"

others_on_gpu() {   # prints why GPU $1 counts as busy ("" when it is free for the workers)
    # Busy: a process of another user (or one ps cannot see), except an MPS server, which is a daemon this
    # project's CUDA bypasses (scripts/common.sh, the MPS fix of 2026-09-24); or more than 2,000 MiB of memory
    # in use, which also catches work run through someone's MPS server, whose clients nvidia-smi may not list.
    local uuid pids n=0 pid args mem
    uuid=$("$NVIDIA_SMI_CMD" --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v g="$1" '$1 == g {print $2}')
    if [ -z "$uuid" ]; then echo "GPU $1 not found"; return; fi
    pids=$("$NVIDIA_SMI_CMD" --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', *' -v u="$uuid" '$1 == u {print $2}')
    for pid in $pids; do
        if [ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$(id -u)" ]; then continue; fi
        args=$(ps -o args= -p "$pid" 2>/dev/null)
        case "$args" in nvidia-cuda-mps-server*|*/nvidia-cuda-mps-server*) continue ;; esac
        n=$((n + 1))
    done
    mem=$("$NVIDIA_SMI_CMD" --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g="$1" '$1 == g {print $2}')
    if [ "$n" -gt 0 ]; then echo "other users' processes: $n"
    elif [ -z "$mem" ] || [ "$mem" -gt 2000 ]; then echo "memory in use: ${mem:-unknown} MiB"
    fi
}

day() {
    touch "$Q/STOP_gpu0" "$Q/STOP_gpu1"
    log "day: STOP_gpu0 and STOP_gpu1 written; GPUs 0 and 1 stop after their current jobs, GPUs 2 and 3 keep running"
}

start_gpus() {   # $1 the step's name, for the log; then the GPUs to give workers
    local step="$1" g busy
    shift
    if [ -e "$Q/STOP" ]; then log "$step: queue/STOP exists, so no GPU is started"; return; fi
    for g in "$@"; do
        if "$TMUX_CMD" has-session -t "gpu$g" 2>/dev/null; then log "$step: GPU $g already has its workers"; continue; fi
        busy=$(others_on_gpu "$g")
        if [ -n "$busy" ]; then log "$step: GPU $g NOT started: $busy"; continue; fi
        rm -f "$Q/STOP_gpu$g"
        "$TMUX_CMD" new -d -s "gpu$g" "bash $ROOT/scripts/worker.sh $g 3"
        log "$step: GPU $g started (3 workers, tmux session gpu$g)"
    done
    log "$step: pending jobs $(ls "$Q/pending" 2>/dev/null | wc -l)"
}

night() { start_gpus night 0 1 2 3; }

day_start() {
    touch "$Q/STOP_gpu0" "$Q/STOP_gpu1"
    log "day_start: STOP_gpu0 and STOP_gpu1 written; GPUs 0 and 1 get no workers by day"
    start_gpus day_start 2 3
}

end() {
    touch "$Q/STOP"
    rm -f "$Q/STOP_gpu0" "$Q/STOP_gpu1" "$Q/STOP_gpu2" "$Q/STOP_gpu3"
    log "end: queue/STOP written (every worker stops after its current job); the weekend schedule is over"
}

log "gpu_schedule started, ${#STEPS[@]} steps: $(for s in "${STEPS[@]}"; do printf '%s %s; ' "$(TZ=America/Los_Angeles date -d "@${s%%:*}" '+%a %m-%d %H:%M')" "${s#*:}"; done)"
for i in "${!STEPS[@]}"; do
    when="${STEPS[$i]%%:*}"; what="${STEPS[$i]#*:}"
    next=""
    if [ $((i + 1)) -lt ${#STEPS[@]} ]; then next="${STEPS[$((i + 1))]%%:*}"; fi
    now=$(date +%s)
    if [ -n "$next" ] && [ "$now" -ge "$next" ]; then log "skipped $what of $(date -d "@$when" -u +%FT%TZ): its time and the next step's have passed"; continue; fi
    if [ "$now" -lt "$when" ]; then sleep $((when - now)); fi
    case "$what" in
        day) day ;;
        night) night ;;
        day_start) day_start ;;
        end) end ;;
        *) log "unknown step $what"; exit 1 ;;
    esac
done
log "gpu_schedule finished"
