#!/usr/bin/env bash
# A week of GPU runs that starts and stops by itself (added 2026-11-02, for the week to Monday 2026-11-09): every GPU
# no other user is on gets 3 workers whenever the time allows it and jobs are pending, and only then; the run ends by
# itself at WEEK_END. Times are Pacific (the node's clock is read with TZ=America/Los_Angeles):
#     weekday nights   Monday to Friday, from NIGHT_START (18:00) to 08:00 the next morning: every free GPU (0 to 3)
#     weekday days     Monday to Friday, 08:00 to NIGHT_START: no GPU
#     weekend          Saturday and Sunday, 08:00 to 22:00: GPUs 2 and 3 only (DAY_GPUS); 22:00 to 08:00: every free
#                      GPU. Friday evening and Monday morning are nights
# Every LOOP_S seconds (300) it looks again. A GPU gets its workers (tmux session gpu<g>, scripts/worker.sh) when it may
# run now and still at the next look, has no workers, is not reserved, jobs are pending, and nothing else is on it
# (gpu_schedule.sh's rule, made stricter on 2026-11-02: a process that is not one of our affect runs, other than an MPS
# server, or more than 2,000 MiB in use beyond our runs' own; a process counts as ours only when it is our user's and
# its command is affect.cli, so another project of ours on a GPU keeps it busy too); so workers that ran the queue empty
# start again once new jobs arrive, and a GPU another user leaves is used the same night.
# A GPU can be reserved for other work (queue/RESERVED_gpu<g>, any text in it; Harrison, 2026-11-02: GPU 2 for another
# project): it gets no workers (ours on it get its stop file), and it is released by itself (the file removed, logged)
# once a full hour (RESERVE_QUIET_S) has passed in which no look saw anything on it but our runs; removing the file
# releases it at once. A GPU that may not run at the next look gets its stop file (queue/STOP_gpu<g>) already now: its workers finish
# the job they are on, then exit, so no job starts after 08:00 on a weekday. Nothing starts while queue/STOP exists.
# A GPU whose last workers stopped because 5 jobs failed in a row, or for lack of disk (their logs say so), is held:
# queue/HELD_gpu<g> is written with the reason and the GPU gets no workers until a person removes that file (worker.sh
# stops after 5 failures so that a broken setup cannot empty the queue into queue/failed; restarting it every 5 minutes
# would). Jobs left in queue/running while no gpu session exists (a worker that was killed) go back to pending after
# 10 minutes. At WEEK_END (at the last look before it): queue/STOP is written (every worker stops after its current
# job), the per-GPU stop files are removed, and the script exits.
# Every HOOK_EVERY loops (15 minutes) it also starts HOOK_FILE (scripts/advance_week.sh, the step that queues each next
# stage once the one before it has finished; it reads validation files only) in a tmux session of its own,
# HOOK_SESSION (advance32), which ends by itself, if that file and ENABLE_FILE exist, queue/STOP does not, the session
# is not running already, and a GPU may run now and still HOOK_MARGIN seconds (30 minutes) from now (its smoke tests
# use a GPU): SMOKE_GPU is the highest-numbered such GPU that has our workers or no other user. Without the two files,
# nothing else is done.
# Run it from a copy outside ~/affect, so that installing code by day (no workers run then) never meets an open file of
# this script (unpack.sh moves the old scripts aside):
#     mkdir -p ~/week32 && cp ~/affect/scripts/gpu_week.sh ~/week32/ && \
#         tmux new -d -s week32 "WEEK_END=<epoch> bash ~/week32/gpu_week.sh"
# Its tmux session is not named gpu..., which unpack.sh and the start lines read as workers. To end it early:
#     tmux kill-session -t week32        (workers already running finish the queue or stop at their stop file)
# Log: logs/gpu_week_<utc>.txt: every change, and a status line each hour. For tests: TEST_TIMES (a list of epochs, one
# per loop, no sleeping), TMUX_CMD, NVIDIA_SMI_CMD and HOOK_FILE replace the real ones.
set -uo pipefail
ROOT="${AFFECT_ROOT:-$HOME/affect}"
TMUX_CMD="${TMUX_CMD:-tmux}"
NVIDIA_SMI_CMD="${NVIDIA_SMI_CMD:-nvidia-smi}"
WEEK_END="${WEEK_END:?WEEK_END (epoch seconds) must be set}"
NIGHT_START="${NIGHT_START:-18}"
DAY_GPUS="${DAY_GPUS:-2 3}"
ALL_GPUS="0 1 2 3"
LOOP_S="${LOOP_S:-300}"
HOOK_EVERY="${HOOK_EVERY:-3}"
HOOK_FILE="${HOOK_FILE:-$ROOT/scripts/advance_week.sh}"
ENABLE_FILE="${ENABLE_FILE:-$ROOT/artifacts/advance_week32/ENABLED}"
HOOK_SESSION="${HOOK_SESSION:-advance32}"
HOOK_MARGIN="${HOOK_MARGIN:-1800}"
ORPHAN_S="${ORPHAN_S:-600}"
RESERVE_QUIET_S="${RESERVE_QUIET_S:-3600}"
WORKERS="${WORKERS:-3}"
Q="$ROOT/queue"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/gpu_week_$(date -u +%Y%m%dT%H%M%SZ).txt"
read -r -a TIMES <<< "${TEST_TIMES:-}"
loop=0
now=0

log() { echo "$(date -u -d "@$now" +%FT%TZ) ($(TZ=America/Los_Angeles date -d "@$now" '+%a %H:%M %Z')) $*" >> "$LOG"; }

others_on_gpu() {   # why GPU $1 counts as busy ("" when free): a process that is not one of our affect runs, or memory
    # beyond theirs (gpu_schedule.sh's rule; a process counts as ours only when it is our user's and runs affect.cli)
    local uuid n=0 ours=0 pid args used mem line uid
    uuid=$("$NVIDIA_SMI_CMD" --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v g="$1" '$1 == g {print $2}')
    if [ -z "$uuid" ]; then echo "GPU $1 not found"; return; fi
    while IFS= read -r line; do
        pid=$(echo "$line" | awk -F', *' '{print $2}')
        used=$(echo "$line" | awk -F', *' '{print $3}' | tr -dc 0-9)
        [ -n "$pid" ] || continue
        uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
        args=$(ps -o args= -p "$pid" 2>/dev/null)
        case "$args" in nvidia-cuda-mps-server*|*/nvidia-cuda-mps-server*) continue ;; esac
        if [ -n "$uid" ] && [ "$uid" = "$(id -u)" ] && [[ "$args" == *affect.cli* ]]; then ours=$((ours + ${used:-0})); continue; fi
        n=$((n + 1))
    done < <("$NVIDIA_SMI_CMD" --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits \
             | awk -F', *' -v u="$uuid" '$1 == u')
    mem=$("$NVIDIA_SMI_CMD" --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g="$1" '$1 == g {print $2}')
    if [ "$n" -gt 0 ]; then echo "processes on it that are not our runs: $n"
    elif [ -z "$mem" ] || [ $((mem - ours)) -gt 2000 ]; then echo "memory in use beyond ours: $((${mem:-0} - ours)) of ${mem:-unknown} MiB"
    fi
}

allowed() {   # the GPUs that may run at time $1, and the window's name
    local wd hr
    wd=$(TZ=America/Los_Angeles date -d "@$1" +%u)          # 1 Monday ... 7 Sunday
    hr=$((10#$(TZ=America/Los_Angeles date -d "@$1" +%H)))
    if [ "$wd" -le 5 ]; then
        if [ "$hr" -lt 8 ] || [ "$hr" -ge "$NIGHT_START" ]; then echo "night|$ALL_GPUS"; else echo "weekday day|"; fi
    else
        if [ "$hr" -lt 8 ] || [ "$hr" -ge 22 ]; then echo "night|$ALL_GPUS"; else echo "weekend day|$DAY_GPUS"; fi
    fi
}

declare -A last_state last_start last_other
note() {   # log $2 for GPU (or key) $1 only when it differs from what was logged last for it
    if [ "${last_state[$1]:-}" != "$2" ]; then last_state[$1]="$2"; log "$2"; fi
}

held_reason() {   # why GPU $1 must not get workers ("" when it may): its HELD file, or its last workers' logs
    local f
    if [ -e "$Q/HELD_gpu$1" ]; then echo "queue/HELD_gpu$1 exists: $(head -c 300 "$Q/HELD_gpu$1")"; return; fi
    for f in "$ROOT"/logs/worker_gpu"$1"_*.log; do
        [ -f "$f" ] || continue
        [ "$(stat -c %Y "$f")" -ge "${last_start[$1]}" ] || continue
        if grep -q "jobs failed in a row" "$f"; then
            echo "its workers stopped after jobs failed in a row ($(basename "$f"))" | tee "$Q/HELD_gpu$1"; return
        fi
        if grep -q "GB free, below" "$f"; then
            echo "its workers stopped for lack of disk ($(basename "$f"))" | tee "$Q/HELD_gpu$1"; return
        fi
    done
}

in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

for g in $ALL_GPUS; do last_start[$g]=0; last_other[$g]=0; done
first=1
log "gpu_week started: ends $(TZ=America/Los_Angeles date -d "@$WEEK_END" '+%a %Y-%m-%d %H:%M %Z'); weekday nights from $NIGHT_START:00, weekend days GPUs $DAY_GPUS, a look every $LOOP_S s"
while :; do
    if [ "${#TIMES[@]}" -gt 0 ]; then
        [ "$loop" -lt "${#TIMES[@]}" ] || { log "test times used up"; exit 0; }
        now="${TIMES[$loop]}"
    else
        now=$(date +%s)
    fi
    loop=$((loop + 1))
    if [ "$first" = 1 ]; then for g in $ALL_GPUS; do last_start[$g]=$now; last_other[$g]=$now; done; first=0; fi
    if [ $((now + LOOP_S)) -gt "$WEEK_END" ]; then
        touch "$Q/STOP"
        rm -f "$Q/STOP_gpu0" "$Q/STOP_gpu1" "$Q/STOP_gpu2" "$Q/STOP_gpu3"
        log "end: queue/STOP written (every worker stops after its current job); the week's run is over"
        exit 0
    fi
    IFS='|' read -r window gpus <<< "$(allowed "$now")"
    IFS='|' read -r _ next <<< "$(allowed $((now + LOOP_S)))"
    note window "window: $window (GPUs that may run: ${gpus:-none})"
    pending=$(ls "$Q/pending" 2>/dev/null | wc -l)
    sessions=0
    for g in $ALL_GPUS; do
        if "$TMUX_CMD" has-session -t "=gpu$g" 2>/dev/null; then sessions=$((sessions + 1)); fi
        if in_list "$g" "$gpus" && in_list "$g" "$next"; then continue; fi
        if "$TMUX_CMD" has-session -t "=gpu$g" 2>/dev/null && [ ! -e "$Q/STOP_gpu$g" ]; then
            touch "$Q/STOP_gpu$g"
            log "GPU $g: stop file written (may not run at the next look): its workers finish their current jobs, then exit"
        fi
        note "gpu$g" "GPU $g: may not run ($window, or at the next look)"
    done
    for g in $ALL_GPUS; do                 # reserved GPUs: kept free while other work is on them, released after a quiet hour
        [ -e "$Q/RESERVED_gpu$g" ] || continue
        if "$TMUX_CMD" has-session -t "=gpu$g" 2>/dev/null && [ ! -e "$Q/STOP_gpu$g" ]; then
            touch "$Q/STOP_gpu$g"
            log "GPU $g: stop file written (reserved): its workers finish their current jobs, then exit"
        fi
        busy=$(others_on_gpu "$g")
        if [ -n "$busy" ]; then
            last_other[$g]=$now
            note "res$g" "GPU $g: reserved, in use ($busy)"
        elif [ $((now - last_other[$g])) -ge "$RESERVE_QUIET_S" ]; then
            rm -f "$Q/RESERVED_gpu$g"
            log "GPU $g: reservation released: nothing but our runs on it at every look since $(TZ=America/Los_Angeles date -d "@${last_other[$g]}" '+%a %H:%M %Z')"
            last_state["res$g"]=""
        else
            note "res$g" "GPU $g: reserved; nothing else on it now; released after $((RESERVE_QUIET_S / 60)) quiet minutes"
        fi
    done
    if [ "$sessions" -eq 0 ]; then         # jobs a killed worker left behind go back to pending
        for f in "$Q"/running/*.json; do
            [ -f "$f" ] || continue
            [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge "$ORPHAN_S" ] || continue
            b=$(basename "$f")
            if mv -n "$f" "$Q/pending/${b%.*.json}.json" 2>/dev/null; then log "requeued $b (no gpu session runs; left in running)"; fi
        done
    fi
    if [ -e "$Q/STOP" ]; then
        note stop "queue/STOP exists: no GPU is started"
    else
        note stop "queue/STOP absent"
        for g in $gpus; do
            in_list "$g" "$next" || continue
            if [ -e "$Q/RESERVED_gpu$g" ]; then note "gpu$g" "GPU $g: reserved, not started"; continue; fi
            if "$TMUX_CMD" has-session -t "=gpu$g" 2>/dev/null; then note "gpu$g" "GPU $g: workers running"; continue; fi
            if [ "$pending" -eq 0 ]; then note "gpu$g" "GPU $g: no pending job, not started"; continue; fi
            held=$(held_reason "$g")
            if [ -n "$held" ]; then note "gpu$g" "GPU $g: HELD, not started: $held (remove queue/HELD_gpu$g to allow it)"; last_start[$g]=$now; continue; fi
            busy=$(others_on_gpu "$g")
            if [ -n "$busy" ]; then note "gpu$g" "GPU $g: NOT started: $busy"; continue; fi
            rm -f "$Q/STOP_gpu$g"
            "$TMUX_CMD" new -d -s "gpu$g" "bash $ROOT/scripts/worker.sh $g $WORKERS"
            last_start[$g]=$now
            last_state["gpu$g"]="GPU $g: workers running"
            log "GPU $g: started ($WORKERS workers, tmux session gpu$g; $pending jobs pending)"
        done
    fi
    if [ $((loop % HOOK_EVERY)) -eq 0 ] && [ -f "$HOOK_FILE" ] && [ -f "$ENABLE_FILE" ]; then
        IFS='|' read -r _ soon <<< "$(allowed $((now + HOOK_MARGIN)))"
        if [ -e "$Q/STOP" ]; then
            note hook "hook: not started (queue/STOP exists)"
        elif "$TMUX_CMD" has-session -t "=$HOOK_SESSION" 2>/dev/null; then
            note hook "hook: $HOOK_SESSION is still running"
        elif [ -z "$gpus" ] || [ -z "$soon" ]; then
            note hook "hook: not started (no GPU may run now or in $((HOOK_MARGIN / 60)) minutes)"
        else
            sg=""
            for g in 3 2 1 0; do
                in_list "$g" "$gpus" && in_list "$g" "$soon" || continue
                [ -e "$Q/RESERVED_gpu$g" ] && continue
                if "$TMUX_CMD" has-session -t "=gpu$g" 2>/dev/null || [ -z "$(others_on_gpu "$g")" ]; then sg=$g; break; fi
            done
            if [ -n "$sg" ]; then
                "$TMUX_CMD" new -d -s "$HOOK_SESSION" "SMOKE_GPU=$sg bash $HOOK_FILE"
                last_state[hook]="started"
                log "hook: $HOOK_SESSION started ($(basename "$HOOK_FILE"), smoke tests on GPU $sg)"
            else
                note hook "hook: not started (no GPU of ours or free to smoke-test on)"
            fi
        fi
    fi
    if [ "$(TZ=America/Los_Angeles date -d "@$now" +%M)" -lt $((LOOP_S / 60 + 0)) ] || [ "$loop" -eq 1 ]; then
        log "status: pending $pending, running $(ls "$Q/running" 2>/dev/null | wc -l), done $(ls "$Q/done" 2>/dev/null | wc -l), failed $(ls "$Q/failed" 2>/dev/null | wc -l); sessions $("$TMUX_CMD" ls 2>/dev/null | cut -d: -f1 | tr '\n' ' '); reserved: $(ls "$Q" | grep -o 'RESERVED_gpu[0-9]' | tr '\n' ' ')$(ls "$Q" | grep -q RESERVED || echo none); $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9) GB free"
    fi
    if [ "${#TIMES[@]}" -eq 0 ]; then sleep "$LOOP_S"; fi
done
