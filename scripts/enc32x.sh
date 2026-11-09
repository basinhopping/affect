#!/usr/bin/env bash
# Week 32's fifth and sixth encoders (added 2026-11-07, RULE_week32_addendum9.txt part N): BUD-E-Whisper and Whisper
# medium, each done as scripts/enc32s.sh does Whisper Small, each step only if its output does not exist yet, so that
# running it again continues where it stopped. Run on the node in its own tmux session; it uses one GPU (ENC_GPU) beside
# that GPU's workers, and the CPUs:
#     tmux new -d -s enc32x "ENC_GPU=3 bash ~/affect/scripts/enc32x.sh bud_e_whisper whisper_medium"
# The models are taken in the order given (each must have exactly one snapshot in ~/cache/huggingface):
#   1. every layer's mean vectors of each model whose layers are not chosen yet, on GPU ENC_GPU, then all their probe
#      sweeps at once on the CPUs (pretrained_night.sh, which waits for them), then each model's layers chosen on
#      validation (choose-layers)
#   then for each model in turn:
#   2. its probe refits and protocol checks written and started on the CPUs, in the background
#   3. its frames at the four-class session-specific probe's layer, checked against that layer's mean vectors
#   4. EmoFilm's probe means and frames of that layer, checked, for its heads' films once they have refits
#   5. its four head screens written, smoke-tested and queued (its priority in affect.sweep.week32), one head after
#      another
#   then it waits for the CPU sweeps of step 2 and ends. A failed step stops it with the reason. A step that uses the GPU
#   waits while ENC_GPU may not run now or 30 minutes later under the week's windows (common.sh gpu_allowed_at, weekday
#   nights from NIGHT_START, 21:00 unless set otherwise: RULE_week32_addendum6.txt), or something that is not our runs
#   is on it. Since the workers stop below 10 GB free (worker.sh --min-free-gb 10), each step that writes features
#   first needs room for them above that: MIN_FREE_GB (11) for mean vectors and EmoFilm's features, and for a model's
#   frames 11 plus their size (about 2 GB for 768 values a frame, 3 GB for 1024), counted in df's GB.
#   Log: logs/enc32x_<utc>.txt.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$AFFECT_ROOT" || exit 1
export NIGHT_START="${NIGHT_START:-21}"
ENC_GPU="${ENC_GPU:?ENC_GPU (0 to 3) must be set}"
case "$ENC_GPU" in 0|1|2|3) ;; *) echo "ENC_GPU must be 0, 1, 2 or 3, not $ENC_GPU"; exit 1 ;; esac
case "$NIGHT_START" in 18|19|20|21|22) ;; *) echo "NIGHT_START must be an hour from 18 to 22, not $NIGHT_START"; exit 1 ;; esac
MIN_FREE_GB="${MIN_FREE_GB:-11}"
[ "$#" -gt 0 ] || { echo "usage: enc32x.sh <model> [<model> ...] (bud_e_whisper, whisper_medium)"; exit 1; }
MODELS=("$@")
LOG="$AFFECT_ROOT/logs/enc32x_$(stamp).txt"
OUT="$AFFECT_ROOT/artifacts/advance_week32"
F="$AFFECT_ROOT/artifacts/features"
TMP="$LOG.tmp"
SLEEP_S="${SLEEP_S:-300}"
say() { echo "$(date -u +%FT%TZ) ($(TZ=America/Los_Angeles date '+%a %H:%M %Z')) $*" | tee -a "$LOG"; }
stop() { say "STOPPED: $*"; wait; exit 1; }
declare -A REPO=([bud_e_whisper]=laion--BUD-E-Whisper [whisper_medium]=openai--whisper-medium)
declare -A FRAMES_GB=([bud_e_whisper]=2 [whisper_medium]=3)
snap() {   # the one snapshot of a Hugging Face repository in ~/cache/huggingface, or nothing
    local s=( "$HOME"/cache/huggingface/models--"$1"/snapshots/* )
    if [ "${#s[@]}" = 1 ] && [ -d "${s[0]}" ]; then echo "${s[0]}"; fi
}
free_gb() { df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9; }
room() {   # <GB needed> <what>: stop unless that much is free
    [ "$(free_gb)" -ge "$1" ] || stop "$(free_gb) GB free, $1 needed before $2 (the workers stop below 10)"
}
gpu_time() {   # before a step that uses the GPU: wait while ENC_GPU may not run now or in 30 minutes, or another user is on it
    local said="" now
    while :; do
        now=$(date +%s)
        if ! gpu_allowed_at "$now" "$ENC_GPU" || ! gpu_allowed_at $((now + 1800)) "$ENC_GPU"; then
            [ "$said" = window ] || say "GPU $ENC_GPU may not run now or in 30 minutes (the week's windows): waiting"
            said=window
        elif [ -n "$(gpu_other_users "$ENC_GPU")" ]; then
            [ "$said" = other ] || say "GPU $ENC_GPU runs something that is not our runs: waiting until it is free"
            said=other
        else
            return 0
        fi
        sleep "$SLEEP_S"
    done
}
emofilm_frames() {   # <model> <layer> <snapshot>: EmoFilm's probe means and frames of one layer, then their check
    local m="$1" k="$2" s="$3"
    if [ -f "$OUT/frames_checked/emofilm_${m}_l${k}.json" ]; then say "EmoFilm: $m layer $k checked already"; return 0; fi
    if [ ! -f "$F/emofilm/${m}_l${k}/record.json" ]; then
        room "$MIN_FREE_GB" "EmoFilm's mean vectors of $m"
        gpu_time
        say "EmoFilm: $m, every layer's mean vectors of the 726 clips from $s"
        CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained.sh extract-pretrained --dataset emofilm \
            --label-set three_no_neutral --subset all --model "$m" --snapshot "$s" --device cuda --pooling mean \
            >> "$LOG" 2>&1 || stop "EmoFilm's probe means of $m FAILED (see the lines above)"
    fi
    if [ ! -f "$F/emofilm/${m}_l${k}_frames/record.json" ]; then
        room "$MIN_FREE_GB" "EmoFilm's frames of $m"
        gpu_time
        say "EmoFilm: $m layer-$k frames of the 726 clips"
        CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained.sh extract-frames --dataset emofilm \
            --label-set three_no_neutral --subset all --model "$m" --layers "$k" --snapshot "$s" --device cuda \
            >> "$LOG" 2>&1 || stop "EmoFilm's frames of $m FAILED (see the lines above)"
    fi
    affect_python -m affect.sweep.week32 check-frames --dataset emofilm --model "$m" --layer "$k" > "$TMP" 2>&1
    cat "$TMP" >> "$LOG"; tail -1 "$TMP"
    grep -q '^FRAMES OK' "$TMP" || stop "EmoFilm's frames of $m layer $k do not match its probe means"
}
queue_screens() {   # <file listing SCREEN lines>: smoke-test and queue those 4 screens, then order them head by head
    local files ids
    mapfile -t files < <(grep '^SCREEN ' "$1" | cut -c8-)
    [ "${#files[@]}" = 4 ] || stop "expected 4 screen files, got ${#files[@]}"
    gpu_time
    SMOKE_GPU="$ENC_GPU" bash scripts/smoke_enqueue_files.sh "${files[@]}" 2>&1 | tee -a "$LOG"
    mapfile -t ids < <(for f in "${files[@]}"; do basename "$f" .json; done)
    bash scripts/order_sweeps.sh "${ids[@]}" 2>&1 | tee -a "$LOG"
    for i in "${ids[@]}"; do [ -f "artifacts/sweeps/$i/jobs.csv" ] || stop "$i is not queued (see the smoke lines above)"; done
}
say "enc32x started on GPU $ENC_GPU (weekday nights from $NIGHT_START:00) for ${MODELS[*]}; $(free_gb) GB free"
for m in "${MODELS[@]}"; do
    [ -n "${REPO[$m]:-}" ] || stop "$m is not one of enc32x's models (bud_e_whisper, whisper_medium)"
    [ -n "$(snap "${REPO[$m]}")" ] || stop "not exactly one snapshot of $m in ~/cache/huggingface (download-pretrained first)"
done

# 1. probes and layers, for the models whose layers are not chosen yet
NEED=(); PROBES=()
for m in "${MODELS[@]}"; do
    [ -f "$OUT/layers_$m.json" ] && continue
    NEED+=("$m")
    for setup in so4 pooled4 so3 pooled3; do PROBES+=("classical_pretrained_${m}_${setup}_1"); done
done
if [ "${#NEED[@]}" -gt 0 ]; then
    room "$MIN_FREE_GB" "the mean vectors of ${NEED[*]}"
    gpu_time
    say "${NEED[*]}: every layer's mean vectors on GPU $ENC_GPU, then their ${#PROBES[@]} probe sweeps on the CPUs"
    CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained_night.sh --label-set four_merged --models "${NEED[@]}" \
        --sweeps "${PROBES[@]}" >> "$LOG" 2>&1
    grep 'ALL DONE' "$(ls -t logs/pretrained_night_*.txt | head -1)" | tail -1 | tee -a "$LOG"
    for m in "${NEED[@]}"; do
        affect_python -m affect.sweep.week32 choose-layers --model "$m" > "$TMP" 2>&1
        cat "$TMP" >> "$LOG"
        [ -f "$OUT/layers_$m.json" ] || stop "$m's layers not chosen: $(tail -1 "$TMP")"
    done
fi

for m in "${MODELS[@]}"; do
    S=$(snap "${REPO[$m]}")
    L=$(affect_python -c "import json; print(json.load(open('$OUT/layers_$m.json'))['setups']['so4']['layer'])")
    say "$m: layers chosen on validation: $(affect_python -c "import json; d=json.load(open('$OUT/layers_$m.json'))['setups']; print(', '.join(f\"{s} {d[s]['layer']} ({d[s]['validation_ua']:.4f})\" for s in d))"); the heads read layer $L"

    # 2. its probe refits and protocol checks, on the CPUs in the background
    affect_python -m affect.sweep.week32 encoder-sweeps --model "$m" > "$TMP" 2>&1 || { cat "$TMP" >> "$LOG"; stop "$m's sweeps not written"; }
    cat "$TMP" >> "$LOG"
    mapfile -t CPU < <(grep '^CLASSICAL ' "$TMP" | cut -c11-)
    [ "${#CPU[@]}" = 8 ] || stop "expected 8 classical sweep files for $m, got ${#CPU[@]}"
    for c in "${CPU[@]}"; do     # a sweep run before resumes: its complete runs are skipped (affect.classical.runner)
        id=$(basename "$c" .json)
        bash scripts/classical.sh classical --sweep-file "$c" --max-consecutive-failures 3 > "logs/classical_${id}_$(stamp).log" 2>&1 &
        say "CPU sweep $id started (process $!)"
        sleep 2
    done
    cp "$TMP" "$TMP.screens"

    # 3. its frames
    if [ ! -f "$F/iemocap/${m}_l${L}_frames/record.json" ]; then
        room $((MIN_FREE_GB + ${FRAMES_GB[$m]})) "$m's frames"
        gpu_time
        say "$m: layer-$L frames of the 5,531 utterances from $S"
        CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained.sh extract-frames --dataset iemocap --label-set four_merged \
            --subset all --model "$m" --layers "$L" --snapshot "$S" --device cuda >> "$LOG" 2>&1 \
            || stop "$m's frames FAILED (see the lines above)"
    fi
    affect_python -m affect.sweep.week32 check-frames --model "$m" --layer "$L" > "$TMP" 2>&1
    cat "$TMP" >> "$LOG"; tail -1 "$TMP"
    grep -q '^FRAMES OK' "$TMP" || stop "$m's frames do not match the probe's means"

    # 4. EmoFilm's frames of its layer
    emofilm_frames "$m" "$L" "$S"

    # 5. its screens
    queue_screens "$TMP.screens"
    say "$m: the four head screens are queued"
done
say "waiting for the CPU sweeps"
failed=0
for job in $(jobs -p); do wait "$job" || failed=$((failed + 1)); done
rm -f "$TMP" "$TMP.screens"
say "ALL DONE: $failed CPU sweeps ended with a failure; pending $(ls queue/pending | wc -l); $(free_gb) GB free"
