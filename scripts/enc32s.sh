#!/usr/bin/env bash
# Week 32's additions of 2026-11-05 (RULE_week32_addendum7.txt): EmoFilm's frames for the films of the heads, and Whisper
# Small as a fourth encoder, each step only if its output does not exist yet, so that running it again continues where
# it stopped. Run on the node in its own tmux session; it uses one GPU (ENC_GPU) beside that GPU's workers, and the CPUs:
#     tmux new -d -s enc32s "ENC_GPU=3 bash ~/affect/scripts/enc32s.sh"
#   1. EmoFilm's frames of the layers the heads read: Whisper large-v3's layer 31, AST's layer 8 and w2v-BERT 2.0's
#      four-class session-specific probe's layer (layers_w2v_bert_2.json). For each encoder: EmoFilm's probe means (every
#      layer's mean vector of its 726 clips, extract-pretrained), the layer's frames (extract-frames), and
#      check-frames --dataset emofilm (every clip's frames against its mean vector), whose record the film stage of
#      affect.sweep.week32 advance waits for (artifacts/advance_week32/frames_checked/emofilm_<model>_l<k>.json)
#   2. Whisper Small (its snapshot must be in ~/cache/huggingface): every layer's mean vectors and its four probe sweeps
#      on the CPUs (pretrained_night.sh), then each setup's layer chosen on validation (choose-layers)
#   3. its probe refits and protocol checks written and started on the CPUs, in the background
#   4. its frames at the four-class session-specific probe's layer, checked against that layer's mean vectors
#   5. EmoFilm's probe means and frames of that layer, checked, for its heads' films once they have refits
#   6. its four head screens written, smoke-tested and queued (priority 163), one head after another
#   then it waits for the CPU sweeps of step 3 and ends. A failed step stops it with the reason (the films' steps come
#   first, so a missing Whisper Small snapshot leaves them done). A step that uses the GPU waits while ENC_GPU may not
#   run now or 30 minutes later under the week's windows (common.sh gpu_allowed_at, weekday nights from NIGHT_START,
#   21:00 unless set otherwise: RULE_week32_addendum6.txt), or something that is not our runs is on it.
#   Log: logs/enc32s_<utc>.txt.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$AFFECT_ROOT" || exit 1
export NIGHT_START="${NIGHT_START:-21}"
ENC_GPU="${ENC_GPU:?ENC_GPU (0 to 3) must be set}"
case "$ENC_GPU" in 0|1|2|3) ;; *) echo "ENC_GPU must be 0, 1, 2 or 3, not $ENC_GPU"; exit 1 ;; esac
case "$NIGHT_START" in 18|19|20|21|22) ;; *) echo "NIGHT_START must be an hour from 18 to 22, not $NIGHT_START"; exit 1 ;; esac
LOG="$AFFECT_ROOT/logs/enc32s_$(stamp).txt"
OUT="$AFFECT_ROOT/artifacts/advance_week32"
F="$AFFECT_ROOT/artifacts/features"
TMP="$LOG.tmp"
SLEEP_S="${SLEEP_S:-300}"
say() { echo "$(date -u +%FT%TZ) ($(TZ=America/Los_Angeles date '+%a %H:%M %Z')) $*" | tee -a "$LOG"; }
stop() { say "STOPPED: $*"; wait; exit 1; }
snap() {   # the one snapshot of a Hugging Face repository in ~/cache/huggingface, or nothing
    local s=( "$HOME"/cache/huggingface/models--"$1"/snapshots/* )
    if [ "${#s[@]}" = 1 ] && [ -d "${s[0]}" ]; then echo "${s[0]}"; fi
}
free_gb() { df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9; }
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
        [ "$(free_gb)" -ge 4 ] || stop "less than 4 GB free on /home"
        gpu_time
        say "EmoFilm: $m, every layer's mean vectors of the 726 clips from $s"
        CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained.sh extract-pretrained --dataset emofilm \
            --label-set three_no_neutral --subset all --model "$m" --snapshot "$s" --device cuda --pooling mean \
            >> "$LOG" 2>&1 || stop "EmoFilm's probe means of $m FAILED (see the lines above)"
    fi
    if [ ! -f "$F/emofilm/${m}_l${k}_frames/record.json" ]; then
        [ "$(free_gb)" -ge 4 ] || stop "less than 4 GB free on /home"
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
queue_screens() {   # smoke-test and queue the screens listed as SCREEN <file> in $TMP, then order them head by head
    local files ids
    mapfile -t files < <(grep '^SCREEN ' "$TMP" | cut -c8-)
    [ "${#files[@]}" = 4 ] || stop "expected 4 screen files, got ${#files[@]}"
    gpu_time
    SMOKE_GPU="$ENC_GPU" bash scripts/smoke_enqueue_files.sh "${files[@]}" 2>&1 | tee -a "$LOG"
    mapfile -t ids < <(for f in "${files[@]}"; do basename "$f" .json; done)
    bash scripts/order_sweeps.sh "${ids[@]}" 2>&1 | tee -a "$LOG"
    for i in "${ids[@]}"; do [ -f "artifacts/sweeps/$i/jobs.csv" ] || stop "$i is not queued (see the smoke lines above)"; done
}
say "enc32s started on GPU $ENC_GPU (weekday nights from $NIGHT_START:00); $(free_gb) GB free"

# 1. EmoFilm's frames of the heads' layers
WL=$(snap openai--whisper-large-v3)
AST=$(snap MIT--ast-finetuned-audioset-10-10-0.4593)
W2V=$(snap facebook--w2v-bert-2.0)
[ -n "$WL" ] && [ -n "$AST" ] && [ -n "$W2V" ] || stop "not exactly one snapshot of Whisper large-v3, AST and w2v-BERT 2.0 each in ~/cache/huggingface"
[ -f "$OUT/layers_w2v_bert_2.json" ] || stop "no layers_w2v_bert_2.json: w2v-BERT 2.0's layer is not chosen"
VL=$(affect_python -c "import json; print(json.load(open('$OUT/layers_w2v_bert_2.json'))['setups']['so4']['layer'])")
emofilm_frames whisper_large_v3 31 "$WL"
emofilm_frames ast_audioset 8 "$AST"
emofilm_frames w2v_bert_2 "$VL" "$W2V"
say "EmoFilm: the frames of the three encoders' heads are checked (the films are written by the next advance pass)"

# 2. Whisper Small: probes and layers
WS=$(snap openai--whisper-small)
[ -n "$WS" ] || stop "no Whisper Small snapshot in ~/cache/huggingface (download-pretrained first); the films' steps are done"
PROBES="classical_pretrained_whisper_small_so4_1 classical_pretrained_whisper_small_pooled4_1 classical_pretrained_whisper_small_so3_1 classical_pretrained_whisper_small_pooled3_1"
if [ ! -f "$OUT/layers_whisper_small.json" ]; then
    [ "$(free_gb)" -ge 6 ] || stop "less than 6 GB free on /home"
    gpu_time
    say "Whisper Small: every layer's mean vectors on GPU $ENC_GPU, then the four probe sweeps on the CPUs"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained_night.sh --label-set four_merged --models whisper_small \
        --sweeps $PROBES >> "$LOG" 2>&1
    grep 'ALL DONE' "$(ls -t logs/pretrained_night_*.txt | head -1)" | tail -1 | tee -a "$LOG"
    affect_python -m affect.sweep.week32 choose-layers --model whisper_small > "$TMP" 2>&1
    cat "$TMP" >> "$LOG"
    [ -f "$OUT/layers_whisper_small.json" ] || stop "Whisper Small's layers not chosen: $(tail -1 "$TMP")"
fi
L=$(affect_python -c "import json; print(json.load(open('$OUT/layers_whisper_small.json'))['setups']['so4']['layer'])")
say "Whisper Small: layers chosen on validation: $(affect_python -c "import json; d=json.load(open('$OUT/layers_whisper_small.json'))['setups']; print(', '.join(f\"{s} {d[s]['layer']} ({d[s]['validation_ua']:.4f})\" for s in d))"); the heads read layer $L"

# 3. its probe refits and protocol checks, on the CPUs in the background
affect_python -m affect.sweep.week32 encoder-sweeps --model whisper_small > "$TMP" 2>&1 || { cat "$TMP" >> "$LOG"; stop "Whisper Small's sweeps not written"; }
cat "$TMP" >> "$LOG"
mapfile -t CPU < <(grep '^CLASSICAL ' "$TMP" | cut -c11-)
[ "${#CPU[@]}" = 8 ] || stop "expected 8 classical sweep files, got ${#CPU[@]}"
for c in "${CPU[@]}"; do     # a sweep run before resumes: its complete runs are skipped (affect.classical.runner)
    id=$(basename "$c" .json)
    bash scripts/classical.sh classical --sweep-file "$c" --max-consecutive-failures 3 > "logs/classical_${id}_$(stamp).log" 2>&1 &
    say "CPU sweep $id started (process $!)"
    sleep 2
done
cp "$TMP" "$TMP.screens"

# 4. its frames
if [ ! -f "$F/iemocap/whisper_small_l${L}_frames/record.json" ]; then
    [ "$(free_gb)" -ge 5 ] || stop "less than 5 GB free on /home"
    gpu_time
    say "Whisper Small: layer-$L frames of the 5,531 utterances from $WS"
    CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained.sh extract-frames --dataset iemocap --label-set four_merged \
        --subset all --model whisper_small --layers "$L" --snapshot "$WS" --device cuda >> "$LOG" 2>&1 \
        || stop "Whisper Small's frames FAILED (see the lines above)"
fi
affect_python -m affect.sweep.week32 check-frames --model whisper_small --layer "$L" > "$TMP" 2>&1
cat "$TMP" >> "$LOG"; tail -1 "$TMP"
grep -q '^FRAMES OK' "$TMP" || stop "Whisper Small's frames do not match the probe's means"

# 5. EmoFilm's frames of its layer
emofilm_frames whisper_small "$L" "$WS"

# 6. its screens
cp "$TMP.screens" "$TMP"
queue_screens
say "Whisper Small: the four head screens are queued (priority 163)"
say "waiting for the CPU sweeps"
failed=0
for job in $(jobs -p); do wait "$job" || failed=$((failed + 1)); done
rm -f "$TMP" "$TMP.screens"
say "ALL DONE: $failed CPU sweeps ended with a failure; pending $(ls queue/pending | wc -l); $(free_gb) GB free"
