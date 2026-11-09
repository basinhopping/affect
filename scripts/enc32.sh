#!/usr/bin/env bash
# Week 32's new encoders (added 2026-11-02; RULE_week32.txt part 6): my four heads on AST's frames and on w2v-BERT 2.0's,
# each step only if its output does not exist yet, so that running it again continues where it stopped. Run on the node
# in its own tmux session; it uses one GPU (ENC_GPU) beside that GPU's workers, and the CPUs:
#     tmux new -d -s enc32 "ENC_GPU=3 bash ~/affect/scripts/enc32.sh"
#   1. AST's layer-8 frames (layer 8: probe30_ast_audioset_so4's layer, chosen on validation on night 30), checked
#      against the probe's layer-8 mean vectors (affect.sweep.week32 check-frames)
#   2. AST's four head screens written (affect.sweep.week32 encoder-sweeps), smoke-tested on ENC_GPU and queued (priority 165),
#      then run one head after another (order_sweeps.sh)
#   3. w2v-BERT 2.0 (its snapshot must be in ~/cache/huggingface): every layer's mean vectors and its four probe sweeps
#      on the CPUs (pretrained_night.sh), then each setup's layer chosen on validation (choose-layers)
#   4. its probe refits and protocol checks written and started on the CPUs, in the background
#   5. its frames at the four-class session-specific probe's layer, checked against that layer's mean vectors
#   6. its four head screens written, smoke-tested and queued (priority 164), one head after another
#   then it waits for the CPU sweeps of step 4 and ends. A failed step stops it with the reason (AST's steps come first,
#   so a missing w2v-BERT 2.0 snapshot leaves them done). A step that uses the GPU waits while ENC_GPU may not run now
#   or 30 minutes later under the week's windows (common.sh gpu_allowed_at), or something that is not our runs is on it.
#   Log: logs/enc32_<utc>.txt.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$AFFECT_ROOT" || exit 1
ENC_GPU="${ENC_GPU:?ENC_GPU (0 to 3) must be set}"
case "$ENC_GPU" in 0|1|2|3) ;; *) echo "ENC_GPU must be 0, 1, 2 or 3, not $ENC_GPU"; exit 1 ;; esac
LOG="$AFFECT_ROOT/logs/enc32_$(stamp).txt"
OUT="$AFFECT_ROOT/artifacts/advance_week32"
F="$AFFECT_ROOT/artifacts/features/iemocap"
TMP="$LOG.tmp"
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
        sleep 300
    done
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
say "enc32 started on GPU $ENC_GPU; $(free_gb) GB free"

# 1. AST's frames
AST=$(snap MIT--ast-finetuned-audioset-10-10-0.4593)
[ -n "$AST" ] || stop "not exactly one AST snapshot in ~/cache/huggingface"
if [ ! -f "$F/ast_audioset_l8_frames/record.json" ]; then
    [ "$(free_gb)" -ge 4 ] || stop "less than 4 GB free on /home"
    gpu_time
    say "AST: layer-8 frames of the 5,531 utterances from $AST"
    CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained.sh extract-frames --dataset iemocap --label-set four_merged \
        --subset all --model ast_audioset --layers 8 --snapshot "$AST" --device cuda >> "$LOG" 2>&1 \
        || stop "AST's frames FAILED (see the lines above)"
fi
affect_python -m affect.sweep.week32 check-frames --model ast_audioset --layer 8 > "$TMP" 2>&1
cat "$TMP" >> "$LOG"; tail -1 "$TMP"
grep -q '^FRAMES OK' "$TMP" || stop "AST's frames do not match the probe's means"

# 2. AST's screens
affect_python -m affect.sweep.week32 encoder-sweeps --model ast_audioset > "$TMP" 2>&1 || { cat "$TMP" >> "$LOG"; stop "AST's sweeps not written"; }
cat "$TMP" >> "$LOG"
queue_screens
say "AST: the four head screens are queued (priority 165)"

# 3. w2v-BERT 2.0: probes and layers
W2V=$(snap facebook--w2v-bert-2.0)
[ -n "$W2V" ] || stop "no w2v-BERT 2.0 snapshot in ~/cache/huggingface (download-pretrained first); AST's steps are done"
PROBES="classical_pretrained_w2v_bert_2_so4_1 classical_pretrained_w2v_bert_2_pooled4_1 classical_pretrained_w2v_bert_2_so3_1 classical_pretrained_w2v_bert_2_pooled3_1"
if [ ! -f "$OUT/layers_w2v_bert_2.json" ]; then
    [ "$(free_gb)" -ge 10 ] || stop "less than 10 GB free on /home"
    gpu_time
    say "w2v-BERT 2.0: every layer's mean vectors on GPU $ENC_GPU, then the four probe sweeps on the CPUs (about 1.5 hours)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained_night.sh --label-set four_merged --models w2v_bert_2 \
        --sweeps $PROBES >> "$LOG" 2>&1
    grep 'ALL DONE' "$(ls -t logs/pretrained_night_*.txt | head -1)" | tail -1 | tee -a "$LOG"
    affect_python -m affect.sweep.week32 choose-layers --model w2v_bert_2 > "$TMP" 2>&1
    cat "$TMP" >> "$LOG"
    [ -f "$OUT/layers_w2v_bert_2.json" ] || stop "w2v-BERT 2.0's layers not chosen: $(tail -1 "$TMP")"
fi
L=$(affect_python -c "import json; print(json.load(open('$OUT/layers_w2v_bert_2.json'))['setups']['so4']['layer'])")
say "w2v-BERT 2.0: layers chosen on validation: $(affect_python -c "import json; d=json.load(open('$OUT/layers_w2v_bert_2.json'))['setups']; print(', '.join(f\"{s} {d[s]['layer']} ({d[s]['validation_ua']:.4f})\" for s in d))"); the heads read layer $L"

# 4. its probe refits and protocol checks, on the CPUs in the background
affect_python -m affect.sweep.week32 encoder-sweeps --model w2v_bert_2 > "$TMP" 2>&1 || { cat "$TMP" >> "$LOG"; stop "w2v-BERT 2.0's sweeps not written"; }
cat "$TMP" >> "$LOG"
mapfile -t CPU < <(grep '^CLASSICAL ' "$TMP" | cut -c11-)
[ "${#CPU[@]}" = 8 ] || stop "expected 8 classical sweep files, got ${#CPU[@]}"
for c in "${CPU[@]}"; do
    id=$(basename "$c" .json)
    bash scripts/classical.sh classical --sweep-file "$c" --max-consecutive-failures 3 > "logs/classical_${id}_$(stamp).log" 2>&1 &
    say "CPU sweep $id started (process $!)"
    sleep 2
done
cp "$TMP" "$TMP.screens"

# 5. its frames
if [ ! -f "$F/w2v_bert_2_l${L}_frames/record.json" ]; then
    [ "$(free_gb)" -ge 8 ] || stop "less than 8 GB free on /home"
    gpu_time
    say "w2v-BERT 2.0: layer-$L frames of the 5,531 utterances from $W2V"
    CUDA_VISIBLE_DEVICES="$ENC_GPU" bash scripts/pretrained.sh extract-frames --dataset iemocap --label-set four_merged \
        --subset all --model w2v_bert_2 --layers "$L" --snapshot "$W2V" --device cuda >> "$LOG" 2>&1 \
        || stop "w2v-BERT 2.0's frames FAILED (see the lines above)"
fi
affect_python -m affect.sweep.week32 check-frames --model w2v_bert_2 --layer "$L" > "$TMP" 2>&1
cat "$TMP" >> "$LOG"; tail -1 "$TMP"
grep -q '^FRAMES OK' "$TMP" || stop "w2v-BERT 2.0's frames do not match the probe's means"

# 6. its screens
cp "$TMP.screens" "$TMP"
queue_screens
say "w2v-BERT 2.0: the four head screens are queued (priority 164)"
say "waiting for the CPU sweeps"
failed=0
for job in $(jobs -p); do wait "$job" || failed=$((failed + 1)); done
rm -f "$TMP" "$TMP.screens"
say "ALL DONE: $failed CPU sweeps ended with a failure; pending $(ls queue/pending | wc -l); $(free_gb) GB free"
