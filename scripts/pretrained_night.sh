#!/usr/bin/env bash
# The pretrained models' night (added 2026-10-19): extract every listed model's layers on one GPU, one model after the
# other, then run the listed classical sweeps at once, one process each (each runs its jobs one after another; the
# SVM uses one CPU thread), and wait for all of them. Run on the node in its own tmux session, from the tree that holds
# the code (here ~/affect_next while GPU workers run from ~/affect):
#     tmux new -d -s pretrained19 "CUDA_VISIBLE_DEVICES=3 bash ~/affect_next/scripts/pretrained_night.sh \
#         [--label-set <label set>] --models wavlm_base_plus hubert_base wav2vec2_base --sweeps <sweep id> [...]"
# The utterances extracted are those of the label set, seven by default (every utterance of every set the project
# uses); four_merged holds those of four_merged and three_no_neutral (night 19c's Large models, 2026-10-19).
# A model whose last feature folder (layer 12 or 24) exists already is not extracted again; a model whose snapshot is
# not in ~/cache/huggingface stops the script before any sweep starts. Each sweep's output goes to
# logs/classical_<sweep id>_<utc>.log; the script's own lines to logs/pretrained_night_<utc>.txt. The session ends by
# itself when every sweep has ended.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/common.sh"
cd "$AFFECT_ROOT"
LOG="$AFFECT_ROOT/logs/pretrained_night_$(stamp).txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
USAGE="usage: pretrained_night.sh [--label-set <label set>] --models <model> [...] --sweeps <sweep id> [...]"
LABEL_SET=seven
if [ "${1:-}" = "--label-set" ]; then LABEL_SET="${2:?$USAGE}"; shift 2; fi
[ "${1:-}" = "--models" ] || { echo "$USAGE"; exit 1; }
shift
MODELS=(); SWEEPS=()
while [ "$#" -gt 0 ] && [ "$1" != "--sweeps" ]; do MODELS+=("$1"); shift; done
[ "${1:-}" = "--sweeps" ] || { echo "no --sweeps"; exit 1; }
shift
SWEEPS=("$@")
[ "${#MODELS[@]}" -gt 0 ] && [ "${#SWEEPS[@]}" -gt 0 ] || { echo "no models or no sweeps"; exit 1; }
declare -A REPO=([wavlm_base_plus]=microsoft--wavlm-base-plus [hubert_base]=facebook--hubert-base-ls960
                 [wav2vec2_base]=facebook--wav2vec2-base [wavlm_large]=microsoft--wavlm-large
                 [hubert_large]=facebook--hubert-large-ll60k [wav2vec2_large]=facebook--wav2vec2-large-lv60)
declare -A LAST=([wavlm_base_plus]=12 [hubert_base]=12 [wav2vec2_base]=12 [wavlm_large]=24 [hubert_large]=24
                 [wav2vec2_large]=24)
for m in "${MODELS[@]}"; do
    [ -n "${REPO[$m]:-}" ] || { say "unknown model $m: stop"; exit 1; }
    if [ -f "artifacts/features/iemocap/${m}_l${LAST[$m]}/record.json" ]; then
        say "$m: features exist already"; continue
    fi
    snaps=( "$HOME"/cache/huggingface/models--"${REPO[$m]}"/snapshots/* )
    if [ "${#snaps[@]}" != 1 ] || [ ! -d "${snaps[0]}" ]; then
        say "$m: not exactly one snapshot in ~/cache/huggingface: stop"; exit 1
    fi
    say "$m: extracting the $LABEL_SET utterances from ${snaps[0]} on GPU ${CUDA_VISIBLE_DEVICES:-unset}"
    if ! bash "$HERE/pretrained.sh" extract-pretrained --dataset iemocap --label-set "$LABEL_SET" --subset all \
            --model "$m" --snapshot "${snaps[0]}" --device cuda --pooling mean >> "$LOG" 2>&1; then
        say "$m: extraction FAILED, see $LOG: stop before any sweep"; exit 1
    fi
    say "$m: extracted"
done
pids=()
for s in "${SWEEPS[@]}"; do
    [ -f "sweeps/$s.json" ] || { say "no sweeps/$s.json: stop"; exit 1; }
done
for s in "${SWEEPS[@]}"; do
    bash "$HERE/classical.sh" classical --sweep-file "sweeps/$s.json" --max-consecutive-failures 3 \
        > "logs/classical_${s}_$(stamp).log" 2>&1 &
    pids+=("$!")
    say "sweep $s started (process $!)"
    sleep 2
done
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then say "sweep ${SWEEPS[$i]} finished"
    else say "sweep ${SWEEPS[$i]} ended with a failure"; failed=$((failed + 1)); fi
done
say "ALL DONE: ${#SWEEPS[@]} sweeps, $failed with a failure"
