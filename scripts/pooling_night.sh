#!/usr/bin/env bash
# The pooling night (added 2026-10-20, the survey's cheapest next steps for the encoders): each listed encoder's
# layers pooled by mean and SD (<model>_l<k>_msd) on one GPU, for IEMOCAP's four_merged utterances (which hold
# three_no_neutral's) and for EmoFilm's clips; each encoder's layer average (<model>_lavg) derived on the CPU from the
# mean layers extracted before (nights 19b, 19c and 20), for IEMOCAP and EmoFilm; then the listed classical sweeps at
# once, one process each. Run on the node in its own tmux session, from the tree that holds the code (~/affect_next
# while GPU workers run from ~/affect):
#     tmux new -d -s pooling21 "CUDA_VISIBLE_DEVICES=3 bash ~/affect_next/scripts/pooling_night.sh \
#         --models wavlm_large hubert_base --sweeps <sweep id> [...]"
# A feature set that exists already is not made again; a model whose snapshot is not the only one in
# ~/cache/huggingface, or less than 8 GiB free on the disk of artifacts/ while a set is still to be made, stops the
# script before anything is made; a step that fails stops it before any sweep. The script's lines go to
# logs/pooling_night_<utc>.txt, its commands' output too; each sweep's to logs/classical_<sweep id>_<utc>.log. The
# session ends by itself when every sweep has ended.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/common.sh"
cd "$AFFECT_ROOT"
LOG="$AFFECT_ROOT/logs/pooling_night_$(stamp).txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
USAGE="usage: pooling_night.sh --models <model> [...] --sweeps <sweep id> [...]"
[ "${1:-}" = "--models" ] || { echo "$USAGE"; exit 1; }
shift
MODELS=(); SWEEPS=()
while [ "$#" -gt 0 ] && [ "$1" != "--sweeps" ]; do MODELS+=("$1"); shift; done
[ "${1:-}" = "--sweeps" ] || { echo "$USAGE"; exit 1; }
shift
SWEEPS=("$@")
[ "${#MODELS[@]}" -gt 0 ] && [ "${#SWEEPS[@]}" -gt 0 ] || { echo "$USAGE"; exit 1; }
declare -A REPO=([wavlm_base_plus]=microsoft--wavlm-base-plus [hubert_base]=facebook--hubert-base-ls960
                 [wav2vec2_base]=facebook--wav2vec2-base [wavlm_large]=microsoft--wavlm-large
                 [hubert_large]=facebook--hubert-large-ll60k [wav2vec2_large]=facebook--wav2vec2-large-lv60)
declare -A LAST=([wavlm_base_plus]=12 [hubert_base]=12 [wav2vec2_base]=12 [wavlm_large]=24 [hubert_large]=24
                 [wav2vec2_large]=24)
declare -A LABELS=([iemocap]=four_merged [emofilm]=three_no_neutral)
for s in "${SWEEPS[@]}"; do
    [ -f "sweeps/$s.json" ] || { say "no sweeps/$s.json: stop"; exit 1; }
done
declare -A SNAP=()
missing=0
for m in "${MODELS[@]}"; do
    [ -n "${REPO[$m]:-}" ] || { say "unknown model $m: stop"; exit 1; }
    snaps=( "$HOME"/cache/huggingface/models--"${REPO[$m]}"/snapshots/* )
    if [ "${#snaps[@]}" != 1 ] || [ ! -d "${snaps[0]}" ]; then
        say "$m: not exactly one snapshot in ~/cache/huggingface: stop"; exit 1
    fi
    SNAP[$m]="${snaps[0]}"
    for ds in iemocap emofilm; do
        for f in "${m}_l${LAST[$m]}_msd" "${m}_lavg"; do
            [ -f "artifacts/features/$ds/$f/record.json" ] || missing=1
        done
    done
done
# The six encoders' mean and SD sets take about 5.5 GB (IEMOCAP 4.7, EmoFilm 0.6, the layer averages 0.2) and the
# sweeps' run folders about 0.5 GB; the disk is shared, so 8 GiB must be free before anything is made.
free_kb=$(df -Pk "$AFFECT_ROOT/artifacts/" | awk 'NR == 2 {print $4}')
if [ "$missing" = 1 ] && [ "${free_kb:-0}" -lt $((8 * 1024 * 1024)) ]; then
    say "$((${free_kb:-0} / 1024 / 1024)) GiB free where artifacts/ is, 8 GiB needed: stop before anything is made"
    exit 1
fi
say "$((free_kb / 1024 / 1024)) GiB free where artifacts/ is"
for m in "${MODELS[@]}"; do
    snaps=( "${SNAP[$m]}" )
    for ds in iemocap emofilm; do
        if [ -f "artifacts/features/$ds/${m}_l${LAST[$m]}_msd/record.json" ]; then
            say "$m: $ds mean and SD exist already"
        else
            say "$m: $ds mean and SD (${LABELS[$ds]}) from ${snaps[0]} on GPU ${CUDA_VISIBLE_DEVICES:-unset}"
            if ! bash "$HERE/pretrained.sh" extract-pretrained --dataset "$ds" --label-set "${LABELS[$ds]}" \
                    --subset all --model "$m" --snapshot "${snaps[0]}" --device cuda --pooling mean_sd >> "$LOG" 2>&1; then
                say "$m: $ds extraction FAILED, see $LOG: stop before any sweep"; exit 1
            fi
            say "$m: $ds mean and SD extracted"
        fi
        if [ -f "artifacts/features/$ds/${m}_lavg/record.json" ]; then
            say "$m: $ds layer average exists already"
        elif ! bash "$HERE/classical.sh" derive-pretrained --dataset "$ds" --model "$m" --kind layer_average \
                >> "$LOG" 2>&1; then
            say "$m: $ds layer average FAILED, see $LOG: stop before any sweep"; exit 1
        else
            say "$m: $ds layer average derived"
        fi
    done
done
pids=()
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
