#!/usr/bin/env bash
# The EmoFilm cold test (added 2026-10-20): EmoFilm's clips through each listed pretrained encoder on one GPU (every
# hidden layer, as for IEMOCAP), the listed openSMILE functionals on the CPU, then affect.cli cold-test of the listed
# selection records in the classical environment, all records in one process. Run on the node in its own tmux session:
#     tmux new -d -s coldtest20 "CUDA_VISIBLE_DEVICES=3 bash ~/affect/scripts/coldtest_night.sh \
#         --models wavlm_large hubert_base --functionals ComParE_2016 --selections <record> [...]"
# (--functionals none: no openSMILE set). data/emofilm/assembled must be there, and every record. A model or feature
# set whose EmoFilm features exist already is not extracted again; a model whose snapshot is not the only one in
# ~/cache/huggingface, or an extraction that fails, stops the script before any cold test. The script's lines go to
# logs/coldtest_night_<utc>.txt, the commands' output to the same file. The session ends by itself.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/common.sh"
cd "$AFFECT_ROOT"
LOG="$AFFECT_ROOT/logs/coldtest_night_$(stamp).txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
USAGE="usage: coldtest_night.sh --models <model> [...] --functionals <feature set> [...]|none --selections <record> [...]"
[ "${1:-}" = "--models" ] || { echo "$USAGE"; exit 1; }
shift
MODELS=(); FUNCTIONALS=(); SELECTIONS=()
while [ "$#" -gt 0 ] && [ "$1" != "--functionals" ]; do MODELS+=("$1"); shift; done
[ "${1:-}" = "--functionals" ] || { echo "$USAGE"; exit 1; }
shift
while [ "$#" -gt 0 ] && [ "$1" != "--selections" ]; do FUNCTIONALS+=("$1"); shift; done
[ "${1:-}" = "--selections" ] || { echo "$USAGE"; exit 1; }
shift
SELECTIONS=("$@")
[ "${#MODELS[@]}" -gt 0 ] && [ "${#FUNCTIONALS[@]}" -gt 0 ] && [ "${#SELECTIONS[@]}" -gt 0 ] || { echo "$USAGE"; exit 1; }
[ "${FUNCTIONALS[*]}" = "none" ] && FUNCTIONALS=()
declare -A REPO=([wavlm_base_plus]=microsoft--wavlm-base-plus [hubert_base]=facebook--hubert-base-ls960
                 [wav2vec2_base]=facebook--wav2vec2-base [wavlm_large]=microsoft--wavlm-large
                 [hubert_large]=facebook--hubert-large-ll60k [wav2vec2_large]=facebook--wav2vec2-large-lv60)
declare -A LAST=([wavlm_base_plus]=12 [hubert_base]=12 [wav2vec2_base]=12 [wavlm_large]=24 [hubert_large]=24
                 [wav2vec2_large]=24)
DATA=(--dataset emofilm --label-set three_no_neutral --subset all)
[ -f data/emofilm/assembled/ASSEMBLY.json ] || { say "no data/emofilm/assembled: stop"; exit 1; }
for s in "${SELECTIONS[@]}"; do
    [ -f "artifacts/selections/$s/selection.json" ] || { say "no record $s: stop"; exit 1; }
done
for m in "${MODELS[@]}"; do
    [ -n "${REPO[$m]:-}" ] || { say "unknown model $m: stop"; exit 1; }
    if [ -f "artifacts/features/emofilm/${m}_l${LAST[$m]}/record.json" ]; then
        say "$m: EmoFilm features exist already"; continue
    fi
    snaps=( "$HOME"/cache/huggingface/models--"${REPO[$m]}"/snapshots/* )
    if [ "${#snaps[@]}" != 1 ] || [ ! -d "${snaps[0]}" ]; then
        say "$m: not exactly one snapshot in ~/cache/huggingface: stop"; exit 1
    fi
    say "$m: extracting EmoFilm from ${snaps[0]} on GPU ${CUDA_VISIBLE_DEVICES:-unset}"
    if ! bash "$HERE/pretrained.sh" extract-pretrained "${DATA[@]}" --model "$m" --snapshot "${snaps[0]}" \
            --device cuda --pooling mean >> "$LOG" 2>&1; then
        say "$m: extraction FAILED, see $LOG: stop before any cold test"; exit 1
    fi
    say "$m: extracted"
done
for f in ${FUNCTIONALS[@]+"${FUNCTIONALS[@]}"}; do
    if [ -f "artifacts/features/emofilm/$f/record.json" ]; then say "$f: EmoFilm features exist already"; continue; fi
    say "$f: extracting EmoFilm's functionals (8 processes)"
    if ! bash "$HERE/classical.sh" extract-functionals "${DATA[@]}" --feature-set "$f" --workers 8 >> "$LOG" 2>&1; then
        say "$f: extraction FAILED, see $LOG: stop before any cold test"; exit 1
    fi
    say "$f: extracted"
done
say "cold tests of ${#SELECTIONS[@]} records started"
if bash "$HERE/classical.sh" cold-test --selections "${SELECTIONS[@]}" --target emofilm --label-set three_no_neutral \
        --subset all >> "$LOG" 2>&1; then
    say "ALL DONE: ${#SELECTIONS[@]} records cold-tested (artifacts/coldtests/)"
else
    say "cold-test FAILED, see $LOG"; exit 1
fi
