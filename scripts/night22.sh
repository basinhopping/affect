#!/usr/bin/env bash
# Night 22 (added 2026-10-21): the valence and arousal probes (va-probe, the six encoders on both four-class splits)
# and the EmoFilm fine-tuning of WavLM Large's two three-class records (emofilm-finetune), four processes at once on
# CPUs, no GPU. Run on the node in its own tmux session, from the tree that holds the code (~/affect_next while GPU
# workers run from ~/affect):
#     tmux new -d -s va22 "bash ~/affect_next/scripts/night22.sh"
# Each process's output goes to logs/night22_<name>_<utc>.log, the script's own lines to logs/night22_<utc>.txt. The
# session ends by itself when all four have ended. RULE_night22.txt (artifacts/analyses/night22_prep_20261021) says
# what is read afterwards. Names given as arguments (va_so, va_pooled, ft_so3, ft_pooled3) run only those (added
# 2026-10-21 for the second start of the probes):
#     tmux new -d -s va22b "bash ~/affect_next/scripts/night22.sh va_so va_pooled"
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/common.sh"
cd "$AFFECT_ROOT"
LOG="$AFFECT_ROOT/logs/night22_$(stamp).txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
MODELS="wavlm_base_plus hubert_base wav2vec2_base wavlm_large hubert_large wav2vec2_large"
NAMES=(va_so va_pooled ft_so3 ft_pooled3)
COMMANDS=(
    "va-probe --models $MODELS --dataset iemocap --label-set four_merged --subset all --split leave_one_session_out"
    "va-probe --models $MODELS --dataset iemocap --label-set four_merged --subset all --split utterance_5fold"
    "emofilm-finetune --selections pretrained19c_wavlm_large_so3 --film-folds 5"
    "emofilm-finetune --selections pretrained19c_wavlm_large_pooled3 --film-folds 5"
)
ONLY=" $* "
for n in "$@"; do
    case " ${NAMES[*]} " in *" $n "*) ;; *) say "unknown name $n: stop"; exit 1 ;; esac
done
pids=()
run_names=()
for i in "${!COMMANDS[@]}"; do
    if [ "$#" -gt 0 ] && [[ "$ONLY" != *" ${NAMES[$i]} "* ]]; then continue; fi
    # the command is split into words on purpose: it holds no quoting
    # shellcheck disable=SC2086
    bash "$HERE/classical.sh" ${COMMANDS[$i]} > "logs/night22_${NAMES[$i]}_$(stamp).log" 2>&1 &
    pids+=("$!")
    run_names+=("${NAMES[$i]}")
    say "${NAMES[$i]} started (process $!): ${COMMANDS[$i]}"
    sleep 1
done
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then say "${run_names[$i]} finished"
    else say "${run_names[$i]} ended with a failure, see its log"; failed=$((failed + 1)); fi
done
say "ALL DONE: ${#pids[@]} processes, $failed with a failure"
