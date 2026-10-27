#!/usr/bin/env bash
# Run a pretrained-model command (download-pretrained, extract-pretrained; added 2026-10-19) with the affect
# environment and the folder that holds transformers, huggingface_hub and their dependencies,
# ~/venvs/affect_pretrained (installed there with pip install --target --no-deps, so ~/venvs/affect itself is not
# changed), from any folder:
#     bash ~/affect/scripts/pretrained.sh download-pretrained --model wavlm_base_plus --cache-dir ~/cache/huggingface
#     CUDA_VISIBLE_DEVICES=3 bash ~/affect/scripts/pretrained.sh extract-pretrained --dataset iemocap \
#         --label-set seven --subset all --model wavlm_base_plus \
#         --snapshot <the folder download-pretrained printed> --device cuda --pooling mean
# The repository's own src comes first on the path, then the pretrained packages, then the environment's.
# Keeps a copy of the output in logs/<command>_<utc>_<process id>.log.
set -euo pipefail
source "$(dirname "$0")/common.sh"
EXTRA="$HOME/venvs/affect_pretrained"
[ -d "$EXTRA" ] || { echo "no $EXTRA: install transformers there first (pip install --target)"; exit 1; }
export PYTHONPATH="$AFFECT_ROOT/src:$EXTRA"
cd "$AFFECT_ROOT"
COMMAND="${1:?usage: pretrained.sh <command> [arguments]}"
affect_python -m affect.cli "$@" 2>&1 | tee -a "$AFFECT_ROOT/logs/${COMMAND}_$(stamp)_$$.log"
exit "${PIPESTATUS[0]}"
