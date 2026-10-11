#!/usr/bin/env bash
# Run any affect command on the node with the project's environment, from any folder:
#     bash ~/affect/scripts/affect.sh status --sweep cnn_screen_1 --top 20
#     bash ~/affect/scripts/affect.sh enqueue --sweep-file sweeps/cnn_screen_1.json
#     CUDA_VISIBLE_DEVICES=0 bash ~/affect/scripts/affect.sh smoke --sweep-file sweeps/cnn_screen_1.json --expected-gpus 1
# Does the cd and the environment itself (common.sh), and keeps a copy of the output in
# logs/<command>_<utc>.log.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$AFFECT_ROOT"
COMMAND="${1:?usage: affect.sh <command> [arguments], for example: affect.sh status --sweep cnn_screen_1 --top 20}"
affect_python -m affect.cli "$@" 2>&1 | tee -a "$AFFECT_ROOT/logs/${COMMAND}_$(stamp).log"
