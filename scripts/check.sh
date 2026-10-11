#!/usr/bin/env bash
# Preflight on the node, before any overnight run:
#     bash ~/affect/scripts/check.sh
# 1. every test, with nothing allowed to skip (missing numpy, scipy, torch, librosa or data fails)
# 2. verify-data: every file of data/iemocap/assembled re-hashed against its manifest, and
#    the dataset checks repeated (plan 5.5, stop condition 1)
# 3. env-check: packages, network, and CUDA plus one training step on each of the 4 GPUs
#    (plan 5.5, stop condition 2)
# Stops at the first failure. Results: artifacts/checks/, and this log in logs/.
set -euo pipefail
source "$(dirname "$0")/common.sh"
LOG="$AFFECT_ROOT/logs/check_$(stamp).log"
exec > >(tee -a "$LOG") 2>&1
cd "$AFFECT_ROOT"

echo "=== 1/3 tests"
AFFECT_REQUIRE=numpy,scipy,torch,librosa,data affect_python -m unittest discover -s tests -t .
echo "=== 2/3 verify-data"
# verify-data stops once after listing the folder (exit 3: run again with the same state), so it
# runs until it exits 0 (checks passed) or 1 (checks failed).
STATE="node_$(stamp)"
for attempt in 1 2 3 4 5; do
    code=0
    affect_python -m affect.cli verify-data --dataset iemocap --state "$STATE" --max-seconds 7200 || code=$?
    if [ "$code" -eq 0 ]; then break; fi
    if [ "$code" -ne 3 ]; then echo "check: verify-data FAILED (exit code $code)"; exit 1; fi
    if [ "$attempt" -eq 5 ]; then echo "check: verify-data was still unfinished after 5 runs"; exit 1; fi
    echo "verify-data not finished yet; running it again with state $STATE"
done
echo "=== 3/3 env-check"
affect_python -m affect.cli env-check --expected-gpus 4
echo "check: all three passed. Log: $LOG"
