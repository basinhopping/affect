#!/usr/bin/env bash
# Run a classical command (extract-functionals, classical, classical-selftest) with the classical
# environment, ~/venvs/classical (built by setup_classical_env.sh), from any folder:
#     bash ~/affect/scripts/classical.sh classical-selftest --dataset iemocap --label-set four_merged --subset all --n-jobs 4
#     bash ~/affect/scripts/classical.sh extract-functionals --dataset iemocap --label-set seven --subset all --feature-set eGeMAPSv02 --workers 16
#     tmux new -d -s classical "bash ~/affect/scripts/classical.sh classical --sweep-file sweeps/classical_svm_egemaps.json --max-consecutive-failures 3"
# One BLAS thread per process, because the extraction and TPOT start many processes themselves.
# Keeps a copy of the output in logs/<command>_<utc>_<process id>.log (the process id keeps two
# sessions started in the same second apart; on 2026-09-26 the SVM and TPOT sessions shared one log).
set -euo pipefail
source "$(dirname "$0")/common.sh"
CLASSICAL_VENV="$HOME/venvs/classical"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$AFFECT_ROOT"
COMMAND="${1:?usage: classical.sh <command> [arguments]}"
"$CLASSICAL_VENV/bin/python" -m affect.cli "$@" 2>&1 | tee -a "$AFFECT_ROOT/logs/${COMMAND}_$(stamp)_$$.log"
