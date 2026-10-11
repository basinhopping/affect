#!/usr/bin/env bash
# Build the classical baselines' own Python environment on the node. Run once:
#     bash ~/affect/scripts/setup_classical_env.sh
# Creates ~/venvs/classical from the system Python 3.12 with scripts/requirements_classical.txt
# (openSMILE, scikit-learn, TPOT 0.12), apart from ~/venvs/affect, so nothing the training jobs
# use changes. Records the exact versions in artifacts/env/pip_freeze_classical_<utc>.txt. It uses
# the network and the CPU only, and touches neither ~/venvs/affect nor running jobs.
set -euo pipefail
source "$(dirname "$0")/common.sh"
CLASSICAL_VENV="$HOME/venvs/classical"
LOG="$AFFECT_ROOT/logs/setup_classical_env_$(stamp).log"
exec > >(tee -a "$LOG") 2>&1

echo "setup_classical_env $(date -u +%FT%TZ)  root=$AFFECT_ROOT  venv=$CLASSICAL_VENV"
if [ -e "$CLASSICAL_VENV" ]; then
    echo "REFUSING: $CLASSICAL_VENV already exists. Move it aside first if you mean to rebuild it."
    exit 1
fi
echo "--- free space before"
df -h "$HOME"
/usr/bin/python3.12 -m venv "$CLASSICAL_VENV"
"$CLASSICAL_VENV/bin/python" -m pip install --no-cache-dir --upgrade pip
"$CLASSICAL_VENV/bin/python" -m pip install --no-cache-dir -r "$AFFECT_ROOT/scripts/requirements_classical.txt"
mkdir -p "$AFFECT_ROOT/artifacts/env"
FREEZE="$AFFECT_ROOT/artifacts/env/pip_freeze_classical_$(stamp).txt"
"$CLASSICAL_VENV/bin/python" -m pip freeze > "$FREEZE"
echo "--- exact versions written to $FREEZE"
du -sh "$CLASSICAL_VENV"
echo "--- free space after"
df -h "$HOME"
echo "setup_classical_env: done. Next: bash $AFFECT_ROOT/scripts/classical.sh classical-selftest --dataset iemocap --label-set four_merged --subset all --n-jobs 4"
