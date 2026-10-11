#!/usr/bin/env bash
# Build this project's Python environment on the node. Run once:
#     bash ~/affect/scripts/setup_env.sh
# Creates ~/venvs/affect from the system Python 3.12 with the packages in requirements.in,
# registers it as the Jupyter kernel "affect", and records the exact versions installed in
# artifacts/env/pip_freeze_<utc>.txt. Pip keeps no download cache, to spare the home quota.
# It prints the environment's size and the free space before and after, for the storage budget.
set -euo pipefail
source "$(dirname "$0")/common.sh"
LOG="$AFFECT_ROOT/logs/setup_env_$(stamp).log"
exec > >(tee -a "$LOG") 2>&1

echo "setup_env $(date -u +%FT%TZ)  root=$AFFECT_ROOT  venv=$AFFECT_VENV"
if [ -e "$AFFECT_VENV" ]; then
    echo "REFUSING: $AFFECT_VENV already exists. Move it aside first if you mean to rebuild it."
    exit 1
fi
echo "--- free space before"
df -h "$HOME"
mkdir -p "$(dirname "$AFFECT_VENV")" "$HF_HOME"
/usr/bin/python3.12 -m venv "$AFFECT_VENV"
affect_python -m pip install --no-cache-dir --upgrade pip
affect_python -m pip install --no-cache-dir -r "$AFFECT_ROOT/requirements.in"
affect_python -m ipykernel install --prefix "$AFFECT_VENV" --name affect --display-name "affect"
mkdir -p "$AFFECT_ROOT/artifacts/env"
FREEZE="$AFFECT_ROOT/artifacts/env/pip_freeze_$(stamp).txt"
affect_python -m pip freeze > "$FREEZE"
echo "--- exact versions written to $FREEZE"
echo "--- sizes"
du -sh "$AFFECT_VENV" "$HF_HOME"
echo "--- free space after"
df -h "$HOME"
echo "setup_env: done. Next: bash $AFFECT_ROOT/scripts/check.sh"
