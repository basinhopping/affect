# Node settings shared by every node script. Sourced by the others, never run on its own.
# This is the one place for the environment location, the Hugging Face cache, the CUDA MPS
# fix and the thread caps. Nothing here or anywhere in the project uses /scratch.

AFFECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AFFECT_VENV="$HOME/venvs/affect"

# Model weights are cached in the home folder, next to the environment (plan 8.4).
export HF_HOME="$HOME/cache/huggingface"

# CUDA MPS fix (plan 8.3). Another user's MPS daemon makes CUDA unavailable unless these are
# empty strings before Python starts. src/affect/__init__.py clears them again on import.
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""

# Thread caps from the summer project's lane.sh: OpenBLAS on the node "is built for fewer
# threads than the box has cores".
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

export PYTHONPATH="$AFFECT_ROOT/src"
export PYTHONUNBUFFERED=1
mkdir -p "$AFFECT_ROOT/logs"

affect_python() { "$AFFECT_VENV/bin/python" "$@"; }
stamp() { date -u +%Y%m%dT%H%M%SZ; }
