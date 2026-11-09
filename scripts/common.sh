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

# The week's GPU windows (scripts/gpu_week.sh, added 2026-11-02), for the scripts that use a GPU beside its workers:
# gpu_allowed_at <epoch> <gpu> is true when that GPU may run then (Pacific time): weekday nights NIGHT_START (18:00, or
# the hour in NIGHT_START, as gpu_week.sh reads it; added 2026-11-05: since RULE_week32_addendum6.txt the week's nights
# start at 21:00, and scripts/enc32s.sh sets 21) to 08:00 every GPU, weekday days none, weekend days 08:00 to 22:00 GPUs
# 2 and 3, weekend nights every GPU.
gpu_allowed_at() {
    local wd hr
    wd=$(TZ=America/Los_Angeles date -d "@$1" +%u)
    hr=$((10#$(TZ=America/Los_Angeles date -d "@$1" +%H)))
    if [ "$wd" -le 5 ]; then [ "$hr" -lt 8 ] || [ "$hr" -ge "${NIGHT_START:-18}" ]; return; fi
    if [ "$hr" -lt 8 ] || [ "$hr" -ge 22 ]; then return 0; fi
    [ "$2" = 2 ] || [ "$2" = 3 ]
}
# gpu_other_users <gpu>: how many processes the GPU runs that are not our affect runs (another user's, or another
# project of ours: a process counts as ours only when it is our user's and runs affect.cli; MPS servers aside); empty
# when none.
gpu_other_users() {
    local uuid pid n=0 args uid
    uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v g="$1" '$1 == g {print $2}')
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', *' -v u="$uuid" '$1 == u {print $2}'); do
        uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
        args=$(ps -o args= -p "$pid" 2>/dev/null)
        case "$args" in nvidia-cuda-mps-server*|*/nvidia-cuda-mps-server*) continue ;; esac
        if [ -n "$uid" ] && [ "$uid" = "$(id -u)" ] && [[ "$args" == *affect.cli* ]]; then continue; fi
        n=$((n + 1))
    done
    if [ "$n" -gt 0 ]; then echo "$n"; fi
}
