"""affect: speech emotion recognition for the F26 Affective Computing project.

Importing this package applies the node's CUDA MPS fix before anything can import torch.
On the shared GPU node another user's MPS daemon makes CUDA unavailable ("Error 805: MPS client
failed to connect") unless CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY are empty
strings when torch loads. Verified live on 2026-09-24 (plan section 8.3). Empty strings are
harmless on machines without the daemon, so the fix is applied everywhere.

The fix is applied three ways so one missed import cannot bring back a silent CPU fallback:
the node scripts export the variables before Python starts (scripts/common.sh), this file
sets them on import, and training refuses to run without the GPUs it expects.
"""
import os
import sys

_MPS_VARIABLES = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")

if "torch" in sys.modules and any(os.environ.get(name) != "" for name in _MPS_VARIABLES):
    raise ImportError(
        "torch was imported before affect, and the CUDA MPS variables were not cleared first. "
        "On the shared GPU node that breaks CUDA. Import affect before torch, or start Python "
        "through scripts/*.sh, which clear the variables before Python starts."
    )

for _name in _MPS_VARIABLES:
    os.environ[_name] = ""
del _name
