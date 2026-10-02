"""Isaac Lab RL post-training for the AIRBOT depth policy."""

import os

# The pinned DA2 transform runs with deterministic CuBLAS; this must be set before Isaac initialises CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
