"""BEHAVIOR PyTorch entrypoint; shares the tested ordinary/subtask trainer."""

import os
from pathlib import Path
import sys

# Set before importing any of the shared JAX-dependent configuration modules.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_pytorch import main

if __name__ == "__main__":
    main()
