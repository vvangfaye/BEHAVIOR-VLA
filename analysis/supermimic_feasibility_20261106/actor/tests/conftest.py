"""Reuse the existing actual Gemma/attention CPU fixture without copying it."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL_TESTS = ROOT / "BEHAVIOR-VLAs/rlinf_adapter/tests"
sys.path.insert(0, str(ORIGINAL_TESTS))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location("existing_b1k_test_setup", ORIGINAL_TESTS / "conftest.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
tokenizer_path = setup.tokenizer_path
norm_path = setup.norm_path

# Two Gemma layers, real KV attention, adaptive norm and Flow-SDE sampling.
from test_model import model as model  # noqa: E402,F401
