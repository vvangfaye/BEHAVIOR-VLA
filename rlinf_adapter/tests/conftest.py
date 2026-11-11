"""CPU tests use the real RLinf model modules without starting Ray workers."""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import sentencepiece

ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADAPTER))
UPSTREAM = Path(
    os.environ.get("RLINF_ROOT", ADAPTER.parent / "analysis/rlinf_adaptation_20261020/upstream")
)
if not (UPSTREAM / "rlinf/models/embodiment/openpi/tasks/rl.py").is_file():
    raise RuntimeError("Set RLINF_ROOT to the pinned RLinf source checkout")
sys.path.insert(0, str(UPSTREAM))

# Only bypass package initializers that register every distributed backend.
# All model, sampling and attention code imported below is the real upstream code.
for name in ("rlinf.models", "rlinf.models.embodiment", "rlinf.models.embodiment.openpi"):
    module = types.ModuleType(name)
    module.__path__ = [str(UPSTREAM.joinpath(*name.split(".")))]
    sys.modules[name] = module


@pytest.fixture(scope="session")
def tokenizer_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("tokenizer")
    corpus = root / "corpus.txt"
    corpus.write_text(
        ("Task: open the radio, State: 0 1 2 127 255; Subtask: move hand\nAction: grasp object\n") * 20
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(root / "tiny"),
        vocab_size=96,
        hard_vocab_limit=False,
        bos_id=2,
        eos_id=1,
        unk_id=0,
        minloglevel=2,
    )
    return root / "tiny.model"


@pytest.fixture
def norm_path(tmp_path):
    import json

    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    name: {"mean": (np.arange(23) / 10).tolist(), "std": (1 + np.arange(23) / 20).tolist()}
                    for name in ("state", "actions")
                }
            }
        )
    )
    return path
