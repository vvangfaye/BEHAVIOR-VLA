import ast
from pathlib import Path

import numpy as np
import pytest

from b1k_rlinf.tokenizer import JointTokenizer


def original_tokenizer_class():
    # Execute only the original class, avoiding unrelated JAX/FAST downloads.
    path = Path(__file__).resolve().parents[2] / "openpi/src/openpi/models/tokenizer.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PaligemmaTokenizer")
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["PaligemmaTokenizer"]


def test_exact_token_and_causal_mask_parity(tokenizer_path):
    adapter = JointTokenizer(tokenizer_path)
    cls = original_tokenizer_class()
    original = cls.__new__(cls)
    original._tokenizer = adapter.processor
    original._max_len = 512
    state = np.linspace(-2, 2, 23)
    label = np.asarray(adapter.processor.encode("move hand", add_eos=True))
    for result, reference in (
        (adapter.prefix("open_the_radio", state), original.tokenize_subtask_prefix("open_the_radio", state)),
        (
            adapter.actions("open_the_radio", state, label),
            original.tokenize_subtask_tokens("open_the_radio", label, state),
        ),
    ):
        for key, expected in zip(("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask"), reference):
            np.testing.assert_array_equal(result[key], expected)


def test_rejects_incomplete_generated_subtask(tokenizer_path):
    tokenizer = JointTokenizer(tokenizer_path)
    with pytest.raises(ValueError, match="EOS"):
        tokenizer.actions("open radio", np.zeros(23), np.array([3, 4]))
    with pytest.raises(ValueError, match="truncation"):
        JointTokenizer(tokenizer_path, max_len=80, max_subtask_len=64).prefix("open radio", np.zeros(23))
