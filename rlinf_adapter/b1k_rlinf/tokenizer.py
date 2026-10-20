"""The checkpoint's task/state/subtask token format, without JAX imports.

Ported from openpi/src/openpi/models/tokenizer.py in this repository. Generated
token IDs are preserved; decoding to text and encoding again changes context.
"""

from pathlib import Path

import numpy as np
import sentencepiece


class JointTokenizer:
    def __init__(self, path, max_len=512, max_subtask_len=64):
        if not 1 < max_subtask_len < max_len:
            raise ValueError("Expected 1 < max_subtask_len < max_len")
        self.max_len = max_len
        self.max_subtask_len = max_subtask_len
        self.processor = sentencepiece.SentencePieceProcessor(model_proto=Path(path).read_bytes())

    @property
    def eos_id(self):
        return self.processor.eos_id()

    def _prefix(self, prompt, state):
        prompt = prompt.replace("_", " ").replace("\n", " ").strip()
        state = np.asarray(state)
        if not prompt or state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("Expected nonempty task and finite one-dimensional state")
        bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {prompt}, State: {' '.join(map(str, bins))};\nSubtask: "
        return self.processor.encode(text, add_bos=True)

    def prefix(self, prompt, state):
        prefix = self._prefix(prompt, state)
        reserve = self.max_subtask_len + len(self.processor.encode("\nAction: "))
        if len(prefix) + reserve > self.max_len:
            raise ValueError("Task/state does not fit the token budget; refusing truncation")
        return self._pack(prefix, [], [])

    def actions(self, prompt, state, subtask_tokens):
        label = np.asarray(subtask_tokens)
        if label.ndim != 1 or not np.issubdtype(label.dtype, np.integer):
            raise ValueError("Expected a one-dimensional integer subtask token array")
        if not 2 <= len(label) <= self.max_subtask_len:
            raise ValueError("Subtask must contain text and EOS within its token budget")
        if label[-1] != self.eos_id or np.any(label[:-1] == self.eos_id):
            raise ValueError("Subtask requires exactly one final EOS")
        if np.any(label < 0) or np.any(label >= self.processor.vocab_size()):
            raise ValueError("Subtask token outside the vocabulary")
        if not self.processor.decode(label.tolist()).strip():
            raise ValueError("Empty generated subtask")
        return self._pack(self._prefix(prompt, state), label.tolist(), self.processor.encode("\nAction: "))

    def _pack(self, prefix, label, cue):
        size = len(prefix) + len(label) + len(cue)
        if size > self.max_len:
            raise ValueError("Joint token sequence exceeds max_len")
        pad = self.max_len - size
        return {
            "tokenized_prompt": np.asarray(prefix + label + cue + [0] * pad, dtype=np.int64),
            "tokenized_prompt_mask": np.asarray([True] * size + [False] * pad),
            "token_ar_mask": np.asarray(
                [False] * len(prefix) + [True] * (len(label) + len(cue)) + [False] * pad
            ),
        }
