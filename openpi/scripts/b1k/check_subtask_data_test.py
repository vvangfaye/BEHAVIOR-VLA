# ruff: noqa: SLF001 -- use a local tokenizer fixture instead of downloading weights.
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.b1k import check_subtask_data

pytest_plugins = ["openpi.models.subtask_test"]


def test_metadata_preflight_checks_selected_original_episode_ids(tmp_path, monkeypatch, tokenizer):
    meta = tmp_path / "meta"
    episodes = meta / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [42, 99], "length": [10, 5]}), episodes / "file-000.parquet")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "turn on radio"}) + "\n")
    segments = meta / "subtask_segments.parquet"
    pq.write_table(
        pa.table(
            {"episode_index": [42, 42], "start": [0, 5], "end": [5, 10], "subtask": ["move to radio", "press radio"]}
        ),
        segments,
    )

    def make_tokenizer(max_len):
        tokenizer._max_len = max_len
        return tokenizer

    monkeypatch.setattr(check_subtask_data, "PaligemmaTokenizer", make_tokenizer)
    result = check_subtask_data.check_dataset(tmp_path, segments, episodes_index=[42])
    assert result["episodes"] == 1
    assert result["frames"] == 10
    assert result["segments"] == 2
    assert result["state_dim"] == 23
    assert result["status"] == "passed"
    with pytest.raises(ValueError, match="Missing subtask annotations"):
        check_subtask_data.check_dataset(tmp_path, segments)
    with pytest.raises(ValueError, match="Unknown selected"):
        check_subtask_data.check_dataset(tmp_path, segments, episodes_index=[0])
    with pytest.raises(ValueError, match="max_token_len"):
        check_subtask_data.check_dataset(tmp_path, segments, episodes_index=[42], max_token_len=8)
