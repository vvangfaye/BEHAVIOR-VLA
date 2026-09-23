# ruff: noqa: SLF001 -- use a local tokenizer fixture instead of downloading weights.
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

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


def _dataset_with_empty_labels(tmp_path):
    meta = tmp_path / "meta"
    episodes = meta / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [5, 42, 99], "length": [576, 10, 5]}), episodes / "file-000.parquet")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "turn on radio"}) + "\n")
    segments = meta / "subtask_segments.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": [5, 42, 42, 99],
                "start": [0, 0, 5, 0],
                "end": [576, 5, 10, 5],
                "subtask": ["", "move to radio", "press radio", " \t"],
                "is_annotated": [True, True, True, True],
                "provenance": ["annotated", "annotated", "annotated", "annotated"],
            }
        ),
        segments,
    )
    return segments


def test_preflight_drops_only_requested_empty_episodes(tmp_path, monkeypatch, tokenizer):
    segments = _dataset_with_empty_labels(tmp_path)
    tokenizer._max_len = 512
    monkeypatch.setattr(check_subtask_data, "PaligemmaTokenizer", lambda max_len: tokenizer)
    original = segments.read_bytes()
    report = check_subtask_data.check_dataset(
        tmp_path, segments, episodes_index=[42, 99], empty_label_policy="drop_episode"
    )
    assert report["requested_episodes"] == 2
    assert report["requested_frames"] == 15
    assert report["selected_episode_indices"] == [42]
    assert report["dropped_episode_indices"] == [99]
    assert report["dropped_episode_frames"] == 5
    assert report["empty_label_segments"] == 1
    assert report["episodes"] == 1
    assert report["frames"] == 10
    assert report["dropped_episodes"] == [
        {"episode_index": 99, "reason": "empty_subtask_label", "empty_intervals": [(0, 5)]}
    ]
    assert segments.read_bytes() == original
    # Filtering on episode IDs also works in strict mode with the original full table.
    strict = check_subtask_data.check_dataset(tmp_path, segments, episodes_index=[42])
    assert strict["selected_episode_indices"] == [42]
    with pytest.raises(ValueError, match="Empty subtask label: episode=5"):
        check_subtask_data.check_dataset(tmp_path, segments)
    with pytest.raises(ValueError, match="No episodes remain"):
        check_subtask_data.check_dataset(tmp_path, segments, episodes_index=[5, 99], empty_label_policy="drop_episode")


def test_cli_writes_exact_validated_ids_for_training(tmp_path, monkeypatch, tokenizer):
    segments = _dataset_with_empty_labels(tmp_path)
    tokenizer._max_len = 512
    monkeypatch.setattr(check_subtask_data, "PaligemmaTokenizer", lambda max_len: tokenizer)
    report_path = tmp_path / "outputs" / "report.json"
    ids_path = tmp_path / "outputs" / "valid.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_subtask_data.py",
            "--dataset-root",
            str(tmp_path),
            "--episodes-index",
            "42",
            "99",
            "--empty-label-policy",
            "drop_episode",
            "--report-file",
            str(report_path),
            "--write-episodes-file",
            str(ids_path),
        ],
    )
    check_subtask_data.main()
    report = json.loads(report_path.read_text())
    assert report["selected_episode_indices"] == [42]
    assert report["dropped_episode_indices"] == [99]
    assert ids_path.read_text() == "42\n"
    # Reproduce the trainer's strict load from the shared, unmodified parquet.
    labels = check_subtask_data.SubtaskFromSegments.from_parquet(
        segments, episodes_index=[int(ep) for ep in ids_path.read_text().splitlines()]
    )
    labels.validate_coverage({42: 10})


def test_training_contract_matches_distributed_drop_last(tmp_path, monkeypatch, tokenizer):
    segments = _dataset_with_empty_labels(tmp_path)
    episode_file = next((tmp_path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    table = pq.read_table(episode_file)
    pq.write_table(table.append_column("task_index", pa.array([0, 0, 1])), episode_file)
    tokenizer._max_len = 512
    monkeypatch.setattr(check_subtask_data, "PaligemmaTokenizer", lambda max_len: tokenizer)

    report = check_subtask_data.check_dataset(
        tmp_path,
        segments,
        empty_label_policy="drop_episode",
        frame_stride=1,
        global_batch_size=4,
        process_count=2,
        epochs=2,
        expected_tasks=1,
    )

    assert report["selected_episode_indices"] == [42]
    assert report["tasks"] == 1
    assert report["selected_episode_indices_sha256"] == check_subtask_data._episode_list_sha256([42])
    assert report["training_contract"] == {
        "annotation_sha256": check_subtask_data._sha256(segments),
        "episode_list_sha256": check_subtask_data._episode_list_sha256([42]),
        "frame_stride": 1,
        "process_count": 2,
        "retained_episodes": 1,
        "retained_frames": 10,
        "retained_tasks": 1,
        "strided_samples": 10,
        "global_batch_size": 4,
        "local_batch_size": 2,
        "samples_per_process": 5,
        "batches_per_epoch": 2,
        "dropped_samples_per_epoch": 2,
        "epochs": 2,
        "train_steps": 4,
    }
    assert len(report["training_contract_sha256"]) == 64


def test_training_contract_rejects_invalid_topology(tmp_path, monkeypatch, tokenizer):
    segments = _dataset_with_empty_labels(tmp_path)
    tokenizer._max_len = 512
    monkeypatch.setattr(check_subtask_data, "PaligemmaTokenizer", lambda max_len: tokenizer)
    with pytest.raises(ValueError, match="divisible by process_count"):
        check_subtask_data.check_dataset(
            tmp_path,
            segments,
            episodes_index=[42],
            global_batch_size=3,
            process_count=2,
            epochs=2,
        )


@pytest.mark.parametrize(("mode", "policy"), [("full", "drop_episode"), ("check", "drop_episode"), ("full", "error")])
def test_launcher_propagates_preflight_subset_to_norm_stats_and_trainer(tmp_path, tokenizer, mode, policy):
    """Run the real shell + preflight; record heavy norm/train invocations instead of running them."""
    _dataset_with_empty_labels(tmp_path)
    (tmp_path / "meta" / "info.json").write_text("{}")
    requested = tmp_path / "requested.txt"
    requested.write_text("42\n99\n")
    fake_openpi = tmp_path / "openpi"
    bin_dir = fake_openpi / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    (bin_dir / "activate").write_text(f"export PATH={shlex.quote(str(bin_dir))}:$PATH\n")
    scripts = fake_openpi / "scripts"
    (scripts / "b1k").mkdir(parents=True)
    real_preflight = Path(check_subtask_data.__file__).resolve()
    (scripts / "b1k" / "check_subtask_data.py").write_text(
        "from pathlib import Path\nimport runpy\nimport openpi.shared.download as download\n"
        f"download.maybe_download = lambda *args, **kwargs: Path({str(tmp_path / 'sp.model')!r})\n"
        f"runpy.run_path({str(real_preflight)!r}, run_name='__main__')\n"
    )
    calls = tmp_path / "calls.jsonl"
    recorder = (
        f"import json, sys\nwith open({str(calls)!r}, 'a') as out:\n    out.write(json.dumps(sys.argv) + '\\n')\n"
    )
    (scripts / "compute_norm_stats.py").write_text(recorder)
    (scripts / "b1k" / "train_b1k.py").write_text(recorder)
    repo = real_preflight.parents[3]
    env = {
        **os.environ,
        "CODE_ROOT": str(repo),
        "OPENPI_DIR": str(fake_openpi),
        "DATASET_ROOT": str(tmp_path),
        "EPISODES_FILE": str(requested),
        "EXP_NAME": "empty_labels_regression",
        "NUM_GPUS": "1",
        "CUDA_DEVICES": "0",
        "BATCH_SIZE": "1",
        "NUM_WORKERS": "0",
        "SUBTASK_EMPTY_LABEL_POLICY": policy,
        "SUBTASK_PROMPTS": "",
        "WHOLE_TASK_PROB": "0",
        "MAX_TOKEN_LEN": "512",
        "MAX_SUBTASK_LEN": "64",
        "PREFLIGHT_ONLY": "0",
        "PYTHONPATH": os.pathsep.join(str(Path(p or ".").resolve()) for p in sys.path),
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "train_pi05_subtask_b1k.sh"), mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert requested.read_text() == "42\n99\n"
    if policy == "error":
        assert result.returncode != 0
        assert "Empty subtask label: episode=99" in result.stderr
        assert not calls.exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    report_dir = fake_openpi / "outputs" / "subtask_preflight"
    report = json.loads((report_dir / "empty_labels_regression.json").read_text())
    assert report["selected_episode_indices"] == [42]
    assert report["dropped_episode_indices"] == [99]
    assert (report_dir / "empty_labels_regression.episodes.txt").read_text() == "42\n"
    if mode == "check":
        assert not calls.exists()
        return
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 2
    for command, flag in zip(commands, ["--episodes-index", "--data.episodes-index"], strict=True):
        index = command.index(flag)
        assert command[index + 1] == "42"
        assert index + 2 == len(command) or command[index + 2].startswith("--")
