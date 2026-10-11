"""Freezing and extending a mainline mixture is local and preserves prior draws."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpi/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openpi.training.recovery_index import build_index
from openpi.training.subtask_mixture import MixturePlan, consumed_contract, compare_consumed_contract
from prepare_subtask_mixture import build_manifest, make_source


@pytest.fixture
def indexes(tmp_path):
    paths = {}
    for number, kind in enumerate(("recovered", "collected", "replay")):
        episode = {"episode_index": number, "length": 100, "task_name": "task", "task_instance_id": number}
        provenance = {}
        if kind == "collected":
            provenance["collected_episodes"] = [{**episode, "source_kind": kind, "trial_id": "trial",
                                                  "content_sha256": hashlib.sha256(b"trial").hexdigest()}]
        segments = [{"episode_index": number, "start": 0, "end": 10, "subtask": "" if kind == "recovered" else "skill"},
                    {"episode_index": number, "start": 10, "end": 100, "subtask": "skill"}]
        index = build_index([episode], segments, dataset_root=tmp_path / kind, repo_id=kind, root_id=kind,
                            source_kind=kind, episode_indices=[number], provenance=provenance)
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(index))
        paths[kind] = path
    return paths


def test_cli_freezes_weights_and_boosts_without_reading_videos(indexes, tmp_path):
    script = Path(__file__).with_name("prepare_subtask_mixture.py")
    output = tmp_path / "mixture.json"
    weak = tmp_path / "weak.json"
    weak.write_text(json.dumps({"requested_groups": [{"task": "task", "subtask": "skill"}]}))
    command = [sys.executable, str(script), "--end-step", "100", "--output", str(output), "--weak-groups", str(weak)]
    for kind, index in indexes.items():
        command += [f"--{kind}-index", str(index)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    assert receipt["videos_decoded"] == 0 and receipt["training_started"] is False
    plan = MixturePlan.load(output)
    assert plan.global_batch_size == 128 and plan.num_steps == 100
    assert [s["weight"] for s in plan.manifest["phases"][0]["sources"]] == [.8, .1, .1]
    assert all(s["boosted_groups"] == 1 for s in receipt["sources"])
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_new_phase_preserves_consumed_prefix(indexes, tmp_path):
    old_sources = [make_source(kind, [path], weight) for (kind, path), weight in zip(indexes.items(), [.8, .1, .1])]
    manifest, initial = build_manifest(old_sources, end_step=100)
    original = tmp_path / "initial.json"
    original.write_text(json.dumps(manifest))
    new_sources = [make_source(kind, [path], weight) for (kind, path), weight in zip(indexes.items(), [.45, .45, .1])]
    manifest, continued = build_manifest(new_sources, end_step=200, previous_path=original, start_step=50)
    compare_consumed_contract(consumed_contract(initial, 50), continued, 50)
    assert [p["end_step"] for p in manifest["phases"]] == [50, 200]
    assert initial.reference(127) == continued.reference(127)
    assert initial.reference(6399) == continued.reference(6399)
    with pytest.raises(ValueError, match="preserve"):
        build_manifest(new_sources, end_step=200, batch_size=64, previous_path=original, start_step=50)
