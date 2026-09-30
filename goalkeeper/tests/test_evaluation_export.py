import json
import numpy as np
import pytest
from goalkeeper.evaluation import summarize
from goalkeeper.prepare import export_episode
from goalkeeper.data import load_manifest, load_record
from goalkeeper.goals import compile_bddl


def test_official_missing_zero_and_duplicate_rejection(tmp_path):
    row = {"task": "p", "instance_id": 0, "rollout_id": 0, "success": True, "q_score": {"final": 1.}}
    (tmp_path / "a.json").write_text(json.dumps(row))
    expected = [["p", 0, 0], ["p", 1, 0]]
    result = summarize(tmp_path, expected, baseline=tmp_path, bootstraps=100)
    assert result["q_score"] == .5 and result["missing_counted_as_zero"] == 1
    assert result["paired_difference"]["q_score"] == 0
    (tmp_path / "repeat.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Duplicate"): summarize(tmp_path, expected)


def test_episode_export_aligns_labels_to_executed_action_prefix(tmp_path):
    class Features:
        feature_id = "mock-v1"
        def encode_queries(self, texts): return np.zeros((len(texts), 8), np.float32)
        def encode_images(self, images): return np.stack([x[0, 0][None].astype(np.float32) for x in images])
    text = "(define (problem p) (:objects a - switch) (:goal (on a)))"
    bddl = tmp_path / "task.bddl"; bddl.write_text(text)
    program = compile_bddl(text)
    steps = 12
    rgb = np.stack([np.full((3, 2, 2, 3), t, np.uint8) for t in range(steps + 1)])
    truth = np.zeros((steps + 1, 1), np.float32); truth[8:] = 1
    episode = tmp_path / "episode.npz"
    np.savez_compressed(episode, rgb=rgb, proprio=np.arange(steps + 1)[:, None].astype(np.float32),
                        actions=np.arange(steps)[:, None].astype(np.float32), truth=truth,
                        visibility=np.ones_like(truth), atom_keys=np.array([a.key for a in program.atoms]))
    manifest = export_episode(episode, bddl, tmp_path / "out", group_id="training-scene-initial-state-0",
                              features=Features(), action_spec="test", horizon=4, history=2)
    records = [load_record(row) for row in load_manifest(manifest)]
    assert len(records) == 2
    first = records[0]
    assert first["proprio"][:, 0].tolist() == [0, 4]
    assert first["actions"][0, :, 0].tolist() == [4, 5, 6, 7]
    assert first["future_truth"].item() == 1  # truth[8], not truth[4] or end of episode
