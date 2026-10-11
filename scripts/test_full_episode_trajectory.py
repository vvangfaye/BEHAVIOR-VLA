"""CPU data-contract tests; synthetic arrays are not real collection evidence."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_episode_trajectory import (
    CAMERAS, KIND, MANIFEST_KIND, NativeFullEpisodeTrajectory,
    capture_observation, inspect_full_episode_trajectory,
)
import eval_b1k_queue as queue_runner
from eval_b1k_queue import run_task_instance, release_uploaded_trajectories


def metadata(**changes):
    result = dict(task="put_cups_away", instance_id=301, rollout_id=0, run_id="fixture-run",
                  attempt_id="attempt-001", checkpoint={"step": 1688977, "sha256": "a" * 64},
                  fps=30, task_prompt="Put the cups away.", source_manifest_sha256="b" * 64)
    result.update(changes)
    return result


def observation(index=0):
    return {"robot_r1::proprio": np.full(57, index, np.float32),
            **{source: np.full((8, 8, 4), index, np.uint8) for source in CAMERAS.values()}}


def result(**changes):
    value = dict(task="put_cups_away", instance_id=301, rollout_id=0, steps=3, success=True,
                 terminated=True, truncated=False, final_goal_state={"success": True},
                 conditioning={"mode": "whole_task", "privileged": False, "oracle": False})
    value.update(changes)
    return value


def write_episode(path, *, success=True):
    with NativeFullEpisodeTrajectory(path, metadata()) as writer:
        for step, (label, source) in enumerate(((None, None), ("", "generated"), ("grasp cup", "cached"))):
            writer.record(capture_observation(observation(step)), np.full(23, step, np.float32),
                          step == 2, False, label, source)
        writer.finish(result(success=success, final_goal_state={"success": success}),
                      capture_observation(observation(3)))


def test_native_missing_labels_and_terminal_observation_roundtrip(tmp_path):
    path = tmp_path / "episode.hdf5"
    write_episode(path)
    info = inspect_full_episode_trajectory(path)
    assert info["metadata"]["official_benchmark"] is False
    assert info["segments"] == [dict(start=0, end=1, subtask=None, source=None),
                                dict(start=1, end=2, subtask="", source="generated"),
                                dict(start=2, end=3, subtask="grasp cup", source="cached")]
    assert info["terminal_observation_recorded"]
    with h5py.File(path) as stream:
        np.testing.assert_array_equal(stream["observation.state"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(stream["action"][:, 0], [0, 1, 2])
        assert stream["terminal_observation/observation.state"][0] == 3


def test_completed_failure_saved_but_not_bc_eligible(tmp_path):
    path = tmp_path / "failure.hdf5"
    write_episode(path, success=False)
    assert not inspect_full_episode_trajectory(path, require_success=False)["result"]["success"]
    with pytest.raises(ValueError, match="successes"):
        inspect_full_episode_trajectory(path)


@pytest.mark.parametrize("changes", [dict(instance_id=300), dict(instance_id=321), dict(rollout_id=True),
                                     dict(checkpoint={}), dict(source_manifest_sha256="mutable"),
                                     dict(official_benchmark=True), dict(conditioning="provided")])
def test_bad_identity_rejected_before_file_creation(tmp_path, changes):
    path = tmp_path / "invalid.hdf5"
    with pytest.raises(ValueError):
        NativeFullEpisodeTrajectory(path, metadata(**changes))
    assert not path.exists()


@pytest.mark.parametrize("source", ["provided", "teacher", "oracle", "cached "])
def test_provided_or_teacher_labels_cannot_be_native(tmp_path, source):
    with NativeFullEpisodeTrajectory(tmp_path / "wrong.hdf5", metadata()) as writer:
        with pytest.raises(ValueError, match="native"):
            writer.record(capture_observation(observation()), np.zeros(23), False, False, "grasp", source)
        assert writer.count == 0


def test_only_executed_vector_and_no_steps_after_terminal(tmp_path):
    with NativeFullEpisodeTrajectory(tmp_path / "wrong.hdf5", metadata()) as writer:
        with pytest.raises(ValueError, match="one executed"):
            writer.record(capture_observation(observation()), np.zeros((16, 23)), False, False, None, None)
        writer.record(capture_observation(observation()), np.zeros(23), True, False, "grasp", "generated")
        with pytest.raises(ValueError, match="terminal"):
            writer.record(capture_observation(observation()), np.zeros(23), False, False, "move", "cached")


@pytest.mark.parametrize("changes", [dict(steps=2), dict(success=False), dict(task="other"),
                                     dict(terminated=False), dict(conditioning={"mode": "external_fixed"})])
def test_bad_outcome_keeps_file_incomplete(tmp_path, changes):
    path = tmp_path / "bad.hdf5"
    with NativeFullEpisodeTrajectory(path, metadata()) as writer:
        for index in range(3):
            writer.record(capture_observation(observation()), np.zeros(23), index == 2, False, None, None)
        with pytest.raises(ValueError):
            writer.finish(result(**changes))
    with pytest.raises(ValueError, match="Unfinished"):
        inspect_full_episode_trajectory(path, require_success=False)


@pytest.mark.parametrize("field", ["action", "subtask", "metadata", "terminal_observation/observation.state"])
def test_tampering_is_rejected(tmp_path, field):
    path = tmp_path / "tampered.hdf5"
    write_episode(path)
    with h5py.File(path, "r+") as stream:
        if field == "metadata":
            value = json.loads(stream.attrs["metadata"]); value["checkpoint"]["step"] += 1
            stream.attrs["metadata"] = json.dumps(value)
        elif field == "subtask":
            stream[field][0] = json.dumps("fake")
        elif field.startswith("terminal_"):
            stream[field][0] += 1
        else:
            stream[field][0, 0] += 1
    with pytest.raises(ValueError):
        inspect_full_episode_trajectory(path)


class FakeEvaluator:
    def __init__(self, *, failure_step=None):
        self.failure_step = failure_step
        self.env = SimpleNamespace(task=SimpleNamespace(success=True), env_config={"rendering_frequency": 30})
        self.metrics = []
        self.conditioning_metadata = result()["conditioning"]
        self.prompt_events = self.subtask_events = self.subtask_vote_events = self.selector_events = []
        self.subtask_vote_summary = self.selector_summary = None
        self.final_goal_state = {"success": True}

    def reset(self, **kwargs):
        self.index = 0
        self.obs = observation()
        self.observed_subtasks, self.observed_subtask_sources = [], []

    def load_task_instance(self, instance):
        assert instance == 301

    def step(self):
        if self.index == self.failure_step:
            raise RuntimeError("synthetic environment failure")
        self.robot_action = np.full(23, self.index + 10, np.float32)
        self.observed_subtasks.append([None, "", "grasp cup"][self.index])
        self.observed_subtask_sources.append([None, "generated", "cached"][self.index])
        self.index += 1
        self.obs = observation(self.index)
        return self.index == 3, False


def setup_runner(tmp_path, *, capture=True, scratch=False):
    attempt_root = tmp_path / "attempts"
    output = attempt_root / "instance-301/attempt-1"
    args = SimpleNamespace(task_name="put_cups_away", num_rollouts=1, write_video=False, video_fps=30,
                           mode="public_test", prompt_mode="whole_task", external_subtask=None,
                           external_subtask_selector=None, subtask_vote_window=None,
                           queue_run_id="fixture-run", queue_attempt_root=str(attempt_root))
    if capture:
        manifest = dict(kind=MANIFEST_KIND, run_id="fixture-run", checkpoint=metadata()["checkpoint"],
                        source_manifest_sha256="b" * 64, task_prompts={"put_cups_away": "Put the cups away."})
        if scratch:
            manifest["trajectory_root"] = str(tmp_path / "scratch")
        args.full_episode_collection_manifest = tmp_path / "manifest.json"
        args.full_episode_collection_manifest.write_text(json.dumps(manifest))
    logger = SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)
    base = SimpleNamespace(logger=logger, verify_external_subtask_rollout=lambda *a, **k: None,
                           verify_subtask_voting_rollout=lambda *a, **k: None)
    return base, args, output


def test_opt_in_queue_capture_starts_at_reset_zero_with_executed_actions(tmp_path):
    base, args, output = setup_runner(tmp_path, scratch=True)
    values = run_task_instance(base, FakeEvaluator(), args, 301, output)
    reference = values[0]["full_episode_trajectory"]
    assert Path(reference["path"]).is_relative_to(tmp_path / "scratch")
    info = inspect_full_episode_trajectory(reference["path"])
    assert info["metadata"]["attempt_id"] == "instance-301/attempt-1/rollout-0"
    assert info["steps"] == 3 and info["terminal_observation_recorded"]
    with h5py.File(reference["path"]) as stream:
        np.testing.assert_array_equal(stream["observation.state"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(stream["action"][:, 0], [10, 11, 12])
    saved = json.loads((output / "json/put_cups_away_301_0.json").read_text())
    assert saved == values[0]


def test_default_queue_path_keeps_legacy_result_and_no_raw_files(tmp_path):
    base, args, output = setup_runner(tmp_path, capture=False)
    values = run_task_instance(base, FakeEvaluator(), args, 301, output)
    assert "full_episode_trajectory" not in values[0]
    assert "task_success" not in values[0]
    assert not list(tmp_path.rglob("*.hdf5"))


def test_exception_does_not_publish_json_or_complete_raw_attempt(tmp_path):
    base, args, output = setup_runner(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic"):
        run_task_instance(base, FakeEvaluator(failure_step=1), args, 301, output)
    assert not list(output.glob("json/*.json"))
    paths = list(output.glob("trajectories/*.hdf5"))
    assert len(paths) == 1
    with h5py.File(paths[0]) as stream:
        assert not stream.attrs["complete"] and len(stream["action"]) == 1


def test_object_checkpoint_happens_before_json_completion(tmp_path, monkeypatch):
    base, args, output = setup_runner(tmp_path)
    manifest = json.loads(args.full_episode_collection_manifest.read_text())
    manifest["object_storage"] = {"fixture": True}
    args.full_episode_collection_manifest.write_text(json.dumps(manifest))
    calls = []

    def checkpoint(path, result, manifest, *, attempt_id):
        assert not list(output.glob("json/*.json"))
        assert inspect_full_episode_trajectory(path)["steps"] == 3
        calls.append(attempt_id)
        return {"object_key": "fixture/episode.hdf5", "verified": True}

    monkeypatch.setitem(sys.modules, "full_episode_checkpoint", SimpleNamespace(checkpoint_episode=checkpoint))
    values = run_task_instance(base, FakeEvaluator(), args, 301, output)
    assert calls == ["instance-301/attempt-1/rollout-0"]
    assert values[0]["full_episode_object_checkpoint"]["verified"]


def test_manifest_changed_mid_episode_leaves_only_incomplete_raw(tmp_path):
    base, args, output = setup_runner(tmp_path)

    class MutableManifestEvaluator(FakeEvaluator):
        def step(self):
            flags = super().step()
            if self.index == 2:
                changed = json.loads(args.full_episode_collection_manifest.read_text())
                changed["checkpoint"]["step"] += 1
                args.full_episode_collection_manifest.write_text(json.dumps(changed))
            return flags

    with pytest.raises(ValueError, match="manifest changed"):
        run_task_instance(base, MutableManifestEvaluator(), args, 301, output)
    assert not list(output.glob("json/*.json"))
    with h5py.File(next(output.glob("trajectories/*.hdf5"))) as stream:
        assert not stream.attrs["complete"]


def uploaded_scratch_fixture(tmp_path, monkeypatch, *, release=True):
    base, args, attempt_dir = setup_runner(tmp_path)
    scratch_base = tmp_path / "scratch-base"
    manifest = json.loads(args.full_episode_collection_manifest.read_text())
    manifest.update(trajectory_root=str(scratch_base / "run/trajectories"),
                    release_uploaded_scratch=release,
                    object_storage={"put_cups_away:301:0": {"archive": {"key": "fixture/archive.tar"}}})
    args.full_episode_collection_manifest.write_text(json.dumps(manifest))
    monkeypatch.setattr(queue_runner, "FULL_EPISODE_SCRATCH_BASES", (scratch_base,))

    def checkpoint(path, result, manifest, *, attempt_id):
        from full_episode_trajectory import file_sha256
        info = inspect_full_episode_trajectory(path)
        return dict(kind="native_full_episode_object_checkpoint", run_id="fixture-run",
                    source_manifest_sha256="b" * 64, attempt_id=attempt_id,
                    task="put_cups_away", instance_id=301, rollout_id=0,
                    trajectory_sha256=file_sha256(path), content_sha256=info["content_sha256"],
                    subtask_sha256=info["subtask_sha256"], result=result.copy(),
                    archive=dict(object_key="fixture/archive.tar", bytes=123, sha256="c" * 64,
                                 etag="fixture-etag", parts=1, part_content_md5_verified=True))

    monkeypatch.setitem(sys.modules, "full_episode_checkpoint", SimpleNamespace(checkpoint_episode=checkpoint))
    values = run_task_instance(base, FakeEvaluator(), args, 301, attempt_dir)
    args.output_dir = str(tmp_path / "canonical")
    canonical_json = Path(args.output_dir) / "json"
    canonical_json.mkdir(parents=True)
    path = canonical_json / "put_cups_away_301_0.json"
    path.write_text(json.dumps(values[0]))
    return args, SimpleNamespace(task="put_cups_away", instance_id=301, attempt_dir=attempt_dir), path, Path(values[0]["full_episode_trajectory"]["path"])


def test_release_only_after_published_json_and_verified_uploaded_scratch(tmp_path, monkeypatch):
    args, lease, canonical, raw = uploaded_scratch_fixture(tmp_path, monkeypatch)
    assert release_uploaded_trajectories(lease, args) == 1
    assert canonical.is_file() and not raw.exists()


def test_default_retains_uploaded_scratch(tmp_path, monkeypatch):
    args, lease, _, raw = uploaded_scratch_fixture(tmp_path, monkeypatch, release=False)
    assert release_uploaded_trajectories(lease, args) == 0 and raw.is_file()


@pytest.mark.parametrize("damage", ["raw_hash", "outside_path", "incomplete_receipt", "wrong_object_key", "missing_json"])
def test_unverified_or_outside_scratch_is_never_released(tmp_path, monkeypatch, damage):
    args, lease, canonical, raw = uploaded_scratch_fixture(tmp_path, monkeypatch)
    value = json.loads(canonical.read_text())
    if damage == "raw_hash":
        with raw.open("ab") as stream:
            stream.write(b"changed")
    elif damage == "outside_path":
        outside = tmp_path / "persistent.hdf5"
        outside.write_bytes(raw.read_bytes())
        value["full_episode_trajectory"]["path"] = str(outside)
    elif damage == "incomplete_receipt":
        value["full_episode_object_checkpoint"]["archive"]["part_content_md5_verified"] = False
    elif damage == "wrong_object_key":
        value["full_episode_object_checkpoint"]["archive"]["object_key"] = "wrong/archive"
    if damage == "missing_json":
        canonical.unlink()
    else:
        canonical.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        release_uploaded_trajectories(lease, args)
    assert raw.exists()
    if damage == "outside_path":
        assert outside.exists()


def test_publish_failure_never_calls_scratch_release(tmp_path, monkeypatch):
    base, args, output = setup_runner(tmp_path, capture=False)
    args.headless = True
    args.queue_worker, args.queue_generation = "fixture", 0
    args.queue_db, args.output_dir = str(tmp_path / "queue"), str(tmp_path / "canonical")
    lease = SimpleNamespace(instance_id=301, attempt_dir=output, task="put_cups_away", attempt=1, lease_token="lease")

    class Queue:
        def __init__(self, *args):
            pass
        def claim(self, *args):
            return lease
        def publish(self, *args, **kwargs):
            raise RuntimeError("synthetic publish failure")
        def reconcile(self, *args, **kwargs):
            return False
        def fail(self, *args):
            return "exhausted"

    class Evaluator:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass

    base.gm = SimpleNamespace(HEADLESS=False)
    base.DEFAULT_EVAL_SEED, base.seed_everything = 0, lambda value: value
    base.Evaluator = lambda cfg: Evaluator()
    base.logger.error = lambda *a, **k: None
    monkeypatch.setattr(queue_runner, "EvaluationQueue", Queue)
    monkeypatch.setattr(queue_runner, "_build_evaluator_cfg", lambda *a: {})
    monkeypatch.setattr(queue_runner, "_validate_queue_metadata", lambda *a: None)
    calls = []
    monkeypatch.setattr(queue_runner, "release_uploaded_trajectories", lambda *a: calls.append(True))
    with pytest.raises(RuntimeError, match="publish failure"):
        queue_runner.run(args, base_module=base, instance_runner=lambda *a: [])
    assert calls == []
