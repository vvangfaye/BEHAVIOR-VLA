"""CPU contract tests for the vendored simulator API; no Isaac Sim launch required."""
import ast
from copy import deepcopy
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch as th

ROOT = Path(os.environ.get("BEHAVIOR_COMPAT_SOURCE", Path(__file__).resolve().parents[1] / "BEHAVIOR-1K"))
EVAL = ROOT / "OmniGibson/omnigibson/eval"


def load_definitions(path, names, **globals_):
    """Execute real CPU definitions without importing simulator-dependent package init."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    scope = dict(Any=Any, Dict=Dict, Optional=Optional, Sequence=Sequence, Tuple=Tuple, th=th)
    scope.update(globals_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**scope)


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location("behavior_single_env", EVAL / "single_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SingleEnvironmentAdapter


class Task:
    def __init__(self):
        self.object_scopes = [{"apple": "scene0-apple"}]
        self.success = th.tensor([False])
        self.goals = [[False, False]]

    def _evaluate_predicate(self, env_idx, name, *entities):
        return env_idx, name, entities

    def get_goal_option_satisfaction(self, env_idx):
        assert env_idx == 0
        return deepcopy(self.goals)

    def assign_object_scope_with_cache(self, env, env_idx):
        self.bound = env, env_idx


class Environment:
    num_envs = 1

    def __init__(self):
        self.task = Task()
        self.scene = SimpleNamespace(robots=[object()])
        self._current_steps = th.tensor([12], dtype=th.int32)
        self._current_episodes = th.tensor([7], dtype=th.int32)

    def get_obs(self):
        return [{"rgb": th.ones(2, 2, 3)}], [{"camera": "head"}]

    def reset(self, get_obs=True):
        return self.get_obs() if get_obs else None

    def step(self, action, **kwargs):
        self.last_action = action
        return [dict(proprio=th.ones(4))], th.tensor([0.5]), th.tensor([True]), th.tensor([False]), [{"done": {"success": True}}]


class Wrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


def test_single_environment_observations_and_actions(adapter):
    raw = Environment()
    env = adapter(Wrapper(raw))
    obs, info = env.get_obs()
    assert obs["rgb"].shape == (2, 2, 3) and info == {"camera": "head"}
    assert isinstance(env.reset()[0], dict)
    assert env.reset(get_obs=False) is None
    obs, reward, terminated, truncated, info = env.step(th.ones(4))
    assert raw.last_action.shape == (1, 4)
    assert reward == 0.5 and terminated is True and truncated is False
    assert info["done"]["success"] is True
    assert env.env is env and env.robots == raw.scene.robots and env.robot is raw.scene.robots[0]


def test_single_environment_rejects_batches(adapter):
    raw = Environment()
    raw.num_envs = 2
    with pytest.raises(ValueError, match="num_envs=1"):
        adapter(raw)


def test_task_scope_predicates_and_goals_bind_environment_zero(adapter):
    raw = Environment()
    env = adapter(raw)
    assert env.task.object_scope == {"apple": "scene0-apple"}
    assert env.task._evaluate_predicate("on", "apple", "table") == (0, "on", ("apple", "table"))
    assert env.get_goal_option_satisfaction() == [[False, False]]
    env.task.assign_object_scope_with_cache(env)
    assert raw.task.bound == (env, 0)
    raw.task.success[0] = True
    assert env.success is True


def test_state_restoration_keeps_vector_counter_shapes(adapter):
    raw = Environment()
    env = adapter(Wrapper(raw))
    steps, episodes = raw._current_steps, raw._current_episodes
    env._current_step = 0
    env._current_episode = 0
    assert raw._current_steps is steps and raw._current_episodes is episodes
    assert steps.tolist() == [0] and episodes.tolist() == [0]
    assert env._current_step == 0 and env._current_episode == 0


def test_local_policy_shapes_for_diagnostics_and_official_batches():
    cls = load_definitions(EVAL / "policies.py", ["LocalPolicy"]).LocalPolicy
    policy = cls(action_dim=23)
    policy.set_batch_size(0)
    assert policy.forward({"proprio": th.zeros(256)}).shape == (23,)
    batch = cls(action_dim=23)
    assert batch.forward({"proprio": th.zeros(3, 256)}).shape == (3, 23)


def test_task_metric_counts_new_goals_and_full_success(adapter):
    root = ROOT / "OmniGibson/omnigibson/metrics"
    base = load_definitions(root / "metric_base.py", ["MetricBase"]).MetricBase
    defs = load_definitions(root / "task_metric.py", ["compute_q_score", "TaskMetric"], MetricBase=base,
                            og=SimpleNamespace(sim=SimpleNamespace(get_rendering_dt=lambda: 1 / 30)))
    env = adapter(Environment())
    # A hashable scene mirrors actual simulator Scene objects.
    env._wrapped_env.scene = type("Scene", (), {"robots": [object()]})()
    metric = defs.TaskMetric({"length": 100}, env_accessor=env)
    metric.reset(env)
    env._wrapped_env.task.goals = [[True, False]]
    result = metric._compute_episode_metrics(env, {"timesteps": [4]})
    assert result["q_score"]["final"] == 0.5
    env._wrapped_env.task.success[0] = True
    assert metric._compute_episode_metrics(env, {})["q_score"]["final"] == 1.0


class Connection:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(value)

    def recv(self):
        return next(self.responses)

    def close(self):
        self.closed = True


@pytest.fixture
def client():
    ws = SimpleNamespace(sync=SimpleNamespace(client=SimpleNamespace(ClientConnection=Connection)),
                         exceptions=SimpleNamespace(ConnectionClosedError=type("Closed", (Exception,), {})))
    return load_definitions(EVAL / "utils/network_utils.py", ["WebsocketClientPolicy"],
                            Packer=lambda: SimpleNamespace(pack=lambda x: x), unpackb=lambda x: x,
                            deepcopy=deepcopy, logger=logging.getLogger(__name__), websockets=ws,
                            ACTION_CHUNK_REQUEST_KEY="__action_chunk_size__").WebsocketClientPolicy


def test_server_identity_mismatch_is_rejected(client):
    policy, conn = client(expected_server_token="expected"), Connection()
    with pytest.raises(RuntimeError, match="identity"):
        policy._validate_server_metadata(conn, {"server_instance_token": "other"})
    assert conn.closed


def test_subtask_voting_contract_is_validated(client):
    policy = client(expected_subtask_vote_action_horizon=16, expected_subtask_vote_interval=1,
                    expected_subtask_vote_window=3, expected_subtask_vote_threshold=2)
    conn = Connection()
    with pytest.raises(RuntimeError, match="voting contract"):
        policy._validate_server_metadata(conn, {"action_horizon": 16})
    assert conn.closed


def test_action_chunk_replay_preserves_subtask_metadata_and_reset_clears_it(client):
    policy = client(action_chunk_size=3)
    chunk = np.arange(6, dtype=np.float32).reshape(3, 2)
    policy._ws = Connection([{"action": chunk[0], "action_chunk": chunk, "subtasks": ["pick apple"],
                              "subtask_sources": ["predicted"], "subtask_votes": [{"id": 4}]}])
    for i in range(3):
        assert th.equal(policy.act({"proprio": np.zeros(4)}), th.from_numpy(chunk[i]))
        assert policy.current_subtask == "pick apple" and policy.current_subtask_vote == {"id": 4}
    assert len(policy._ws.sent) == 1 and policy._ws.sent[0]["__action_chunk_size__"] == 3
    policy.reset()
    assert policy._ws.sent[-1] == {"reset": True}
    assert policy.current_subtask is None and policy.current_subtask_vote is None
    assert policy.current_subtask_source is None and policy._action_chunk is None


@pytest.mark.parametrize("shape", [(2, 2), (3, 3)])
def test_invalid_action_chunk_shape_is_rejected(client, shape):
    policy = client(action_chunk_size=3)
    policy._ws = Connection([{"action": np.zeros(2, np.float32), "action_chunk": np.zeros(shape, np.float32)}])
    with pytest.raises(RuntimeError, match="shape"):
        policy.act({})


def test_action_chunk_must_start_with_the_returned_action(client):
    policy = client(action_chunk_size=3)
    policy._ws = Connection([{"action": np.ones(2, np.float32), "action_chunk": np.zeros((3, 2), np.float32)}])
    with pytest.raises(RuntimeError, match="exactly equal"):
        policy.act({})


def test_exact_state_camera_config_preserves_recorded_sensor_settings():
    tree = ast.parse((EVAL / "evaluator.py").read_text())
    evaluator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Evaluator")
    method = next(n for n in evaluator.body if isinstance(n, ast.FunctionDef) and n.name == "configure_robot_cameras")
    scope = {"DictConfig": dict, "get_class": lambda _: SimpleNamespace(camera_spec=lambda: {
        "modalities": ["rgb"], "resolution": {"head": (224, 224)}})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(EVAL / "evaluator.py"), "exec"), scope)
    owner = SimpleNamespace(robot_eval_config={"camera_sensor_names": {"head": "robot:head:Camera:0"}})
    robot = {"sensor_config": {"head:Camera:0": {"sensor_kwargs": {"focal_length": 20}}}}
    scope["configure_robot_cameras"](owner, robot, {"_target_": "wrapper"})
    assert robot["obs_modalities"] == ["proprio", "rgb"]
    assert robot["sensor_config"]["head:Camera:0"]["sensor_kwargs"] == {
        "focal_length": 20, "image_height": 224, "image_width": 224}
