# ruff: noqa: SLF001 -- controlled generation outcomes and inspection of model inputs in tests.
import asyncio
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import tyro
import websockets

from openpi import transforms
from openpi.configs import ROBOT_REGISTRY
from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import policy_config
from openpi.policies.b1k_policy import B1KOutputs
from openpi.policies.policy import PolicyRecorder
from openpi.policies.subtask_policy import SubtaskPolicy
from openpi.serving import websocket_b1k_server
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as config_lib
from scripts.b1k import serve_b1k

pytest_plugins = ["openpi.models.subtask_test"]


def simple_input(obs):
    return {
        "image": jax.tree.map(lambda x: np.asarray(x[0]), obs.images),
        "image_mask": jax.tree.map(lambda x: np.asarray(x[0]), obs.image_masks),
        "state": np.zeros(4),
        "prompt": "turn on radio",
    }


def test_policy_auto_provided_refresh_reset_and_missing_eos(model_setup, tokenizer, monkeypatch):
    model, obs, _, _ = model_setup
    policy = SubtaskPolicy(
        model,
        tokenizer_transform=transforms.TokenizeSubtaskPrompt(tokenizer, max_subtask_len=8),
        subtask_interval=2,
        output_transforms=[B1KOutputs(action_dim=4)],
        sample_kwargs={"num_steps": 2},
        rng=jax.random.key(3),
    )
    calls = []
    label = tokenizer.tokenize_subtask_training("turn on radio", "press radio", max_subtask_len=8)
    ids = label[0][label[3]]

    def generate(key, observation, **kwargs):
        calls.append(observation)
        assert not np.asarray(observation.token_loss_mask).any()
        return {
            "tokens": np.pad(ids, (0, 8 - len(ids)))[None],
            "lengths": np.array([len(ids)]),
            "has_eos": np.array([True]),
            "valid_prefix": np.array([True]),
        }

    monkeypatch.setattr(policy, "_generate", generate)
    source = simple_input(obs)
    result = policy.infer(source)
    assert result["subtask"] == "press radio"
    assert result["subtask_source"] == "generated"
    assert result["actions"].shape == (2, 4)
    assert np.isfinite(result["actions"]).all()
    assert policy.infer(source)["subtask_source"] == "cached"
    assert len(calls) == 1
    assert policy.infer(source)["subtask_source"] == "generated"
    policy.infer({**source, "prompt": "radio"})
    assert len(calls) == 3
    policy.reset()
    policy.infer(source)
    assert len(calls) == 4
    explicit = policy.infer({**source, "subtask": "move to radio"})
    assert explicit["subtask_source"] == "provided"
    assert len(calls) == 4
    policy.infer(source)
    assert len(calls) == 5
    policy.reset()
    monkeypatch.setattr(
        policy, "_generate", lambda *a, **kw: {"has_eos": np.array([False]), "valid_prefix": np.array([True])}
    )
    monkeypatch.setattr(policy, "_sample_actions", lambda *a, **kw: pytest.fail("Must not act on a truncated subtask"))
    with pytest.raises(RuntimeError, match="EOS"):
        policy.infer(source)
    assert policy._cached_tokens is None
    assert "subtask" not in source


def test_factory_uses_autonomous_policy_with_b1k_adapters_and_real_generation(
    model_setup,
    tokenizer,
    monkeypatch,
    tmp_path,
):
    _, _, _, model_config = model_setup
    model_config = dataclasses.replace(model_config, action_dim=32)
    model = model_config.create(jax.random.key(0))
    word = tokenizer._tokenizer.encode("press")[0]

    def head(self, hidden):
        return jnp.zeros((*hidden.shape[:2], 64)).at[..., word].set(5).at[..., 1].set(10)

    monkeypatch.setattr(gemma.Module, "deembed", head)
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda *a, **kw: model)
    monkeypatch.setattr(model_lib, "restore_params", lambda *a, **kw: {})
    monkeypatch.setattr(tokenizer_lib, "PaligemmaTokenizer", lambda max_len: tokenizer)
    config = dataclasses.replace(config_lib.get_config("pi05_b1k"), model=model_config)
    policy = policy_config.create_trained_policy(
        config, tmp_path, norm_stats={}, default_prompt="turn on radio", sample_kwargs={"num_steps": 2}
    )
    assert isinstance(policy, SubtaskPolicy)
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    source = {"observation/state": np.zeros(57)}
    source.update({f"observation/{key}": np.zeros((224, 224, 3), dtype=np.uint8) for key in robot.observations})
    result = policy.infer(source)
    assert result["subtask"] == "press"
    assert result["subtask_source"] == "generated"
    assert np.isfinite(result["actions"]).all()
    assert policy.metadata["subtask_generation"]


def test_b1k_wrapper_keeps_text_and_resets_on_instruction_change(tmp_path):
    class FakePolicy:
        def __init__(self):
            self.metadata = {"subtask_interval": 1}
            self.calls = []
            self.resets = 0

        def infer(self, obs):
            self.calls.append(obs)
            return {"actions": np.zeros((2, 23)), "subtask": obs.get("subtask", "press radio")}

        def reset(self):
            self.resets += 1

    policy = FakePolicy()
    recorder = PolicyRecorder(policy, str(tmp_path))
    wrapper = B1KPolicyWrapper(recorder, "b1k/R1Pro", "turn on radio", "receding_horizon", 2, 2)
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    source = {f"{robot.name}::proprio": np.zeros(57)}
    source.update({camera.obs_key: np.zeros((224, 224, 3), dtype=np.uint8) for camera in robot.observations.values()})
    wrapper.act(source)
    assert wrapper.last_subtasks == {0: "press radio"}
    wrapper.act(source)
    assert len(policy.calls) == 1
    wrapper.act({**source, "subtask": "move to radio"})
    assert policy.calls[-1]["subtask"] == "move to radio"
    assert wrapper.last_subtasks == {0: "move to radio"}
    assert policy.resets == 1
    wrapper.reset()
    assert wrapper.last_subtasks == {}
    assert policy.resets == 2


def test_server_protocol_preserves_subtasks_and_episode_reset():
    class Policy:
        def __init__(self):
            self.last_subtasks = {0: "press radio"}
            self.resets = 0

        def act(self, obs):
            return torch.zeros(23)

        def reset(self):
            self.resets += 1

    class Socket:
        remote_address = "test-client"

        def __init__(self):
            self.requests = iter([{}, {"reset": True}])
            self.sent = []

        async def recv(self):
            try:
                return websocket_b1k_server.packb(next(self.requests))
            except StopIteration:
                raise websockets.ConnectionClosed(None, None) from None

        async def send(self, data):
            self.sent.append(websocket_b1k_server.unpackb(data))

    policy, socket = Policy(), Socket()
    server = websocket_b1k_server.WebsocketPolicyServer(policy)
    asyncio.run(server._handler(socket))
    assert socket.sent[1]["action"].shape == (23,)
    assert socket.sent[1]["subtasks"] == ["press radio"]
    assert policy.resets == 1


def test_b1k_server_cli_exposes_subtask_settings():
    args = tyro.cli(
        serve_b1k.Args,
        args=[
            "--robot",
            "b1k/R1Pro",
            "--task",
            "b1k/turning_on_radio",
            "--policy.config",
            "pi05_b1k",
            "--policy.dir",
            "/tmp/subtask-checkpoint",
            "--subtask-prediction",
            "--max-token-len",
            "512",
            "--max-subtask-len",
            "64",
            "--subtask-interval",
            "2",
            "--subtask-temperature",
            "0.2",
        ],
    )
    assert args.subtask_prediction
    assert args.subtask_interval == 2
    assert args.subtask_temperature == 0.2
