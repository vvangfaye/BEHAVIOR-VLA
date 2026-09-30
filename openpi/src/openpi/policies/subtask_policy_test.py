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
from openpi.shared import frame_memory
from openpi.shared import subtask_contract
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


@pytest.mark.parametrize("conditioning", ["task_and_subtask", "subtask_only"])
def test_policy_auto_provided_refresh_reset_and_missing_eos(model_setup, tokenizer, monkeypatch, conditioning):
    model, obs, _, _ = model_setup
    model.subtask_action_conditioning = conditioning
    policy = SubtaskPolicy(
        model,
        tokenizer_transform=transforms.TokenizeSubtaskPrompt(
            tokenizer, max_subtask_len=8, subtask_action_conditioning=conditioning
        ),
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


@pytest.mark.parametrize("conditioning", ["task_and_subtask", "subtask_only"])
@pytest.mark.parametrize("memory_enabled", [False, True])
def test_factory_uses_autonomous_policy_with_b1k_adapters_and_real_generation(
    model_setup,
    tokenizer,
    monkeypatch,
    tmp_path,
    conditioning,
    memory_enabled,
):
    _, _, _, model_config = model_setup
    model_config = dataclasses.replace(model_config, action_dim=32, framesamp_modul=memory_enabled,
                                       memory_num_frames=3, memory_grid_size=1)
    model = model_config.create(jax.random.key(0))
    model.subtask_action_conditioning = conditioning
    if conditioning == "subtask_only":
        subtask_contract.save(tmp_path / "assets", conditioning)
    frame_memory.save(tmp_path / "assets", frame_memory.model_contract(model_config))
    word = tokenizer._tokenizer.encode("press")[0]

    def head(self, hidden):
        return jnp.zeros((*hidden.shape[:2], 64)).at[..., word].set(5).at[..., 1].set(10)

    monkeypatch.setattr(gemma.Module, "deembed", head)
    def load(config, *args, **kwargs):
        assert config.framesamp_modul == memory_enabled
        return model
    monkeypatch.setattr(pi0_config.Pi0Config, "load", load)
    monkeypatch.setattr(model_lib, "restore_params", lambda *a, **kw: {})
    monkeypatch.setattr(tokenizer_lib, "PaligemmaTokenizer", lambda max_len: tokenizer)
    # The service reconstructs memory settings from checkpoint metadata.
    config = dataclasses.replace(config_lib.get_config("pi05_b1k"),
                                 model=dataclasses.replace(model_config, framesamp_modul=False))
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
    if memory_enabled:
        assert policy.metadata["framesamp_modul"]
        assert result["policy_timing"]["memory_ms"] > 0
        assert len(policy._frame_memory.features) == 1
        policy.reset()
        assert len(policy._frame_memory.features) == 0
    assert policy.metadata["subtask_action_conditioning"] == conditioning
    assert policy.metadata["subtask_interval"] == (2 if conditioning == "subtask_only" else 1)
    if conditioning == "subtask_only":
        with pytest.raises(ValueError, match="disagrees"):
            policy_config.create_trained_policy(
                config, tmp_path, norm_stats={}, subtask_action_conditioning="task_and_subtask"
            )


def test_factorized_policy_uses_current_observation_and_exact_tokens(model_setup, tokenizer, monkeypatch):
    model, obs, _, _ = model_setup
    model.subtask_action_conditioning = "subtask_only"
    transform = transforms.TokenizeSubtaskPrompt(
        tokenizer, max_subtask_len=8, subtask_action_conditioning="subtask_only"
    )
    policy = SubtaskPolicy(model, tokenizer_transform=transform, subtask_interval=2)
    source = simple_input(obs)
    training = transform({**source, "subtask": "press radio"})
    ids = training["tokenized_prompt"][training["token_loss_mask"]]
    high_inputs, action_inputs = [], []

    def generate(key, observation, **kwargs):
        high_inputs.append(observation)
        return {
            "tokens": ids[None],
            "lengths": np.array([len(ids)]),
            "has_eos": np.array([True]),
            "valid_prefix": np.array([True]),
        }

    def sample(key, observation, **kwargs):
        action_inputs.append(observation)
        return jnp.zeros((1, 2, 4))

    monkeypatch.setattr(policy, "_generate", generate)
    monkeypatch.setattr(policy, "_sample_actions", sample)
    assert policy.infer(source)["subtask_source"] == "generated"
    np.testing.assert_array_equal(action_inputs[-1].action_tokenized_prompt[0], training["action_tokenized_prompt"])
    changed = {**source, "state": np.ones(4), "image": {k: np.ones_like(v) for k, v in source["image"].items()}}
    assert policy.infer(changed)["subtask_source"] == "cached"
    assert len(high_inputs) == 1
    assert not np.array_equal(action_inputs[0].action_tokenized_prompt, action_inputs[1].action_tokenized_prompt)
    assert np.asarray(action_inputs[1].images["base_0_rgb"]).mean() == 1
    assert policy.infer({**source, "subtask": "press radio"})["subtask_source"] == "provided"
    np.testing.assert_array_equal(action_inputs[-1].action_tokenized_prompt, action_inputs[0].action_tokenized_prompt)
    assert policy.infer({**source, "prompt": "move to radio"})["subtask_source"] == "generated"
    np.testing.assert_array_equal(action_inputs[-1].action_tokenized_prompt, action_inputs[0].action_tokenized_prompt)
    assert not np.array_equal(high_inputs[0].tokenized_prompt, high_inputs[-1].tokenized_prompt)


def test_joint_candidates_share_one_subtask_decision(model_setup, tokenizer, monkeypatch):
    model, obs, _, _ = model_setup
    policy = SubtaskPolicy(
        model,
        tokenizer_transform=transforms.TokenizeSubtaskPrompt(tokenizer, max_subtask_len=8),
        subtask_interval=2,
        output_transforms=[B1KOutputs(action_dim=4)],
        rng=jax.random.key(3),
    )
    arrays = tokenizer.tokenize_subtask_training("turn on radio", "press radio", max_subtask_len=8)
    ids = arrays[0][arrays[3]]
    generations, samples = [], []

    def generate(key, observation, **kwargs):
        generations.append(observation)
        return {"tokens": ids[None], "lengths": np.array([len(ids)]),
                "has_eos": np.array([True]), "valid_prefix": np.array([True])}

    def sample(key, observation, **kwargs):
        samples.append((np.asarray(jax.random.key_data(key)), np.asarray(observation.tokenized_prompt)))
        return jnp.full((1, 2, 4), len(samples), dtype=jnp.float32)

    monkeypatch.setattr(policy, "_generate", generate)
    monkeypatch.setattr(policy, "_sample_actions", sample)
    source = simple_input(obs)
    result = policy.infer_candidates(source, 4)
    assert result["actions"].shape == (4, 2, 4)
    np.testing.assert_array_equal(result["actions"][:, 0, 0], [1, 2, 3, 4])
    assert result["subtask"] == "press radio"
    assert result["subtask_source"] == "generated"
    assert len(generations) == 1
    assert policy._calls_since_generation == 1
    assert all(np.array_equal(samples[0][1], item[1]) for item in samples[1:])
    assert all(not np.array_equal(samples[0][0], item[0]) for item in samples[1:])
    assert policy.infer_candidates(source, 2)["subtask_source"] == "cached"
    assert len(generations) == 1
    assert policy.infer_candidates(source, 2)["subtask_source"] == "generated"
    assert len(generations) == 2
    with pytest.raises(ValueError, match="count"):
        policy.infer_candidates(source, 0)
    policy.reset()
    assert policy.infer_candidates(source, 2)["subtask_source"] == "generated"


def test_b1k_wrapper_keeps_text_and_resets_on_instruction_change(tmp_path):
    class FakePolicy:
        def __init__(self):
            self.metadata = {"subtask_interval": 1}
            self.calls = []
            self.resets = 0

        def infer(self, obs):
            self.calls.append(obs)
            result = {"actions": np.zeros((2, 23)), "subtask": obs.get("subtask", "press radio")}
            if obs.get("subtask") != "legacy override":
                result["subtask_source"] = "provided" if "subtask" in obs else "generated"
            return result

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
    assert wrapper.last_subtask_sources == {0: "generated"}
    wrapper.act(source)
    assert len(policy.calls) == 1
    wrapper.act({**source, "subtask": "move to radio"})
    assert policy.calls[-1]["subtask"] == "move to radio"
    assert wrapper.last_subtasks == {0: "move to radio"}
    assert wrapper.last_subtask_sources == {0: "provided"}
    assert policy.resets == 1
    wrapper.act({**source, "subtask": "legacy override"})
    assert wrapper.last_subtasks == {0: "legacy override"}
    assert wrapper.last_subtask_sources == {}
    assert policy.resets == 2
    wrapper.reset()
    assert wrapper.last_subtasks == {}
    assert wrapper.last_subtask_sources == {}
    assert policy.resets == 3


def test_server_protocol_preserves_subtasks_and_episode_reset():
    class Policy:
        def __init__(self):
            self.last_subtasks = {0: "press radio"}
            self.last_subtask_sources = {0: "provided"}
            self.resets = 0

        def act(self, obs):
            return torch.zeros(23)

        def reset(self):
            self.last_subtasks = {}
            self.last_subtask_sources = {}
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
    server = websocket_b1k_server.WebsocketPolicyServer(
        policy,
        metadata={"server_instance_token": "stale"},
        server_token="expected",
    )
    asyncio.run(server._handler(socket))
    assert socket.sent[0]["server_instance_token"] == "expected"
    assert socket.sent[1]["action"].shape == (23,)
    assert socket.sent[1]["subtasks"] == ["press radio"]
    assert socket.sent[1]["subtask_sources"] == ["provided"]
    assert policy.last_subtasks == {}
    assert policy.last_subtask_sources == {}
    assert policy.resets == 1


def test_server_protocol_keeps_legacy_subtask_response_compatible():
    class Policy:
        def __init__(self):
            self.last_subtasks = {0: "press radio"}

        def act(self, obs):
            return torch.zeros(23)

        def reset(self):
            pass

    class Socket:
        remote_address = "test-client"

        def __init__(self):
            self.requests = iter([{}])
            self.sent = []

        async def recv(self):
            try:
                return websocket_b1k_server.packb(next(self.requests))
            except StopIteration:
                raise websockets.ConnectionClosed(None, None) from None

        async def send(self, data):
            self.sent.append(websocket_b1k_server.unpackb(data))

    socket = Socket()
    server = websocket_b1k_server.WebsocketPolicyServer(Policy())
    asyncio.run(server._handler(socket))
    assert socket.sent[1]["subtasks"] == ["press radio"]
    assert "subtask_sources" not in socket.sent[1]


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
            "--server-token",
            "test-token",
        ],
    )
    assert args.subtask_prediction
    assert args.subtask_interval == 2
    assert args.subtask_temperature == 0.2
    assert args.server_token == "test-token"
