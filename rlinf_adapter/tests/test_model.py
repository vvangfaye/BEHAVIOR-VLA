"""Actual small Gemma network: generation, attention and PPO gradient contracts."""

import numpy as np
import pytest
import torch
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.modules import gemma
from rlinf.models.embodiment.openpi.modules.model import BaseModel, Observation
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.tasks.rl import Pi0RLConfig
from torch import nn

from b1k_rlinf.io import IMAGE_KEYS, CheckpointIO
from b1k_rlinf.model import JointPi0RL
from b1k_rlinf.tokenizer import JointTokenizer


class TinyVision(nn.Module):
    """Small real image encoder; only model width and depth differ in tests."""

    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(3, width)

    def forward(self, image):
        return self.projection(image.mean((1, 2)))[:, None], None


@pytest.fixture
def model(norm_path, tokenizer_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    set_torch_compile(False)
    tokenizer = JointTokenizer(tokenizer_path)
    # Use the same Gemma attention, KV cache, adaptive norm and sampler as production.
    network = JointPi0RL.__new__(JointPi0RL)
    BaseModel.__init__(network, 32, 32, 512)
    network.model_action_dim = 32
    network.pi05, network.pcd = True, False
    network.embed_dtype = torch.float32
    width = 32
    cfg = gemma.Config(width=width, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16)
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    gemma.PALIGEMMA_VOCAB_SIZE = tokenizer.processor.vocab_size()
    try:
        network.llm = gemma.Module(
            [cfg, cfg], embed_dtype="float32", adarms=[False, True], use_gradient_checkpointing=False
        )
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab
    network.img = TinyVision(width)
    network.action_in_proj = nn.Linear(32, width)
    network.action_out_proj = nn.Linear(width, 32)
    network.time_mlp_in = nn.Linear(width, width)
    network.time_mlp_out = nn.Linear(width, width)
    network.value_head = ValueHead(
        input_dim=width, hidden_sizes=(32, 16), output_dim=1, activation="relu", bias_last=True
    )
    network.rl_cfg = Pi0RLConfig(
        add_value_head=True,
        value_after_vlm=True,
        train_expert_only=True,
        noise_method="flow_sde",
        noise_level=0.4,
        ignore_last=True,
    )
    network.global_step = 0
    Pi0._init_rlinf_runtime(
        network,
        num_steps=4,
        action_env_dim=23,
        action_chunk=16,
        config_name="pi05_behavior",
        state_indices=None,
        rlt_cfg=None,
    )
    network.install_joint_io(CheckpointIO(norm_path, "policy23"), tokenizer)
    network.freeze_vlm()
    network.eval()
    return network


def observation(network):
    rng = np.random.default_rng(4)
    images = {name: rng.integers(0, 255, (2, 8, 8, 3), dtype=np.uint8) for name in IMAGE_KEYS}
    states = np.zeros((2, 23), dtype=np.float32)
    labels = network.joint_tokenizer.processor.encode("move hand", add_eos=True)
    arrays = [network.joint_tokenizer.actions("open radio", s, labels) for s in states]
    return network._token_observation(images, states, arrays)


def pack_replay(obs, result):
    data = result["forward_inputs"]
    data["obs_state"] = obs.state
    data["token_ar_mask"] = obs.token_ar_mask
    for name in IMAGE_KEYS:
        data[f"obs_image__{name}"] = obs.images[name]
        data[f"obs_image_mask__{name}"] = obs.image_masks[name]
    return data


def test_causal_context_blocks_future_subtask_tokens(model):
    obs = observation(model)
    before, _, _ = model.build_prefix_cache(obs)
    changed = Observation.from_dict(obs.to_dict())
    changed.tokenized_prompt = obs.tokenized_prompt.clone()
    label_start = obs.token_ar_mask[0].nonzero()[0, 0].item()
    changed.tokenized_prompt[:, label_start] = 4
    after, _, _ = model.build_prefix_cache(changed)
    image_count = len(IMAGE_KEYS)
    torch.testing.assert_close(before[:, : image_count + label_start], after[:, : image_count + label_start])
    assert not torch.allclose(before[:, image_count + label_start :], after[:, image_count + label_start :])


def test_replay_ratio_and_frozen_language_after_actual_ppo_update(model):
    obs = observation(model)
    with torch.no_grad():
        _, rollout = model._predict_train(
            obs, noise=None, rng=torch.Generator().manual_seed(9), compute_values=True
        )
    replay = pack_replay(obs, rollout)
    recomputed = model.default_forward(replay)
    assert torch.isfinite(rollout["prev_logprobs"]).all()
    torch.testing.assert_close(recomputed["logprobs"], rollout["prev_logprobs"], atol=1e-5, rtol=1e-5)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    # Opposite advantages exercise both sides of PPO rather than imitating actions.
    old = rollout["prev_logprobs"].mean((1, 2))
    new = recomputed["logprobs"].mean((1, 2))
    advantage = torch.tensor([1.0, -1.0])
    ratio = (new - old).exp()
    policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
    value_loss = (recomputed["values"].flatten() - torch.tensor([1.0, 0.0])).square().mean()
    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if n.startswith("action_")
    )
    assert all(p.grad is None for n, p in model.named_parameters() if n in frozen)
    optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="joint context"):
        model.default_forward({k: v for k, v in replay.items() if k != "token_ar_mask"})


def test_complete_generation_and_rollout_use_identical_stored_context(model):
    # Force a valid two-token greedy subtask using real tied embeddings.
    text_id = next(
        i
        for i in range(3, model.joint_tokenizer.processor.vocab_size())
        if model.joint_tokenizer.processor.decode([i]).strip()
    )
    with torch.no_grad():
        for block in model.llm.layers:
            for p in block.parameters():
                p.zero_()
        weights = model.llm.embedder.embedding.weight
        weights.fill_(0.1)
        weights[text_id].fill_(0.5)
        weights[model.joint_tokenizer.eos_id].fill_(1.0)
    env = {
        "states": np.zeros((2, 23), dtype=np.float32),
        "main_images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 2, 8, 8, 3), dtype=np.uint8),
        "task_descriptions": ["open radio", "grasp object"],
    }
    actions, rollout = model.predict_action_batch(env, mode="train", rng=torch.Generator().manual_seed(3))
    assert actions.shape == (2, 16, 23)
    assert rollout["forward_inputs"]["subtask_lengths"].tolist() == [2, 2]
    replay = rollout["forward_inputs"]
    before_tokens = replay["tokenized_prompt"].clone()
    env["task_descriptions"] = ["a different task", "a different task"]
    result = model.default_forward(replay)
    torch.testing.assert_close(result["logprobs"], rollout["prev_logprobs"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(replay["tokenized_prompt"], before_tokens)
