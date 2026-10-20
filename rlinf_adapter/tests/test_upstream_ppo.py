"""Exercise RLinf's actual GAE, chunk reduction and PPO loss on the real tiny model."""

import torch
from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
from rlinf.algorithms.utils import preprocess_loss_inputs
from test_model import model as model
from test_model import observation, pack_replay


def test_upstream_ppo_backpropagates_into_expert_and_value_only(model):
    obs = observation(model)
    with torch.no_grad():
        _, rollout = model._predict_train(
            obs, noise=None, rng=torch.Generator().manual_seed(21), compute_values=True
        )
    old_values = rollout["prev_values"].reshape(2)
    advantages, returns = compute_gae_advantages_and_returns(
        rewards=torch.tensor([[1.0, -1.0]]),
        values=torch.stack([old_values, torch.zeros_like(old_values)]),
        dones=torch.tensor([[False, False], [True, True]]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    output = model.default_forward(pack_replay(obs, rollout))
    args = preprocess_loss_inputs(
        logprobs=output["logprobs"],
        old_logprobs=rollout["prev_logprobs"],
        values=output["values"],
        prev_values=rollout["prev_values"],
        advantages=advantages[0],
        returns=returns[0],
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=23,
    )
    loss, _ = compute_ppo_actor_critic_loss(
        **args, clip_ratio_low=0.2, clip_ratio_high=0.2, value_clip=0.2, huber_delta=10.0
    )
    loss.backward()
    assert torch.isfinite(loss)
    for prefix in ["action_", "value_head."]:
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for name, p in model.named_parameters()
            if name.startswith(prefix)
        )
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
