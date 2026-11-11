import random

import pytest
import torch
from torch import nn

from actor import model_actor_loss, sample_chunk_with_grad
from test_model import observation, pack_replay


def recorded_inputs(model):
    obs = observation(model)
    with torch.no_grad():
        _, collected = model._predict_train(
            obs, noise=None, rng=torch.Generator().manual_seed(8), compute_values=True,
        )
    return obs, pack_replay(obs, collected)


def test_flow_sde_matches_existing_behavior_chain(model):
    obs, replay = recorded_inputs(model)
    noise = torch.randn((2, 32, 32), generator=torch.Generator().manual_seed(10))
    random.seed(19)
    with torch.no_grad():
        actions, native = model._predict_train(
            obs, noise=noise, rng=torch.Generator().manual_seed(20), compute_values=False,
        )
    random.seed(19)
    sample = sample_chunk_with_grad(
        model, replay, noise=noise, generator=torch.Generator().manual_seed(20),
    )
    assert sample.sampling_law == "flow_sde"
    assert sample.denoise_index == int(native["forward_inputs"]["denoise_inds"][0, 0])
    torch.testing.assert_close(sample.chains, native["forward_inputs"]["chains"], atol=0, rtol=0)
    torch.testing.assert_close(sample.policy_actions, actions, atol=0, rtol=0)
    assert sample.execution_actions.requires_grad
    assert not sample.prefix_features.requires_grad
    repeated = sample_chunk_with_grad(
        model, replay, noise=sample.initial_noise,
        step_noises=sample.step_noises, denoise_index=sample.denoise_index,
    )
    torch.testing.assert_close(repeated.model_actions, sample.model_actions, atol=0, rtol=0)


def test_explicit_ode_matches_evaluation_law(model):
    obs, replay = recorded_inputs(model)
    noise = torch.randn((2, 32, 32), generator=torch.Generator().manual_seed(10))
    with torch.no_grad():
        expected, _ = model._predict_eval(obs, noise=noise)
    sample = sample_chunk_with_grad(model, replay, sampling_law="flow_ode", noise=noise)
    assert sample.denoise_index is None
    torch.testing.assert_close(sample.policy_actions, expected, atol=0, rtol=0)
    assert sample.policy_actions.requires_grad


class SmallWorldModel(nn.Module):
    """Nonlinear action-conditioned state/reward model for the autograd contract."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(23 + 16 * 23, 32), nn.Tanh(), nn.Linear(32, 24))

    def forward(self, state, actions):
        predicted = self.net(torch.cat([state, actions.flatten(1)], dim=1))
        return predicted[:, :23], predicted[:, 23]


@pytest.mark.parametrize("law", ["flow_sde", "flow_ode"])
def test_model_return_updates_actual_gemma_expert_only(model, law):
    _, replay = recorded_inputs(model)
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    expert_before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("action_")}
    world = SmallWorldModel()
    critic = nn.Sequential(nn.Linear(23, 16), nn.Tanh(), nn.Linear(16, 1))
    fixed_state = torch.zeros((2, 23))
    sample = sample_chunk_with_grad(
        model, replay, sampling_law=law, generator=torch.Generator().manual_seed(11),
    )

    def score(actions):
        predicted_state, reward = world(fixed_state, actions)
        return reward + critic(predicted_state).flatten()

    loss, metrics = model_actor_loss(
        sample, score, frozen_modules=[world, critic],
        reference_actions=sample.execution_actions.detach() + 0.01, trust_coef=0.01,
    )
    assert loss.isfinite() and metrics["action_trust_mse"].isfinite()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if n.startswith("action_"))
    assert all(p.grad is None for n, p in model.named_parameters() if n in frozen_before)
    assert all(p.grad is None for p in world.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert all(p.requires_grad for p in world.parameters())  # original flags restored
    optimizer.step()
    assert any(not torch.equal(p, expert_before[n]) for n, p in model.named_parameters() if n in expert_before)
    for name, parameter in model.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(parameter, frozen_before[name], atol=0, rtol=0)


def test_reject_disconnected_model_score(model):
    _, replay = recorded_inputs(model)
    sample = sample_chunk_with_grad(model, replay)
    with pytest.raises(ValueError, match="retain its action gradient"):
        model_actor_loss(sample, lambda actions: torch.zeros(actions.shape[0]))


def test_actual_gemma_through_checkpoint_bridge(model):
    """Join both real module implementations; physics remains a synthetic model."""
    from bridge.score import FrozenWorldModelScore, world_model_class

    _, replay = recorded_inputs(model)
    config = {"goals": 3, "visual_dim": 4, "use_action": True, "use_visual": True, "hidden": 16}
    torch.manual_seed(23)
    dynamics = world_model_class()(**config)
    transform = {
        "state_mean": torch.zeros(23), "state_scale": torch.ones(23),
        "action_mean": torch.zeros(368), "action_scale": torch.ones(368),
        "reward_mean": torch.zeros(1), "reward_scale": torch.ones(1),
        "visual_mean": torch.zeros(576), "visual_scale": torch.ones(4),
        "components": torch.randn(576, 4) * .01,
    }
    score = FrozenWorldModelScore({"schema": "b1k_small_world_model_probe_v1", "name": "full",
        "config": config, "transform": transform, "model_states": [dynamics.state_dict()]})
    sample = sample_chunk_with_grad(model, replay, generator=torch.Generator().manual_seed(24))
    obs = replay["obs_state"][:, :23].detach()
    goals, visual = torch.zeros((len(obs), 3)), torch.zeros((len(obs), 576))
    loss, _ = model_actor_loss(sample, lambda actions: score(obs, goals, visual, actions),
                               frozen_modules=[score])
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if n.startswith("action_"))
    assert all(p.grad is None for p in score.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
