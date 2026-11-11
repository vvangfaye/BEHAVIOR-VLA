"""Experimental marginal flow with a noisy-action-dependent support posterior.

The prior remains full-dimensional Gaussian. Activity labels supervise a
posterior during training; no activity labels are accepted during sampling.
The population identity used here is the ordinary law of total expectation,
not a new theorem. This file is separate from the frozen first pilot.
"""
import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.openpi.modules import model as model_utils
from rlinf.models.embodiment.openpi.pi0 import make_attn_mask


def mix_velocity(raw, noisy, time, probability, physical_dim=23):
    """Native reverse-time velocity (time=1 noise, time=0 data)."""
    mean = raw.mean(1, keepdim=True)
    residual = raw - mean
    normal = (noisy - noisy.mean(1, keepdim=True)) / time[:, None, None]
    physical = (mean[..., :physical_dim]
                + probability[:, None] * residual[..., :physical_dim]
                + (1-probability[:, None]) * normal[..., :physical_dim])
    return torch.cat((physical, raw[..., physical_dim:]), -1)


class PosteriorSupportAdapter(nn.Module):
    def __init__(self, base, physical_dim=23, aux_weight=.1):
        super().__init__()
        self.base = base
        self.physical_dim = physical_dim
        self.aux_weight = aux_weight
        width = base.llm.embedder.embedding.embedding_dim
        flat = base.action_horizon * base.action_dim
        self.context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.SiLU())
        self.noisy_context = nn.Sequential(nn.Linear(flat, 256), nn.SiLU())
        self.posterior = nn.Sequential(nn.Linear(128+256+4, 128), nn.SiLU(),
                                       nn.Linear(128, physical_dim))
        # Near-native initialization; not a hard or oracle support choice.
        nn.init.zeros_(self.posterior[-1].weight)
        nn.init.constant_(self.posterior[-1].bias, 5.)

    def prepare_context(self, observation):
        observation = model_utils.preprocess_observation(observation, train=False)
        observation = model_utils._observation_to_dtype(observation, self.base.embed_dtype)
        with torch.no_grad():
            hidden, valid, cache = self.base.build_prefix_cache(observation)
            pooled = ((hidden.float() * valid[..., None]).sum(1)
                      / valid.sum(1, keepdim=True).clamp_min(1))
        return observation, self.context(pooled), valid, cache

    def velocity_and_probability(self, observation, noisy, time, context, valid, cache):
        tokens, suffix_valid, ar, cond = self.base.embed_suffix(observation, noisy, time)
        suffix_attention = make_attn_mask(suffix_valid, ar)
        prefix_attention = valid[:, None, :].expand(-1, tokens.shape[1], -1)
        attention = torch.cat((prefix_attention, suffix_attention), -1)
        positions = valid.sum(-1)[:, None] + suffix_valid.int().cumsum(-1) - 1
        output, _ = self.base.llm([None, tokens], positions=positions, mask=attention,
                                  kv_cache=cache, adarms_cond=[None, cond])
        raw = self.base.velocity_from_suffix(output[1][:, -self.base.action_horizon:])
        temporal = torch.stack((time, time.square(),
                                torch.sin(torch.pi*time), torch.cos(torch.pi*time)), -1)
        encoded = torch.cat((context, self.noisy_context(noisy.flatten(1).float()), temporal), -1)
        logits = self.posterior(encoded)
        return raw.float(), logits.float()

    def loss(self, observation, actions, *, noise, time, mode='posterior', raw_active=None):
        if mode not in ('ambient_aux', 'posterior'):
            raise ValueError('Unknown posterior-support pilot method')
        if raw_active is None or raw_active.shape != (len(actions), self.physical_dim):
            raise ValueError('Training requires pre-normalization activity labels')
        if not torch.all((raw_active == 0) | (raw_active == 1)) or torch.any(time <= 0):
            raise ValueError('Invalid activity labels or flow time')
        span = actions[..., :self.physical_dim].amax(1)-actions[..., :self.physical_dim].amin(1)
        if torch.any((raw_active == 0) & (span != 0)):
            raise ValueError('Inactive label has a varying normalized target')
        observation, context, valid, cache = self.prepare_context(observation)
        noisy = time[:, None, None]*noise + (1-time[:, None, None])*actions
        raw, logits = self.velocity_and_probability(observation, noisy, time, context, valid, cache)
        target = (noise-actions).float()
        if mode == 'ambient_aux':
            flow_loss = (raw-target).square().mean()
            predicted = raw
        else:
            # Orthogonal time-mean/residual decomposition. The active residual
            # expert estimates E[v_residual | noisy, context, M_j=1]. Inactive
            # residual velocities are known exactly as noisy_residual / time.
            error = raw-target
            mean_error = error.mean(1, keepdim=True)
            residual_error = error-mean_error
            weight = torch.ones((len(actions), self.base.action_dim), device=actions.device)
            weight[:, :self.physical_dim] = raw_active
            flow_loss = mean_error.square().mean() + (residual_error.square()*weight[:, None]).mean()
            predicted = mix_velocity(raw, noisy.float(), time.float(), logits.sigmoid(), self.physical_dim)
        mask_loss = F.binary_cross_entropy_with_logits(logits, raw_active.float())
        return dict(loss=flow_loss+self.aux_weight*mask_loss,
                    flow_loss=flow_loss, mask_loss=mask_loss,
                    predicted_velocity=predicted, mask_logits=logits, active=raw_active)

    @torch.no_grad()
    def sample(self, observation, *, noise, steps=10, rng=None, mode='posterior'):
        if mode not in ('ambient_aux', 'posterior') or steps < 1:
            raise ValueError('Invalid sampling configuration')
        observation, context, valid, cache = self.prepare_context(observation)
        actions = noise.float()
        for k in range(steps):
            time = torch.full((len(actions),), 1-k/steps, device=actions.device)
            raw, logits = self.velocity_and_probability(observation, actions, time, context, valid, cache)
            probability = logits.sigmoid()
            velocity = (raw if mode == 'ambient_aux' else
                        mix_velocity(raw, actions, time, probability, self.physical_dim))
            actions = actions-velocity/steps
        # This probability is a diagnostic at the last visited noisy state,
        # not an oracle mask and not the observation-only support prior.
        return actions, probability >= .5, probability
