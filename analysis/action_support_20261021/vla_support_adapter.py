"""Experimental support-conditioned SFT on an existing Pi0 action expert.

This module is isolated from the baseline evaluation and RL implementations.
The returned actions remain in the original model's normalized coordinates.
"""
import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.openpi.modules import model as model_utils
from rlinf.models.embodiment.openpi.pi0 import make_attn_mask

from support_flow import project


class SupportActionAdapter(nn.Module):
    def __init__(self, base, physical_dim=23, aux_weight=.1):
        super().__init__()
        if not base.pi05:
            raise ValueError('This experimental adapter requires Pi0.5')
        self.base = base
        self.physical_dim = physical_dim
        self.aux_weight = aux_weight
        prefix_width = base.llm.embedder.embedding.embedding_dim
        expert_width = base.action_out_proj.in_features
        self.mask_head = nn.Sequential(nn.LayerNorm(prefix_width), nn.Linear(prefix_width, physical_dim))
        self.mask_condition = nn.Linear(physical_dim, expert_width, bias=False)
        nn.init.zeros_(self.mask_condition.weight)

    def project_physical(self, actions, active):
        first = project(actions[..., :self.physical_dim], active)
        # Padding coordinates follow the baseline's original noise/objective.
        return torch.cat([first, actions[..., self.physical_dim:]], -1)

    def prepare_context(self, observation):
        observation = model_utils.preprocess_observation(observation, train=False)
        observation = model_utils._observation_to_dtype(observation, self.base.embed_dtype)
        # The base VLM is frozen; reuse one prefix pass for structure and flow.
        with torch.no_grad():
            hidden, valid, cache = self.base.build_prefix_cache(observation)
            pooled = (hidden.float() * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return observation, self.mask_head(pooled), valid, cache

    def velocity(self, observation, x_t, t, active, valid, cache, *, condition=True):
        tokens, suffix_valid, ar, cond = self.base.embed_suffix(observation, x_t, t)
        if condition:
            cond = cond + self.mask_condition(active.float()).to(cond.dtype)
        suffix_attention = make_attn_mask(suffix_valid, ar)
        prefix_attention = valid[:, None, :].expand(-1, tokens.shape[1], -1)
        attention = torch.cat([prefix_attention, suffix_attention], -1)
        positions = valid.sum(-1)[:, None] + suffix_valid.int().cumsum(-1) - 1
        output, _ = self.base.llm([None, tokens], positions=positions, mask=attention,
                                   kv_cache=cache, adarms_cond=[None, cond])
        return self.base.velocity_from_suffix(output[1][:, -self.base.action_horizon:])

    def loss(self, observation, actions, *, noise, time, mode='projected', raw_active=None):
        if mode not in ('projected', 'conditioned', 'standard'):
            raise ValueError('Expected projected, conditioned or standard training')
        if actions.shape != noise.shape or actions.shape[1:] != (self.base.action_horizon, self.base.action_dim):
            raise ValueError('Expected a complete, unpadded-time model action chunk')
        if not torch.isfinite(actions).all():
            raise ValueError('Nonfinite action target')
        observation, logits, valid, cache = self.prepare_context(observation)
        span = actions[..., :self.physical_dim].amax(1) - actions[..., :self.physical_dim].amin(1)
        if raw_active is None:
            active = (span != 0).float()
        else:
            # Derive labels before normalization in the data pipeline, so
            # floating point rounding cannot silently relabel a small motion.
            active = raw_active.to(device=actions.device, dtype=torch.float32)
            if active.shape != span.shape or not torch.all((active == 0) | (active == 1)):
                raise ValueError('Expected binary raw-action activity labels [B,D]')
            if torch.any((active == 0) & (span != 0)):
                raise ValueError('An inactive raw-action label has a varying normalized target')
        x0 = self.project_physical(noise, active) if mode == 'projected' else noise
        # Pi0 uses t=1 for noise and integrates towards t=0 for data.
        x_t = time[:, None, None] * x0 + (1 - time[:, None, None]) * actions
        predicted = self.velocity(observation, x_t, time, active, valid, cache,
                                  condition=mode != 'standard')
        if mode == 'projected':
            predicted = self.project_physical(predicted, active)
        flow_loss = (predicted - (x0 - actions)).square().mean()
        mask_loss = F.binary_cross_entropy_with_logits(logits, active)
        # Pure SFT has no extra head gradients altering its global clipping.
        total = flow_loss if mode == 'standard' else flow_loss + self.aux_weight * mask_loss
        return {'loss': total,
                'flow_loss': flow_loss, 'mask_loss': mask_loss, 'active': active,
                'predicted_velocity': predicted, 'mask_logits': logits}

    @torch.no_grad()
    def sample(self, observation, *, noise, steps=10, rng=None, mode='projected'):
        if steps < 1:
            raise ValueError('At least one Euler step is required')
        if mode not in ('projected', 'conditioned', 'standard'):
            raise ValueError('Unknown action generation method')
        observation, logits, valid, cache = self.prepare_context(observation)
        active = torch.bernoulli(logits.sigmoid(), generator=rng)
        actions = self.project_physical(noise, active) if mode == 'projected' else noise
        for k in range(steps):
            time = torch.full((len(actions),), 1 - k / steps, device=actions.device)
            velocity = self.velocity(observation, actions, time, active, valid, cache,
                                     condition=mode != 'standard')
            if mode == 'projected':
                velocity = self.project_physical(velocity, active)
            actions = actions - velocity / steps
        return actions, active, logits.sigmoid()

    @torch.no_grad()
    def predict_actions(self, env_obs, *, method='projected', noise=None, rng=None, steps=10):
        """Full deployment path: original greedy subtask, then generated mask.

        Only current environment observations and the model RNG enter this
        interface. It accepts neither expert subtask text nor activity labels.
        """
        images, normalized, prompts = self.base.checkpoint_io.prepare(env_obs)
        prefixes = [self.base.joint_tokenizer.prefix(p, s) for p, s in zip(prompts, normalized)]
        prefix_observation = self.base._token_observation(images, normalized, prefixes)
        generated, lengths = self.base.generate_subtask(prefix_observation)
        labels = [generated[i, :int(lengths[i])].cpu().numpy() for i in range(len(prompts))]
        arrays = [self.base.joint_tokenizer.actions(p, s, label)
                  for p, s, label in zip(prompts, normalized, labels)]
        observation = self.base._token_observation(images, normalized, arrays)
        if noise is None:
            noise = torch.randn(len(prompts), self.base.action_horizon, self.base.action_dim,
                                device=observation.state.device, generator=rng)
        normalized_actions, active, probabilities = self.sample(observation, noise=noise,
                                                                steps=steps, rng=rng, mode=method)
        actions = self.base.decode_actions(normalized_actions, observation.state)
        return actions, {'sampled_active': active, 'active_probabilities': probabilities,
                         'subtask_tokens': generated, 'subtask_lengths': lengths}
