"""Native Pi0 action-expert adapter for a fully stochastic path policy.

Every denoising transition must have positive variance. Its entire latent path
is the augmented action, and the environment executes only its decoded prefix.
This intentionally differs from the usual one-random-transition RLinf surrogate.
"""
import torch
from torch import nn
from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
from rlinf.models.embodiment.openpi.sampling import rl_sampler


class NativeObjective(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, mode, observation, *, actions=None, noise=None, time=None, chain=None):
        net = self.network
        # This adapter is deliberately restricted to a frozen VLM.
        if not net.rl_cfg.train_expert_only:
            raise ValueError("NativeObjective currently requires frozen VLM")
        observation = preprocess_observation(observation, train=False)
        with torch.no_grad():
            _, prefix_mask, cache = net.build_prefix_cache(observation)
        if mode == "fm":
            if actions is None or noise is None or time is None:
                raise ValueError("Explicit demonstration, FM noise and time required")
            x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
            v = net.velocity_from_suffix(net.run_suffix(observation, x, time, cache, prefix_mask))
            return (v - (noise - actions)).square().mean()
        if mode not in ("path_logp", "path_logp_elements") or chain is None:
            raise ValueError("Expected fm or path_logp with explicit inputs")
        if not net.rl_cfg.joint_logprob or net.rl_cfg.noise_method != "flow_sde":
            raise ValueError("Full stochastic Flow-SDE path required")
        if chain.requires_grad:
            raise ValueError("REINFORCE replay must hold the sampled path fixed")
        if chain.shape[1] != net.num_steps + 1:
            raise ValueError("Incomplete denoising path")
        times = rl_sampler.get_timesteps(net.num_steps, chain.device)
        # Parameter-independent initial density is retained for numerical parity.
        result = rl_sampler.gaussian_logprob(
            chain[:, 0], torch.zeros_like(chain[:, 0]), torch.ones_like(chain[:, 0])
        )
        for index in range(net.num_steps):
            x = chain[:, index]
            t = times[index].expand(x.shape[0])
            v = net.velocity_from_suffix(net.run_suffix(observation, x, t, cache, prefix_mask))
            mean, std = rl_sampler.sample_mean_var(
                x, v, torch.full((x.shape[0],), index, device=x.device, dtype=torch.long),
                noise_method="flow_sde", noise_level=net.rl_cfg.noise_level,
                num_steps=net.num_steps,
            )
            if not torch.all(std > 0):
                raise ValueError("Zero-variance transition invalidates path-density estimator")
            result = result + rl_sampler.gaussian_logprob(chain[:, index + 1], mean, std)
        return result if mode == "path_logp_elements" else result.flatten(1).sum(-1)
