"""Bridge existing Pi0 velocity evaluation to the generic posterior sampler."""
import torch


class OpenPiGaussianTransition:
    def __init__(self, adapter, prepared, sample_mean_var, get_timesteps, *,
                 steps=10, noise_level=0.1):
        if steps < 2 or noise_level <= 0:
            raise ValueError('Need at least two nondegenerate stochastic transitions')
        self.adapter = adapter
        self.prepared = prepared
        self.sample_mean_var = sample_mean_var
        self.steps = steps
        self.noise_level = noise_level
        self.horizon = adapter.base.action_horizon
        self.dim = adapter.base.action_dim
        self.timesteps = get_timesteps(steps, prepared[0].state.device)

    def __call__(self, flat, step):
        if flat.shape != (self.prepared[0].state.shape[0], self.horizon * self.dim):
            raise ValueError('Flat actions do not match the cached observation batch')
        x = flat.reshape(-1, self.horizon, self.dim)
        indices = torch.full((len(x),), step, dtype=torch.long, device=x.device)
        times = self.timesteps[indices].to(dtype=torch.float32)
        velocity = self.adapter.velocity(self.prepared, x, times)
        mean, std = self.sample_mean_var(
            x, velocity, indices, noise_method='flow_sde', noise_level=self.noise_level,
            num_steps=self.steps)
        return mean.flatten(1), std.flatten(1)
