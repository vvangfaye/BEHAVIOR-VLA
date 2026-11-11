"""Deployment-only bindings; leave the frozen training module untouched."""
import torch

from posterior_support_adapter import PosteriorSupportAdapter
from vla_native_policy import TorchRawPolicy


class DeployedPosteriorAdapter(PosteriorSupportAdapter):
    @torch.no_grad()
    def predict_actions(self, env_obs, *, method, noise=None, rng=None, steps=10):
        if method not in ('ambient_aux','posterior'):
            raise ValueError('Unknown posterior-support method')
        images, normalized, prompts = self.base.checkpoint_io.prepare(env_obs)
        prefixes = [self.base.joint_tokenizer.prefix(p,s) for p,s in zip(prompts,normalized)]
        prefix_observation = self.base._token_observation(images,normalized,prefixes)
        generated,lengths = self.base.generate_subtask(prefix_observation)
        labels = [generated[i,:int(lengths[i])].cpu().numpy() for i in range(len(prompts))]
        arrays = [self.base.joint_tokenizer.actions(p,s,label)
                  for p,s,label in zip(prompts,normalized,labels)]
        observation = self.base._token_observation(images,normalized,arrays)
        if noise is None:
            noise = torch.randn(len(prompts),self.base.action_horizon,self.base.action_dim,
                                device=observation.state.device,generator=rng)
        normalized_actions,_,probability = self.sample(observation,noise=noise,steps=steps,
                                                       rng=rng,mode=method)
        actions = self.base.decode_actions(normalized_actions,observation.state)
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Nonfinite decoded action')
        return actions,dict(last_activity_posterior=probability,
                            subtask_tokens=generated,subtask_lengths=lengths)


class PosteriorRawPolicy(TorchRawPolicy):
    def __init__(self,adapter,method,*,seed=0):
        if method not in ('ambient_aux','posterior'):
            raise ValueError('Unknown posterior-support method')
        self.adapter,self.method=adapter,method
        self.device=next(adapter.parameters()).device
        self.metadata=dict(subtask_interval=1,model_horizon=32,action_horizon=16,
                           conditioning_mode='task_and_subtask',subtask_temperature=0.,
                           subtask_generation=True,subtask_action_conditioning='task_and_subtask',
                           posterior_support_method=method,provided_activity_labels=False)
        self.set_seed(seed)
