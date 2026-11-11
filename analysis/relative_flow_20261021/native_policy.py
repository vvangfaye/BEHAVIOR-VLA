"""Native deployed policy for an explicitly frozen closed-loop study."""
import torch

from adapter import NativeAdapter
from vla_native_policy import TorchRawPolicy


class DeployedAdapter(NativeAdapter):
    @torch.no_grad()
    def predict_actions(self,env_obs,*,method,noise=None,rng=None,steps=10):
        if method not in ('extra_fm','relative_difference'):
            raise ValueError('Unknown fixed study method')
        images,normalized,prompts=self.base.checkpoint_io.prepare(env_obs)
        prefixes=[self.base.joint_tokenizer.prefix(p,s) for p,s in zip(prompts,normalized)]
        prefix=self.base._token_observation(images,normalized,prefixes)
        generated,lengths=self.base.generate_subtask(prefix)
        labels=[generated[i,:int(lengths[i])].cpu().numpy() for i in range(len(prompts))]
        tokens=[self.base.joint_tokenizer.actions(p,s,label) for p,s,label in zip(prompts,normalized,labels)]
        observation=self.base._token_observation(images,normalized,tokens)
        if noise is None:
            noise=torch.randn(len(prompts),32,32,device=observation.state.device,generator=rng)
        normalized_actions=self.sample(observation,noise,steps)
        actions=self.base.decode_actions(normalized_actions,observation.state)
        if not torch.isfinite(actions).all():raise FloatingPointError('Nonfinite native actions')
        return actions,dict(subtask_tokens=generated,subtask_lengths=lengths)


class RawPolicy(TorchRawPolicy):
    def __init__(self,adapter,method,*,seed=0):
        if method not in ('extra_fm','relative_difference'):
            raise ValueError('Unknown fixed study method')
        self.adapter,self.method=adapter,method
        self.device=next(adapter.parameters()).device
        self.metadata=dict(subtask_interval=1,model_horizon=32,action_horizon=16,
                           conditioning_mode='task_and_subtask',subtask_temperature=0.,
                           subtask_generation=True,subtask_action_conditioning='task_and_subtask',
                           supplied_subtasks=False)
        self.set_seed(seed)
