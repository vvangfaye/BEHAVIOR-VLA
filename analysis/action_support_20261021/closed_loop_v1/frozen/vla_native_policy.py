"""Expose the experimental VLA through the unchanged native B1K wrapper."""
from contextlib import nullcontext

import numpy as np
import torch

SEED_FIELD = '_collection_policy_seed'
SEED_ACK = 'full_episode_policy_seed_ack'
SEED_PROTOCOL = 'native_full_episode_seed_ack_v1'


class TorchRawPolicy:
    """Consume the native wrapper's processed observations, generate actions."""
    def __init__(self, adapter, method, *, seed=0):
        if method not in ('standard', 'conditioned', 'projected'):
            raise ValueError('Unknown VLA method')
        self.adapter, self.method = adapter, method
        self.device = next(adapter.parameters()).device
        self.metadata = {'subtask_interval': 1, 'model_horizon': 32, 'action_horizon': 16,
                         'conditioning_mode': 'task_and_subtask', 'subtask_temperature': 0.,
                         'subtask_generation': True,
                         'subtask_action_conditioning': 'task_and_subtask'}
        self.set_seed(seed)

    def set_seed(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Policy seed must be uint32')
        self.seed = seed
        self.reset()

    def reset(self):
        # Keep Gaussian action noise independent of how many Bernoulli draws a
        # method uses. This defines a reproducible paired operational protocol.
        self.noise_rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.mask_rng = torch.Generator(device=self.device).manual_seed(self.seed ^ 0x5A17)

    @torch.inference_mode()
    def infer(self, observation):
        if 'subtask' in observation:
            raise ValueError('This evaluation requires generated, not supplied, subtasks')
        env = {'states': np.asarray(observation['observation/state'])[None],
               'main_images': np.asarray(observation['observation/image_0'])[None],
               'wrist_images': np.stack([observation['observation/image_1'],
                                          observation['observation/image_2']])[None],
               'task_descriptions': [observation['prompt']]}
        noise = torch.randn(1, 32, 32, device=self.device, generator=self.noise_rng)
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
        with precision:
            actions, info = self.adapter.predict_actions(env, method=self.method, noise=noise,
                                                         rng=self.mask_rng, steps=10)
        n = int(info['subtask_lengths'][0])
        text = self.adapter.base.joint_tokenizer.processor.decode(info['subtask_tokens'][0, :n].tolist())
        return {'actions': actions[0].float().cpu().numpy(), 'subtask': text, 'subtask_source': 'generated'}


class SeededNativeWrapper:
    """Acknowledge the native client's explicit seed before its first action."""
    def __init__(self, wrapped, raw_policy):
        self.wrapped, self.raw_policy = wrapped, raw_policy
        self.applied_seed = None

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def reset(self):
        self.applied_seed = None
        self.wrapped.reset()

    def act(self, observation):
        observation = dict(observation)
        if SEED_FIELD in observation:
            seed = observation.pop(SEED_FIELD)
            self.reset()
            self.raw_policy.set_seed(seed)
            self.applied_seed = seed
        if self.applied_seed is None:
            raise ValueError('First action requires an explicit acknowledged episode seed')
        return self.wrapped.act(observation)


def acknowledged_server_class(module):
    original = module.WebsocketPolicyServer
    class Socket:
        def __init__(self, wrapped, policy):
            self.wrapped, self.policy = wrapped, policy
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
        async def send(self, raw):
            message = module.unpackb(raw, strict_map_key=False)
            if 'action' in message:
                seed = self.policy.applied_seed
                if seed is None:
                    raise ValueError('Cannot acknowledge an unapplied seed')
                message[SEED_ACK] = seed
                raw = module.Packer().pack(message)
            return await self.wrapped.send(raw)
    class Server(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._metadata['full_episode_seed_protocol'] = SEED_PROTOCOL
            self._metadata['policy_rng_backend'] = 'torch-separated-action-and-structure'
        async def _handler(self, socket):
            return await super()._handler(Socket(socket, self._policy))
    return Server
