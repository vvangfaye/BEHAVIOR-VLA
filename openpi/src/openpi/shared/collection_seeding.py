"""Explicitly enabled per-trajectory JAX sampling seeds for data collection."""
from __future__ import annotations
FIELD = '_collection_policy_seed'


class SeededCollectionPolicy:
    def __init__(self, wrapper, key_factory=None):
        self.wrapper = wrapper
        self.key_factory = key_factory
        if getattr(wrapper.policy, '_is_pytorch_model', True):
            raise ValueError('Collection seeding currently requires a JAX policy')

    def __getattr__(self, key):
        return getattr(self.wrapper, key)

    def reset(self):
        self.wrapper.reset()

    def act(self, obs):
        obs = dict(obs)
        if FIELD in obs:
            seed = obs.pop(FIELD)
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError('Collection seed must be a uint32 integer')
            self.wrapper.reset()
            if self.key_factory is None:
                import jax
                key_factory = jax.random.key
            else:
                key_factory = self.key_factory
            self.wrapper.policy._rng = key_factory(seed)
        return self.wrapper.act(obs)
