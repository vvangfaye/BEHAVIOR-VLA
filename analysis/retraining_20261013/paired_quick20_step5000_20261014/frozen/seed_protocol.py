"""Explicit, acknowledged policy seeds for additional native episode rounds."""
from copy import deepcopy

FIELD = '_collection_policy_seed'
ACK = 'full_episode_policy_seed_ack'
PROTOCOL = 'native_full_episode_seed_ack_v1'


def seed_spec(manifest, task, instance):
    round_id = manifest['sampling_round']
    seed = manifest['policy_seeds'][f'{task}:{instance}']
    if type(round_id) is not int or round_id not in (1, 2):
        raise ValueError('Only the two explicitly authorized additional rounds are supported')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Policy seed must be uint32')
    return dict(protocol=PROTOCOL, sampling_round=round_id, policy_seed=seed,
                parent_plan_fingerprint=manifest['parent_plan_fingerprint'])


class SeededPolicy:
    """Reset action/text caches and JAX RNG before the episode's first action."""
    def __init__(self, wrapped, key_factory=None):
        if getattr(wrapped.policy, '_is_pytorch_model', True):
            raise ValueError('Repeat collection requires the verified JAX policy')
        self.wrapped, self.key_factory, self.applied_seed = wrapped, key_factory, None

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def reset(self):
        self.applied_seed = None
        self.wrapped.reset()

    def act(self, obs):
        obs = dict(obs)
        if FIELD in obs:
            seed = obs.pop(FIELD)
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError('Invalid repeat policy seed')
            self.reset()
            factory = self.key_factory
            if factory is None:
                import jax
                factory = jax.random.key
            self.wrapped.policy._rng = factory(seed)
            self.applied_seed = seed
        if self.applied_seed is None:
            raise ValueError('Repeat episode started without its explicit policy seed')
        return self.wrapped.act(obs)


def install_server_protocol(server_module, wrapper_module):
    """Adapt the frozen baseline without editing shared server/runtime files."""
    original_server = server_module.WebsocketPolicyServer
    original_wrapper = wrapper_module.B1KPolicyWrapper

    class AcknowledgedSocket:
        def __init__(self, socket, policy):
            self.socket, self.policy = socket, policy

        def __getattr__(self, name):
            return getattr(self.socket, name)

        async def send(self, raw):
            message = server_module.unpackb(raw)
            if 'action' in message:
                seed = self.policy.applied_seed
                if seed is None:
                    raise ValueError('Cannot acknowledge an unapplied seed')
                message[ACK] = seed
                raw = server_module.Packer().pack(message)
            return await self.socket.send(raw)

    class Server(original_server):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._metadata['full_episode_seed_protocol'] = PROTOCOL

        async def _handler(self, socket):
            return await super()._handler(AcknowledgedSocket(socket, self._policy))

    def wrapper(*args, **kwargs):
        return SeededPolicy(original_wrapper(*args, **kwargs))

    server_module.WebsocketPolicyServer = Server
    wrapper_module.B1KPolicyWrapper = wrapper


class SeededClient:
    def __init__(self, wrapped, spec, unpack=None):
        self.wrapped, self.spec, self.unpack = wrapped, deepcopy(spec), unpack
        self.acknowledged = False

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def reset(self):
        self.acknowledged = False
        self.wrapped.reset()

    def forward(self, obs, *args, **kwargs):
        if self.acknowledged:
            return self.wrapped.forward(obs, *args, **kwargs)
        client = self.wrapped.policy
        if client.get_server_metadata().get('full_episode_seed_protocol') != PROTOCOL:
            raise RuntimeError('Server does not acknowledge repeat policy seeds')
        unpack = self.unpack
        if unpack is None:
            from omnigibson.eval.utils.network_utils import unpackb
            unpack = unpackb
        original = client._ws
        seed = self.spec['policy_seed']
        outer = self

        class CheckedSocket:
            def __getattr__(self, name):
                return getattr(original, name)

            def recv(self, *a, **k):
                raw = original.recv(*a, **k)
                message = unpack(raw)
                if type(message.get(ACK)) is not int or message[ACK] != seed:
                    raise RuntimeError('Server did not apply the requested episode seed')
                outer.acknowledged = True
                return raw

        client._ws = CheckedSocket()
        try:
            action = self.wrapped.forward({**obs, FIELD: seed}, *args, **kwargs)
            if not self.acknowledged:
                raise RuntimeError('First episode action bypassed seed acknowledgment')
            return action
        finally:
            client._ws = original

    def proof(self):
        if not self.acknowledged:
            raise ValueError('Episode lacks a server seed acknowledgment')
        return dict(self.spec, server_seed_acknowledged=True)


def begin_episode(evaluator, manifest, task, instance):
    spec = seed_spec(manifest, task, instance)
    wrapped = evaluator.policy
    if isinstance(wrapped, SeededClient):
        wrapped = wrapped.wrapped
    evaluator.policy = SeededClient(wrapped, spec)
    return spec


def validate_result_seed(result, metadata):
    expected = metadata.get('repeat_sampling')
    if expected is not None and result.get('repeat_sampling') != dict(expected, server_seed_acknowledged=True):
        raise ValueError('Completed episode seed proof differs from raw metadata')
