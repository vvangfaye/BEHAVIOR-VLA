import asyncio
import json
from types import SimpleNamespace

import pytest

from full_episode_repeat_seeding import (ACK, FIELD, PROTOCOL, SeededClient, SeededPolicy,
    install_server_protocol, seed_spec, validate_result_seed)


def test_policy_reseeds_actual_rng_and_resets_cache_once():
    calls = []
    raw = SimpleNamespace(_is_pytorch_model=False, _rng='original')
    wrapped = SimpleNamespace(policy=raw, reset=lambda: calls.append('reset'),
        act=lambda obs: calls.append((dict(obs), raw._rng)) or 'action')
    policy = SeededPolicy(wrapped, key_factory=lambda seed: ('jax-key', seed))
    with pytest.raises(ValueError, match='without'):
        policy.act({'state': 1})
    assert policy.act({FIELD: 17, 'state': 1}) == 'action'
    policy.act({'state': 2})
    assert calls == ['reset', ({'state': 1}, ('jax-key', 17)), ({'state': 2}, ('jax-key', 17))]
    policy.reset()
    with pytest.raises(ValueError):
        policy.act({'state': 3})


@pytest.mark.parametrize('ack', [None, 18, True, '17'])
def test_client_refuses_missing_or_wrong_server_ack(ack):
    socket = SimpleNamespace(recv=lambda: json.dumps({ACK: ack}))
    network = SimpleNamespace(_ws=socket, get_server_metadata=lambda: {'full_episode_seed_protocol': PROTOCOL})
    client = SimpleNamespace(policy=network,
        forward=lambda obs: network._ws.recv(), reset=lambda: None)
    wrapped = SeededClient(client, {'policy_seed': 17}, unpack=json.loads)
    with pytest.raises(RuntimeError, match='requested episode seed'):
        wrapped.forward({'state': 1})
    assert network._ws is socket and not wrapped.acknowledged


def test_seed_sent_only_once_and_ack_is_bound_into_result():
    sent = []
    socket = SimpleNamespace(recv=lambda: json.dumps({ACK: 17}))
    network = SimpleNamespace(_ws=socket, get_server_metadata=lambda: {'full_episode_seed_protocol': PROTOCOL})
    def forward(obs):
        sent.append(dict(obs))
        if FIELD in obs:
            network._ws.recv()
        return 'action'
    client = SimpleNamespace(policy=network, forward=forward, reset=lambda: None)
    spec = dict(protocol=PROTOCOL, policy_seed=17, sampling_round=1, parent_plan_fingerprint='abc')
    wrapped = SeededClient(client, spec, unpack=json.loads)
    wrapped.forward({'state': 1}); wrapped.forward({'state': 2})
    assert sent == [{'state': 1, FIELD: 17}, {'state': 2}]
    validate_result_seed({'repeat_sampling': wrapped.proof()}, {'repeat_sampling': spec})
    with pytest.raises(ValueError, match='seed proof'):
        validate_result_seed({'repeat_sampling': dict(wrapped.proof(), policy_seed=18)}, {'repeat_sampling': spec})


def test_server_sends_ack_of_the_applied_seed_not_just_requested_seed():
    messages = []
    class Server:
        def __init__(self, policy):
            self._policy, self._metadata = policy, {}
        async def _handler(self, socket):
            await socket.send(json.dumps({'action': [1]}))
    module = SimpleNamespace(WebsocketPolicyServer=Server, unpackb=json.loads,
        Packer=lambda: SimpleNamespace(pack=json.dumps))
    wrappers = SimpleNamespace(B1KPolicyWrapper=lambda: None)
    install_server_protocol(module, wrappers)
    policy = SimpleNamespace(applied_seed=23)
    server = module.WebsocketPolicyServer(policy)
    async def send(message): messages.append(json.loads(message))
    asyncio.run(server._handler(SimpleNamespace(send=send)))
    assert messages == [{'action': [1], ACK: 23}]
    assert server._metadata['full_episode_seed_protocol'] == PROTOCOL


def test_seed_scope_rejects_unapproved_round():
    with pytest.raises(ValueError):
        seed_spec(dict(sampling_round=3, policy_seeds={'task:301': 1}, parent_plan_fingerprint='x'), 'task', 301)
