"""Wire-level test of the actual native WebSocket handler and seed adapter."""
import asyncio
import json
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from verify_native_wrapper import CaptureAdapter, B1KPolicyWrapper
from vla_native_policy import (TorchRawPolicy, SeededNativeWrapper, SEED_FIELD,
                               SEED_ACK, SEED_PROTOCOL, acknowledged_server_class)
from openpi.serving import websocket_b1k_server as native


async def check():
    adapter = CaptureAdapter(31)
    raw = TorchRawPolicy(adapter, 'standard')
    wrapped = B1KPolicyWrapper(raw, 'b1k/R1Pro', 'task', 'receding_horizon', 16, 32)
    policy = SeededNativeWrapper(wrapped, raw)
    server = acknowledged_server_class(native)(policy, host='127.0.0.1', port=0,
                                                 server_token='local-vla-wire-check',
                                                 metadata=raw.metadata)
    obs = {'robot_r1::proprio': np.zeros(61, dtype=np.float32)}
    for name in ['zed_link', 'left_realsense_link', 'right_realsense_link']:
        obs[f'robot_r1::robot_r1:{name}:Camera:0::rgb'] = np.zeros((240, 240, 4), dtype=np.uint8)
    packer = native.Packer()
    async with serve(server._handler, '127.0.0.1', 0, compression=None,
                     max_size=None, ping_interval=None) as endpoint:
        port = endpoint.sockets[0].getsockname()[1]
        assert port >= 1000
        async with connect(f'ws://127.0.0.1:{port}', compression=None,
                           max_size=None, ping_interval=None) as client:
            metadata = native.unpackb(await client.recv(), strict_map_key=False)
            assert metadata['full_episode_seed_protocol'] == SEED_PROTOCOL
            assert metadata['server_instance_token'] == 'local-vla-wire-check'
            # These are the real evaluator worker's readiness conditions.
            assert metadata['subtask_generation'] is True
            assert metadata['subtask_action_conditioning'] == 'task_and_subtask'
            await client.send(packer.pack({'reset': True}))
            await client.send(packer.pack({**obs, SEED_FIELD: 55}))
            first = native.unpackb(await client.recv(), strict_map_key=False)
            assert first[SEED_ACK] == 55 and first['action'].shape == (23,)
            assert first['subtasks'] == ['generated subtask']
            assert first['subtask_sources'] == ['generated']
            await client.send(packer.pack(obs))
            second = native.unpackb(await client.recv(), strict_map_key=False)
            assert second[SEED_ACK] == 55 and len(adapter.calls) == 1
            await client.send(packer.pack({'reset': True}))
            await client.send(packer.pack({**obs, SEED_FIELD: 55}))
            repeated = native.unpackb(await client.recv(), strict_map_key=False)
            np.testing.assert_array_equal(first['action'], repeated['action'])
            assert len(adapter.calls) == 2
    return {'passed': True, 'metadata_protocol_verified': True,
            'seed_ack_on_first_and_buffered_actions': True, 'reset_repeat_exact': True,
            'scope': 'Actual native WebSocket handler plus action stub; no trained-policy or simulation claim'}


if __name__ == '__main__':
    result = asyncio.run(check())
    Path(__file__).with_name('native_websocket_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
