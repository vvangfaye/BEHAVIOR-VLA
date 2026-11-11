"""Exercise the frozen evaluation transport using a stateful, slow CPU policy.

The short keepalive timings only accelerate the reproduction. No VLA model,
GPU, simulation, original source, or published evaluation result is modified.
"""
import asyncio
import importlib.util
import json
import logging
from pathlib import Path
import socket
import sys
import threading
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime51/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/network_utils.py"
og = types.ModuleType("omnigibson")
macros = types.ModuleType("omnigibson.macros")
macros.gm = types.SimpleNamespace(DEBUG=False)
sys.modules["omnigibson"] = og
sys.modules["omnigibson.macros"] = macros
spec = importlib.util.spec_from_file_location("frozen_network", SOURCE)
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)
logging.disable(logging.CRITICAL)


def run_case(disable_client_keepalive, expect_abort=False):
    class StatefulPolicy:
        def __init__(self):
            self.calls = []

        def act(self, observation):
            self.calls.append(observation["observation_id"])
            if len(self.calls) == 1:
                time.sleep(0.45)
            return torch.tensor([float(len(self.calls))])

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    policy = StatefulPolicy()
    server = network.WebsocketPolicyServer(policy, "127.0.0.1", port)
    ready = threading.Event()
    context = {}

    async def serve():
        context["loop"] = asyncio.get_running_loop()
        context["stop"] = asyncio.Event()
        async with network._server.serve(
            server._handler, "127.0.0.1", port, compression=None,
            max_size=None, process_request=network._health_check,
            ping_interval=None,
        ):
            ready.set()
            await context["stop"].wait()

    thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    thread.start()
    assert ready.wait(5), "Test server did not start"
    original = network.websockets.sync.client.connect

    def connect(*args, **kwargs):
        kwargs.update(ping_interval=None if disable_client_keepalive else 0.04,
                      ping_timeout=0.04, close_timeout=0.1)
        return original(*args, **kwargs)

    network.websockets.sync.client.connect = connect
    client = None
    try:
        client = network.WebsocketClientPolicy("127.0.0.1", port)
        try:
            result = client.act({"observation_id": 123})
        except RuntimeError as exc:
            if not expect_abort:
                raise
            return {
                "client_keepalive_disabled": disable_client_keepalive,
                "requests_executed_by_stateful_policy": policy.calls,
                "aborted": True,
                "error": str(exc),
            }
        return {
            "client_keepalive_disabled": disable_client_keepalive,
            "requests_executed_by_stateful_policy": policy.calls,
            "received_action_index": int(result.item()),
        }
    finally:
        network.websockets.sync.client.connect = original
        if client is not None:
            client._ws.close()
        context["loop"].call_soon_threadsafe(context["stop"].set)
        thread.join(5)
        assert not thread.is_alive()


if __name__ == "__main__":
    original = run_case(False)
    mitigated = run_case(True)
    assert original["requests_executed_by_stateful_policy"] == [123, 123]
    assert original["received_action_index"] == 2
    assert mitigated["requests_executed_by_stateful_policy"] == [123]
    assert mitigated["received_action_index"] == 1
    candidate_path = Path(__file__).parent / "network_utils_candidate.py"
    spec = importlib.util.spec_from_file_location("candidate_network", candidate_path)
    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)
    candidate_normal = run_case(True)
    candidate_disconnect = run_case(False, expect_abort=True)
    assert candidate_normal["received_action_index"] == 1
    assert candidate_disconnect["aborted"]
    assert candidate_disconnect["requests_executed_by_stateful_policy"] == [123]
    result = {"passed": True, "scope": "CPU transport reproduction; not a rerun of the six episodes",
              "original": original, "keepalive_mitigation": mitigated,
              "candidate_normal": candidate_normal,
              "candidate_forced_disconnect": candidate_disconnect}
    (Path(__file__).parent / "protocol_retry_test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
