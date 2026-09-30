"""One resettable policy state per websocket connection, bound locally by default."""
import asyncio
import http
import logging
import time
import websockets
from websockets.asyncio.server import serve
from .protocol import packb, unpackb


def health_check(connection, request):
    if request.path == "/healthz": return connection.respond(http.HTTPStatus.OK, "OK\n")


def handler_factory(policy_factory, metadata=None):
    async def handler(connection):
        policy = policy_factory()
        await connection.send(packb(metadata or {"protocol": "b1k-v3.9.2", "implementation": "goalkeeper-v1"}))
        try:
            async for message in connection:
                if not isinstance(message, bytes): raise ValueError("Expected binary msgpack message")
                payload = unpackb(message)
                if not isinstance(payload, dict): raise ValueError("Expected observation dictionary")
                if "reset" in payload:
                    if payload != {"reset": True}: raise ValueError("Invalid reset message")
                    policy.reset()
                    continue  # Official protocol expects no reset acknowledgement.
                started = time.perf_counter()
                action = policy.act(payload)
                response = {"action": action,
                            "server_timing": {"infer_ms": 1000 * (time.perf_counter() - started)}}
                if subtasks := getattr(policy, "last_subtasks", None):
                    indices = sorted(subtasks)
                    response["subtasks"] = [subtasks[i] for i in indices]
                    sources = getattr(policy, "last_subtask_sources", None)
                    if sources and all(i in sources for i in indices):
                        response["subtask_sources"] = [sources[i] for i in indices]
                await connection.send(packb(response))
        except websockets.ConnectionClosed:
            return
        except Exception:
            logging.exception("Inference failed")
            await connection.close(code=1011, reason="Inference error; inspect server log")
    return handler


async def run_server(policy_factory, host="127.0.0.1", port=8000):
    async with serve(handler_factory(policy_factory), host, port, compression=None,
                     max_size=32 * 1024 * 1024, process_request=health_check) as server:
        await server.serve_forever()


def serve_forever(policy_factory, host="127.0.0.1", port=8000):
    asyncio.run(run_server(policy_factory, host, port))
