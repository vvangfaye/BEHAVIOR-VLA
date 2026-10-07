"""
Adapted from https://github.com/Physical-Intelligence/openpi
"""

import asyncio
import functools
import http
import logging
import msgpack
import numpy as np
import requests
import time
import torch as th
import traceback
import websockets
import websockets.asyncio.server as _server
import websockets.sync.client
from copy import deepcopy
from omnigibson.macros import gm
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["WebsocketClientPolicy", "WebsocketPolicyServer"]

ACTION_CHUNK_REQUEST_KEY = "__action_chunk_size__"


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        scheme: str = "ws",
        api_key: Optional[str] = None,
        allow_reconnect: bool = False,
        expected_server_token: Optional[str] = None,
        expected_subtask_vote_action_horizon: Optional[int] = None,
        expected_subtask_vote_interval: Optional[int] = None,
        expected_subtask_vote_window: Optional[int] = None,
        expected_subtask_vote_threshold: Optional[int] = None,
        server_wait_timeout: Optional[float] = None,
        action_chunk_size: int = 0,
    ) -> None:
        """
        Initializes the websocket client policy.

        Args:
            host (str): Hostname of the websocket server to connect to.
            port (int): Port of the websocket server to connect to. Defaults to 8000.
            scheme (str): WebSocket scheme to use. Either "wss" (secure) or "ws" (insecure). Defaults to "ws".
            api_key (str, optional): API key to include in the Authorization header when connecting to the websocket server, if required.
            allow_reconnect (bool): Whether to allow automatic reconnection if the websocket connection is lost.
                If False, the client will raise an error if the connection is lost.
                If True, the client will attempt to reconnect indefinitely with a delay between attempts.
            action_chunk_size (int): Number of server-returned actions to execute before requesting a new
                observation-conditioned action. Values <= 1 disable chunk requests. Only enable this when the
                server returns an exact open-loop action sequence produced from the current observation.
        """
        if expected_server_token is not None and not expected_server_token:
            raise ValueError("expected_server_token must be nonempty")
        expected_vote_values = (
            expected_subtask_vote_action_horizon,
            expected_subtask_vote_interval,
            expected_subtask_vote_window,
            expected_subtask_vote_threshold,
        )
        if any(value is not None for value in expected_vote_values) and not all(
            type(value) is int and value > 0 for value in expected_vote_values
        ):
            raise ValueError(
                "Expected subtask voting action horizon, interval, window, and threshold must be positive integers."
            )
        if server_wait_timeout is not None and server_wait_timeout <= 0:
            raise ValueError("server_wait_timeout must be positive")
        self._uri = f"{scheme}://{host}:{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = None, None
        self._allow_reconnect = allow_reconnect
        self._expected_server_token = expected_server_token
        self._expected_subtask_voting = (
            {
                "action_horizon": expected_subtask_vote_action_horizon,
                "subtask_interval": expected_subtask_vote_interval,
                "subtask_vote_mode": "temporal_majority",
                "subtask_vote_window": expected_subtask_vote_window,
                "subtask_vote_threshold": expected_subtask_vote_threshold,
            }
            if expected_subtask_vote_interval is not None
            else None
        )
        self._server_wait_timeout = (
            server_wait_timeout
            if server_wait_timeout is not None
            else (60.0 if expected_server_token is not None else None)
        )
        self.current_subtask = None
        self.current_subtask_source = None
        self.current_subtask_vote = None
        self._action_chunk_size = max(0, int(action_chunk_size))
        self._action_chunk = None
        self._action_chunk_index = 0
        self._chunk_requests_supported = self._action_chunk_size > 1

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _retry_wait(self, deadline: Optional[float]) -> None:
        if deadline is None:
            time.sleep(5)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for policy server at {self._uri}")
        time.sleep(min(5, remaining))

    def _validate_server_metadata(self, conn, metadata: Any) -> None:
        if self._expected_server_token is not None and (
            not isinstance(metadata, dict) or metadata.get("server_instance_token") != self._expected_server_token
        ):
            conn.close()
            raise RuntimeError("Connected policy server identity does not match this evaluator")
        if not isinstance(metadata, dict):
            conn.close()
            raise RuntimeError("Connected policy server returned invalid metadata")
        if self._expected_subtask_voting is not None:
            mismatches = {
                key: (metadata.get(key), expected)
                for key, expected in self._expected_subtask_voting.items()
                if metadata.get(key) != expected
            }
            if mismatches:
                conn.close()
                raise RuntimeError(f"Connected policy server subtask voting contract does not match: {mismatches}")

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        parsed = urlparse(self._uri)
        host = parsed.hostname
        port = parsed.port
        http_scheme = "https" if parsed.scheme == "wss" else "http"
        health_url = f"{http_scheme}://{host}:{port}/healthz"
        deadline = time.monotonic() + self._server_wait_timeout if self._server_wait_timeout is not None else None

        while True:
            try:
                response = requests.get(health_url, timeout=2)
                if response.ok:
                    logger.info("Health check passed, attempting websocket connection...")
                    break
            except Exception:
                pass
            logger.info(f"Health check failed, waiting for server at {http_scheme}://{host}:{port}...")
            self._retry_wait(deadline)

        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=2,
                    # Long scene restoration or first policy compilation is
                    # supervised by the evaluator, not a heartbeat deadline.
                    ping_interval=None,
                )
                metadata = unpackb(conn.recv(timeout=2))
                self._validate_server_metadata(conn, metadata)
                logger.info("Connected to server!")
                return conn, metadata
            except (OSError, TimeoutError, websockets.exceptions.InvalidMessage, EOFError) as e:
                logger.info(f"Websocket connection failed ({e}), retrying...")
                self._retry_wait(deadline)

    def act(self, obs: Dict) -> th.Tensor:
        if self._action_chunk is not None and self._action_chunk_index < self._action_chunk.shape[-2]:
            action = self._action_chunk[..., self._action_chunk_index, :].clone()
            self._action_chunk_index += 1
            return action

        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        request = obs
        if self._chunk_requests_supported:
            request = dict(obs)
            request[ACTION_CHUNK_REQUEST_KEY] = self._action_chunk_size
        data = self._packer.pack(request)
        max_retries = 2
        response = None

        for attempt in range(max_retries + 1):
            try:
                self._ws.send(data)
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")

                action_dict = unpackb(response)
                if "action" not in action_dict:
                    if attempt < max_retries:
                        logger.warning(
                            f"Server response missing 'action' key, retrying ({attempt + 1}/{max_retries})..."
                        )
                        continue
                    raise RuntimeError(f"Server response missing 'action' key: {action_dict}")
                subtasks = action_dict.get("subtasks")
                valid_subtask = (
                    isinstance(subtasks, list)
                    and bool(subtasks)
                    and isinstance(subtasks[0], str)
                    and bool(subtasks[0].strip())
                )
                self.current_subtask = subtasks[0].strip() if valid_subtask else None
                subtask_sources = action_dict.get("subtask_sources")
                valid_subtask_source = (
                    isinstance(subtask_sources, list)
                    and bool(subtask_sources)
                    and isinstance(subtask_sources[0], str)
                    and bool(subtask_sources[0].strip())
                )
                self.current_subtask_source = subtask_sources[0].strip() if valid_subtask_source else None
                subtask_votes = action_dict.get("subtask_votes")
                if subtask_votes is None:
                    self.current_subtask_vote = None
                elif isinstance(subtask_votes, list) and len(subtask_votes) == 1 and isinstance(subtask_votes[0], dict):
                    self.current_subtask_vote = deepcopy(subtask_votes[0])
                else:
                    raise RuntimeError(f"Server response contains invalid subtask_votes: {subtask_votes!r}")
                action = th.from_numpy(deepcopy(action_dict["action"])).to(th.float32)
                if self._chunk_requests_supported:
                    if "action_chunk" in action_dict:
                        chunk = th.from_numpy(deepcopy(action_dict["action_chunk"])).to(th.float32)
                        expected_shape = (*action.shape[:-1], self._action_chunk_size, action.shape[-1])
                        if chunk.shape != expected_shape:
                            raise RuntimeError(
                                f"Server returned action_chunk shape {tuple(chunk.shape)}, expected {expected_shape}."
                            )
                        if not th.equal(chunk[..., 0, :], action):
                            raise RuntimeError("Server action must exactly equal action_chunk[..., 0, :].")
                        self._action_chunk = chunk
                        self._action_chunk_index = 1
                    else:
                        logger.warning(
                            "Policy server does not support action chunks; falling back to one request per step."
                        )
                        self._chunk_requests_supported = False
                return action

            except websockets.exceptions.ConnectionClosedError as e:
                if self._allow_reconnect and attempt < max_retries:
                    logger.warning(f"Connection lost, reconnecting (attempt {attempt + 1}/{max_retries})...")
                    self._ws, self._server_metadata = self._wait_for_server()
                    continue
                raise RuntimeError(f"Websocket connection error: {e}")

    def reset(self) -> None:
        self.current_subtask = None
        self.current_subtask_source = None
        self.current_subtask_vote = None
        self._action_chunk = None
        self._action_chunk_index = 0
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        data = self._packer.pack({"reset": True})
        self._ws.send(data)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}...")
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue

                obs = deepcopy(result)

                infer_time = time.monotonic()
                action = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time

                action = {
                    "action": action.cpu().numpy(),
                }
                current_subtask = getattr(self._policy, "current_subtask", None)
                if current_subtask:
                    action["subtasks"] = [current_subtask]
                    current_subtask_source = getattr(self._policy, "current_subtask_source", None)
                    if current_subtask_source:
                        action["subtask_sources"] = [current_subtask_source]
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                if gm.DEBUG:
                    await websocket.send(traceback.format_exc())
                try:
                    # Try new websockets API first
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:
                    # Fallback for older websockets versions
                    await websocket.close(code=1011, reason="Internal server error")
                raise


def _health_check(connection, request) -> Optional[Any]:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        else:
            # For older websockets versions, return a simple response
            return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    # Continue with the normal request handling.
    return None


"""
Adds NumPy array and PyTorch tensor support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

This module supports serializing both NumPy arrays and PyTorch tensors. PyTorch tensors are converted to
NumPy arrays (zero-copy when possible) before serialization. On deserialization, arrays are returned as NumPy arrays.

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""


def pack_data(obj):
    if isinstance(obj, th.Tensor):
        data = obj.detach().cpu().numpy()
        return {
            b"__ndarray__": True,
            b"data": data.tobytes(),
            b"dtype": data.dtype.str,
            b"shape": data.shape,
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
packb = functools.partial(msgpack.packb, default=pack_data)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)
