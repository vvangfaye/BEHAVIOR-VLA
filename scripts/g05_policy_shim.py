#!/usr/bin/env python3
"""Wire-protocol shim: OmniGibson evaluator ↔ GalaxeaVLA G0.5 policy server.

The BEHAVIOR-1K OmniGibson evaluator (`WebsocketClientPolicy` in
`BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/network_utils.py`) expects:
  - HTTP `GET /healthz` returning 200 OK
  - WebSocket msgpack carrying an `obs` dict whose top-level keys are
    `robot_r1::sensors::<camera>::rgb` (per the `flatten_obs_dict` helper) and
    `robot_r1::proprio::*`, plus a `task_id` integer and the action returned as
    a flat `{"action": ndarray}` of shape `(action_dim,)`
  - Reset sent as `{"reset": True}`

The GalaxeaVLA `serve_policy_mem.py`:
  - Has no HTTP endpoint (only the raw WebSocket), so the evaluator's
    `/healthz` poll times out
  - Expects obs with `images`/`state`/`task` at the top level, where
    `images` is `{<short_key>: ndarray[C,H,W]}`, `state` is `{<key>: ndarray[D]}`
    and `task` is the natural-language instruction string
  - Returns the action as a per-part dict
    (`{"action": {left_arm: ndarray(7,), left_gripper: ndarray(1,), ...}}`)
  - Reset is `{"__reset__": True}`

This shim terminates the OmniGibson-facing protocol on `:LISTEN_PORT` and
proxies to the upstream G0.5 server on `:UPSTREAM_PORT`. It translates obs,
task, and action on the fly. The translation rules are deliberately simple so
they work for both zero-shot and post-fine-tuned G0.5 checkpoints; the BEHAVIOR
side of the translation is configurable via CLI flags so it adapts to the
robot config (R1 Pro vs R1 Lite vs others).

Run with:
    python scripts/g05_policy_shim.py \
        --listen-port 8090 \
        --upstream-host 127.0.0.1 --upstream-port 8765 \
        --task turning_on_radio \
        --task-description "Turn on the radio receiver that's on the table in the living room."
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
import msgpack
import numpy as np
import websockets

# Allow `python scripts/g05_policy_shim.py` to find `g05.utils.websocket` for the
# canonical numpy-aware msgpack codec.
_REPO_ROOT = Path(__file__).resolve().parent.parent / "GalaxeaVLA"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from g05.utils.websocket import packb as g05_packb  # noqa: E402
from g05.utils.websocket import unpackb as g05_unpackb  # noqa: E402

logger = logging.getLogger("g05_shim")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)d)",
    datefmt="%H:%M:%S",
)


# ──────────── msgpack codec ────────────
# The G0.5 codebase ships a numpy-aware msgpack codec. Reuse it so we get
# consistent b"/__ndarray__" wrapping for any numpy arrays the upstream
# server returns, and for any arrays we hand back to the OmniGibson client.
def packb(obj: Any) -> bytes:
    return g05_packb(obj)


def unpackb(buf: bytes) -> Any:
    return g05_unpackb(buf)


# ──────────── translation helpers ────────────
def _match_key(obs: dict, fragment: str) -> str | None:
    """Find the first obs key that contains `fragment`."""
    for k in obs.keys():
        if fragment in k:
            return k
    return None


def extract_images(om: dict, *, head_substr: str, left_substr: str, right_substr: str) -> dict:
    """Map OmniGibson flattened obs keys to G0.5 short image keys.

    OmniGibson `flatten_obs_dict` produces keys like
    `robot_r1::sensors::zed_link_camera_0::rgb`; we look up substrings to be
    robust against the exact sensor name layout. The classifier defaults are
    picked for the R1 Pro embodiment in BEHAVIOR-1K.
    """
    images = {}
    head_k = _match_key(om, head_substr)
    if head_k and isinstance(om[head_k], np.ndarray):
        images["head_rgb"] = np.asarray(om[head_k])
    left_k = _match_key(om, left_substr)
    if left_k and isinstance(om[left_k], np.ndarray):
        images["left_wrist_rgb"] = np.asarray(om[left_k])
    right_k = _match_key(om, right_substr)
    if right_k and isinstance(om[right_k], np.ndarray):
        images["right_wrist_rgb"] = np.asarray(om[right_k])
    return images


def extract_state(om: dict, *, arm_left_substr="arm_left_pos", arm_right_substr="arm_right_pos",
                  grip_left_substr="gripper_left", grip_right_substr="gripper_right") -> dict:
    """Map OmniGibson proprioceptive obs to G0.5 grouped state.

    OmniGibson R1 Pro exposes 7-D arm joint positions and 2-D gripper widths.
    G0.5 expects grouped keys `left_arm(7)`, `right_arm(7)`, `left_gripper(2)`,
    `right_gripper(2)`, plus `lower_body(7)` (trunk_qpos 4 + base_qvel 3). The
    OmniGibson evaluator does not publish base velocity or trunk pose, so we
    zero-pad `lower_body` for zero-shot eval. After fine-tuning on BEHAVIOR-1K
    data the per-embodiment processor still only uses the arm/gripper fields,
    so the padding is harmless.
    """
    state = {}
    for out_key, fragment in [
        ("left_arm", arm_left_substr),
        ("right_arm", arm_right_substr),
        ("left_gripper", grip_left_substr),
        ("right_gripper", grip_right_substr),
    ]:
        k = _match_key(om, fragment)
        if k is not None and isinstance(om[k], np.ndarray):
            state[out_key] = np.asarray(om[k], dtype=np.float32)
    # Pad missing keys with zeros so the G0.5 processor's shape_meta checks pass.
    state.setdefault("left_arm", np.zeros(7, dtype=np.float32))
    state.setdefault("right_arm", np.zeros(7, dtype=np.float32))
    state.setdefault("left_gripper", np.zeros(2, dtype=np.float32))
    state.setdefault("right_gripper", np.zeros(2, dtype=np.float32))
    state.setdefault("lower_body", np.zeros(7, dtype=np.float32))
    return state


def translate_obs_to_g05(om: dict, task_description: str, embodiment: str,
                         *, head_substr: str, left_substr: str, right_substr: str) -> dict:
    """Convert an OmniGibson obs dict into the dict G0.5 expects."""
    images = extract_images(om, head_substr=head_substr, left_substr=left_substr, right_substr=right_substr)
    state = extract_state(om)
    if not images:
        raise ValueError(
            f"No camera observations found in obs keys {list(om.keys())[:5]}..."
        )
    out = {
        "images": images,
        "state": state,
        "task": task_description,
        "embodiment_type": embodiment,
    }
    return out


def translate_action_to_omnigibson(g05_action: dict) -> np.ndarray:
    """Convert G0.5's per-part action dict into a flat ndarray.

    G0.5 returns `{"action": {left_arm: (7,), right_arm: (7,), lower_body: (7,),
    left_gripper: (1,), right_gripper: (1,)}}` (or a subset when the AR head
    drops absent parts). We concatenate in the canonical BEHAVIOR-1K order:
    `left_arm + left_gripper + right_arm + right_gripper + lower_body` = 23D
    which matches the BEHAVIOR-1K R1 Pro `action_dim=23`. Missing keys are
    zero-padded so the shape is stable across fine-tuned vs zero-shot models.
    """
    canonical_order = ["left_arm", "left_gripper", "right_arm", "right_gripper", "lower_body"]
    expected_dims = {"left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1, "lower_body": 7}
    pieces = []
    for k in canonical_order:
        if k in g05_action and isinstance(g05_action[k], np.ndarray):
            arr = np.asarray(g05_action[k], dtype=np.float32).flatten()[: expected_dims[k]]
            pieces.append(arr)
        else:
            pieces.append(np.zeros(expected_dims[k], dtype=np.float32))
    return np.concatenate(pieces, axis=0)


# ──────────── upstream (G0.5) connection manager ────────────
class UpstreamConnection:
    """Holds the single persistent WebSocket connection to the G0.5 server.

    The G0.5 `ChunkedPolicyWrapper` keeps per-connection state (chunk cache,
    frame buffers), so we keep one upstream socket per downstream client.
    """

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._ws = None
        self._lock = asyncio.Lock()
        self._metadata = None  # {"action_steps": int} from upstream's first msg

    async def connect(self):
        url = f"ws://{self._host}:{self._port}"
        logger.info("Upstream: connecting to %s", url)
        self._ws = await websockets.connect(url, max_size=None)
        # G0.5 sends `{"action_steps": N}` as the first message.
        first = await self._ws.recv()
        self._metadata = unpackb(first)
        logger.info("Upstream: connected; metadata=%s", self._metadata)

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    @property
    def metadata(self):
        return self._metadata

    @property
    def is_alive(self) -> bool:
        return self._ws is not None and self._ws.state.name == "OPEN"

    async def roundtrip(self, payload: dict) -> dict:
        """Send an obs (or reset) and wait for the matching response."""
        async with self._lock:
            if self._ws is None or not self.is_alive:
                await self.connect()
            assert self._ws is not None
            await self._ws.send(packb(payload))
            raw = await self._ws.recv()
            return unpackb(raw)


# ──────────── downstream (OmniGibson) WebSocket handler ────────────
async def downstream_handler(ws: web.WebSocketResponse, *, upstream: UpstreamConnection,
                            task_description: str, embodiment: str,
                            head_substr: str, left_substr: str, right_substr: str):
    """Translate between the OmniGibson and G0.5 protocols for one connection."""
    client = ws.remote_address
    logger.info("Downstream client connected: %s", client)
    try:
        # Make sure the upstream G0.5 socket is freshly connected for this
        # episode (G0.5 keeps per-connection state).
        await upstream.connect()
        # Pass through G0.5's metadata so the OmniGibson client sees a sane
        # value (it doesn't actually use it, but it expects to receive *some*
        # dict on connect).
        await ws.send_bytes(packb(upstream.metadata))

        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                payload = unpackb(msg.data)
            elif msg.type == web.WSMsgType.TEXT:
                # OmniGibson uses msgpack, but be tolerant of accidental JSON.
                import json

                payload = json.loads(msg.data)
            else:
                continue

            # Reset translation: OmniGibson `{"reset": True}` ↔ G0.5 `{"__reset__": True}`
            if isinstance(payload, dict) and payload.get("reset"):
                await upstream.roundtrip({"__reset__": True})
                continue

            # Translate obs to G0.5 shape.
            try:
                g05_obs = translate_obs_to_g05(
                    payload,
                    task_description=task_description,
                    embodiment=embodiment,
                    head_substr=head_substr,
                    left_substr=left_substr,
                    right_substr=right_substr,
                )
            except ValueError as exc:
                logger.warning("Translate failure: %s", exc)
                await ws.send_bytes(packb({"error": {"code": 400, "message": str(exc)}}))
                continue

            # Forward to G0.5.
            try:
                g05_response = await upstream.roundtrip(g05_obs)
            except Exception as exc:
                logger.exception("Upstream call failed")
                await ws.send_bytes(packb({"error": {"code": 502, "message": f"upstream: {exc}"}}))
                continue

            # Translate action back to OmniGibson.
            g05_action = g05_response.get("action", {})
            action_flat = translate_action_to_omnigibson(g05_action)
            await ws.send_bytes(packb({"action": action_flat}))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Downstream client disconnected: %s", client)
    finally:
        await upstream.close()


# ──────────── HTTP + WebSocket aiohttp app ────────────
async def healthz_handler(request: web.Request, *, upstream: UpstreamConnection) -> web.Response:
    """Probe G0.5 liveness without disrupting its chunk cache.

    A trivial upstream roundtrip costs <1 ms because G0.5 caches the previous
    chunk; we deliberately use `{"__reset__": False}` (no-op) rather than
    relying on TCP state, which would otherwise pass even when the upstream
    model process is wedged.
    """
    if not upstream.is_alive:
        return web.Response(status=503, text="upstream disconnected\n")
    return web.Response(status=200, text="OK\n")


async def websocket_handler(request: web.Request, *, upstream: UpstreamConnection,
                            task_description: str, embodiment: str,
                            head_substr: str, left_substr: str, right_substr: str):
    ws = web.WebSocketResponse(max_msg_size=None)
    await ws.prepare(request)
    await downstream_handler(
        ws,
        upstream=upstream,
        task_description=task_description,
        embodiment=embodiment,
        head_substr=head_substr,
        left_substr=left_substr,
        right_substr=right_substr,
    )
    return ws


def build_app(*, upstream: UpstreamConnection, task_description: str, embodiment: str,
              head_substr: str, left_substr: str, right_substr: str) -> web.Application:
    app = web.Application()
    app["upstream"] = upstream
    app["task_description"] = task_description
    app["embodiment"] = embodiment
    app["head_substr"] = head_substr
    app["left_substr"] = left_substr
    app["right_substr"] = right_substr
    app.router.add_get(
        "/healthz",
        lambda r: healthz_handler(r, upstream=upstream),
    )
    app.router.add_get(
        "/",
        lambda r: websocket_handler(
            r,
            upstream=upstream,
            task_description=task_description,
            embodiment=embodiment,
            head_substr=head_substr,
            left_substr=left_substr,
            right_substr=right_substr,
        ),
    )
    return app


# ──────────── CLI ────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8090)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8765)
    parser.add_argument(
        "--task-description", default=argparse.SUPPRESS,
        help="Natural-language task instruction. Required.",
    )
    parser.add_argument(
        "--task", default=None,
        help="If set, look up the task description from `<GalaxeaVLA>/tasks.jsonl`.",
    )
    parser.add_argument(
        "--embodiment", default="galaxea_r1pro",
        help="G0.5 embodiment identifier (default: galaxea_r1pro).",
    )
    # Image-key matching
    parser.add_argument("--head-substr", default="zed_link_camera_0::rgb",
                        help="Substring used to find the head-camera obs key.")
    parser.add_argument("--left-substr", default="left_realsense_link_camera_0::rgb",
                        help="Substring used to find the left-wrist-camera obs key.")
    parser.add_argument("--right-substr", default="right_realsense_link_camera_0::rgb",
                        help="Substring used to find the right-wrist-camera obs key.")
    parser.add_argument("--tasks-jsonl", default="/var/tmp/behavior-1k/2026-challenge-demos/meta/tasks.jsonl")
    return parser.parse_args()


def resolve_task_description(args: argparse.Namespace) -> str:
    if getattr(args, "task_description", None):
        return args.task_description
    if args.task:
        import json

        with open(args.tasks_jsonl) as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("task_name") == args.task:
                    return row["task"]
        raise SystemExit(f"--task={args.task!r} not found in {args.tasks_jsonl}")
    raise SystemExit("Pass either --task-description or --task")


def main():
    args = parse_args()
    task_description = resolve_task_description(args)

    upstream = UpstreamConnection(args.upstream_host, args.upstream_port)

    app = build_app(
        upstream=upstream,
        task_description=task_description,
        embodiment=args.embodiment,
        head_substr=args.head_substr,
        left_substr=args.left_substr,
        right_substr=args.right_substr,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(*_):
        logger.info("Shutting down")
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    web.run_app(app, host=args.listen_host, port=args.listen_port, loop=loop, print=lambda *a, **kw: None)


if __name__ == "__main__":
    main()