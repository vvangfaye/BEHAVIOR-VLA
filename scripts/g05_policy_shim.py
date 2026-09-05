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
    """Map OmniGibson proprioceptive obs to the b1k state keys G0.5 expects.

    OmniGibson publishes the whole 61-D proprio vector under `<robot>::proprio`
    (the GR00T eval wrapper reads the same array), so slice it with the indices
    the training data uses. These match both `meta/modality.json` and
    `GalaxeaVLA/configs/data/b1k.yaml`, which is what keeps eval-time state
    layout identical to training:

        base_qvel [0:3]  left_arm [3:10]   left_gripper  [24:26]
        trunk_qpos[53:57] right_arm[28:35]  right_gripper [49:51]

    Widths must equal `raw_shape` (grippers stay 2-D here; the processor's
    BehaviorPerKeyTransform collapses them to 1-D).
    """
    # dataset state layout: key -> (start, width)
    PROPRIO_SLICES = {
        "base_qvel": (0, 3),
        "left_arm": (3, 7),
        "left_gripper": (24, 2),
        "right_arm": (28, 7),
        "right_gripper": (49, 2),
        "trunk_qpos": (53, 4),
    }

    prop_k = _match_key(om, "proprio")
    prop = np.asarray(om[prop_k], dtype=np.float32).reshape(-1) if prop_k is not None else None

    state = {}
    if prop is not None and prop.size >= 57:
        for key, (start, width) in PROPRIO_SLICES.items():
            state[key] = prop[start : start + width].astype(np.float32)
        return state

    # Fallback: no flat proprio vector, recover what we can from per-joint keys.
    for out_key, fragment in [
        ("left_arm", arm_left_substr),
        ("right_arm", arm_right_substr),
        ("left_gripper", grip_left_substr),
        ("right_gripper", grip_right_substr),
    ]:
        k = _match_key(om, fragment)
        if k is not None and isinstance(om[k], np.ndarray):
            state[out_key] = np.asarray(om[k], dtype=np.float32).reshape(-1)
    for key, (_, width) in PROPRIO_SLICES.items():
        state.setdefault(key, np.zeros(width, dtype=np.float32))
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


def translate_action_to_omnigibson(g05_action: dict, prev: np.ndarray | None = None,
                                   state: dict | None = None,
                                   gripper_max_width: float = 0.1) -> np.ndarray:
    """Convert G0.5's per-part action dict into the flat 23-D BEHAVIOR-1K action.

    Raw layout (meta/modality.json, configs/data/b1k.yaml):
        [base(3) | torso(4) | left_arm(7) | left_gripper(1) | right_arm(7) | right_gripper(1)]

    G0.5 returns `{left_arm(7), left_gripper(1), right_arm(7), right_gripper(1),
    lower_body(7)}`, where `BehaviorPerKeyTransform.backward` has already
    reordered `lower_body` to `[base_qvel(3), trunk_qpos(4)]`.

    Absent parts are *held*, not zeroed. The tokenizer drops a part whose value
    does not change within the chunk, and on this dataset both grippers are
    constant in 95.3% of 32-step chunks. The gripper command is strictly binary
    -1/+1 and never 0, so a zero fill would be out-of-distribution and would
    break grasping; zeroing arm/torso joints would command the zero pose.
    Precedence per absent part: previous command, else measured state. Only base
    velocity falls back to zero, where zero legitimately means "stop".
    """
    out = np.zeros(23, dtype=np.float32)
    # destination slice -> (part key, offset within that part)
    placements = [
        (slice(0, 3), "lower_body", 0),    # base_qvel
        (slice(3, 7), "lower_body", 3),    # trunk_qpos
        (slice(7, 14), "left_arm", 0),
        (slice(14, 15), "left_gripper", 0),
        (slice(15, 22), "right_arm", 0),
        (slice(22, 23), "right_gripper", 0),
    ]
    # measured-state fallback, keyed by destination slice start
    def _from_state(start: int) -> np.ndarray | None:
        if not state:
            return None
        if start == 3:
            v = state.get("trunk_qpos")
            return np.asarray(v, dtype=np.float32).reshape(-1)[:4] if v is not None else None
        if start in (7, 15):
            v = state.get("left_arm" if start == 7 else "right_arm")
            return np.asarray(v, dtype=np.float32).reshape(-1)[:7] if v is not None else None
        if start in (14, 22):
            v = state.get("left_gripper" if start == 14 else "right_gripper")
            if v is None:
                return None
            width = float(np.asarray(v, dtype=np.float32).sum())
            # match BehaviorPerKeyTransform: 2*(width/max)-1, then binarize to +-1
            return np.array([1.0 if 2.0 * (width / gripper_max_width) - 1.0 >= 0.0 else -1.0],
                            dtype=np.float32)
        return None

    for dst, key, src_off in placements:
        width = dst.stop - dst.start
        part = g05_action.get(key)
        if part is not None:
            arr = np.asarray(part, dtype=np.float32).reshape(-1)
            chunk = arr[src_off : src_off + width]
            out[dst.start : dst.start + len(chunk)] = chunk
            continue
        if prev is not None:
            out[dst] = np.asarray(prev, dtype=np.float32).reshape(-1)[dst]
            continue
        held = _from_state(dst.start)
        if held is not None:
            out[dst.start : dst.start + len(held)] = held
    return out


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

        # Last 23-D command, used to hold parts the AR head omits this step.
        prev_action: np.ndarray | None = None

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
                prev_action = None
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
            action_flat = translate_action_to_omnigibson(
                g05_action, prev=prev_action, state=g05_obs.get("state")
            )
            prev_action = action_flat
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