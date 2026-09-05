"""Serve G0.5 to the BEHAVIOR-1K (OmniGibson) evaluator over a websocket.

The evaluator in ``omnigibson.eval.eval`` talks a different protocol from
``scripts/serve_policy.py``:

  * it sends the simulator's *flat* observation dict, keyed like
    ``robot_r1::proprio`` and ``robot_r1::robot_r1:zed_link:Camera:0::rgb``,
    rather than G0.5's ``{"images", "state", "task"}`` schema;
  * it expects ``{"action": <23-dim vector>}`` back — one action per call, not a
    chunk — and converts it straight into ``env.step``;
  * it encodes numpy with **bytes** msgpack keys (``b"__ndarray__"``), while
    ``g05.utils.websocket`` uses **str** keys. The two are wire-incompatible, so
    the codec below is a copy of the evaluator's, not an import of G0.5's.

Everything model-side is reused from ``scripts/serve_policy.py`` so behaviour
matches training: checkpoint loading, the normalizer built from
``dataset_stats.json``, the processor, and ``ChunkedPolicyWrapper``'s
receding-horizon cache.

Layout mappings are read from the processor's ``shape_meta`` at runtime rather
than hardcoded, so they follow ``configs/data/b1k.yaml`` if it changes.

Usage:
    python scripts/b1k/serve_b1k.py \
        --ckpt_path /path/to/checkpoints/step_50000/model_state_dict.pt \
        --task turning_on_radio \
        --tasks-jsonl /var/tmp/behavior-1k/2026-challenge-demos/meta/tasks.jsonl \
        --host 127.0.0.1 --port 8000

Then point the evaluator at it:
    python -m omnigibson.eval.eval --task-name turning_on_radio \
        --host 127.0.0.1 --port 8000 ...
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as _ws_server

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_SERVE_POLICY = _HERE.parent.parent / "serve_policy.py"

# The evaluator's robot is named robot_r1 in BEHAVIOR-1K 3.9.0 and every
# observation it publishes is namespaced under that name.
ROBOT_NAME = "robot_r1"


def _load_serve_policy():
    """Import scripts/serve_policy.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("g05_serve_policy", _SERVE_POLICY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SERVE_POLICY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g05_serve_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


# ──────── msgpack codec (must match omnigibson/eval/utils/network_utils.py) ────────


def _pack(obj):
    if isinstance(obj, torch.Tensor):
        data = obj.detach().cpu().numpy()
        return {b"__ndarray__": True, b"data": data.tobytes(),
                b"dtype": data.dtype.str, b"shape": data.shape}
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)


# ──────── observation / action adapter ────────


def _camera_obs_key(lerobot_key: str) -> str:
    """Map a dataset video column to the evaluator's observation key.

    ``observation.rgb.zed_link_camera_0`` describes the same sensor the simulator
    publishes as ``robot_r1::robot_r1:zed_link:Camera:0::rgb``.
    """
    link = lerobot_key.removeprefix("observation.rgb.").removesuffix("_camera_0")
    return f"{ROBOT_NAME}::{ROBOT_NAME}:{link}:Camera:0::rgb"


def _resize_hwc(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize uint8 HWC to (height, width) with the processor's own backend.

    The simulator renders 1080x1080 while the dataset stored 720x720 (head) and
    480x480 (wrists) — same aspect ratio, so a plain resize is faithful and no
    padding is needed.
    """
    if img.shape[0] == height and img.shape[1] == width:
        return img
    tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(
        tensor.float(), size=(height, width), mode="bilinear", align_corners=False
    )
    return out[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


class B1KAdapter:
    """Translates between the evaluator's protocol and G0.5's raw_obs schema."""

    def __init__(self, wrapper, processor, task_prompt: str, frequency: float,
                 embodiment_type: str = "behavior_1k", gripper_default: float = 1.0):
        self.wrapper = wrapper
        self.task_prompt = task_prompt
        self.frequency = frequency
        self.embodiment_type = embodiment_type

        p = processor
        if hasattr(processor, "processors"):
            p = processor.processors.get(embodiment_type) or next(iter(processor.processors.values()))
        sm = p.shape_meta

        # State groups are slices of the simulator's 61-dim proprio vector; the
        # dataset built them from the same column, so start_index carries over.
        self.state_spec = [
            (m["key"], int(m["start_index"]), int(m["raw_shape"]))
            for m in sm["state"] if m["lerobot_key"] == "observation.state"
        ]
        # Action groups are slices of the 23-dim action the evaluator expects.
        self.action_spec = [
            (m["key"], int(m["start_index"]), int(m["raw_shape"]))
            for m in sm["action"] if m["lerobot_key"] == "action"
        ]
        self.action_dim = max(s + n for _, s, n in self.action_spec)
        self.image_spec = [
            (m["key"], _camera_obs_key(m["lerobot_key"]),
             int(m["raw_shape"][1]), int(m["raw_shape"][2]))
            for m in sm["images"]
        ]
        # How to fill an action group the AR head did not predict. Three cases:
        #
        #   base_qvel  velocity command -> hold means zero (stop).
        #   grippers   normalised command in the controller's [-1, 1] range. It must
        #              NOT be filled from the measured state: state carries a
        #              physical width in metres, so 0.05 m read as a command is a
        #              half-closed gripper that can never grasp. Hold the last
        #              command instead, defaulting to fully open.
        #   arms/torso absolute joint targets -> repeating the measured joint
        #              positions is correct; verified against ground truth.
        self.gripper_groups = {k for k, _, _ in self.action_spec if "gripper" in k}
        self.hold_from_state = {
            k for k, _, _ in self.action_spec
            if k != "base_qvel" and k not in self.gripper_groups
        }
        # Last command actually sent, per group, for gripper hold.
        self._last_command: dict[str, np.ndarray] = {}
        self._last_absent: tuple[str, ...] | None = None
        self.gripper_default = float(gripper_default)

        logger.info("state groups : %s", [(k, s, n) for k, s, n in self.state_spec])
        logger.info("action groups: %s (total %dD)",
                    [(k, s, n) for k, s, n in self.action_spec], self.action_dim)
        logger.info("cameras      : %s", [(k, o) for k, o, _, _ in self.image_spec])

    def reset(self) -> None:
        # Episode boundary: a command carried over from the previous episode would
        # be wrong, so drop the history and let the gripper default apply again.
        self._last_command.clear()
        self._last_absent = None
        self.wrapper.reset()

    def _raw_obs(self, obs: dict) -> tuple[dict, dict[str, np.ndarray]]:
        proprio = obs.get(f"{ROBOT_NAME}::proprio")
        if proprio is None:
            raise KeyError(
                f"observation has no '{ROBOT_NAME}::proprio'; got {sorted(obs)[:8]}"
            )
        proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)

        state = {k: proprio[s:s + n].copy() for k, s, n in self.state_spec}

        images = {}
        for key, obs_key, height, width in self.image_spec:
            frame = obs.get(obs_key)
            if frame is None:
                raise KeyError(f"observation has no camera '{obs_key}'")
            frame = np.asarray(frame)[..., :3]  # drop alpha if present
            frame = _resize_hwc(frame.astype(np.uint8, copy=False), height, width)
            images[key] = np.ascontiguousarray(frame.transpose(2, 0, 1))  # HWC -> CHW

        raw_obs = {
            "images": images,
            "state": state,
            "task": self.task_prompt,
            "frequency": self.frequency,
            "embodiment_type": self.embodiment_type,
        }
        return raw_obs, state

    async def act(self, obs: dict) -> np.ndarray:
        raw_obs, state = self._raw_obs(obs)
        # ChunkedPolicyWrapper only needs a real observation when its cache is
        # exhausted; passing raw_obs every call is harmless and keeps this simple.
        parts, _cot = await self.wrapper.get_action(raw_obs)

        action = np.zeros(self.action_dim, dtype=np.float32)
        missing = []
        for key, start, n in self.action_spec:
            value = parts.get(key)
            if value is None:
                # With dropout_noop_parts the AR head omits a group's marker when
                # that group has no motion, so an absent group usually means "do
                # not move" and holding is the intended semantics, not a patch.
                # It cannot, however, distinguish that from the model simply
                # failing to predict a group that should have moved.
                missing.append(key)
                if key in self.gripper_groups:
                    prev = self._last_command.get(key)
                    value = (prev if prev is not None
                             else np.full(n, self.gripper_default, dtype=np.float32))
                elif key in self.hold_from_state and key in state:
                    value = np.asarray(state[key], dtype=np.float32).reshape(-1)[:n]
                else:
                    value = np.zeros(n, dtype=np.float32)
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            if value.shape[0] != n:
                raise ValueError(
                    f"action group '{key}' has {value.shape[0]} dims, expected {n}"
                )
            action[start:start + n] = value
            self._last_command[key] = value.copy()
        if missing:
            # Fires on nearly every step, so only report when the set changes.
            absent = tuple(missing)
            if absent != self._last_absent:
                logger.info("absent action groups now %s (held)", missing)
                self._last_absent = absent
        return action


# ──────── websocket server (protocol matches omnigibson.eval) ────────


def _health_check(connection, request):
    if getattr(request, "path", None) == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _handler(websocket, adapter: B1KAdapter, metadata: dict):
    logger.info("connection from %s opened", websocket.remote_address)
    packer = Packer()
    await websocket.send(packer.pack(metadata))
    while True:
        try:
            payload = unpackb(await websocket.recv(), strict_map_key=False)
            if "reset" in payload:
                adapter.reset()
                continue
            t0 = time.monotonic()
            action = await adapter.act(payload)
            await websocket.send(packer.pack({
                "action": action,
                "server_timing": {"infer_ms": (time.monotonic() - t0) * 1000},
            }))
        except websockets.ConnectionClosed:
            logger.info("connection from %s closed", websocket.remote_address)
            break
        except Exception:
            logger.error("error in connection from %s:\n%s",
                         websocket.remote_address, traceback.format_exc())
            await websocket.close(code=1011, reason="Internal server error")
            raise


def _task_prompt(task_name: str, tasks_jsonl: Path) -> str:
    """Read the task's instruction from the dataset's meta/tasks.jsonl.

    Using the dataset's own text keeps the prompt identical to the one training
    conditioned on.
    """
    if not tasks_jsonl.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {tasks_jsonl}")
    for line in tasks_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_name") == task_name:
            return row["task"]
    raise KeyError(f"task '{task_name}' not present in {tasks_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--task", required=True, help="BEHAVIOR task name, e.g. turning_on_radio")
    parser.add_argument("--tasks-jsonl", default=None,
                        help="dataset meta/tasks.jsonl (default: derived from the config's dataset dir)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action_steps", type=int, default=16,
                        help="steps served per inference; 16 matches the π0.5 baseline's horizon")
    parser.add_argument("--frequency", type=float, default=30.0,
                        help="control rate; the challenge demos are 30 fps")
    parser.add_argument("--embodiment", default="behavior_1k")
    parser.add_argument("--gripper-default", type=float, default=1.0,
                        help="command used for an absent gripper group before any has been "
                             "sent; MultiFingerGripperController takes [-1, 1] where 1 is open")
    args, remaining = parser.parse_known_args()
    overrides = [r for r in remaining if "=" in r]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    sp = _load_serve_policy()

    run_dir = sp.find_run_dir(args.ckpt_path)
    logger.info("run dir: %s", run_dir)
    cfg = sp.load_config_from_run_dir(run_dir, args.ckpt_path, overrides)
    if cfg.get("eval_embodiment", None) and "embodiment_datasets" in cfg.data:
        sp.filter_embodiment(cfg, cfg.eval_embodiment)

    tasks_jsonl = args.tasks_jsonl
    if tasks_jsonl is None:
        # embodiment_datasets.<name>.dataset_groups[].dataset_dirs[] holds the
        # dataset root the checkpoint was trained on.
        ds = cfg.data.embodiment_datasets[args.embodiment]
        tasks_jsonl = Path(ds.dataset_groups[0].dataset_dirs[0]) / "meta" / "tasks.jsonl"
    prompt = _task_prompt(args.task, Path(tasks_jsonl))
    logger.info("task %s -> %r", args.task, prompt)

    policy, processor = sp.setup(cfg, device=args.device)
    inferencer = sp.PolicyInferencer(policy, processor, device=args.device)
    wrapper = sp.ChunkedPolicyWrapper(inferencer, processor, action_steps=args.action_steps)
    adapter = B1KAdapter(wrapper, processor, prompt, args.frequency, args.embodiment,
                         gripper_default=args.gripper_default)

    metadata = {
        "model": "g05",
        "task": args.task,
        "action_steps": args.action_steps,
        "action_dim": adapter.action_dim,
    }

    async def run() -> None:
        handler = functools.partial(_handler, adapter=adapter, metadata=metadata)
        # ping_interval=None: autoregressive generation blocks the event loop for
        # seconds at a time, so the default 20 s keepalive fires and the server
        # drops the connection with 1011 mid-rollout. That killed two whole tasks
        # of the step-280000 sweep (assembling_gift_baskets, storing_food).
        async with _ws_server.serve(handler, args.host, args.port, compression=None,
                                    max_size=None, ping_interval=None,
                                    process_request=_health_check):
            logger.info("serving g05 on ws://%s:%d (chunk=%d)",
                        args.host, args.port, args.action_steps)
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
