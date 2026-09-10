# Evaluating G0.5 (G05) on BEHAVIOR-1K

This document covers how to serve the OpenGalaxea G05 policy and run closed-loop
OmniGibson rollouts against it. The G05 model lives in
[`GalaxeaVLA/`](../GalaxeaVLA); everything here is wired into the
BEHAVIOR-Challenge wrapper scripts.

## Status

There is a **wire-protocol mismatch** between the G05 inference server and the
OmniGibson evaluator's `WebsocketClientPolicy`:

- OmniGibson (in `BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/network_utils.py`)
  polls `GET /healthz` before opening the WebSocket, then expects a flat
  `{action: ndarray}` reply to a flat `obs` payload, with `{"reset": True}` to
  reset.
- G05 (`GalaxeaVLA/scripts/serve_policy_mem.py`) is a pure msgpack WebSocket
  with no HTTP layer and expects `images: {cam: CHW}`, `state: {part: 1D}`,
  `task: str`, `embodiment_type: str`, and replies with
  `{body_part: {T, D}}`.

Until an adapter is written, the wrapper scripts will start the G05 server
correctly but the evaluator will time out at the `/healthz` probe. This is
called out as a blocker in `scripts/eval_g05_b1k.sh`.

## venv

The env-setup agent builds the G05 venv at:

```text
/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/GalaxeaVLA/.venv
```

via `uv sync` from the repo root. Override with `GALAXEA_VENV=...` if it lands
elsewhere.

The OmniGibson evaluator runs out of the existing BEHAVIOR-1K conda env
(`/dev/shm/wangfaye/conda_envs/b1k-eval` by default, or the `behavior` conda
env at `/mnt/afs/visitor34/venvs/miniforge3/envs/behavior/`). Override with
`EVAL_ENV=...`.

## Checkpoint

G05 weights (~33 GB) are downloaded by an external agent to:

```text
/mnt/workspace/wangfaye/cache/huggingface/hub/OpenGalaxea/G05/checkpoints/model_state_dict.pt
```

Override per-run with `G05_CKPT=/path/to/some_other.pt`. To use a fine-tuned
checkpoint later, just point `G05_CKPT` at it; no other knob needs to change.
The same `--ckpt_path` is the only difference between a base run and a
post-fine-tune run.

## Port

Default `PORT=8090`. The pi05 and gr00t servers use 8000 by convention, so
8090 keeps them runnable in parallel and avoids accidental cross-talk if both
ever start on the same host. Override per-run with `PORT=...`.

## Launching just the server

```bash
./scripts/run_g05_server.sh
```

Prints the listening URL (`ws://127.0.0.1:8090`) and tees logs to
`eval_results/g05/server_8090.log`. Useful when you want to poke the server
manually with `websocat`, or run multiple eval tasks against a long-lived
server.

Stop with Ctrl-C; the script does not background itself, and a SIGTERM is
delivered to the python process automatically.

## Launching a closed-loop eval

```bash
./scripts/eval_g05_b1k.sh turning_on_radio
# or several tasks:
./scripts/eval_g05_b1k.sh turning_on_radio picking_up_toys
```

Same env knobs as `scripts/eval_b1k.sh`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUM_ROLLOUTS` | `1` | Rollouts per instance |
| `INSTANCE_INDICES` | empty | Restrict to a subset of the split (e.g. `"0 1"`) |
| `MODE` | `public_test` | `train` / `public_test` / `hidden_test` |
| `SERVER_GPU` | `0` | GPU index for the G05 inference server |
| `EVAL_GPU` | `1` | GPU index for Isaac Sim |
| `OUTPUT_ROOT` | `<repo>/eval_results` | Where rollouts land |

Outputs:

- Per-instance JSONs and MP4 videos: `eval_results/g05/<task>/json/`,
  `eval_results/g05/<task>/videos/`
- Per-task eval log: `eval_results/g05/<task>/eval.log`
- Server log: `eval_results/g05/server_<port>.log`

## Stopping the server

The eval wrapper installs an `EXIT` trap (`trap stop_server EXIT`), so killing
the wrapper (Ctrl-C or `kill`) cleans up the server. If you started the server
standalone with `run_g05_server.sh`, Ctrl-C is enough. For a runaway process,
find the pid from the printed line or `pgrep -f serve_policy_mem.py` and
`kill -TERM <pid>`.

## Switching to a fine-tuned checkpoint

No code changes are required -- the server only needs `--ckpt_path`. Replace
the default with the fine-tuned file:

```bash
G05_CKPT=/path/to/finetune/checkpoints/step_50000/model_state_dict.pt \
    ./scripts/eval_g05_b1k.sh turning_on_radio
```

The same script will load the fine-tuned weights, the matching dataset stats
(next to the checkpoint) and run the evaluator unchanged. If the fine-tune
moved to a different embodiment tag, also add `eval_embodiment=...` to the
launch (the R1 Pro default is `galaxea_r1pro`).

## Environment variables set by the wrapper

The wrapper exports:

- `PYTHONPATH=$GALAXEA_REPO/src:$GALAXEA_REPO` (so `g05` resolves)
- `HF_ENDPOINT=https://hf-mirror.com` (override with `HF_ENDPOINT=...`)
- `HYDRA_FULL_ERROR=1` (full Hydra tracebacks from the config loader)
- `CUDA_VISIBLE_DEVICES=$SERVER_GPU` for the server, `$EVAL_GPU` for the eval
- `OMNIGIBSON_DATA_PATH` (must point at the BEHAVIOR-1K data root)
- The Isaac Sim graphics env: `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`,
  `__EGL_VENDOR_LIBRARY_FILENAMES`, `OMNI_KIT_ACCEPT_EULA=YES`,
  `PYTHONNOUSERSITE=1` (same set as `scripts/eval_b1k.sh`).