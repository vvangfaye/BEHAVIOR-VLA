---
name: restore-pi05-inference
description: Restore and verify the BEHAVIOR-Challenge π0.5 inference and OmniGibson closed-loop evaluation environment after a notebook or pod restart. Use when /dev/shm was cleared, b1k-pi05 or b1k-eval is missing, simulator assets need re-extraction, or the user asks to recover the pi05 inference/eval runtime.
---

# Restore π0.5 Inference

Restore from persistent OSS archives; do not rebuild or redownload large dependencies unless an archive is missing.

## Restore

1. From the repository root, check for active evaluation processes and ports. Do not stop unrelated processes. After a notebook restart there should normally be none.
2. Run:

```bash
.qoder/skills/restore-pi05-inference/scripts/restore.sh restore
```

The script:

- restores `b1k-pi05.tar.gz` and `b1k-eval.tar.gz` from `/data/oss_bucket_0/wangfaye/conda_env/` into `/dev/shm/wangfaye/conda_envs/`;
- runs `conda-unpack` only for newly extracted environments;
- links `openpi/.venv` to `b1k-pi05`;
- writes Vulkan/EGL manifests for the active NVIDIA driver;
- extracts all three OmniGibson bundles from `BEHAVIOR-1K/.downloads/hf-assets` without network access;
- links `BEHAVIOR-1K/omnigibson-data` to tmpfs;
- downloads only the small encryption key if it is absent and links it into the runtime data directory;
- repairs the archived `packaging`/`setuptools` payload only if its import check fails;
- checks π0.5 imports, JAX GPUs, eval imports, CUDA, and Vulkan.

If the script reports a non-empty incomplete target, inspect it first. Prefer a fresh target for incomplete environments. Within the authorized competition workflow, repair only owned temporary environments after preserving needed evidence; no repeated confirmation is required. Preserve unrelated or actively used environments. Do not fall back to `setup_envs.sh` or `setup_eval_env.sh` while the archives exist; those scripts can contact GitHub and PyPI.

Use check-only mode at any time:

```bash
.qoder/skills/restore-pi05-inference/scripts/restore.sh check
```

## Runtime verification

Use the actual public interface, not an import-only test. Start one co-located server/simulator smoke with a unique port and output root:

```bash
timeout --signal=TERM --kill-after=30s 300s env \
  MODEL=pi05 \
  PI05_CKPT="$PWD/openpi/outputs/checkpoints/pi05_b1k/pi05_b1k_subtask_mix_s32_20260825_185859/314999" \
  PORT=8120 SERVER_GPU=0 EVAL_GPU=0 \
  INSTANCE_INDICES=0 NUM_ROLLOUTS=1 \
  PROMPT_MODE=whole_task \
  OUTPUT_ROOT="$PWD/eval_results/pi05_env_restore_smoke_$(date +%Y%m%d_%H%M%S)" \
  "$PWD/scripts/eval_b1k.sh" turning_on_radio
```

Launch it in the background and monitor its generated `eval.log` and `server_8120.log`. Require all of the following:

- checkpoint restoration completes;
- websocket server listens on the selected port;
- evaluator log contains `app ready`;
- no `CUDA out of memory`, `Segmentation fault`, `Failed to create the raytracing pipeline`, or `vkCreateInstance failed` appears.

`INSTANCE_INDICES` is a zero-based public-test index in `0..19`, not the actual instance ID such as `301`.

Once `app ready` is observed, stop the tracked smoke task; a full rollout is unnecessary for environment recovery. Confirm the server, simulator, and selected port are gone. Do not kill processes by broad name matching.

## Report

Report archive paths, restored environment paths, driver version, Vulkan device count, checkpoint, `app ready` evidence, and cleanup status. If the key download is the only blocker, state that assets and environments are otherwise restored rather than rebuilding them.
