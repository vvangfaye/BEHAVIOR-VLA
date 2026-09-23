---
name: evaluate-pi05-b1k
description: Safely run, submit, monitor, validate, and summarize BEHAVIOR-Challenge π0.5 OmniGibson closed-loop evaluations. Use for local smoke or multi-GPU evaluation, Nebula/AIHub evaluation submission, whole-task or joint predicted-subtask policies, fixed/external selector diagnostics, rollout progress, videos, success rate, or q-score analysis.
---

# Evaluate π0.5 on BEHAVIOR-1K

Reuse the repository launchers and validator. Do not recreate policy serving, simulator setup, result parsing, or Nebula packaging.

## Safety gates

1. Work from `/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge` and inspect `git status` before relying on local code.
2. Never read, print, copy, upload, summarize, or commit `nebula/config.sh`. Let `nebula/submit_pi05_eval.sh` source it internally. Keep shell tracing disabled and never print a complete remote environment.
3. Require explicit user approval immediately before an actual Nebula submission, job stop, output deletion, Git push, or other shared/destructive action. A prior submission approval or dry run does not authorize another submission.
4. Use a fresh run ID, output root, ports, and runtime roots. Never launch into an existing result directory; stale JSON can produce false completeness or duplicate rollouts.
5. Before launching locally, inspect all GPUs and compute processes. Never stop or reuse unrelated processes:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

Use `ps -o user,pid,lstart,cmd -p "$PID"` for ambiguous processes. Check intended ports independently.
6. Track exact task/process IDs. Never kill by broad process-name matching.
7. Treat `state_machine_v1` as a privileged diagnostic, not an official benchmark. It supports only `carrying_in_groceries` and `make_rose_centerpieces`.
8. Never repeat a Nebula submission while its status is ambiguous. Inspect the saved receipt, ordinary targeted AIHub status, and persistent output markers first. Never use `status --detail`.

## Select the conditioning mode

Keep checkpoint training semantics and inference semantics aligned.

| Evaluation | Required settings | Meaning |
|---|---|---|
| Whole-task action-only | `SUBTASK_PREDICTION=0`, `PROMPT_MODE=whole_task` | Whole task → actions |
| Joint predicted subtask | `SUBTASK_PREDICTION=1`, `MAX_TOKEN_LEN=512`, `MAX_SUBTASK_LEN=64` | Model predicts subtask and then actions |
| Fixed subtask, joint | Joint settings plus `EXTERNAL_SUBTASK='<text>'` | Diagnostic using one supplied skill |
| Correct-skill joint diagnostic | Joint settings plus `EXTERNAL_SUBTASK_SELECTOR=state_machine_v1` | State selector bypasses high-level prediction |
| Unsatisfied-goal oracle | `PROMPT_MODE=oracle_unsatisfied_goal` | Privileged BDDL prompt; not comparable to whole-task results |

External text and an external selector are mutually exclusive. Keep `PROMPT_MODE=whole_task` for external subtask conditioning.

For joint checkpoints, inspect `assets/subtask_conditioning.json`: preserve `task_and_subtask` for legacy joint checkpoints and `subtask_only` for factorized joint checkpoints. Do not override metadata inconsistently.

## Recover and verify the runtime

If `/dev/shm` environments, cache, simulator assets, or Vulkan manifests are missing, invoke `/restore-pi05-inference` or run:

```bash
.qoder/skills/restore-pi05-inference/scripts/restore.sh restore
```

Otherwise use its `check` mode. Prefer the persistent archives; do not rebuild environments while valid archives exist.

Keep the OpenPI cache on a local filesystem such as `/dev/shm`; OSS mounts may not support tokenizer lock operations. If copying a staged checkpoint to tmpfs for speed, use a new destination and verify `SHA256SUMS.inference` there before evaluation. Never overwrite persistent checkpoint siblings.

Confirm:

- the intended GPU count and driver;
- JAX sees the intended GPUs;
- OmniGibson, CUDA, Vulkan, and EGL checks pass;
- the checkpoint contains exactly one supported model format;
- `PI05_REPO_ID` matches `checkpoint/assets/<repo-id>/norm_stats.json`;
- staged checkpoints pass their checksum manifest and `_READY` contract.

## Run a local smoke

Use a fresh output directory and unique port. For an evaluation-contract smoke, complete one short rollout rather than stopping at `app ready`:

```bash
REPO=/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge
RUN_ID=<unique-run-id>
OUT="$REPO/eval_results/$RUN_ID"

timeout --signal=TERM --kill-after=30s 1800s env \
  MODEL=pi05 PI05_CKPT=<absolute-checkpoint> PI05_REPO_ID=<assets-repo-id> \
  PI05_ENV="$REPO/openpi/.venv" EVAL_ENV=/dev/shm/wangfaye/conda_envs/b1k-eval \
  OPENPI_DATA_HOME=/dev/shm/wangfaye/openpi-cache \
  OMNIGIBSON_DATA_PATH="$REPO/BEHAVIOR-1K/omnigibson-data" \
  SERVER_GPU=0 EVAL_GPU=0 PORT=8130 PI05_MEM_FRACTION=0.35 \
  INSTANCE_INDICES=0 NUM_ROLLOUTS=1 MAX_STEPS=100 WRITE_VIDEO=1 \
  PROMPT_MODE=whole_task \
  <conditioning-settings> \
  OUTPUT_ROOT="$OUT" \
  bash "$REPO/scripts/eval_b1k.sh" <task>
```

For selector smoke, choose one of the two supported tasks and add `EXTERNAL_SUBTASK_SELECTOR=state_machine_v1` plus the matching joint or action-only settings.

Require:

- server startup and `app ready`;
- one parseable rollout JSON and one nonempty video;
- finite q-score and no fatal CUDA/Vulkan/segfault signature;
- predicted joint: nonempty generated subtask events when generation is expected;
- external selector: nonempty selector/subtask events, source `provided`, valid selector summary, and `validated_action_requests == steps`.

The repeated Isaac/OmniGraph `No module named 'coverage'` test-extension messages can be nonfatal if rollout progress continues; do not ignore actual simulator termination or missing artifacts.

## Run two tasks locally on eight GPUs

Use the dynamic launcher for the formal two-task contract. It creates 40 `(task, instance)` jobs; each job loads one public-test instance once and runs rollout IDs `0,1,2`. Eight co-located workers claim the next available instance, which avoids the static-shard idle tail while preserving the existing three-rollout order.

Require all selected GPUs and ports to be idle, and use a fresh run root:

```bash
REPO=/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge
RUN_ROOT="$REPO/eval_results/<unique-run-id>"

OPENPI_DATA_HOME=/dev/shm/wangfaye/openpi-cache \
  python "$REPO/scripts/eval_b1k_dynamic.py" start \
    --run-root "$RUN_ROOT" \
    --checkpoint <absolute-checkpoint> \
    --repo-id <assets-repo-id> \
    --gpus 0 1 2 3 4 5 6 7 \
    --base-port 8140 \
    --max-attempts 2 \
    <conditioning-options>
```

Conditioning options are explicit and immutable:

- whole task: no additional options;
- joint prediction: `--subtask-prediction --max-token-len 512`;
- selector on a joint model: `--subtask-prediction --external-subtask-selector state_machine_v1 --max-token-len 512`.

The launcher fixes `PI05_MEM_FRACTION=0.35`, assigns one GPU and unique port/runtime directory per slot, isolates failed attempts, retries whole instances, and publishes only complete three-rollout bundles. `_SUCCESS` is written only after all 120 unique JSON/video pairs pass the strict validator.

Inspect progress without contacting worker processes:

```bash
python "$REPO/scripts/eval_b1k_dynamic.py" status --run-root "$RUN_ROOT"
```

After an interrupted or failed supervisor has stopped, resume the immutable run without rerunning successful instances:

```bash
python "$REPO/scripts/eval_b1k_dynamic.py" resume --run-root "$RUN_ROOT"
```

Do not resume while the recorded supervisor is active. The launcher refuses occupied GPUs and ports; inspect rather than killing unrelated processes.

Use the old modulo-eight static shards only as a fallback when diagnosing the scheduler itself. Do not use `eval_sweep.sh` for this two-task instance-parallel contract because it shards tasks rather than instances. Continue using `eval_sweep.sh` for broad task-list sweeps.

## Monitor local progress

Count only completed JSON files; a video can exist while its rollout is still running. Report:

- completed and expected JSON by worker;
- nonempty video count;
- active tracked workers;
- interim success count/rate and mean q-score, clearly labeled partial;
- per-task counts so partial results are not mistaken for a balanced final estimate.

Inspect log growth and selected signatures with dedicated search tools. Expected progress includes `app ready`, `Result:`, and `Eval summary:`. Treat `CUDA out of memory`, segmentation faults, ray-tracing pipeline failures, invalid Vulkan initialization, Python tracebacks after rollout startup, or a worker exit without its expected JSON count as failures.

Do not restart successful shards. For a failed shard, preserve evidence, determine the missing rollout keys, and obtain approval before a replacement run if it could duplicate or overwrite results.

## Submit through Nebula

Use `nebula/submit_pi05_eval.sh`; do not call the worker entry point directly. Supported profiles are `smoke`, `speed`, `two_tasks`, `reduced`, and `full`.

For the formal two-task profile, the launcher uses four co-located workers with modulo-four shards and expects 120 rollouts. A selector run is a privileged diagnostic.

Before submission:

1. Confirm the checkpoint is staged in a unique persistent directory with `SHA256SUMS.inference` verified and `_READY` written only after verification.
2. Confirm model format, repo ID, conditioning metadata, action horizon, and profile compatibility.
3. Choose a fresh `RUN_ID`, `JOB_NAME`, and persistent output root.
4. Check that neither the persistent output nor `.submission_receipts/<run-id>.*` exists.
5. Run a dry run:

```bash
DRY_RUN=1 KEEP_BUNDLE=1 \
  EVAL_PROFILE=<smoke-or-two_tasks> \
  RUN_ID=<unique-run-id> JOB_NAME=<unique-job-name> \
  CHECKPOINT_SOURCE=<staged-checkpoint> PI05_REPO_ID=<assets-repo-id> \
  PI05_MODEL_FORMAT=<jax-or-torch> \
  <conditioning-settings> \
  bash nebula/submit_pi05_eval.sh
```

Inspect only bundle filenames and contract output. Verify the bundle does not contain `nebula/config.sh`. Do not print file contents that may contain credentials.

After the dry run passes, summarize the exact checkpoint, profile, rollout count, worker/GPU topology, conditioning, output root, and cost/blast radius. Obtain explicit user approval, then repeat the same command without `DRY_RUN` and `KEEP_BUNDLE`.

Save and report the receipt path and submitted IID. If CLI output is interrupted or ambiguous, do not resubmit. Check the receipt and use ordinary targeted AIHub status only; never request detailed status or print the remote environment.

## Monitor and validate Nebula

Use the exact IID and persistent output root from the receipt. Require `_SUCCESS`; `summary.json` alone is insufficient. Inspect worker status files and targeted logs without dumping environments.

Expected persistent artifacts include:

```text
<output>/staging.log
<output>/manifest.json
<output>/status/
<output>/shards/worker_<n>/
<output>/parallel_metrics.json
<output>/summary.json
<output>/_RUNNING | _FAILED | _SUCCESS
```

Do not automatically resubmit `_FAILED`, incomplete, or ambiguous jobs. Diagnose first and ask before a new submission. Never delete completed formal results unless explicitly requested.

## Validate and summarize final results

Use the strict validator, not `scripts/summarize_closed_loop.py`:

```bash
"$REPO/openpi/.venv/bin/python" "$REPO/nebula/summarize_pi05_eval.py" \
  --results-root <run-root-or-shards-root> \
  --output <run-root>/summary.json \
  --run-id <run-id> --profile <profile> \
  --checkpoint <checkpoint> \
  --expected-rollouts 120 \
  --expected-tasks "carrying_in_groceries make_rose_centerpieces" \
  --expected-instance-ids "301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320" \
  --expected-rollout-ids "0 1 2" \
  --require-videos --require-complete \
  <mode-specific-validation-flags>
```

Add:

- predicted joint: `--require-subtask-events --require-generated-subtasks` when generation is required for every rollout;
- selector: `--require-subtask-events --require-selector-events`;
- fixed supplied text: `--require-subtask-events`.

For Nebula, also pass `--expected-workers` and `--status-root` when worker status files are available. Local manually sharded runs lack those status files, so validate worker counts separately against the shard table.

Require exactly 120 unique rollout keys, 120 nonempty videos, finite metrics, no invalid JSON, no missing/unexpected keys, and no duplicates. For selector runs, require matching provenance hashes, nonempty transitions, `source=provided`, and validated action request counts.

Report:

- total and per-task success count/rate;
- total and per-task mean q-score;
- rollout/video completeness;
- checkpoint identity and conditioning mode;
- for predicted policies, subtask-generation coverage;
- for selector diagnostics, phase/stall/recovery distributions and the explicit privileged/non-official caveat;
- comparison against only evaluations with matching task, instances, rollouts, prompt mode, and evaluator contract.

Do not infer closed-loop success from open-loop L1/RMSE, and do not treat partial-run metrics as final.