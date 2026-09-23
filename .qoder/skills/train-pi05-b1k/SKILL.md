---
name: train-pi05-b1k
description: Safely prepare, preflight, launch, monitor, resume, and validate BEHAVIOR-Challenge π0.5 local JAX/Torch training and Nebula/AIHub submissions. Use for whole-task, joint, strict subtask-only, pure executor, multihost, checkpoint, or training-status requests.
---

# Train π0.5 on BEHAVIOR-1K

Use the repository's launchers; do not recreate their data loading, distributed setup, or checkpoint logic.

## Safety gates

1. Work from `/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge` and inspect `git status` before relying on current code.
2. Before local training, inspect every GPU and compute process:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

Use `ps -o user,pid,lstart,cmd -p "$PID"` for ambiguous processes. Never stop, reuse, or oversubscribe a device owned by an unrelated process. Recheck immediately before launch.

3. Give every new run a fresh `EXP_NAME`, log path, assets ID, checkpoint root, and Nebula output root. Reuse them only for an intentional resume.
4. Never read, print, copy, upload, or commit `nebula/config.sh`. Let the approved launcher source it internally. Never hardcode credentials or enable shell tracing.
5. Never call `ai-hub-cli train status --detail`; detailed metadata can expose worker environment values. Use ordinary targeted status only.
6. Require explicit user approval before an actual Nebula submission, job stop, checkpoint deletion, Git push, or another externally visible/destructive action. A dry run is not approval to submit.

## Choose the training objective

The following JAX switches are mutually exclusive: `SUBTASK_PREDICTION`, `SUBTASK_EXECUTOR`, and `STRICT_SUBTASK_ACTION_CHUNKS`.

| Objective | Entry point / switch | Text and action semantics |
|---|---|---|
| Whole-task action-only | `scripts/train_pi05_b1k.sh` with all three switches `0` | Full task prompt → actions; no subtask CE |
| Legacy joint | `SUBTASK_ACTION_CONDITIONING=task_and_subtask scripts/train_pi05_subtask_b1k.sh` | Full task predicts subtask; action branch sees full task and GT subtask |
| Factorized joint | `scripts/train_pi05_subtask_b1k.sh` | Default `subtask_only`: full task predicts subtask; independent image/state/subtask branch predicts actions |
| Strict subtask-only | `STRICT_SUBTASK_ACTION_CHUNKS=1 scripts/train_pi05_b1k.sh` | Current GT subtask replaces task prompt; action loss only; reject every anchor whose full horizon crosses a segment boundary |
| Pure executor | `scripts/train_pi05_subtask_executor_b1k.sh` | Current GT subtask prompt; action loss only; retain boundary anchors and mask steps outside the segment or episode |
| Torch | `scripts/train_pi05_pytorch_b1k.sh` | Supports whole-task and joint; this shell path does not expose strict or pure executor |

Treat a joint checkpoint without `assets/subtask_conditioning.json` as legacy `task_and_subtask`. Do not resume it as factorized. Local legacy resume must set `SUBTASK_ACTION_CONDITIONING=task_and_subtask`; the current Nebula profile does not carry that override, so do not resume an old legacy Nebula run with the current default bundle without first adding and testing explicit support.

Do not confuse the legacy action-only compatibility path (`SUBTASK_PROMPTS` set while all three switches are zero) with joint supervision.

## Prepare subtask data

Set reusable absolute paths:

```bash
REPO=/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge
PY="$REPO/openpi/.venv/bin/python"
DATASET_ROOT=<absolute-dataset-root>
RUN_ROOT=<fresh-persistent-output-root>
SUBTASK_PROMPTS="$RUN_ROOT/subtask_segments.role_aware_v1.parquet"
```

Generate role-aware labels for pure executor training. The implemented option is `--out`, not `--output`:

```bash
"$PY" "$REPO/scripts/gen_subtask_prompts.py" \
  --dataset "$DATASET_ROOT" \
  --task-indices 14 56 \
  --label-format role_aware_v1 \
  --out "$SUBTASK_PROMPTS"
```

Pure executor requires the role/provenance columns. Keep the generated segment table immutable after preflight.

First validate the requested episode set and write the retained original IDs with `openpi/scripts/b1k/check_subtask_data.py`. Pass explicit episode IDs when the segment table contains only selected tasks; otherwise the checker will correctly reject unannotated episodes elsewhere in the dataset.

Create a held-out split only when the user wants offline validation:

```bash
"$PY" "$REPO/openpi/scripts/b1k/split_subtask_episodes.py" \
  --dataset-root "$DATASET_ROOT" \
  --retained-episode-file "$RETAINED_EPISODES" \
  --task-indices 14 56 \
  --output-dir "$SPLIT_DIR" \
  --seed 1 --val-fraction 0.2
```

The split keeps each `(task_index, task_instance_id)` group on one side. It refuses a nonempty output directory. For training all retained episodes, pass `EPISODES_FILE=$RETAINED_EPISODES` and omit `VAL_EPISODES_FILE`.

## Decide pure-executor sampling explicitly

The pure-executor wrapper defaults to at most 64 evenly spaced anchors per annotated segment:

```text
MAX_SAMPLES_PER_SEGMENT=64
```

This balances long and short subtasks but can reduce millions of eligible frame anchors to hundreds of thousands. “All episodes” does not mean “all anchors.” Report both `eligible_action_anchors` and `capped_sample_count` before launch.

For the cap, use `scripts/train_pi05_subtask_executor_b1k.sh`. For every annotated anchor, bypass that wrapper and invoke the base launcher with:

```bash
SUBTASK_EXECUTOR=1 \
SUBTASK_PREDICTION=0 \
STRICT_SUBTASK_ACTION_CHUNKS=0 \
WHOLE_TASK_PROB=0 \
MAX_SAMPLES_PER_SEGMENT= \
  bash "$REPO/scripts/train_pi05_b1k.sh" full
```

Do not choose capped versus all-anchor sampling without telling the user the resulting sample and step counts.

## Run exact preflight

Use `check` before normalization or model initialization:

```bash
# Joint (set task_and_subtask explicitly for legacy)
bash "$REPO/scripts/train_pi05_subtask_b1k.sh" check

# Pure executor with the wrapper's 64-anchor cap
bash "$REPO/scripts/train_pi05_subtask_executor_b1k.sh" check

# Strict subtask-only
STRICT_SUBTASK_ACTION_CHUNKS=1 PREFLIGHT_ONLY=1 \
  bash "$REPO/scripts/train_pi05_b1k.sh" full
```

Supply the same `DATASET_ROOT`, `SUBTASK_PROMPTS`, `EPISODES_FILE`, `BATCH_SIZE`, device/process topology, and output paths intended for training.

For exact epochs:

1. Read the JSON report's retained episodes/frames, training samples, batches per epoch, train steps, episode-list SHA, and contract SHA.
2. Set `TRAIN_EPOCHS`, `TRAIN_STEPS=<epochs*batches_per_epoch>`, and `LR_DECAY_STEPS=$TRAIN_STEPS`.
3. Set all applicable `EXPECTED_*` values (`TASKS`, `TASK_INDICES`, `RETAINED_EPISODES`, `RETAINED_FRAMES`, `TRAINING_SAMPLES`, `BATCHES_PER_EPOCH`, `TRAIN_STEPS`).
4. Rerun preflight and abort on any mismatch.

Global batch determines step count; GPU count only changes per-device batch when global batch is fixed.

## Launch locally

Use explicit paths and a fresh tmux session. Prefer persistent NAS output, and initialize a new experiment from an explicit base checkpoint:

```bash
SESSION=<unique-session>
ENTRYPOINT="$REPO/scripts/train_pi05_subtask_executor_b1k.sh"  # choose from the objective table
LOG="$RUN_ROOT/train.log"

# Recheck nvidia-smi here.
tmux new-session -d -s "$SESSION" bash -lc \
  "set -o pipefail; cd '$REPO'; env \
    DATASET_ROOT='$DATASET_ROOT' OPENPI_DATA_HOME='<cache-root>' \
    SUBTASK_PROMPTS='$SUBTASK_PROMPTS' EPISODES_FILE='$EPISODES_FILE' \
    INIT_PARAMS='<pi05-base-params>' REPO_ID='$REPO_ID' EXP_NAME='$EXP_NAME' \
    ASSETS_BASE_DIR='$ASSETS_BASE_DIR' CHECKPOINT_BASE_DIR='$CHECKPOINT_BASE_DIR' \
    NUM_GPUS='$NUM_GPUS' CUDA_DEVICES='$CUDA_DEVICES' FSDP_DEVICES='$FSDP_DEVICES' \
    BATCH_SIZE='$BATCH_SIZE' TRAIN_EPOCHS='$TRAIN_EPOCHS' TRAIN_STEPS='$TRAIN_STEPS' \
    LR_DECAY_STEPS='$TRAIN_STEPS' LOG_INTERVAL=1 \
    OPENPI_REQUIRE_FINITE_METRICS=1 OPENPI_VERIFY_FINAL_CHECKPOINT=1 \
    OPENPI_SUCCESS_PATH='$RUN_ROOT/_SUCCESS.json' \
    bash '$ENTRYPOINT' full 2>&1 | tee '$LOG'"
```

Add objective-specific variables and the exact `EXPECTED_*` contract. Do not reuse an existing checkpoint directory with a new experiment.

Monitor the tracked tmux session and log, not broad process names. Require the first optimizer update to report finite `action_loss`, `loss`, `grad_norm`, and `param_norm`:

- pure executor: also require `0 < valid_action_fraction <= 1` and no subtask CE metrics;
- joint: require finite `subtask_loss`, positive `subtask_tokens`, and finite token accuracy;
- action-only modes: require total loss to match action loss within logging precision.

At completion, require `_SUCCESS.json` and final restore verification. For JAX, physical checkpoint directory `TRAIN_STEPS-1` restores to logical `train_state.step == TRAIN_STEPS`. Torch checkpoint directories use completed optimizer updates, so the final directory is `TRAIN_STEPS`.

For JAX resume, keep the same `EXP_NAME`, paths, topology, batch, objective, conditioning contract, and dataset contract; set `RESUME=1` and treat `TRAIN_STEPS` as the new total target. Never combine `RESUME=1` with overwrite.

## Submit through Nebula

Use `nebula/submit_pi05_multihost.sh`. Current pinned profiles are:

| Profile | Queue/topology | Global batch | Exact full run |
|---|---|---:|---:|
| `two_tasks` | development, normally 2×8 | 64 | unpinned/non-production |
| `two_tasks_joint_h200` | `demand_general`, 2×8 H200/L20Z | 256 | 400 episodes / 4 epochs / 58,768 steps |
| `two_tasks_subtask_only_h200` | `demand_general`, 2×8 H200/L20Z | 256 | 3 epochs / 28,587 steps |
| `two_tasks_subtask_only` | H20 queue, 1×8 H20 | 64 | 3 epochs / 114,357 steps |
| `full100` | `demand_general`, 2×8 H200/L20Z | 192 | 2 epochs / 1,688,978 steps |

The two `subtask_only` profiles are strict full-horizon filtering, not pure executor. There is currently no pure-executor Nebula profile. Current new joint submissions use factorized conditioning; historical joint checkpoints may be legacy.

Before any actual submission, build and inspect a dry-run bundle:

```bash
DRY_RUN=1 KEEP_BUNDLE=1 \
DATASET_PROFILE="$PROFILE" MODE="$PHASE" \
RUN_ID="$RUN_ID" ATTEMPT_ID="$ATTEMPT_ID" JOB_NAME="$JOB_NAME" \
EXP_NAME="$EXP_NAME" OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash "$REPO/nebula/submit_pi05_multihost.sh"
```

Inspect only bundle filenames and verify the secret config is absent:

```bash
find "$BUNDLE_DIR" -type f -printf '%P\n' | LC_ALL=C sort
test ! -e "$BUNDLE_DIR/nebula/config.sh"
```

Use only known cluster JSON files: normally `nebula/multihost_cluster.json`; use `nebula/multihost_cluster_4gpu.json` only for a compatible development contract. Do not pass arbitrary files through `CLUSTER_FILE`.

After explicit submission approval, repeat the exact command without `DRY_RUN` and `KEEP_BUNDLE`. Use separate run/output roots for `discover`, `preflight`, `smoke`, and `full` unless resuming:

```text
discover → preflight → smoke → full
```

Validate each phase from shared outputs:

- discover: `_DISCOVERY_OK.json`;
- pinned preflight: `_PREFLIGHT_OK.json`, distinct hosts, expected local/global devices, mesh, and identical rank contracts;
- smoke/full: finite metrics, no rank/NCCL/rendezvous failure, complete checkpoint, and `_SUCCESS.rank0.json`.

Use ordinary targeted status only, with credentials already available through the approved secret mechanism:

```bash
ai-hub-cli train --token "$AI_HUB_TOKEN" --namespace "$AI_HUB_NAMESPACE" \
  status --iid "$IID"
```

Never append the detailed metadata flag. Prefer the rank-0 log and attempt-scoped files under:

```text
$OUTPUT_ROOT/logs/$ATTEMPT_ID/
$OUTPUT_ROOT/contracts/$ATTEMPT_ID/
$OUTPUT_ROOT/rendezvous/$ATTEMPT_ID/
```

## Resume or stop Nebula safely

A Nebula resume is a new submission, not an AIHub restart. Before resuming, verify the latest numeric Orbax directory contains:

```text
_CHECKPOINT_METADATA
params/_METADATA
train_state/_METADATA
assets/
```

Also verify all ranks agree on the latest step and that the original profile, dataset, topology, batch, repo ID, conditioning mode, and contract are unchanged.

Resume with the same `RUN_ID`, `OUTPUT_ROOT`, `EXP_NAME`, `REPO_ID`, and `DATASET_PROFILE`; set `MODE=full RESUME=1`; create a fresh `ATTEMPT_ID` and `JOB_NAME`. Dry-run and obtain approval again before submitting.

Only stop a job after an explicit request. First wait for and verify the desired complete checkpoint, then use ordinary status and:

```bash
ai-hub-cli train --token "$AI_HUB_TOKEN" --namespace "$AI_HUB_NAMESPACE" \
  stop --iid "$IID"
```

Confirm it is no longer running. Never delete or rewrite output roots as part of stop/resume.

## Report

Report the objective and conditioning mode, dataset/annotation/episode hashes, retained episodes and samples, anchor cap, global/per-device batch, topology, epochs/steps, initialization or resume checkpoint, output/log paths, first finite metrics, current status, and latest complete/restored checkpoint. State explicitly when preflight passed but optimizer training never began.
