---
name: train-pi05-b1k
description: Safely prepare, preflight, launch, monitor, resume, and validate BEHAVIOR-Challenge π0.5 local JAX/Torch training and Nebula/AIHub submissions. Use for whole-task, joint, FrameSamp visual memory, multihost, checkpoint, or training-status requests.
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
4. Never read, print, copy, upload, or commit `nebula/config.sh`. Let the repository launcher source it internally. Never hardcode credentials or enable shell tracing.
5. Never call `ai-hub-cli train status --detail`; detailed metadata can expose worker environment values. Use ordinary targeted status only.
6. Proceed autonomously with preflighted training submissions, failed-job stops, necessary resumes/retries and Git pushes within the user-authorized competition workflow. The user has requested no repeated confirmations. Preserve needed checkpoints and formal results, use fresh attempts/output roots, and keep actions within the requested scope.

## Choose the training objective

Use `SUBTASK_PREDICTION=0` for ordinary action training and `1` for joint training.

| Objective | Entry point / switch | Text and action semantics |
|---|---|---|
| Whole-task action-only | `scripts/train_pi05_b1k.sh` with `SUBTASK_PREDICTION=0` and no `SUBTASK_PROMPTS` | Full task prompt → actions; no subtask CE |
| Legacy joint | `SUBTASK_ACTION_CONDITIONING=task_and_subtask scripts/train_pi05_subtask_b1k.sh` | Full task predicts subtask; action branch sees full task and GT subtask |
| Factorized joint | `scripts/train_pi05_subtask_b1k.sh` | Default `subtask_only`: full task predicts subtask; independent image/state/subtask branch predicts actions |
| Torch | `scripts/train_pi05_pytorch_b1k.sh` | Supports whole-task and joint |

Treat a joint checkpoint without `assets/subtask_conditioning.json` as legacy `task_and_subtask`. Do not resume it as factorized. Legacy resume must set `SUBTASK_ACTION_CONDITIONING=task_and_subtask` explicitly in the local or Nebula launcher and match the saved checkpoint contract.

Do not confuse the legacy action-only compatibility path (`SUBTASK_PROMPTS` set while `SUBTASK_PREDICTION=0`) with joint supervision.

Strict subtask-only and pure executor support has been removed. Do not use the deleted `SUBTASK_EXECUTOR`, `STRICT_SUBTASK_ACTION_CHUNKS`, `PROVIDED_SUBTASK_AS_PROMPT`, `scripts/train_pi05_subtask_executor_b1k.sh`, or removed Nebula profiles.

## Prepare subtask data

Set reusable absolute paths:

```bash
REPO=/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge
PY="$REPO/openpi/.venv/bin/python"
DATASET_ROOT=<absolute-dataset-root>
RUN_ROOT=<fresh-persistent-output-root>
SUBTASK_PROMPTS="$RUN_ROOT/subtask_segments.role_aware_v1.parquet"
```

Generate role-aware labels when the experiment calls for object-role information; use the legacy format to preserve existing label semantics. The implemented option is `--out`, not `--output`:

```bash
"$PY" "$REPO/scripts/gen_subtask_prompts.py" \
  --dataset "$DATASET_ROOT" \
  --task-indices 14 56 \
  --label-format role_aware_v1 \
  --out "$SUBTASK_PROMPTS"
```

Keep the generated segment table immutable after preflight.

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

## Bounded collection fine-tuning

For the low-success-subtask experiment initialized from verified checkpoint 1375000, use `nebula/submit_subtask_finetune.sh prepare` followed by `submit` after preflight, under the user's existing autonomous authorization. See `docs/subtask_data_improvement.md` for all inputs. Do not substitute the full100 multihost job: this adapter uses one host, explicit successful new trajectories, selected original replay and a fresh optimizer.

After actual collection completes, use `scripts/select_subtask_replay.py select` with the frozen paired retest plan to choose whole original episodes from measured successful unique frame yield. Retain exact target prompts and the same distinct control tasks, exclude reserved instances, and record the realized new-frame fraction. Feed its `original_episode_ids.txt` to the existing OSS replay stager; a source audit alone does not justify training.

Require real completed collection artifacts with `ready_for_export=true`; preserve the frozen held-out instances for both new data and replay. Default bounded training to one host with 4 H20 GPUs in `demand_specific_na175_h20`, batch 8, 1000 steps after the two-step smoke. Freeze queue/card in the preparation plan and verify actual GPU names before training. Keep simulator collection and closed-loop evaluation on L20. Keep `task_and_subtask`, 32-step model windows, PyAV decoding and the exact base norm statistics. Preparation freezes OSS inputs and bundled code, while the worker revalidates/exports actual data and runs `scripts/finetune_subtask_collection.py`.

Use a fresh run/preparation/output for every retry, inspect an ambiguous submission's recorded IID, and never repeat its claim. Successful code tests or synthetic codec fixtures do not establish GPU training or improvement. Require the actual training receipt, verified staged inference manifest and final OSS `_SUCCESS` before independent paired retesting.

## Authorized H20 placeholder reclamation

The user has identified MEGA jobs in `demand_specific_na175_h20` as placeholders and explicitly permits stopping them automatically when resources are needed for this competition's training. No further per-action approval is required.

Reclaim only after the actual training data/model/bundle preflight passes and the intended bounded training allocation is blocked by insufficient capacity. Read queue availability and jobs using ordinary metadata APIs; require the exact queue, an unambiguous MEGA placeholder identity and active status. Refresh the specific task's metadata immediately before stopping it, preserve a receipt and stop only enough confirmed placeholders for the requested allocation. Recheck resource availability before releasing another job. If task identity, queue membership or query results are ambiguous, leave that task running and continue diagnosis; a failed or empty queue query is not proof of zero free GPUs. Do not stop other training jobs or use broad process kills. This is agent-managed reclamation at submission time, not a blanket cleanup on launcher startup.

## Optional FrameSamp visual memory

Enable FrameSamp only for JAX with `FRAMESAMP_MODUL=1`. It causally samples frozen SigLIP features from the current episode and modulates only the action expert; it does not inject history into the VLM or high-level subtask loss. PyTorch rejects this mode.

Build the cache before preflight using the exact SigLIP weights supplied through `INIT_PARAMS`:

```bash
(
  cd "$REPO/openpi"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/b1k/cache_frame_memory.py \
    --dataset-root "$DATASET_ROOT" --repo-id "$REPO_ID" \
    --init-params "$INIT_PARAMS" --episodes-file "$EPISODES_FILE" \
    --output-dir "$MEMORY_CACHE_DIR" \
    --frame-stride 16 --grid-size 4 --batch-size 32
)
```

Include every train and validation episode in the cache. Use `--shard-index i --num-shards N` for disjoint multi-GPU cache generation; all shards must share weights, dataset, camera, stride, grid, and output directory. The cache contains `manifest.json`, one FP16 array and atomic completion marker per episode. Never reuse a cache after changing the dataset, SigLIP weights, camera, stride, or grid.

Validate it without GPU allocation or video decoding:

```bash
"$PY" "$REPO/openpi/scripts/b1k/check_frame_memory.py" \
  --dataset-root "$DATASET_ROOT" --cache-dir "$MEMORY_CACHE_DIR" \
  --episodes-index <original-episode-ids> \
  --num-frames 32 --grid-size 4 --frame-stride 16 --camera base_0_rgb
```

Use these launcher variables:

| Variable | Default | Contract |
|---|---:|---|
| `FRAMESAMP_MODUL` | `0` | Set to `1` to enable memory |
| `MEMORY_CACHE_DIR` | none | Completed cache path |
| `MEMORY_NUM_FRAMES` | `32` | Maximum uniformly sampled history frames |
| `MEMORY_GRID_SIZE` | `4` | Pooling grid side; 16 tokens per frame |
| `MEMORY_FRAME_STRIDE` | `16` | Historical candidate interval in original control frames |
| `MEMORY_CAMERA` | `base_0_rgb` | Historical camera |

Do not confuse `MEMORY_FRAME_STRIDE` with the action-training `FRAME_STRIDE`. The former controls cached history candidates; the latter selects action-training anchors.

When adding memory to an existing non-memory model, create a new `EXP_NAME`, set `INIT_PARAMS=<old-checkpoint>/params`, and do not use full-state resume; only the new memory parameters are initialized. Resume an existing memory experiment only with identical memory configuration and cache provenance. Require `assets/frame_memory.json` in its checkpoints. Changing only `MEMORY_NUM_FRAMES` may reuse the cache but still requires a new experiment.

The implementation has CPU/unit coverage but no committed evidence of a full-data GPU smoke, multihost throughput, or closed-loop gain. Run a target-hardware smoke and measure finite metrics, peak memory, and steady-state step time before formal training. See `docs/frame_memory_training.md` for architecture and evaluation details.

## Run exact preflight

Use `check` before normalization or model initialization:

```bash
# Joint (set task_and_subtask explicitly for legacy)
bash "$REPO/scripts/train_pi05_subtask_b1k.sh" check
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
ENTRYPOINT="$REPO/scripts/train_pi05_subtask_b1k.sh"  # choose from the objective table
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
| `full100` | `demand_general`, 2×8 H200/L20Z | 192 | 2 epochs / 1,688,978 steps |

Current new joint submissions use factorized conditioning; historical joint checkpoints may be legacy.

For FrameSamp submission, pass `FRAMESAMP_MODUL=1`, `MEMORY_CACHE_DIR`, all `MEMORY_*` contract values, and `INIT_PARAMS` to the existing command. The cache path must be mounted read-only at the same location on every worker; the submission does not generate or upload it, and each worker validates cache completeness before training.

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

After the concrete submission preflight passes, repeat the exact command without `DRY_RUN` and `KEEP_BUNDLE` under the ongoing task authorization; do not ask for per-submission confirmation. Use separate run/output roots for `discover`, `preflight`, `smoke`, and `full` unless resuming:

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

Resume with the same `RUN_ID`, `OUTPUT_ROOT`, `EXP_NAME`, `REPO_ID`, and `DATASET_PROFILE`; set `MODE=full RESUME=1`; create a fresh `ATTEMPT_ID` and `JOB_NAME`. Dry-run the concrete resume configuration and submit under the ongoing authorization without repeated confirmation.

Only stop a job after an explicit request. First wait for and verify the desired complete checkpoint, then use ordinary status and:

```bash
ai-hub-cli train --token "$AI_HUB_TOKEN" --namespace "$AI_HUB_NAMESPACE" \
  stop --iid "$IID"
```

Confirm it is no longer running. Never delete or rewrite output roots as part of stop/resume.

## Report

Report the objective and conditioning mode, dataset/annotation/episode hashes, retained episodes and samples, global/per-device batch, topology, epochs/steps, initialization or resume checkpoint, output/log paths, first finite metrics, current status, and latest complete/restored checkpoint. For FrameSamp, also report the cache path, encoder/dataset fingerprints, camera, stride, grid, frame budget, cache completeness, and whether the run is a weight migration or full-state resume. State explicitly when preflight passed but optimizer training never began.
