# Codex-assisted full-task data: isolated training pilot

The experiment collects only instances 311–320, mixes verified successful full
trajectories with original demonstrations for the selected tasks, and initializes
joint subtask/action fine-tuning from checkpoint 1688977. Instances 301–310 are
reserved for evaluation with the learned policy and no collection teacher.

The selected tasks are `turning_on_radio` and `putting_shoes_on_rack`. The teacher
is **gpt-6.1-sol, medium**, using the explicitly selected
`~/.codex/auth_weizhi.json` ChatGPT account, without automatic account failover.
Its CLI executes inside the user's existing SenseCore image, with a private Codex
home and a worker-owned loopback mihomo process. It does not rely on a Mac relay.
The source proxy configuration is copied privately for the worker and is not
modified; credentials and proxy secrets are excluded from code/data bundles.

The first radio 311 attempt is bounded to 32 teacher decisions, 4500 control
steps and 7200 seconds, with a normal teacher interval of 256 steps. Correction
completion can trigger an earlier decision. First validate training and paired evaluation with one full-task success per
selected task; five successes per task remain a later expansion target. Keep
at most ten attempts per task and 640 teacher decisions overall. Shoes collection is capped at 10000 steps per
attempt; subsequent code-v8 shoes attempts use a 320-step normal observation
interval so 32 continue decisions can cover the cap. Corrections can consume
that budget sooner. Radio keeps its 256-step interval. These limits are distinct from official evaluation limits. See
[collection details](codex_teacher_collection.md).

**Status at 2026-10-11:** implementation and CPU contract tests are completed;
the live cluster proxy, selected ChatGPT account, and one Sol medium call have
been verified. All six original demonstrations passed real reader validation,
using the existing training reader's 5 ms timestamp tolerance for shoe videos.
Radio 311 `pilot-001` failed during startup with zero actions and zero teacher
decisions. `pilot-002` executed 3339 actions and 17 validated teacher calls,
then stopped at an unexecuted IK correction rejection. `pilot-003` completed
4500 actions and 23 teacher calls without task success. Its two rejected IK
corrections were returned for replanning without advancing the simulator; the
complete trajectory and event audit passed, but it is not training eligible.
Radio 317 `collect-001` then executed 1792 actions and 8 validated teacher calls,
but its first gripper correction was blocked by a confirmed interface mismatch:
the correction wrapper rejected a finite raw VLA gripper command before applying
the clipping already used by the robot controller. It is not training eligible.
A corrected native-policy collection overlay (`collection_code_v9_r2.tar.gz`,
SHA256 `46bb77c810fde0cbdd5779a7cf49658b2066540edad71238fc86726d885fa512`)
was submitted for a fresh radio 317 `native-001` attempt on one reserved GPU.
It preserves the model's own subtask labels and action execution during ordinary
segments; Codex supplies only bounded explicit corrections. The first v9
candidate was never installed or submitted. There are no new successful collection
trajectories, and this experiment has not yet run GPU
training or a trained-policy paired evaluation. Original-data readiness must be
established by `_ORIGINAL_DATA_READY.json`; downloads or unit tests alone do not
establish it.

This path is separate from `finetune_subtask_collection.py`, whose experiment is
fixed to checkpoint 1375000 and fixed-subtask trajectories. No old optimizer is
resumed, no existing checkpoint or dataset is overwritten, and these helpers do
not submit cloud jobs.

Ordinary finite ACP batches default to Spot. The user additionally authorized
reserved resources for experimental validation, subject to a fresh audit of
their current standard allocation and applicable quota. This does not change the
default for all later batches or authorize stopping other jobs.

Teacher CLI calls narrow known tool surfaces through per-call configuration,
disable configured MCP servers, and use a read-only sandbox. Completed JSON
events are then audited and decisions containing tool/failure/unknown events are
rejected. The event audit is an acceptance check after the process finishes; it
is not proof that all CLI versions have no tools or that a prior tool execution
was prevented.

## Required inputs

- Successful `CodexCollectionTrajectory` HDF5 files, with environment-confirmed
  full-task success, dynamic subtask labels, and the actual executed controls.
- Original LeRobot v3 data **and videos** for explicitly selected retained train
  episodes. Metadata alone cannot produce a training batch. Each collected task
  must have original replay; original replay excludes all public instances
  301–320. Labels must exactly cover the complete original episodes.
- The verified 1688977 JAX parameters and their matching normalization assets.
  The deployed AOSS checkpoint uses `_READY.checkpoint_step`; pass the original
  `aoss_checkpoint_upload_plan.json` to verify every parameter/asset SHA256. Its
  hash is distinct from `_PUBLIC_TEST_SOURCE_READY`, which includes code and
  other runtime provenance.
- The actual repository LeRobot/video and OpenPI JAX training environment.

The reusable original-data root is
`/mnt/afs/260010165/behavior_1k/codex_pilot_20261011/original-v3`.
Its `selected_episodes.txt` identifies the retained selection, and
`labels/subtask_segments.parquet` contains the original labels. The collection
code bundle layers independent files over the immutable base source and reuses
this data; it does not transfer the original dataset again. The aggregate reader
receipt and prepared teacher demonstrations are verified separately at
`original-reader-v4`; its readiness receipt explicitly identifies `original-v3`
as the underlying dataset root.

The baseline normalization asset ID is
`b1k_full100_joint_subtask_v1_drop_episode_s1`. State and action statistics are
copied byte-for-byte from that baseline; normalization is not recomputed on the
small mixture.

## Export and real reader gate

The frozen `code-v8` overlay retains the training reader-tolerance fix from
`code-v7`; only collection and correction code differs. Its bundle SHA256 is
`b2f054f13ccc76f2db46fa990e5d40ef3bfc8ca4cfaaae18baf6b231f42803e0`.
Installation is part of the submitted recovery job. Verify its actual code-ready
receipt and source hashes before training; do not modify the running overlay. Set the Bash array
`SUCCESS_TRAJECTORIES` to explicit environment-verified successful HDF5 files.
The current radio attempt's candidate path is
`/mnt/afs/260010165/behavior_1k/codex_pilot_20261011/collection/radio-311-pilot003/attempt/trajectory.hdf5`;
its existence alone is not success evidence.

The current native collection uses the independently frozen `code-v9` r2 overlay
described above. Its HDF5 files preserve mixed native/teacher label provenance.
Use `code_version: "v9"` with `run_codex_pilot_data_preparation.py` after successful
audits, and use `TRAINING_CODE="$PILOT/code-v9"` for the equivalent manual reader
commands below. The older reader does not accept native provenance. The training
worker and evaluation source remain frozen; evaluation never loads the teacher.

The six-line `selected_episodes.txt` below is valid only after successful new
trajectories exist for **both** tasks. If only radio has succeeded, create a
separate fresh episode-list file containing `28`, `68`, and `102`, and point
`BASE_EPISODES` at that file. Preserve the original six-line selection. The
mixed exporter rejects original tasks absent from the new successful data.

```bash
PILOT=/mnt/afs/260010165/behavior_1k/codex_pilot_20261011
RUNTIME=/mnt/afs/260010165/behavior_1k/runtime_20261007_1306
PYTHON="$RUNTIME/envs/b1k-eval/bin/python"
TRAINING_CODE="$PILOT/code-v8"
BASE_EPISODES="${BASE_EPISODES:-$PILOT/original-v3/selected_episodes.txt}"
MIXED="$PILOT/mixed-lerobot-pilot001"
export PYTHONDONTWRITEBYTECODE=1
(( ${#SUCCESS_TRAJECTORIES[@]} > 0 )) || exit 1

"$PYTHON" "$TRAINING_CODE/scripts/prepare_codex_training_data.py" export \
  --trajectories "${SUCCESS_TRAJECTORIES[@]}" \
  --base-dataset "$PILOT/original-v3" \
  --base-repo-id behavior-1k/2026-challenge-demos \
  --base-segments "$PILOT/original-v3/labels/subtask_segments.parquet" \
  --base-episodes "$BASE_EPISODES" \
  --output "$MIXED" --repo-id b1k_codex_pilot

"$PYTHON" "$TRAINING_CODE/scripts/prepare_codex_training_data.py" verify \
  --dataset-root "$MIXED" \
  --report "$PILOT/mixed-reader-verification-pilot001.json"
```

These output names are proposed fresh paths; they have not been created. The
original reader receipt is
`$PILOT/original-reader-v4/_ORIGINAL_DATA_READY.json`, while the LeRobot data stays
at `original-v3`. Do not pass `original-reader-v4` as `--base-dataset` or use the
old collection bundle's `original_data=original-v2` field to select training data.
Running `export_codex_collection.py` first is unnecessary: the mixed exporter
accepts successful collection HDF5 files directly.

Both sources are rewritten to one fresh dense episode index space, with preserved
task-instance grouping and dynamic labels. The report records the actual new
frame fraction; ordinary shuffled frame sampling then uses this mixture. There
is no implicit trajectory duplication or hidden reweighting.

The reader gate compares **all** numeric observations and actions to the original
sources. It checks a 32-action training horizon, terminal padding, frame/task IDs
and complete subtask coverage. It decodes cameras at every subtask boundary plus
the first, middle and final frames, comparing RGB with source frames under a
mean absolute codec tolerance of 12/255. A mock reader/writer is never accepted
as evidence for training. The export report remains a preparation record; actual
reader evidence is generated separately and rerun during training preparation.
Export, mixed-data readback, original-source readback, and the `pi05_b1k` training
configuration all use `tolerance_s=0.005`. Export and verification reports record
`reader_tolerance_s`; verification rejects a different value. Timestamp errors
outside that bound still fail without frame skipping or fallback.

Use the existing `runtime_20261007_1306/envs/b1k-eval/bin/python` for data
preparation and the training-helper parent process: this environment has the
verified HDF5, LeRobot, PyAV and YAML dependencies. Set `PI05_ENV` explicitly to
`/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05`; the helper's
JAX child activates that environment, and the script rejects an unset value.
These are separate environments; the pi05 environment does not
have an identical data-preparation dependency set. The training child explicitly
uses this experiment's `openpi/src` and client paths, overriding stale
editable-install pointers. This arrangement does not modify shared environments.

## Finite smoke and training

```bash
export PI05_ENV="$RUNTIME/envs/b1k-pi05"
export OPENPI_DATA_HOME="$RUNTIME/openpi-cache"
export PYTHONDONTWRITEBYTECODE=1
TRAIN_RUN="$PILOT/training/pilot001"

"$PYTHON" "$TRAINING_CODE/scripts/finetune_codex_collection.py" prepare \
  --dataset-root "$MIXED" \
  --checkpoint /mnt/afs/260010165/behavior_1k/checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010 \
  --checkpoint-manifest "$PILOT/bundle-v2/checkpoint_upload_plan.json" \
  --expected-manifest-sha256 156f0508b558f150402072e393ceca840554d3f345b3414c81c138fe1ed84a20 \
  --base-repo-id b1k_full100_joint_subtask_v1_drop_episode_s1 \
  --run-root "$TRAIN_RUN" --run-id codex_pilot001 \
  --devices 4 --batch-size 4 --steps 256

"$PYTHON" "$TRAINING_CODE/scripts/finetune_codex_collection.py" run \
  --plan "$TRAIN_RUN/training_plan.json" --phase smoke
"$PYTHON" "$TRAINING_CODE/scripts/finetune_codex_collection.py" run \
  --plan "$TRAIN_RUN/training_plan.json" --phase full
"$PYTHON" "$TRAINING_CODE/scripts/finetune_codex_collection.py" stage \
  --plan "$TRAIN_RUN/training_plan.json" --destination "$PILOT/inference/pilot001"
```

The required smoke trains two steps and restores the actual saved state. The full
phase starts again from 1688977 with a fresh optimizer and requires a successful
smoke receipt for the exact same model, data and device configuration. The pilot
defaults to 256 full steps, learning rate 1e-5 decaying to 1e-6, seed 42, full-task
plus subtask action conditioning, subtask CE weight 1.0, 32-action model horizon
and a global batch of four on four FSDP devices. Configured full steps are bounded
to 128–5000.

The exact checkpoint contains 3,353,433,872 float32 parameters. Parameters,
Adam's two moment arrays and the default EMA require about **49.97 GiB** before
gradients or activations. Therefore one 32 or 48 GiB device cannot run this
unchanged full-model recipe. The four-device default shards that persistent
state to about 12.51 GiB per GPU (15.64 GiB including gradients), before unmeasured
activations, communication buffers and compilation workspace. This estimate
comes from global Zarr array metadata in the actual checkpoint, not compressed
file size or the per-writer chunks in `_METADATA`.

The two-step smoke on the actual four-device allocation remains the feasibility
gate. An OOM cannot produce a completed training receipt or trigger automatic
retries on unrelated GPUs. Cluster execution also needs a finite job wall-time
limit. A deliberately different device plan must be prepared under a fresh
experiment root and pass its own smoke before full training.

The verified existing four-GPU template is `pt-4ewqyd0e` in
`computing-cluster-5090-01g`, with SKU `n12lp.nn.i10a.4`: one worker, four 32 GiB
GPUs, 56 CPUs and 240 GiB memory. This verifies the resource shape, not current
availability, a new allocation, or memory feasibility of this training recipe.
The collection cloud helper accepts only zero or one GPU. Training therefore
requires a separate finite four-GPU job using this verified resource shape;
do not reuse the one-GPU collection submission as the training allocation.

Only a successfully trained and restored checkpoint can be staged to a fresh
inference directory. Staging emits a full checksum manifest and `_READY`, with
the original step 1688977 and the new run's own step recorded separately. The
training plan records selected tasks, 311–320 collection, 301–310 evaluation,
and `evaluation_agent_enabled=false` for the paired no-Agent evaluation runner.

Final staging reruns real mixed-data readback, verifies that trained parameter
files differ from the base checkpoint, checks copied parameter hashes, and
preserves the base normalization byte hash. Only after those gates does
`training_receipt.json` use the canonical kind
`codex_pilot_trained_checkpoint_v1`. Its checkpoint ID is
`pi05_b1k_codex_<run_id>_<steps>`; it records `parent_checkpoint_step=1688977`,
positive `optimizer_steps`, `ready_for_evaluation=true`,
`real_lerobot_readback_verified=true`, `base_normalization_preserved=true`, the
`base_norm_sha256`, and a `checkpoint_manifest` reference with absolute path,
SHA256 and byte size. `_READY.restored_train_state_step` must equal the actual
optimizer steps. The receipt also references the frozen plan, full-phase receipt
and checkpoint restore proof. A smoke checkpoint cannot produce this receipt.

## Paired evaluation without the teacher

`sco/run_codex_pilot_evaluation.py` requires that canonical training receipt and
the complete inference file manifest. It runs the original immutable public-test
source snapshot, with `_PUBLIC_TEST_SOURCE_READY` SHA256
`7f2a16738961dfb1ca5b95d280ee61450f996757c5514b8c144e86dce3c982d4`.
The collector overlay is not the evaluator source. The child environments remove
teacher, collection, provided-subtask and API-key settings; this entry point
imports no teacher module and starts no Codex process.

The fixed panel contains both tasks on 301–310, one rollout each. To match the
baseline's seed-0 process scope, use five independently started policy/evaluator
shards in this order: radio 301; radio 302–305; radio 306–310; shoes 301–305; shoes
306–310. Each server uses native subtask generation, `task_and_subtask`, and an
execution horizon of 16. The original evaluator determines the official limits:
3225 steps for radio and 11590 for shoes. No collection limit overrides them.

`sco/submit_codex_pilot_evaluation.py` creates these five one-GPU jobs in that
fixed order under one panel lock, with up to four jobs active concurrently on
the same trained checkpoint. Each submission checks every prior exact job UID
and freshly queries all owner standard-GPU usage. Both capacity checks combine
the owner list with known active panel jobs by UID, counting each UID once and
rejecting conflicting identities or GPU allocations. They admit one job only
when that combined usage is at most three GPUs and the panel has fewer than
four active jobs.
A full initial capacity check creates no shard intent. After saving the unique
intent and verifying the uploaded source, a second capacity check must pass.
A capacity conflict at that point records `not_submitted` and retains the intent
without an automatic retry. Unknown submission outcomes block further jobs.
Each job retains its four-hour outer limit, and the final comparison still
requires all five receipts in the declared order, regardless of completion order.

`scripts/evaluate_codex_pilot.py` verifies all 20 original baseline result files,
all candidate result files, the five ordered process receipts and their model,
runtime and no-Agent provenance before comparison. The retained baseline is
radio 6/10 and shoes 1/10; it is not a new-model result. Preserve every outcome,
including failures, with the predeclared task/instance pairing. The first radio
shard additionally requires decoded first/middle/final video-frame proof.

The tests exercise contracts with synthetic data. They do not establish real
codec compatibility, GPU memory usage, successful collection, or score gains.
