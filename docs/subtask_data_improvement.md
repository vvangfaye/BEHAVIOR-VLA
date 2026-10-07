# Improving low-success subtasks with additional trajectories

This is a training-state diagnostic and data collection workflow. It does not report the official competition score. The current checkpoint uses `task_and_subtask`; preserve that contract during collection, export and fine-tuning. The current `pi05_b1k` model config trains/predicts 32-step action windows, while `serve_b1k.py` executes the first 16 actions before replanning. Export read-back therefore verifies 32-step windows, including episode-end padding. Do not treat the 16 executed actions as the training horizon.

## Acceptance criteria

1. Exact-state restoration must finish an eligible model rollout with finite metrics and a nonempty video. A restoration-only run does not pass.
2. Rank subtasks using a completed strict baseline, its `_SUCCESS` marker and matching submission receipt. Infrastructure failures, unsupported predicates and already-satisfied starts are excluded from the success denominator. Groups below the minimum evidence count are reported separately.
3. Freeze independent `(task, task_instance_id)` groups before collection. Do not collect or replay them into the fine-tuning dataset. These are held out from **this fine-tuning experiment**; the original checkpoint may already have seen the public training instances.
4. Record RGB/proprioception before each actually executed 23-dimensional R1Pro action. Each attempt has a deterministic policy seed, source-state identity and content hash. New seeds vary policy sampling; initial scene diversity comes from distinct recorded training starts, without an unvalidated physics perturbation.
5. Export only eligible successes with finite, aligned arrays, fixed annotated conditioning, complete provenance and nonempty video. Failed attempts remain diagnostics. A completed collection with zero accepted attempts is not a trainable data deliverable; obtain teacher demonstrations for such groups.
6. Export one LeRobot dataset root with dense global episode IDs, complete `[0,length)` subtask labels and original instance groups. Optional original replay uses explicit original episode IDs and rejects reserved instances. Report the actual fraction of new frames.
7. Run real LeRobot video encoding/read-back and training-data preflight in the worker environment. Local tests include a real LeRobot/AV1/PyAV round trip on synthetic arrays; this verifies format compatibility locally, not simulator success or GPU training. Recheck in the actual archived worker environment.
8. Initialize a new experiment from checkpoint 1375000 parameters, preserve conditioning and action transforms, and validate normalization. First run a two-step training smoke, then a bounded fine-tuning experiment. Do not resume the old optimizer on changed data.
9. Independently compare base and fine-tuned checkpoints on exactly the same frozen held-out samples/seeds and matched limits; include control tasks to detect forgetting. Report counts and uncertainty, including no improvement or regressions. Passing code tests alone does not finish the goal.

## Freeze a collection pilot

Use the completed **baseline** files, not the six-sample 128-step restoration validation. The strict producer records `run_id` in the summary and binds its canonical digest, panel fingerprint and scheduled counts into `_SUCCESS`; collection selection checks these against the submission receipt. A stale marker, mismatched run, or model-name string in place of a manifest checksum is rejected. The default pilot selects up to three sufficiently evaluated groups, two starts per group and three seeds per start (at most 18 attempts). Fewer available starts are reported, not invented. Increase evidence or expand the source panel if necessary.

```bash
python scripts/subtask_improvement.py \
  --panel /absolute/baseline/panel.json \
  --summary /absolute/baseline/summary.json \
  --success-marker /absolute/baseline/_SUCCESS \
  --submission-receipt /absolute/baseline/submission_receipt.json \
  --held-out-instances /absolute/experiment/held_out_instances.json \
  --output /absolute/experiment/collection_plan.json
```

`held_out_instances.json` is a JSON list of `[task_name, task_instance_id]` pairs. Plans freeze the checkpoint step and manifest digest from the matched baseline receipt. Collection starts exclude the strict baseline's per-sample `setup_exclusions`, as well as reserved instance groups. Those excluded sample IDs are retained in the frozen plan and checked again on load, so a plan cannot schedule an already-satisfied or otherwise invalid baseline start even with a recomputed fingerprint. Success filtering uses simulator predicates for verification; those privileged values do not become model input features.

## Nebula collection preflight

Reuse `nebula/submit_pi05_eval.sh` and the existing simulator/policy launchers:

```bash
DRY_RUN=1 KEEP_BUNDLE=1 EVAL_PROFILE=subtasks100 CONFIRM_FULL=1 \
EVAL_WORKERS=1 CLUSTER_FILE=nebula/eval_cluster_1gpu.json \
SUBTASK_PREDICTION=1 WRITE_VIDEO=1 PI05_MODEL_FORMAT=jax \
SUBTASK_PANEL_PATH=/absolute/baseline/panel.json \
SUBTASK_RAW_INVENTORY_PATH=/absolute/baseline/raw_inventory.json \
SUBTASK_RAW_SOURCE=oss://traffic-model/wangfaye/embodied/datasets/behavior-1k/2026-challenge-rawdata_20260921230138 \
SUBTASK_COLLECTION_PLAN_PATH=/absolute/experiment/collection_plan.json \
SUBTASK_SAMPLE_MAX_STEPS=2048 RUN_ID=<fresh-run-id> JOB_NAME=<fresh-job-name> \
CHECKPOINT_SOURCE=<verified-staged-checkpoint> \
PI05_REPO_ID=b1k_full100_joint_subtask_v1_drop_episode_s1 \
bash nebula/submit_pi05_eval.sh
```

Do not set a validation plan or production queue during collection. Only the selected raw episodes are staged. The pilot is capped at 300 attempts, but its conservative RGB/HDF5/video budget can require batching well below that cap. The worker checks expanded asset size, trained checkpoint size, selected raw size, maximum trajectory output and a 16 GiB reserve. It restores only the tokenizer from the old cache archive and extracts already-verified assets directly instead of retaining duplicate zip files. It prints required and available disk space before staging.

The collection server advertises seed support and resets the JAX sampling key once at the first request of each attempt. Normal evaluation does not enable this extra protocol. A real GPU round trip remains required.

The user has authorized autonomous execution of this competition workflow. After each concrete preflight passes, submit and monitor without requesting another confirmation. Preserve the receipt and use its exact IID. Collection `_SUCCESS` means **all planned attempts completed**, and `ready_for_export` separately states whether any succeeded; it does not indicate a competition evaluation or a score improvement.

## Export and fine-tune

Run export in an environment with the repository's BEHAVIOR LeRobot fork, h5py, pyarrow and video encoding dependencies:

```bash
python scripts/export_subtask_collection.py \
  --collection-root /absolute/finished-collection \
  --panel /absolute/baseline/panel.json \
  --collection-plan /absolute/experiment/collection_plan.json \
  --output /absolute/new-dataset-root --repo-id <fresh-dataset-id> \
  --base-dataset /absolute/original-dataset \
  --base-segments /absolute/original-subtask-segments.parquet \
  --base-episodes /absolute/replay-original-episode-ids.txt
```

Omit all three replay flags only for a data-format smoke. For the fine-tuning experiment, target about 20% new frames and 80% original replay, measure the realized ratio, and include representative control-task data. The exporter does not implement weighted sampling or claim an exact mixture from these flags. Original episode IDs are remapped only in the exported root; provenance retains their source IDs and grouping.

Real export automatically reads the result with PyAV and verifies numerical action/state alignment against accepted HDF5 sources, dense episode IDs, exact label coverage, prompts, and 32-step action windows including padding at episode boundaries. It decodes first/middle/last frames of every episode. Only after this passes are `format_verification.json` and `_VERIFIED_EXPORT` written. These markers record a successful check at export time; they do not authorize training or prove that files remain unchanged.

Before training, rerun the verifier against the finalized data and preserve a fresh report:

```bash
python scripts/verify_subtask_collection_export.py \
  --dataset-root /absolute/new-dataset-root --repo-id <fresh-dataset-id> \
  --collection-root /absolute/finished-collection \
  --report /absolute/experiment/pretraining_format_verification.json
```

After the fresh read-back and `openpi/scripts/b1k/check_subtask_data.py` pass, reuse `scripts/train_pi05_subtask_b1k.sh`. Set `SUBTASK_ACTION_CONDITIONING=task_and_subtask`, `INIT_PARAMS=<1375000 params>`, a fresh `EXP_NAME`, the exported `DATASET_ROOT`, its `meta/subtask_segments.parquet`, and retained episode IDs. Verify the norm-stats asset against the unchanged action/state transform before training. The base 1375000 manifest verifies `norm_stats.json` SHA256 `52d63e89f8359143d793ca80d0661c7facb0daff43c7d8c08f0d05572dd528fb`; both transformed state and action statistics have 23 finite values. Reuse these base statistics for this paired fine-tuning experiment, under the fresh assets ID, rather than silently recomputing normalization on the small new dataset. Submit the bounded training experiment after its data and runtime preflights pass under the user's ongoing authorization.


## Bounded training from actual exported collection data

Use `scripts/finetune_subtask_collection.py` in the restored training environment. It revalidates completed collection attempts and their runtime contract, accepted trajectory content, real LeRobot read-back with a 32-step training horizon, every exported file hash and held-out instance exclusions. It verifies all initialization parameter hashes against the actual 1375000 manifest and keeps the base 23-dimensional normalization statistics under a fresh assets ID.

```bash
python scripts/finetune_subtask_collection.py prepare \
  --dataset-root /absolute/verified-mixed-export \
  --collection-root /absolute/completed-collection \
  --checkpoint /absolute/verified-1375000-checkpoint \
  --base-repo-id b1k_full100_joint_subtask_v1_drop_episode_s1 \
  --panel /absolute/baseline/panel.json \
  --collection-plan /absolute/experiment/collection_plan.json \
  --run-root /absolute/fresh-training-experiment \
  --run-id pi05_collection_YYYYMMDD_HHMMSS --devices 4 --batch-size 8 --steps 1000
python scripts/finetune_subtask_collection.py run \
  --plan /absolute/fresh-training-experiment/training_plan.json --phase smoke
python scripts/finetune_subtask_collection.py run \
  --plan /absolute/fresh-training-experiment/training_plan.json --phase full
python scripts/finetune_subtask_collection.py stage \
  --plan /absolute/fresh-training-experiment/training_plan.json \
  --destination /absolute/fresh-inference-checkpoint
```

These commands operate inside the worker. The OSS-only submission adapter below restores the existing `openpi/.venv` and its tokenizer cache, preserving `OPENPI_DATA_HOME`. It uses a single host and the explicit small export rather than the original full100 multihost template.

The existing training launcher performs data/tokenizer preflight, runs two training steps first, and restores the final saved train state. The full phase then starts from 1375000 with a fresh optimizer and the same verified export, rather than continuing the smoke optimizer. Pilot steps are bounded to 128..5000, default 1000. Explicit settings preserve `task_and_subtask`, the 32-step model horizon, seed 42, and a batch divisible by the allocated device count. Inherited original-data, epoch, validation, factorized-conditioning and distributed-resume settings are cleared. Existing compute processes prevent launch; unrelated allocations are never stopped.

The bounded training plan freezes `video_backend=pyav` and the launcher passes `B1K_VIDEO_BACKEND=pyav` as `--data.video-backend pyav` to the actual B1K training config. This matches export read-back and preserves the existing 5 ms frame tolerance. Other training runs retain their current automatic decoder selection unless they set this option.

The default learning rate is 1e-5, decaying to 1e-6 after a 100-step warmup. This is an initial experiment setting, not an observed optimum. A failed preflight, nonfinite logged metrics, failed checkpoint restore, changed data or normalization cannot produce a completed phase receipt. Staging checks actual completion evidence, copies only inference parameters/assets into a fresh root, preserves exact base normalization bytes, and writes a checksum manifest and `_READY`. Only then does it write `training_receipt.json` linking actual export provenance and the two model manifests. The initial model step is 1375000; the new run's saved directory step is its own training-step count minus one, recorded separately from the restored train-state count.

CPU tests cover model/data tampering, reserved original replay, inherited configuration, live GPU processes, failed training and staging identity. The real codec integration prepares the bounded training plan through actual LeRobot files and reads six first/intermediate/last samples through `create_b1k_dataset`, checking all RGB cameras, state, 32-step executed-action windows, full-task prompts and subtask labels. Five focused factory/CLI checks verify that explicit PyAV reaches the reader and preserves existing defaults/tolerance; all six real-config/codec checks pass. Those fixtures use synthetic trajectories and dummy model bytes; they do not prove simulator collection or GPU training. Actual new-data collection, GPU fine-tuning and paired outcomes remain required.

## Submit bounded fine-tuning through OSS

After the real collection completes with `ready_for_export=true`, select explicit original replay episodes against the unchanged holdout list and freeze them with `stage_subtask_replay.py freeze`. Prepare a fresh job using `nebula/submit_subtask_finetune.sh`; the launcher alone sources the private configuration. No submission-host NAS or OSS mount is needed.

```bash
NEBULA_SUBMIT_PYTHON=/absolute/submission-env/bin/python \
  bash nebula/submit_subtask_finetune.sh prepare \
  --panel /absolute/baseline/panel.json \
  --collection-plan /absolute/experiment/collection_plan.json \
  --collection-source oss://traffic-model/path/to/completed-collection \
  --metadata /absolute/frozen-original-metadata \
  --replay-plan /absolute/experiment/replay_staging_plan.json \
  --labels /absolute/original/subtask_segments.v1.parquet \
  --checkpoint oss://traffic-model/path/to/verified-1375000-inference \
  --base-repo-id b1k_full100_joint_subtask_v1_drop_episode_s1 \
  --policy-archive oss://traffic-model/wangfaye/conda_env/b1k-pi05.tar.gz \
  --cache-archive oss://traffic-model/wangfaye/conda_env/b1k-pi05-openpi-cache.tar \
  --output oss://traffic-model/path/to/fresh-finetuning-output \
  --run-id <fresh-run-id> --devices 4 --batch-size 8 --steps 1000 \
  --destination /absolute/fresh-submission-preparation

NEBULA_SUBMIT_PYTHON=/absolute/submission-env/bin/python \
  bash nebula/submit_subtask_finetune.sh submit \
  --prepared /absolute/fresh-submission-preparation
```

Use the same cache archive as the baseline launcher configuration; the example uses its current default. Preparation freezes all input ETags/byte counts, model SHA256, source metadata/labels and bundled code. The worker streams each input with `If-Match`, checks actual bytes and known hashes, restores the policy archive and only the tokenizer, validates every completed collection attempt, exports mixed data and runs the existing bounded training bridge. Scratch gates use actual archive expansion, downloaded inputs, accepted/replay frame dimensions and a conservative eight-initial-model-byte allowance for smoke/full optimizer checkpoints plus inference staging, with a 16 GiB reserve.

The bounded training adapter defaults to one host with four H20 GPUs in `demand_specific_na175_h20`. Preparation freezes the queue/card contract; submission uses that frozen contract and the worker checks actual GPU names before exporting/training. Override with `--queue` / `--gpu-card-type` at preparation, or `TRAIN_QUEUE` / `TRAIN_GPU_CARD_TYPE` in the training launcher. Generic caller-provided `QUEUE` / `GPU_CARD_TYPE` are also captured before private config is sourced. Simulator collection and closed-loop evaluation continue to use L20. Algorithm selection remains configurable through the existing private launcher environment. It never starts a second submission for an existing claim, including an unknown CLI outcome. Inspect `submission_receipt.json` and the recorded IID; retries use new preparation/run/output roots. CLI output is redacted before persistence.

The user authorizes reclaiming confirmed MEGA placeholders only in `demand_specific_na175_h20` when a preflighted training allocation needs capacity. The agent refreshes exact task/queue/status metadata, stops only enough identified placeholders and saves receipts. This is submission-time resource management; the launcher does not indiscriminately stop jobs when preparing data. Current local queue API queries returned no valid capacity/jobs (NOP response status 13), so no free-GPU count or placeholder identity has been verified and no placeholder has been stopped.

The worker uploads a live log while training runs. Final output retains the exported dataset, provenance, full training checkpoint/optimizer, phase logs/proofs and `inference_checkpoint`; the temporary smoke checkpoint is omitted from OSS. `_SUCCESS` is written only after actual training and inference staging receipts, all data/model hashes and uploaded byte counts pass. It does not claim an official score improvement. Failed workers write `_FAILED` and preserve their log; the immutable source collection/replay/checkpoint is untouched.

The adapter checks cover changed/incomplete/zero-success collection, checkpoint identity, replay holdout, escaping downloads, bounded budgets, private-output redaction, ambiguous submissions, archive traversal and premature/mutated model publication. H20 profile checks additionally verify that queue/card choices survive private-config sourcing and later environment changes, that known mismatched queue/card pairs fail, and that actual H200/L20 or insufficient GPU allocations cannot pass H20 training preflight. A separate real LeRobot codec test stages actual encoded files through a simulated OSS SDK and exercises the worker export/preparation path on three episodes / 44 synthetic frames, including 40 original-replay frames. It verifies 32-step windows and no held-out replay leakage, and leaves GPU training explicitly unverified.

## Independent paired comparison

Reserve task-instance groups before collecting or replaying data. Once target groups are selected, freeze both target samples and representative control tasks from those reserved groups:

```bash
python scripts/subtask_paired_retest.py freeze \
  --panel /absolute/baseline/panel.json \
  --collection-plan /absolute/experiment/collection_plan.json \
  --control-tasks <control-task-name> <another-control-task-name> \
  --seed 1 --max-steps 2048 --samples-per-group 10 \
  --output /absolute/experiment/retest_plan.json
```

The plan prefers distinct instances. If a requested group has no reserved samples, it fails rather than quietly omitting the group. Actual selected counts and distinct instances are frozen; ten samples do not imply ten independent instances.

Run both models through the existing full100 queue profile with `PI05_PAIRED_RETEST_SEED=1`, matching this plan's seed and step limit. This is a **four/eight GPU full-panel evaluation**, not the one-GPU restoration validation or collection pilot. Each actual run needs its own concrete preflight, then proceeds under the user's ongoing authorization. A complete baseline can be reused for the base comparison only if it already used the same explicit per-sample seeding contract. When enabled, the evaluator derives a stable uint32 seed from the root seed, panel fingerprint and sample ID; the first policy request resets the JAX sampling key and wrapper history for that sample. It also reseeds simulator-side Python/NumPy/Torch sampling. Eligible rollout JSON must retain the derived seed and protocol. Ordinary evaluation remains unchanged when the variable is absent.

The worker records content hashes of policy/evaluation environment archives, the verified asset checksum manifest, runtime wheel requirements and normalization statistics, together with relevant evaluator source hashes. These must match between base and fine-tuned runs. The comparison recomputes strict summaries from actual JSON and nonempty videos, checks the completion marker, and pairs only the frozen reserved samples. Setup eligibility and raw-state restoration evidence must agree; infrastructure failures invalidate the comparison instead of becoming policy failures.

After a real bounded training run completes and its output checkpoint is staged and checksum-verified, preserve a `training_receipt.json` with `training_run_id`, `complete=true`, `conditioning=task_and_subtask`, `collection_plan_fingerprint`, the canonical JSON SHA256 of the actual `collection_provenance.json` as `export_provenance_sha256`, and verified `init_checkpoint_manifest_sha256` / `output_checkpoint_manifest_sha256`. Populate it from the actual training configuration, completion/checkpoint evidence and export audit; do not create a successful receipt from intended settings. The comparison rejects missing provenance, unrelated models or any reserved instance in collected **or original replay** episodes.

```bash
python scripts/subtask_paired_retest.py compare \
  --panel /absolute/baseline/panel.json \
  --collection-plan /absolute/experiment/collection_plan.json \
  --plan /absolute/experiment/retest_plan.json \
  --base-root /absolute/base-completed-run --fine-root /absolute/fine-completed-run \
  --base-receipt /absolute/base-submission-receipt.json \
  --fine-receipt /absolute/fine-submission-receipt.json \
  --provenance /absolute/new-dataset-root/meta/collection_provenance.json \
  --training-receipt /absolute/experiment/training_receipt.json \
  --output /absolute/experiment/paired_comparison.json
```

The report preserves every matched outcome, gains/losses, eligible counts, setup exclusions, target/control rates and per-task/subtask comparisons. It resamples whole `(task, task_instance_id)` groups for exploratory bootstrap intervals, so correlated subtasks cannot inflate sample size. It also reports a conservative Hoeffding interval with margin `sqrt(2 * log(40) * sum(n_instance**2) / n_pairs**2)` under independence between instance groups. This interval avoids false certainty when a bootstrap sample contains only wins or ties. Fewer than eight groups are explicitly underpowered; this review threshold is not a promise of statistical power. All setup-excluded targets have a null rate, not a zero success rate. An observed gain does not become confirmed improvement when controls or uncertainty fail the evidence gate.

These fixed annotated-subtask comparisons measure action execution on public training states reserved from this fine-tuning experiment. They do not evaluate high-level subtask choice or establish an official competition score improvement.

## Current evidence

- Validation job 217033 failed before simulator startup: `ERROR: /var/tmp has less than 120 GiB free`.
- Its model and asset SHA256 checks passed; it produced zero rollout JSON/video artifacts.
- OSS zip-directory metadata gives approximately 63.37 GiB needed for the same six-sample validation including a 16 GiB reserve, after removing duplicate zip/base-cache copies. Actual free space on the next worker is still checked at runtime.
- 215 collection, evaluator, packaging, storage, replay, paired-comparison and bounded-training regression checks pass; the existing seven instance-split checks also passed.
- A real CPU integration using the repository-specified LeRobot `release/b1k` fork (commit `436812bd8ee39b768c645c248c91f1330834e687`) encoded AV1 training videos and read three mixed episodes / 44 synthetic frames (including a 40-step original replay) with 32-step action windows. Deliberately changed controls were rejected; the fixture is restored and reverified afterward. This uses synthetic arrays, not successful simulator trajectories.
- Replay decoding now explicitly uses PyAV. Automatic TorchCodec selection failed on the local FFmpeg installation; no host codecs or original training environment were modified.
- The archive used by the GPU worker still needs its own round trip; the isolated CPU fork does not prove its dependency versions.
- Validation job 217086 passed its 63.37 GiB disk budget (116.15 GiB available), CUDA kernel/Torch interoperation, checkpoint checksums and asset extraction. It failed before rollout because the simulator import check selected GPU 1 on a one-GPU worker. Single-worker `subtasks100` checks now select GPU 0; an executed shell-block regression covers the device selection. Follow-up validation job 217101 completed successfully: all six sample results were strictly verified, five had eligible model rollouts and nonempty videos, and one was excluded because its predicate was already true. The short 128-step run achieved 0/5 successes; this verifies state restoration and runtime, not the full success-rate baseline.
- Full100 baseline job 217150 used checkpoint 1375000, four L20 GPUs and fixed per-sample seed 1. Its frozen panel contains 7223 supported rollouts and 889 unsupported annotation exclusions. It was stopped on 2026-10-07 after two gift-basket workers failed at their first policy reset with WebSocket keepalive ping timeout; no canonical rollout JSON had been published. Its logs are evidence of an infrastructure failure, not low model success.
- Late-state GPU validation job 217277 finished successfully with verified checkpoint 1375000, OmniGibson 3.9.3 and one L20. All four samples were eligible and produced actual JSON/video artifacts; the two late starts reached frames 20,682 and 21,675. One of four succeeded within the short 128-step limit. This passes the long-idle restoration/transport gate, not the full baseline or a score-improvement claim. Episode 5278 / instance 83 is reserved for evaluation and must not become new training data.
- Full100 retry 217388 was submitted only after the strict late-state GPU gate passed. It preserves the 100-task / 400-instance panel, 7,223 supported samples, checkpoint 1375000, four L20 GPUs, 2,048-step limits and paired seed 1. Live prefix-restoration logs are progress evidence; difficulty selection still requires the completed canonical baseline and its matching success marker.
- A real original-demonstration diagnostic used the existing OSS replay stager on episode 12161 (`scrubbing_bathroom_floor`, instance 248), outside the frozen holdout. Frozen ETags, bytes and SHA256 were checked before reading the selected 1,066 rows. Actual `create_b1k_dataset` / LeRobot / PyAV reads at frames 0, 533 and 1,065 decoded all three RGB cameras and preserved exact original Parquet state/action values, 32-step action windows with end padding, full-task prompts and provided subtask labels. Raw state has 61 values and the R1Pro model transform selects 23. This verifies one real original source episode; it is not a difficult-group replay selection, new GPU collection, mixed export or fine-tuning result.
- Collection/export CPU tests do not establish real GPU collection, successful fine-tuning, or a higher success rate. Those phases remain pending.

## Replay selection from measured successful yield

Once the collection and independent retest plans are frozen and the actual collection is complete, choose original replay with `scripts/select_subtask_replay.py`. It revalidates completion records, unique HDF5 trajectories, videos and the collection summary; its new-frame count is measured from accepted unique trajectories. Active, failed, incomplete or zero-success collection cannot drive selection.

```bash
python scripts/select_subtask_replay.py audit \
  --metadata-root /absolute/frozen-original-metadata \
  --labels /absolute/original/subtask_segments.v1.parquet \
  --retained-episodes /absolute/original/retained_original_episodes.txt \
  --held-out-instances /absolute/experiment/held_out_instances.json \
  --output /absolute/fresh-source-audit

python scripts/select_subtask_replay.py select \
  --metadata-root /absolute/frozen-original-metadata \
  --labels /absolute/original/subtask_segments.v1.parquet \
  --retained-episodes /absolute/original/retained_original_episodes.txt \
  --panel /absolute/baseline/panel.json \
  --collection-plan /absolute/experiment/collection_plan.json \
  --collection-root /absolute/completed-collection \
  --retest-plan /absolute/experiment/retest_plan.json \
  --target-new-fraction 0.2 --seed 42 \
  --max-episodes 24 --max-original-frames 200000 \
  --output /absolute/fresh-replay-selection
```

Use the resulting `original_episode_ids.txt` as `stage_subtask_replay.py freeze --episodes`. The selector requires original replay for every selected exact task/subtask prompt and the same distinct control tasks as the retest plan. It excludes all reserved task-instance groups, uses the original training's retained IDs and validates complete label tiling. The source audit freezes metadata/label/retained-file hashes; selection also freezes successful new-trajectory content hashes and the collection/model identities.

The default aims for 20% new frames using complete original episodes, with a maximum of 24 episodes / 200,000 original frames. Required target/control coverage must fit this budget or selection fails. Complete episodes can be longer than the desired replay count, or the budget/source pool can prevent reaching it: `selection.json` records the actual fraction and whether it is within five percentage points of the target. It never truncates original demonstrations or claims weighted resampling. If the actual ratio is unsuitable, increase successful collection yield or deliberately freeze a different bounded budget before training.

The real original-source audit on 2026-10-07 found 16,051 retained episodes, excluded 100 reserved instance episodes and validated label tiling on all retained IDs. The remaining 15,951 episodes cover all 100 tasks. This is source availability evidence; no new simulator trajectories, source video readability, training or success-rate improvement is inferred from it. The selector/stager checks separately exercise measured synthetic trajectory yield, prompt/control coverage, holdout exclusion, malformed inputs and staging-plan compatibility.

## Stage original replay through OSS

The original dataset metadata is frozen separately from the working copy. Its 100 episode-metadata files match the full100 panel's `episode_metadata_sha256` (`c67ab350a4311e2f6b00a42792ba7e679893c1dcac98468e84a3c5bcdd890817`). The original raw proprioception has 61 values, including trunk velocity; verify that real collected data has the same feature contract before mixed export. Synthetic format fixtures alone cannot prove this.

Use explicit replay episode IDs selected after measuring accepted new-frame yield. `scripts/stage_subtask_replay.py` resolves the per-episode data/video **file indices**, deduplicates shared files, and checks each selected `(task, instance)` against the frozen holdout list. An episode can share Parquet/video files with other episodes. Transferred shared files may therefore contain extra frames; the exporter still reads only the explicit selected episode IDs and never includes a reserved episode in the training output.

```bash
# Run in the existing submission/worker environment with OSSClient credentials
# already supplied privately by the launcher; never print or copy config.sh.
python scripts/stage_subtask_replay.py freeze \
  --metadata-root /absolute/frozen-original-metadata \
  --source oss://traffic-model/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542 \
  --episodes /absolute/experiment/replay-original-episode-ids.txt \
  --held-out-instances /absolute/experiment/held_out_instances.json \
  --output /absolute/experiment/replay_staging_plan.json

python scripts/stage_subtask_replay.py stage \
  --metadata-root /absolute/frozen-original-metadata \
  --plan /absolute/experiment/replay_staging_plan.json \
  --output /absolute/fresh-working-original-dataset --workers 4
```

The plan freezes metadata SHA256, OSS object ETags, sizes and required local disk. Each API download uses `If-Match`, checks the byte count and records a SHA256. A changed object, truncated transfer or stale metadata cannot produce a complete staging receipt. Metadata retains global episode IDs and shared-file timestamps. The existing `strip_depth_from_info.py` removes unused depth video features only from the new working copy, with a backup; frozen source metadata remains unchanged. All three RGB cameras remain required. This avoids LeRobot looking for unstaged depth videos or falling back to Hugging Face.

Pass the staged root and its `selected_episodes.txt` to the mixed exporter. Staging does not establish video readability: its receipt records `reader_verified=false`. The actual mixed exporter and fresh training preflight must still decode the selected RGB frames and verify labels/actions. Eight staging checks and the real LeRobot/PyAV mixed-export integration pass locally; the integration uses synthetic frames and actual codec files, not new simulator demonstrations.

## Long restoration WebSocket regression

The original six-sample restoration probe reached only frames 0–541. Full-panel episodes can materialize starts beyond frame 20,000 before requesting the first policy action. On job 217150, the idle connection closed with `1011 keepalive ping timeout`, then reset failed and episode results were not published. Both production endpoints now disable periodic heartbeat timers; existing task/episode process supervision and TCP closure handling still manage actual failures. No automatic stochastic inference retry was added.

Three real localhost WebSocket checks reproduce the historical deadline failure, verify restoration followed by reset/action on the fixed connection, and verify blocking first inference followed by another reset/action. Together with the 13 vendored API checks and 68 state/probe/paired comparison checks, 84 checks pass. CPU network evidence does not establish GPU simulator restoration or model quality.

The one-L20 probe 217277 completed gift-basket episode 5278 at frames 0, 411, 20,682 and 21,675 with named state restoration and actual JSON/video outputs. Prefix materialization loaded 21,676 states and captured four starts in 611.7 seconds. All four rollouts were eligible; the late starts executed 128 and six controls, respectively, and the last succeeded. Strict local verification checked the actual completion marker, validation contract/summary, four JSON files, four videos and checkpoint manifest before full100 retry 217388 was submitted. Preserve this proof separately from full-panel score evidence. Future restoration or networking changes still need eligible late-state GPU action evidence before a full retry. The simulator emits `STATE_REPLAY_BEGIN`, periodic loaded/captured frame counts and `STATE_REPLAY_COMPLETE` elapsed time. Networking source files are included in paired-evaluator hashes, so changes cannot silently reuse incompatible run contracts.

## 2026-10-07：原始原生状态六起点验证完成，正式基线重新准备

L20 作业 218183（代码 7bb1af4、checkpoint 1375000）已 FINISH/FINISHED。六个原始起点全部通过独立原始 HDF5 对照：28 个关节、根位姿、8 份原生抓取约束及注释对象来源一致；实际模型控制共 1,520 步，六段视频共 1,520 帧。结果为 1 条成功、5 条在 256 步上限超时。该短测只验证恢复与推理流程，不能作为 2048 步正式困难组基线，也没有生成可直接训练的新 RGB 轨迹。未保存的物理缓存与计时器仍不在验证范围内。

原始六起点证据：`eval_results/subtask_recorded_native_1375000_20261007_081519/validated_outputs/local_gpu_verification.json`。

正式 100 任务面板仍为 400 个原始实例、7,223 个可解析起点及 889 项不支持注释。新增严格的 `source_excluded` 记录，区分旧数据缺失抓取状态与模型失败：`success=null`、0 模型动作、不声明恢复快照或谓词测量，不进入成功率分母，也不会进入新增样本采集。每条记录绑定原始整文件及选中状态行 SHA256。CPU 预处理逐实例发布这些记录，避免为缺失源状态启动模拟器；现代数据继续执行 GPU 原生恢复。

新正式运行强制 `recorded_native_fields_v1`：每个已恢复起点先核对保存字段，结果附实际 61 维观察、23 维控制及固定子任务响应审计；汇总需匹配独立 HDF5 预检哈希。困难组采样需完成且通过原生状态验证的 2048 步基线，保留至少 10 个有效起点门槛，拒绝短测或旧版未验证汇总。

204 项针对性 CPU 检查通过。另用真实旧版文件验证 18 个起点均只能发布源数据排除记录，并对实际六起点 GPU 结果核对新正式结果格式；此格式验证不把短测改算成正式结果。完整正式基线、新成功 RGB 数据、实际补训及独立复测尚未完成。


## Bounded feasibility experiment (2026-10-07)

A human-requested feasibility pilot can proceed before the full 100-task
baseline completes. Use `subtask_improvement.freeze_pilot_plan`, with explicit
sample IDs, original full-file/selected-state hashes, and a completed native GPU
validation for each target group on checkpoint 1375000. This mode has no
estimated difficulty or success-rate selection claim. It is labelled
`small_data_augmentation_feasibility_v1`, with both difficulty selection and
production baseline flags false. Formal `freeze_plan` requirements remain intact.

The first experiment targets `organizing_art_supplies`: placing the marker and
glue stick inside the tote. Collect from instances 130 and 215, three independent
policy seeds per start (12 attempts), with a 768-control cap on one L20. Reserve
instances 213 and 228 from this fine-tune, in addition to existing global instance
holds. Reuse the success/finite RGB-state-action/video checks and LeRobot export.
Every attempt verifies original native fields before its first policy request;
record the executed controls alongside training arrays and verify video hashes.

After successful collection/export, run 256 steps of joint fine-tuning on four
H20 GPUs, then compare base and fine-tuned checkpoints on the same held-out
starts/seeds with an independent control task. These small paired results are
feasibility evidence only; they cannot establish whole-benchmark improvement.
If collection produces no accepted successes, preserve that negative result and
do not submit a training job or relabel failed trials as BC demonstrations.
