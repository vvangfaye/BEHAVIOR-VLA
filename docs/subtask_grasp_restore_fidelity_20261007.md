# Assisted-grasp fidelity issue in legacy raw subtask starts

## User-approved scoring with action-history recovery (2026-10-08 14:31)

The user explicitly approved scoring the original old-format H5 and waived
additional grasp diagnostics. The earlier exact historical grasp-anchor
verification gate no longer blocks this scoring run. Original raw data,
sample identities, starts, 2048-step horizon and three-attempt limit remain.

A finished physical diagnostic recovered native grasps at four original
starts from two files by replaying only preceding recorded actions while
restoring original physical poses. It executed 3807 prefix physics steps and
256 paired continuation steps. All eight recovered grasps existed immediately
and persisted for 32 steps; seven match unique source motion and one is
motion-ambiguous. This does not prove original historical anchor-byte identity.
Accepted bounded evidence: `legacy_history_physical_diagnostic_acceptance.json`.

The production protocol is
`recorded_actions_before_start_native_grasp_recovery_v1`.
Each H5 materializes every original start using the preceding recorded actions,
with temporary compressed snapshots on worker-local storage. Grasp counters
and native constraint parameters are preserved at each captured start.
Results disclose that constraints are recovered through physics, rather than
claiming that their anchors were directly recorded in the original H5.

All 50 one-GPU Spot scoring jobs for 200 H5 / 4118 starts were submitted.
At 14:31, the platform snapshot confirmed 30 running, 19 starting and one
unavailable status query; 27 had reported actual 5090 allocations, with no
failed assignment report. This is ongoing scoring, not completed success rates.
The original frozen 100-task panel and unsupported annotations are preserved.
Authorization and exact receipts live under
`eval_results/sco_migration_20261007/legacy_history_scoring_authorization_current.json`
and `legacy_history_evaluations/behavior-legacy-history-eval-20261008-140818`.

The historical observations below retain their original scope and date.

## Official 2026 data identity and latest checks (2026-10-08 11:31)

“Legacy” describes the recording/serialization format, not the release year.
All 400 original H5 files on AFS match both the size and SHA256 published by
the official `behavior-1k/2026-challenge-rawdata` Hugging Face repository at
revision `f9d2901112993d75684223820eec70722806c2c4`. They were transferred to
SCO from the existing OSS mirror. The metadata-only comparison fetched no
H5 bytes; evidence is `eval_results/sco_migration_20261007/official_hf_raw_provenance_current.json`.
Of the 200 legacy-format files, 174 record `3.7.0-alpha` and 26 record
`1.2.0-alpha` as their simulator version.

The two-version, two-H5, four-midpoint physical diagnostic and independent
audit have both finished successfully: 128 recorded-action steps and 128
decoded nonblank video frames, with exact saved joints and root poses. The
native held-object relation is empty immediately after restoration and is
reestablished for some arms only after 11–20 actions. These are diagnostic
results, not model success rates or verification of all 4118 legacy starts.

A separate 0-GPU audit inventoried all 15568 datasets and structured metadata
in all 200 files with no budget skips. It found no string datasets or separate
grasp/constraint-named numeric datasets; all 200 initial named constraint
fields are empty. No auxiliary per-frame grasp source was found. This does
not prove absence from the main numeric state or exact restoration fidelity.
Accepted evidence is `eval_results/sco_migration_20261007/legacy_aux_state_audit_acceptance.json`.

## Evidence correction (2026-10-08)

The earlier blanket claim that all 50 legacy tasks lack assisted-grasp state
and cannot be restored is withdrawn. All 200 legacy H5 files contain
`ag_obj_constraint_params` in their initial named scene state, with empty
per-arm values. The current guard identifies the old `controllers` layout
and absence of the newer `AG_MAGIC` serialization marker; it is a screening
rule, not proof that every original start is irrecoverable.

All 200 initial frame-zero annotations and the remaining 3,918 supported
later starts remain in the original evaluation scope. Initial empty values
do not establish the grasp relationship at an arbitrary later frame.
The exact later-start restoration remains unverified, rather than proven
impossible. Neither guard-only records nor diagnostic replay enters the
model success denominator.

The 3.7.0-alpha representative frame-zero start has passed a real 32-step
recorded-action replay and an independent CPU audit of raw and row hashes,
the exact 32-by-23 actions, and nonblank RGB video. No model actions were
requested. A separate 1.2.0-alpha representative is undergoing real replay
on SCO. These checks establish bounded format compatibility, not the
success rate or exact grasp restoration of all legacy starts.

The evidence inventory and source-path request are maintained in
`eval_results/sco_migration_20261007/MISSING_NATIVE_STATE_DATA.md`.
The historical observations below retain their original scope; they must
not be generalized to all 200 H5 files or used to discard the 50 tasks.

The full100 evaluation 217388 was stopped on 2026-10-07 at 04:36 China time.
AIHub subsequently reported `STOPPED` and the application reported `KILLED`.
Existing OSS results and logs were preserved. This run must not drive hard
subtask selection or be treated as a verified exact-start baseline.

## Actual evidence

The downloaded source for `wash_dog_toys`, instance 33, raw episode 00330330
has 75,008,990 bytes and SHA256
`ebce71ad4852304038ab52e683ba04f699bf7317d2f23f40c29092ccd097f437`,
matching the live worker's result provenance. Its robot uses `assisted`
grasping. All 11,990 serialized frames lack the native assisted-grasp sentinel;
the initial named robot state uses the legacy `controllers` layout and both
initial arm constraints are empty.

The current native `Robot.deserialize` sets `ag_obj_constraint_params` to an
empty dictionary when the serialized grasp block is absent. `Robot._load_state`
then releases existing constraints. Replaying these legacy rows into full named
snapshots therefore does not supply the missing held-object state.

The first instance's videos show a teddy bear near the right hand at placement
start, followed by a drop while the other arm moves a ball. Both failed teddy
placements and successful ball placements report `grasped=false` initially.
Pixels alone do not prove the original grasp or the identity of an assisted
constraint. The source evidence proves a fidelity gap; it does not establish
that every observed policy failure was caused by it.

The 36 native-schema-validated result files and 31 eligible videos remain useful
for diagnosis. Previous partial success percentages are not established as
model success rates from faithful original starts. The previous 7–14 day ETA
was for the abandoned configuration and is no longer an estimate for an
active evaluation.

## Protection and verification

`load_episode_cache` now audits recorded grasp serialization and rejects
legacy assisted/sticky starts with no grasp sentinel in the inspected prefix.
An explicit `allow_unverified_grasp_state=True` permits diagnostic reading only;
`materialize_episode_start_states` independently prevents such a cache from
entering exact scoring or collection. Sentinel presence is a capability signal,
not proof of a correctly decoded grasp, and runtime verification is still needed.

The actual raw episode is rejected by the ordinary cache path and can be read
through the explicit diagnostic path. The 51 targeted CPU checks include the
native robot's legacy deserialize/release behavior, recreation from an actual
serialized grasp payload, padded-array handling, prevention of diagnostic cache
reuse, and the existing named-state restoration checks. CPU fixtures do not
verify simulator contacts or recover legacy constraints.

Local audit records are under
`eval_results/subtask100_1375000_goal_baseline_idlefix_20261007_005449/`:

- `grasp_fidelity_stop_receipt.json`
- `first_failure_visual_audit/raw_grasp_serialization_audit.json`
- `first_failure_visual_audit/raw_fidelity_gate_verification.json`
- `first_failure_visual_audit/frame_extraction.json`

The earlier late-state GPU gate verified transport, state-loader execution,
actions and videos, but did not prove restoration of legacy held-object
constraints. It cannot authorize another full baseline without this additional
verification.

## Remaining work

Use a bounded L20 diagnostic to inspect original source motion, gripper state,
contact and candidate-object evidence for the critical starts. Any reconstruction
must use source/runtime evidence rather than the desired completion predicate,
report its method explicitly, and fail on ambiguous objects or unsupported joint
constraints. Verify full snapshot restore after a real constraint is established
and measure object retention after physics advances. Establish a defensible
legacy-start restoration protocol before another full100 baseline, difficulty
selection, real collection, fine-tuning and paired retest.

No new collection trajectories, GPU fine-tuning or paired comparison were
completed by this diagnosis.

## Bounded GPU observation diagnostic

`subtask_grasp_restoration_diagnostic` is an explicit validation-plan kind,
separate from `subtask_state_validation`. It reuses the existing Nebula staging,
policy-server identity/runtime setup and cleanup. It reads a diagnostic cache
without entering scoring materialization and writes under `grasp_diagnostic/`,
not the canonical scored `json/` directory.

The diagnostic records the original sparse-state prefix, the last 64 source
frames before each selected start, end-effector and movable-object relative
poses, gripper joint positions/limits and the native held-object constraints.
It advances at most 32 **recorded demonstration actions** per start and then
records fresh contacts/native candidate identities. Before a physical advance,
contact caches are explicitly marked unverified. Movable candidates come from
recorded scene metadata, independently of desired predicate targets. The
program does not create grasp constraints or call model action inference.

Validation requires complete source windows and per-arm/object pose evidence,
finite 23-wide actions, consistent observation timing, a recomputed source motion
summary and videos matching their hashes. Diagnostic completion always retains
`metric_eligible=false` and `exact_legacy_grasp_restoration_verified=false`.
It cannot satisfy the model validation gate or justify a full scoring retry.

The 38 targeted diagnostic/probe/grasp-fidelity checks passed before submission.
Actual source-motion, contact and video evidence still requires the GPU run.
The prepared pilot uses wash_dog_toys instance 33, source frames
0, 4356, 6147, 6870, 9104 and 9785: one L20 and at most 192 recorded-action
control steps, plus the bounded original prefix restoration.


## Actual L20 diagnostic 217709

The one-L20 job finished successfully on 2026-10-07. The worker published
`_SUCCESS` at 05:53 China time; targeted AIHub status subsequently reported
`FINISH`, application `FINISHED`, worker `FINISHED`, and task `success`.
All six starts completed 32 recorded demonstration actions each. The local
verifier decoded all 192 video frames and checked source/checkpoint identity,
frozen inputs, runtime source hashes and the actual 23-wide actions.
There were no model action requests and this is not a success-rate estimate.

Independent decoding of 9,786 actual HDF rows yielded 326 watched source
frames. All 2,608 observed object-pose comparisons had zero position error
against the latest explicitly stored raw pose; orientation differences were
at float rounding level. Of these, 604 comparisons use a carried sparse pose,
which does not prove the original physical pose on a frame that omitted it.
All 326 observed raw gripper joint vectors matched exactly. Full named
snapshot restoration also reproduced the observed EEF poses at the six starts.
These checks establish the pose/state transport but do not establish grasp
constraints absent from the legacy source.

The two critical right-hand teddy starts show a separate physical failure:

| Raw start | Object co-moving with right EEF | Source relative position deviation | Native candidate steps | Native hold in 32 actions | Final relative position change |
| --- | --- | --- | --- | --- | --- |
| 6147 | teddy_bear_190 | 0.0000156 m | 1–3 | none | 0.308 m |
| 9104 | teddy_bear_191 | 0.0000178 m | 1–4 | none | 0.267 m |

The source EEF moves 0.128 m and 0.049 m respectively over each 65-frame
window, while the teddy/EEF transform stays nearly constant. This motion
is strong source evidence of coupling, not an original grasp annotation.
After restoration, physical native candidates are briefly detected but no
right-hand constraint is established before contact is lost. Extracted actual
video frames show both teddy toys dropping. Left-hand balls naturally acquire
a native `FixedJoint` after action 11 and stay held through action 32.
The current native handler requires a 0.3-second grasp window, so relying on
natural reacquisition alone cannot preserve all demonstrated starts.

The independent original Parquet episode has 11,989 rows with 61-wide state;
all its 23-wide actions exactly match the HDF source. Its joint observations
are not exactly the raw before-action state. Current replay code advances
physics before exporting observations; the historical exporter version is
not independently established. Original EEF observations are therefore an
additional numeric diagnostic, not a strict before-physics pose ground truth.
Their interhand pose comparison also cannot verify the common world/base pose.

The local validator initially rejected summaries recomputed from identical
source windows because Linux/macOS NumPy/BLAS differ in a derived `acos` near
zero by about 1e-8 radians. The comparator now permits at most 1e-7 radians
only for that derived angular field, and 1e-12 m for derived distances.
Copied source joints, identity, integer counts and categorical fields remain
exact. All 17 targeted diagnostic tests passed, including rejection of altered
positions, joints, object names, counts and larger angular changes. The actual
six-result verifier then passed without changes to any GPU result.

Reproducible evidence is in
`eval_results/subtask_grasp_diagnostic_1375000_20261007_045015/`:

- `diagnostic_findings.json`
- `validated_outputs/local_gpu_verification.json`
- `validated_outputs/download_receipt.json`
- `validated_outputs/right_teddy_actual_recorded_action_contact_sheet.jpg`
- `independent_recorded_object_pose_audit.json`
- `original_source_proprio/original_proprio_alignment_audit.json`
- `original_source_proprio/original_proprio_timing_and_numeric_audit.json`
- `actual_gpu_source_kinematic_comparison.json`

The next bounded experiment must select any coupled object from source motion
over all movable objects, independently of the desired target predicate;
require a matching physical native candidate and finger contact point; check
the supported native joint type and unchanged initial poses; and verify a
known native grasp through full named snapshot release/load and subsequent
physical retention. Compare recorded-action retention with the unmodified
restoration at the same two starts. None of this completed diagnostic permits
labeling legacy grasp restoration as exact or restarting the full100 score.
Real new training trajectories, fine-tunes and paired model retests remain zero.


## Source/contact paired reconstruction experiment

`subtask_grasp_reconstruction_diagnostic` selects a unique nearby object from
a complete 65-frame source motion window, requiring at least 3 cm EEF motion,
less than 1 mm relative position deviation and less than 0.02 rad relative
orientation deviation. Fully open grippers, ambiguous objects and shared
objects inferred by both arms are rejected. The selection API receives no
task, predicate or desired target label.

At starts 6147 and 9104, each original arm has a unique candidate under this
rule. A single original physical action must produce a matching native contact
candidate and a real finger contact point with a supported FixedJoint. The
program restores the untouched start before adding the constraint, verifies
unchanged initial poses and all robot joints, dumps a native named snapshot,
releases the grasps, moves the objects 20 cm, and reloads the known snapshot.
Both restored constraints and subsequent physical retention are measured.

Unmodified and reconstructed branches run the same 32 recorded actions and
seed 1. Two starts plus one contact probe each give a maximum 130 control steps.
Four videos and two native snapshot files accompany the two result JSONs.
The diagnostic stays `metric_eligible=false` and never claims original legacy
constraints were verified. The 65 targeted CPU checks cover selection,
artifact integrity, diagnostic dispatch and preservation of the scoring guard;
Actual results from that L20 run are recorded below.


## Actual paired reconstruction result 217810

The one-L20 run finished on 2026-10-07, with `_SUCCESS` at 06:46 China
time and targeted AIHub status confirming task `success`, application
`FINISHED`, and worker `FINISHED`. It used source commit `a9794b0`,
checkpoint 1375000 and 130 actual recorded-action control steps. There were
zero model action requests. Submission to published completion took about
38 minutes; original source replay restored 9,105 states.

Immutable OSS HEAD/ETag/SHA receipts cover both JSONs, both native `.pt`
snapshots and all four videos. The independent CPU verifier decoded all
128 RGB frames, checked all actual 23-wide actions against the raw HDF,
and safely loaded the saved native states with `torch.load(weights_only=True)`.
Both native files contain the expected left/right FixedJoint grasp parameters,
including real link identities and finite native anchors. Full initial robot
joint vectors are unchanged, and initial object position difference is zero.
Release, deliberate 20 cm object displacement, and full named snapshot reload
were verified before the reconstructed branch was advanced.

| Source start | Right-hand object | Unmodified native hold | Unmodified final EEF-relative displacement | Reconstructed native hold | Reconstructed maximum EEF-relative displacement |
| --- | --- | --- | --- | --- | --- |
| 6147 | teddy_bear_190 | 0/32 actions | 0.308 m | 32/32 actions | 0.0000146 m |
| 9104 | teddy_bear_191 | 0/32 actions | 0.267 m | 32/32 actions | 0.0000821 m |

The reconstructed left-hand tennis ball and softball also remained held for
all 32 actions. All four reconstructed arm/object pairs passed the bounded
retention check. Unmodified left-hand grasps were held only in the final
22 actions, consistent with the earlier diagnostic. Across both source
windows, 1,040 object-pose comparisons have zero position error against the
latest explicitly recorded raw pose; 260 comparisons are carried sparse poses
and do not independently establish the original omitted-frame physical pose.

Actual evidence is in
`eval_results/subtask_grasp_reconstruction_1375000_20261007_060343/`:

- `diagnostic_findings.json`
- `validated_outputs/local_gpu_verification.json`
- `validated_outputs/download_receipt.json`
- `validated_outputs/paired_actual_recorded_action_contact_sheet.jpg`
- `validated_outputs/paired_contact_sheet_receipt.json`
- `validated_outputs/pi05/wash_dog_toys/grasp_reconstruction/native_snapshots/`

This verifies the known reconstructed native grasps at two starts. It does
not recover the missing original grasp annotation or prove all100-task
fidelity: `exact_legacy_grasp_restoration_verified=false`,
`metric_eligible=false` and `official_benchmark=false` remain unchanged.
The next experiment must execute actual checkpoint1375000 model actions
from these independently verified native snapshots, explicitly distinguishing
reconstructed starts from historical legacy states. Full100 scoring remains
withheld until its required state protocol is validated. Real new accepted
training trajectories, actual fine-tunes and paired model retests remain zero.


## Model validation from verified native snapshots

`subtask_native_snapshot_model_validation` reuses the production policy server,
full named restore and fixed-subtask completion monitor. Its two reconstructed
starts come exclusively from the completed paired experiment217810, whose
13 immutable source artifacts are SHA-bound in a manifest. Preflight recomputes
the completed source proof and retention summary. Input packing copies only
listed artifacts, rejects symlinks/path traversal/extra files and caps input
bytes at128MiB. The checkpoint remains1375000 with its frozen manifest.

Each runtime uses safe weights-only loading, moves native tensors to the actual
robot device, restores the complete named scene and then checks every robot
joint, all watched EEF/object poses, recorded held objects, FixedJoint types,
release counters, target link names and the native robot source hash before any
model action. A failed check aborts before the completion monitor or policy.
The actual model control audit pairs each finite23-wide executed action with
its preceding61-wide proprioception. Policy-reported provided-subtask text and
video hashes must agree with every evaluated control step.

The bounded pilot uses oneL20, seed1 and at most2048 actions per start. It skips
the expensive legacy state-prefix simulator replay by loading the proven native
snapshots. Its top-level outputs remain`metric_eligible=false` and
`exact_legacy_grasp_restoration_verified=false`, with the actual reconstructed
start rollout nested separately. No complete100-task score or training-data
acceptance can be inferred from this diagnostic. The117 targeted CPU checks
passed; actual model inference results are recorded below.


## Actual native-start model result 217886

The one-L20 job finished successfully on2026-10-07, publishing`_SUCCESS`
at07:24 China time. Targeted status confirmed application/worker`FINISHED`
and task`success`. The frozen code was`6022b2f`, checkpoint1375000,
conditioning`task_and_subtask`, seed1 and the two native inputs from217810.
Unlike the preceding experiments, this executed actual model actions.

| Original start | Target | Actual model controls | Subtask result | Initial state | Final debounced state |
| --- | --- | --- | --- | --- | --- |
| 6147 | teddy_bear_190 | 670 | successful | held; outside washer | released; Inside washer for3 steps |
| 9104 | teddy_bear_191 | 396 | successful | held; outside washer | released; Inside washer for3 steps |

The independent verifier froze OSS object identities, reproduced the worker
summary with the saved source bundle, checked the actual saved native grasp
bytes/full joint positions against independently decoded raw joints, checked
runtime initial gripper positions, native links and EEF/object poses, and
decoded all1066 real video frames. Initial position error is zero and maximum
orientation difference is below6e-8 radians. Frozen GPU code compares the
complete runtime joint vector exactly before the first model control; the CPU
verifier directly checks saved full joints and runtime gripper joints/poses.
Every executed23-wide control has an aligned preceding61-wide state audit and
provided-subtask report. Final monitor targets were independently inspected:
the intended teddy is released and Inside the intended washer, with dwell3.

Both whole-task success flags remain false and whole-task q-score remains0:
these are individual placement subtasks, not a completed washing episode.
The actual two-video contact sheet shows the executed model progression.
No RGB training trajectory was accepted or exported by this diagnostic.

Evidence is in
`eval_results/subtask_native_snapshot_model_1375000_20261007_065529/`:

- `diagnostic_findings.json`
- `validated_outputs/local_gpu_verification.json`
- `validated_outputs/download_receipt.json`
- `validated_outputs/actual_model_placement_contact_sheet.jpg`
- `validated_outputs/actual_model_contact_sheet_receipt.json`
- `validated_outputs/pi05/wash_dog_toys/native_snapshot_validation/`

The same unchanged model succeeds at these two starts once valid native grasps
are restored. The old failed legacy-state runs must not be used to label these
as hard model groups or to claim a training improvement. Original grasp
annotations remain absent:`exact_legacy_grasp_restoration_verified=false`.
The top-level diagnostic stays`metric_eligible=false`; two starts from one
episode do not provide a complete100-task baseline or a statistically supported
hard-group rate. Next, a declared audited native-start protocol must cover the
full selected panel, reporting any unsupported starts and coverage, before
valid rate selection and actual diverse RGB sampling. Fine-tune and independent
paired retest remain unexecuted.


## Full-panel raw source capability audit

`scripts/audit_b1k_grasp_sources.py` reads the immutable OSS inventory directly.
It uses bounded byte ranges with `If-Match`, checks object size and ETag before
and after reading, and retains a SHA256 receipt for each downloaded range.
It inspects every scheduled raw start, its valid state width, and the next
32 recorded actions without downloading entire HDF5 episodes. A real-source
pilot checked all 18 starts in wash_dog_toys instance 33 with 801,810 downloaded
bytes; every fetched range matched the previously downloaded complete source.
All 18 selected rows lacked the assisted-grasp sentinel.

The reader reuses `_raw_episode_structure` from the formal evaluator: select
the final numeric `demo_*` group, verify its action length, and accept either
aligned states or one trailing state. Several real source files contain
multiple recording attempts; assuming one group incorrectly rejects them.
Those schema failures must be retried with the correct reader and retained
separately from any model outcome. Nine targeted CPU checks cover actual HDF5
offsets, multiple versions, aligned states, padded sentinel exclusion, changed
ETags, short reads, byte budgets, incomplete coverage and private SDK errors.

The full audit preserves the original 100-task, 400-episode, 7,223-start panel
and its 889 predicate exclusions. It reports per-task coverage and whether the
current two-arm reconstruction's history/action preconditions are available.
This is a source-format audit: no model inference, physical contact check or
success-rate estimate occurs. Missing sentinels do not prove free hands, and
present sentinels do not prove correct grasp decoding. Whole-file SHA256 is
explicitly unverified when only ranges were fetched.

Run the reusable command with the standard OSS credential environment supplied
by the caller; do not read or copy `nebula/config.sh`:

```bash
python scripts/audit_b1k_grasp_sources.py \
  --panel /absolute/baseline/panel.json \
  --inventory /absolute/baseline/raw_inventory.json \
  --output /absolute/fresh-source-audit --workers 4
```

A `complete_selected_row_audit` result requires every scheduled episode and
start. Partial reads cannot authorize scoring, hard-group selection or training.
The next simulator phase must establish and verify native initial states with
explicit reconstructed-state coverage before computing model success rates.


## Actual full100 source audit and original native candidates

The completed two-phase OSS audit covered **all 100 tasks, 400 episodes and
7,223 scheduled starts**, with zero remaining read failures. It retained the
889 predicate exclusions. Independent integrity verification matched every
start identity, per-episode record, frozen ETag/size and range receipt against
the original panel and inventory. The 212 first-pass schema rejections were
read successfully using the evaluator's final numeric demo selection rule;
the 188 initially successful source records were preserved.

The source population is mixed, so the missing-grasp problem must not be
extrapolated from the original wash_dog_toys episode to every task:

| Selected raw rows | Count | Meaning |
| --- | ---: | --- |
| Assisted-grasp sentinel present | 1,875 | Format capability signal across 49 tasks and 195 episodes; runtime decoding is still unverified |
| Legacy assisted layout, no selected-row sentinel | 4,118 | Missing saved grasp evidence; cannot assume free hands |
| Other rows without a sentinel | 1,230 | Require schema and sparse-history checks; absence alone does not prove free hands |

Downloaded range contents retained in the combined receipts total 279,402,252
bytes; total requested bytes including the initial failed schema reads were
338,127,312. Only ranges were fetched for this audit, so no whole-file SHA256
verification is claimed for the full400 population. The report and integrity
check are under
`eval_results/subtask_improvement_goal/full100_grasp_source_audit_repaired_20261007_074449/`.
The exact current folder is recorded in `goal_protocol.json`; the report SHA256
is `91fee7e17237fca9f9c2af728362fa52d7a8367f18de4e7ae29e8cc45c3c3a81`.

A subsequent real-source capture downloaded three whole HDF5 files, totaling
120,898,928 bytes, from freeze_fruit, tidying_living_room and
organizing_art_supplies. Their complete checksums and pinned ETag receipts were
saved. Six placement starts contain eight literal native constraint packets,
including object UUIDs, arm/link indices, both local anchors and fixed-joint
types. All six selected rows match the earlier range hashes, their named
robot schema is `controller_groups`, and their robot joint vectors have 28
values. These are original saved packets, not source/contact-inferred grasps.

Some newer annotations use generic object IDs such as `tupperware` or `marker`,
while saved constraints identify objects such as `tupperware_238` or `marker_89`.
Before model evaluation, verify the actual task-scope binding of these roles.
Do not weaken the exact/unique resolver or choose a category member based on
the desired result. Any necessary disambiguation must have source evidence.

The next gate is a real L20 restore-and-model validation: compare the source's
28 joint positions and saved constraint fields with the simulator immediately
before policy inference, then require actual model actions and videos. Neither
the full source audit nor the six decoded candidate starts establishes a model
success rate. No new RGB training trajectories, fine-tune or paired model
comparison was produced by this source investigation.


## Recorded-native source checks and annotation binding

The source-backed validation path now verifies the recorded full 28 joint
positions, root pose (quaternion sign equivalence allowed), each native arm's
held object UUID and name, sorted target-link index, joint type and all four
local anchor fields immediately after restoration and before the completion
observer or policy can run. A changed field aborts the run. Sparse frames use
the latest explicitly saved robot payload. Legacy controller layouts remain
excluded from ordinary scoring; the custom reconstructed-native validation
hook is unchanged. Unrecorded physics caches and release/grasp timers are not
claimed as historically verified.

Generic annotation roles bind only through the source scene's current
`metadata.task.inst_to_name` mapping, using its BDDL instance names, concrete
scene names and recorded categories. Placement/release targets with multiple
source task instances may resolve through exactly one initially recorded held
object. Reference roles cannot use that hint. Ambiguous free-hand targets,
ambiguous containers and two same-category held objects remain setup
exclusions. The general exact/unique runtime resolver, original predicate,
provided subtask text and full100 parent panel stay unchanged. Binding receipts
record source identities and explicitly state that desired predicate truth
was not used.

All six captured original native starts passed independent CPU decoding,
full-file/selected-row hash comparison and source identity preview. The two
freeze_fruit targets resolve to tupperware_238 and tupperware_237; the living
room targets resolve to newspaper_53 and pot_plant_54; art-supply targets
resolve to marker_89 and glue_stick_92. This source-only preview is not simulator
restoration evidence. Actual GPU verification is still required.

The validation plan freezes every selected source file and row plus checkpoint
1375000's manifest hash. The worker checks these before inference; the output
validator requires the recorded-native proof and source-role receipt for
eligible actions. The launcher packages the new module and checks checkpoint
identity during submission preflight. The targeted regression run passed 158
checks, including malformed sources, changed native runtime state, ambiguous
roles, tampered receipts, plan identity, existing full100 behavior and the
previous reconstructed-native validation path. The next actual pilot uses one
L20, three tasks, six starts, at most 256 model controls per start (1,536 total),
with JSON and videos. It does not establish a full100 difficulty baseline.


The recorded-native pilot also reuses the existing actual action audit: each
model control records its preceding finite 61-value state and executed finite
23-value action. Result verification requires one audit row and a provided
subtask response per actual control, plus a checksum-bound video. The targeted
checks passed 163 cases after this evidence requirement was added. These are
pilot evidence artifacts, not newly exported training trajectories.

## 2026-10-07：原始原生状态六起点验证完成，正式基线重新准备

L20 作业 218183（代码 7bb1af4、checkpoint 1375000）已 FINISH/FINISHED。六个原始起点全部通过独立原始 HDF5 对照：28 个关节、根位姿、8 份原生抓取约束及注释对象来源一致；实际模型控制共 1,520 步，六段视频共 1,520 帧。结果为 1 条成功、5 条在 256 步上限超时。该短测只验证恢复与推理流程，不能作为 2048 步正式困难组基线，也没有生成可直接训练的新 RGB 轨迹。未保存的物理缓存与计时器仍不在验证范围内。

原始六起点证据：`eval_results/subtask_recorded_native_1375000_20261007_081519/validated_outputs/local_gpu_verification.json`。

正式 100 任务面板仍为 400 个原始实例、7,223 个可解析起点及 889 项不支持注释。新增严格的 `source_excluded` 记录，区分旧数据缺失抓取状态与模型失败：`success=null`、0 模型动作、不声明恢复快照或谓词测量，不进入成功率分母，也不会进入新增样本采集。每条记录绑定原始整文件及选中状态行 SHA256。CPU 预处理逐实例发布这些记录，避免为缺失源状态启动模拟器；现代数据继续执行 GPU 原生恢复。

新正式运行强制 `recorded_native_fields_v1`：每个已恢复起点先核对保存字段，结果附实际 61 维观察、23 维控制及固定子任务响应审计；汇总需匹配独立 HDF5 预检哈希。困难组采样需完成且通过原生状态验证的 2048 步基线，保留至少 10 个有效起点门槛，拒绝短测或旧版未验证汇总。

204 项针对性 CPU 检查通过。另用真实旧版文件验证 18 个起点均只能发布源数据排除记录，并对实际六起点 GPU 结果核对新正式结果格式；此格式验证不把短测改算成正式结果。完整正式基线、新成功 RGB 数据、实际补训及独立复测尚未完成。
