# SenseCore subtask trajectory collection

Pre-migration repository snapshot: `89276b4`. The reference repository is
`/Users/faye/code/BEHAVIOR-VLAs` at `e3f4fb7`; only collection and episode
isolation are being migrated. Planning and teacher changes are outside this work.

`scripts/accepted_subtask_collection.py` consumes the immutable accepted-rate
export rather than treating the unfinished full100 aggregate as complete. The
2026-10-10 20:12 Shanghai export has 98 accepted tasks and two pending tasks.
Groups use eligible starts as the denominator. Groups with fewer than ten
eligible starts retain an explicit `underpowered` flag. The initial native-state
adapter defers physical-history groups and older acceptance records that omit
the producing receipt; it does not reinterpret them as model failures.

The measured checkpoint is 1375000. Source HDF5 and selected native row hashes,
accepted baseline result hashes, task/subtask labels, held-out task instances,
policy seeds, and budgets are frozen into each plan. The existing strict full100
selection gate remains available unchanged.

`sco/prepare_subtask_collection.py` creates a separate source snapshot, runs
CPU regression tests in the archived worker environment, checks the original
baseline artifacts, and freezes a finite pilot. `sco/submit_subtask_collection.py`
records submission intent before mutation, checks current owner-wide standard
GPU use and personal storage headroom, and submits one GPU via the known
5090 template. Standard pilot output must finish real LeRobot export/read-back
before the submission tool admits Spot collection.

`sco/prepare_subtask_collection_wave.py` freezes at most 30 attempts for the
first Spot wave using the same verified code and measured checkpoint. It unions
the reserved-instance lists, selects two distinct original starts per group
where available, and creates disjoint task shards without changing seeds. The
shared contract is written once before GPU submission. Each task has its own
writer lock and completion receipt; `sco/finalize_subtask_collection.py` accepts
only exhaustive, disjoint, successful platform executions, then deduplicates and
exports one dataset. Admission reserves twice the plan's raw-output allowance
plus 16 GiB for export and other overhead. The Spot gate queries the standard
pilot's exact name and UID again and requires successful real dataset export.

`sco/run_subtask_collection.py` reuses the existing policy/simulator launcher.
Training arrays contain three 224px RGB cameras, actual proprioception, and the
executed 23-dimensional actions paired with the preceding observations. Only
eligible successful trajectories can enter the exported dataset. Failed trials
remain diagnostics. Export includes the existing video and 32-action-window
read-back checks. Zero successes never count as a training-data deliverable.

Each source episode runs in a separate simulator process. Completed attempts
are validated before a resume; incomplete directories are preserved separately.
The worker takes an exclusive file lock. Recovery requires the exact prior UID,
terminal status and evidence that its writer stopped. The submit CLI accepts
`--recover-receipt` after querying the exact old UID and terminal worker. It
preserves the output root and completed attempts, archives the prior status,
and permits at most three executions per shard. Shared-task recovery moves only
that task's incomplete attempts. Never launch recovery alongside an old writer.

The simulator environment deliberately lacks some training packages. Run
`sco/verify_collection_training_reader.py` with the archived policy Python to
check the actual OpenPI reader without modifying either environment. This CPU
check verifies decoded observations, labels and 32-action windows; it does not
claim tokenization, GPU training or model improvement.

`sco/continue_subtask_collection.py` is a finite local dispatcher. It waits for
the exact standard-pilot UID, submits a 0-GPU artifact acceptance job, and only
then submits all disjoint Spot shards sequentially. After successful shards it
runs one 0-GPU finalizer and checks the actual OpenPI reader. It queries current
personal storage quota before each GPU submission, freezes its helper sources,
keeps an exclusive dispatcher lock, and saves every intent before mutation.
Unexpected submissions, failed pilot acceptance, exhausted recovery attempts,
or its explicit deadline preserve all jobs and outputs and set `needs_attention`.
It does not change a queued Spot shard into a standard job.

Example (use the Python containing the configured private AOSS client):

```bash
/private/tmp/behavior-transfer-env/bin/python sco/continue_subtask_collection.py \
  --pilot-receipt eval_results/sco_subtask_collection_20261010/pilot_fixed_submission.json \
  --wave-prepared eval_results/sco_subtask_collection_20261010/wave_fixed_prepared_current.json \
  --output-dir eval_results/sco_subtask_collection_20261010/dispatch \
  --max-hours 28
```

The dispatcher does not install a recurring automation. Its status, log and
finite submission receipts belong to this run; `dispatch/status.json` records
completion, the final dataset path, or the exact reason for attention.

Initial pilot: two `cook_a_brisket` groups, one non-held-out start per group,
three policy seeds per start, maximum 2048 actions per attempt. This is six
attempts, with a conservative uncompressed output allowance recorded in the
plan. Source-scene diversity comes from distinct original starts; new policy
seeds alone do not create new scenes. Fine-tuning is outside this collection
migration request.

Operational receipts, failed preparation attempts and current progress live in
`eval_results/sco_subtask_collection_20261010/` (ignored generated artifacts).
The online cluster rules page was inaccessible during this run. The local
2026-09-11 rules snapshot was read; submissions retain the four-standard-GPU
owner limit and explicit per-job timeout. AFS filesystem free space must not be
mistaken for the personal directory quota.

## Full coverage requested on 2026-10-11

The current campaign supersedes the initial 70-group selection. It includes all
299 accepted `(task, subtask)` groups whose measured completion rate is strictly
below 80%, across 89 tasks, using the same frozen 98-task export and checkpoint
1375000. No minimum of four eligible samples or nonzero-success filter applies.
The two pending baseline tasks are not silently treated as failures. Counts and
source identities remain tied to the original export.

`scripts/full_subtask_collection.py` freezes one group per shard, at most three
non-held-out training starts and ten policy seeds per start. The original native
protocol and the disclosed physical-history protocol have separate hashes and
validation contracts. Historical recovery only uses recorded controls before the
selected start; its receipts explicitly state that original unrecorded grasp
anchor bytes cannot be verified. Original baseline JSON, raw HDF5 rows, history
action prefixes, and videos are independently audited by
`sco/prepare_full_subtask_collection.py`; their recounted eligible/success totals
must match every selected group. Older accepted records receive additional source
evidence instead of fabricated producing-job metadata.

A group whose eligible original starts all belong to held-out instances requires
additional public training demonstrations. `scripts/additional_training_panel.py`
uses a separate collection-only schema, checks literal annotations and public
training metadata, and excludes held-out instances. Additional source preflights
are identified separately from measured baseline results. The sorting-books
notebook-to-bookcase group has three such public training starts; its baseline
rate is retained, without inventing a new measured rate.

`sco/run_full_collection_shard.py` writes large outputs to worker-local scratch.
`scripts/collection_checkpoint.py` saves each completed attempt to the existing
private AOSS bucket with content MD5, length and SHA256 receipts. Recovery checks
the prior exact UID and terminal worker before restoring the verified objects;
it resumes completed trial identities without resetting seeds or attempts. Each
group permits at most three executions, with a 24-hour timeout per Spot execution.
Small status files and logs remain in the personal AFS directory.

The standard gates cover native restoration, physical history, and additional
training starts. They download completed object checkpoints and validate the
restored collection, export successes through LeRobot, and run the actual OpenPI
reader. Final dataset archives use multipart MD5/ETag verification, and acceptance
JSON is read back exactly. CPU nodes observed in this run could not resolve AOSS;
CPU preparation inputs are therefore staged on AFS through an existing owned GPU
worker. GPU storage access is tested directly before Spot admission.

`sco/submit_full_collection_shard.py` records exclusive intent before submitting
one GPU. Reserved admission freshly checks all owner-wide standard requests;
Spot admission requires the exact successful reserved gate UIDs, matching frozen
source/driver identities, and real object acceptance. Personal AFS headroom covers
small logs, while each worker independently verifies its scratch budget.

`sco/continue_full_subtask_collection.py` waits for all three reserved gates and a
complete 299-group preparation, then sequentially submits all independent Spot
shards without a free-capacity gate. Submission prioritizes demonstrated success
yield, retaining every zero-success group. It holds one local writer lock, preserves
receipts and recovery generations, and has a finite six-day deadline. Reports
separate successful attempts from unique exported episodes and frames. Completed
zero-success shards remain diagnostics and never become training data. The
manifest lists independently readable per-group datasets and standard-gate data;
no training or planning-method changes are part of this campaign.

A standard-gate failure exposed lowercase AOSS HTTP response headers. Receipt
checks now normalize header names; compatibility tests cover canonical and
lowercase responses. A real file PUT and two-part multipart upload verified byte
counts, content MD5 and the final multipart ETag. Failure paths retain bounded
worker-log tails and completed-trial metadata on AFS for diagnosis.

`sco/revise_full_collection_source.py` creates a descendant snapshot for such
storage-only fixes. It restricts changed files to storage/runtime helpers and
tests, verifies all source hashes, reruns CPU tests, and copies the unchanged
299 plans and source evidence byte-for-byte. The descendant report records the
original audit's source identity and path; it does not claim the baseline audit
was rerun under a new version. Small reserved gates must pass again before Spot.

Production acceptance additionally reads every attempt's committed object receipt
and checks archive headers against it before packaging data. This catches a final
upload exception even when the simulator exits with status zero after logging a
Python error. A regression check specifically rejects a complete local collection
whose last object receipt is missing.

The current execution uses a shorter independently audited history probe:
`chop_an_onion-instance-97-episode-8457-segment-2-interval-0`, starting at frame
548 with a nonempty recovered initial grasp. It replaces the 5,079-frame
gift-basket probe to reduce standard-resource test time. Its prepared receipt
cites the original full source audit and the same frozen runtime version; the
299 production plans and 8,700 policy trials are unchanged. Stop receipts retain
the exact UIDs of the superseded standard tests.

ACP reports a stopped worker as `Deleted` after removing its pod. Recovery
accepts this observed terminal phase alongside `Failed` and `Succeeded`, while
still requiring the exact old job UID and a `FAILED` or `SUSPENDED` job state.
Active, uncertain, absent, or duplicate worker rows cannot authorize recovery.
Submission-client hashes are recorded separately from the unchanged simulation,
collection, and training-reader driver identities.

## Verified execution on 2026-10-11

The frozen runtime source is
`acf7124409380ff2869c749097540a17251928811c6daf7fe463f7b7915841c6`.
All three reserved gate jobs reached `SUCCEEDED` with verified object receipts,
restoration, and final archives:

| Gate | Job UID | Successful attempts | Exported frames |
| --- | --- | ---: | ---: |
| Native state | `fe0d1f7543284c23ada95c10d2e76656` | 1 / 1 | 131 |
| Recorded history | `1a9a794024db4804bb2e045cf9d973e1` | 1 / 1 | 1,795 |
| Additional public training start | `0240543fa10d4a1897c92041e3aeeaf9` | 0 / 1 | 0 |

The additional-start gate completed its 2,048-step policy attempt and storage
validation without achieving the subtask; it contributes no training episode.
The two successful gates also passed the actual OpenPI training reader. Together
with the earlier verified three-episode standard pilot, content-hash deduplication
and archive readback confirm five distinct successful trajectories and 2,275
frames. This validates training-data loading, not a new model training run.

Generated execution records are in
`eval_results/sco_subtask_collection_20261011/`: `collection_plan_299.md` lists
every selected group, `currently_verified_training_data.json` identifies the
verified dataset archives, and `RUN_MANIFEST.json` connects frozen sources,
standard tests, Spot receipts, and the finite dispatcher. The private AOSS prefix
is `behavior1k/subtask-collection/full299-20261011/` in the existing personal
bucket. The Spot campaign's evolving dataset index is
`d2f2cd51e380/dataset_manifest.json` under this prefix.

## Additional low-success attempts on 2026-10-11

`scripts/extend_subtask_collection.py` creates a separate deterministic campaign
from the frozen 299-group inputs. The authorized extension adds 1,080 attempts
in 36 independent 30-attempt shards for 32 groups:

| Original measured rate | Groups | Additional attempts per group | Total attempts per group |
| --- | ---: | ---: | ---: |
| Greater than 0%, below 20% | 4 | 60 | 90 |
| At least 20%, below 50% | 28 | 30 | 60 |

The original 8,700 attempts remain intact, giving a cumulative budget of 9,780.
The 43 zero-success groups retain their first-round budgets pending restoration
and failure evidence. Groups at or above 50% receive no extension in this wave.
These are attempt budgets, not promises of successful training trajectories.

The extension retains the original three audited starts per group, held-out
exclusions, checkpoint, source hashes and restoration protocols. Fresh policy
seeds and trial IDs are disjoint from the original campaign and every other new
shard. This adds policy variation; it does not add scene instances. Reconstructing
on AFS requires the locally frozen fingerprint, and all outputs use new prepared
directories and object namespaces. Identical collection drivers allow reuse of
the three standard gates, whose exact UIDs and accepted objects are checked again
at each Spot submission.

`sco/continue_collection_extension.py` sequentially submits the finite extension
and can monitor both the original 299 and new 36 shards. It shares the original
dispatcher's lock, reads the highest saved recovery generation, preserves unknown
submission intents for exact-name reconciliation, and retains the original
deadline (2026-10-17 02:15:05 Asia/Shanghai). Each execution remains one Spot GPU
for at most 24 hours, with at most three executions of each frozen shard. New
attempts start as generation zero; recovery never changes their frozen seeds.

Generated plans, AFS staging proof, validation and per-job submission receipts
are under `eval_results/sco_subtask_collection_20261011/extension_low50_v1/`.
The campaign fingerprint is
`05524aa5e8d9e8ad9bee3680e54bf0fadda090f1533499582de046b2d70ee541`.
Its `dataset_manifest.json` becomes the combined index for the original and
additional shards plus standard gates; the old index is retained unchanged.
It binds accepted results to the highest execution generation and deduplicates
whole archives by SHA256. Cross-archive episode-content deduplication still
requires reading the exported trajectory provenance during dataset assembly.

On 2026-10-11 the user cancelled all collection jobs that had not entered
`RUNNING`. The local automatic dispatcher was stopped before cancellation;
`USER_CANCEL_NONRUNNING.json` in each dispatcher directory now blocks automatic
submission and recovery until the user explicitly reauthorizes them. Existing
running workers and collected objects are preserved. Stop intents, exact UIDs
and live verification are recorded in `extension_low50_v1/cancel_nonrunning/`.

The user subsequently reauthorized submission and recovery with a **64 GPU cap**
for this campaign. Already-running workers drain naturally; no replacement is
admitted until total requested GPUs fall below 64. Waiting, starting, restarting,
recovering and stopping jobs consume slots alongside running jobs. The bounded
controller requires `--admission-policy`, bound to the existing cancellation
markers, frozen campaign and original deadline. The markers remain in place to
block old unbounded entry points. A later cancellation invalidates the policy.

`sco/bounded_collection_admission.py` reads every active owner job with bounded
pagination, counts only this finite campaign, and reserves a slot before helper
preflight. Reservations survive controller restarts and cover unknown submission
outcomes and platform-list visibility delays. Actual cloud creation rechecks the
cap; incomplete or failed platform queries cannot authorize submission. One
shared controller lock covers both initial submissions and recovery generations.
The current policy, frozen controller and operational evidence are recorded in
`extension_low50_v1/bounded64/`; the 9,780-attempt budget and original execution
deadline are unchanged.

The user later replaced the 64-GPU limit for this collection with admission
against current free Spot capacity. `sco/free_capacity_collection.py` resumes
only the 36 cancelled low50 extension shards, preserving their 1,080 frozen
attempt identities and using generation 1. It requires an explicit authorization
file and a target-pool capacity observation no older than 120 seconds. Complete
slots are the minimum of GPU, vCPU/14 and GiB/60; owner pending requests and
persistent unknown or not-yet-visible submissions consume slots. Capacity is
checked before preparation and again at the actual create RPC. Stale or
insufficient observations pause creation until refreshed. Both existing writer
locks are retained; the old automatic controller stays stopped and a separate
observation-only monitor resumes after the finite batch releases the locks.
This change does not authorize recovery of checkpoint evaluation jobs.

All 299 independent Spot jobs were submitted for campaign `d2f2cd51e380`,
covering 870 training starts and 8,700 distinct policy trials. Each initial
receipt retains its exact job UID, frozen plan, source and driver hashes, one-GPU
Spot request, and 24-hour timeout. The complete initial submission index and
fresh platform identity verification are saved locally as
`spot_initial_submissions_299.json` and
`spot_initial_platform_verification_299.json`. An additional AOSS copy of these
internal operational metadata files was declined by automatic approval review;
the local records are the current handoff. Dataset checkpointing and the finite
collection dispatcher continue separately as already submitted.
