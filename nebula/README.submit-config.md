# Local Nebula submission configuration

The repository launchers source `nebula/config.sh`. This file is ignored by Git.
To configure another machine, copy `nebula/config.example.sh` to
`nebula/config.sh` and run `chmod 600 nebula/config.sh`.

The configuration reuses credentials from `~/.ai-hub-cli/config.yaml` and
`~/.ossutilconfig`. No credentials are stored in the repository. Explicit
`AI_HUB_TOKEN`, `AI_HUB_NAMESPACE`, `OSS_ACCESS_ID`, and `OSS_ACCESS_KEY`
environment variables override credentials from those stores. Alternative
profile files can be selected with `AI_HUB_CONFIG_PATH` and
`OSSUTIL_CONFIG_PATH`.

This project's defaults are:

- AI-Hub namespace: loaded from the existing CLI profile (`eta` on this machine).
- Queue: `aios_specific_nm125_l20`; GPU type: `l20`.
- Runtime: `pytorch2120`.
- OSS bucket: `traffic-model`; endpoint: `http://cn-shanghai.oss.aliyun-inc.com`.

Use `NEBULA_OSS_BUCKET` and `NEBULA_OSS_ENDPOINT` to change the project storage
location. These settings deliberately do not inherit unrelated global
`OSS_BUCKET` or `OSS_ENDPOINT` values. Queue, GPU type, and runtime can be
overridden with `QUEUE`, `GPU_CARD_TYPE`, and `ALGO_NAME`.

The current AI-Hub CLI is discovered through `PATH`, with a fallback to
`~/.local/bin/ai-hub-cli`. The configuration requires Python 3 but no third-party
Python packages.

`nebula/submit_pi05_eval.sh` now defaults to `SUBMIT_STORAGE_MODE=oss`:
submission uses the OSS Python SDK for metadata checks, so it needs no local
`/data/oss_bucket_0` mount. Install `nebula/requirements-submit.txt` into a
Python environment and set `NEBULA_SUBMIT_PYTHON` to that environment's Python.
The launcher also discovers this project's parent `.venv` automatically.

Inputs can use `oss://traffic-model/...` URIs or their equivalent container
paths. URI inputs are converted to `/data/oss_bucket_0/...` only for the
worker's runtime settings. Preflight checks environment archives, model format,
normalization statistics, checksum manifest entries, `_READY`'s manifest digest,
and unused output and receipt prefixes. Large models and asset archives are
not downloaded on the Mac. Full content SHA256 verification remains mandatory
in the Nebula worker before evaluation.

The archived simulator environment predates OmniGibson 3.9.3's Warp imports.
Prepare the official pinned Linux wheel once with
`python3 nebula/eval_runtime.py fetch`. It is ignored by Git. Every submission
verifies its published SHA256 before any run reservation and bundles it with a
hashed requirement. The worker installs `warp-lang==1.12.0` offline into its
private restored evaluation environment using `--no-index --no-deps
--require-hashes`. It then executes a Warp CUDA kernel, checks Torch sharing and
imports the bundled simulator before staging large checkpoints/assets. The
complete evaluator import is checked after task metadata extraction. The base
OSS environment archive remains reusable; this repair does not replace it.

`DRY_RUN=1` performs read-only OSS checks and local code packaging, then exits
without reserving a run or submitting a GPU job. `KEEP_BUNDLE=1` preserves the
code bundle for inspection. Credentials and `nebula/config.sh` are excluded
from the bundle.

Code packaging includes the tracked `BEHAVIOR-1K` runtime source. Its NAS
baseline is preserved in Git, and the runtime is upgraded to OmniGibson 3.9.3
(upstream tag `v3.9.3-post1`). See `BEHAVIOR-1K/VENDORING.md` and
`BEHAVIOR-1K/UPSTREAM.json` for provenance. Large simulator assets remain on OSS.

`EVAL_PROFILE=subtasks100` runs independent exact-state subtask diagnostics over
all 100 challenge tasks with a joint JAX policy. Set `CONFIRM_FULL=1`,
`SUBTASK_PREDICTION=1`, `WRITE_VIDEO=1`, and 4 or 8 workers. Supply a frozen full100
`SUBTASK_PANEL_PATH`, its matching `SUBTASK_RAW_INVENTORY_PATH`, and
`SUBTASK_RAW_SOURCE`. The submission checks all selected raw objects against the
frozen size, ETag and CRC64 before packaging. The worker copies only selected
HDF5 files, validates their schema and frame ranges, and records full SHA256
before any rollout. `SUBTASK_SAMPLE_MAX_STEPS` defaults to 2048.

Each annotated segment starts from its recorded demonstration state and uses its
fixed annotated subtask with the checkpoint's `task_and_subtask` conditioning.
A supported completion predicate must remain true for three consecutive control
steps. Unsupported skills, ambiguous object bindings, unobservable states and
predicates already true at reset are explicitly excluded from the success-rate
denominator. This is a training-set diagnostic, not an official benchmark score.
`summary.json`, `task_success.csv` and `subtask_success.csv` report counts and
rates, including unsupported-only prompts with empty rates. Videos are mandatory
for eligible rollouts; `_SUCCESS` is published only after strict validation.

Actual submissions save local receipts under
`eval_results/.submission_receipts` (or a local `RECEIPT_ROOT`). Before invoking
AI-Hub, a unique `.claim.json` is created on OSS with the no-overwrite header.
After the CLI returns, its receipt and log are synced to OSS under
`<OUTPUT_BASE>/.submission_receipts`. An existing output, claim, or receipt
blocks reuse of that run ID. If submission or receipt sync is interrupted,
inspect the local receipt and task status before taking further action.

`SUBMIT_STORAGE_MODE=mounted` retains the earlier Linux submission path with
local checks through an existing OSS mount. The other training launchers still
use their original storage checks.

The default configuration selects L20. Select GPU counts in the launcher's
cluster file separately; `nebula/multihost_cluster_4gpu.json` requests one worker
with 4 GPUs, 16 CPUs, and 100000 MB of memory.

For `subtasks100`, result synchronization uses Python's standard library and
does not require `rsync` in the worker image. Only changed, stable files are
copied through temporary destinations; attempts, live SQLite files and runtime
markers are excluded. `_SUCCESS` is published only after the existing strict
summary and completion checks pass. A failed final sync preserves the original
supervisor error.

Full snapshots captured after replaying an old demonstration use named state
dictionaries (`named_simulator_state_v1`) with detached tensors. Raw HDF5 rows
continue to use their recorded flat field order. This avoids interpreting new
runtime state fields as the next object's UUID. Restoration checks the complete
object set before and after loading and keeps cached snapshots immutable.

Before a full retry, supply `SUBTASK_VALIDATION_PLAN_PATH` with a version-1
`subtask_state_validation` plan referencing 1-12 sample IDs from the unchanged
full100 panel. Set one worker, `nebula/eval_cluster_1gpu.json`, and a matching
`SUBTASK_SAMPLE_MAX_STEPS` of at most 256. The worker stages only the selected
raw episodes and reuses the production evaluator and policy launcher. Validation
requires all selected JSON files, actions and nonempty videos for eligible
samples, and at least one eligible model rollout. Its separate
`validation_summary.json` and labeled `_SUCCESS` confirm only this limited
validation; they do not represent the full panel's success rate.
