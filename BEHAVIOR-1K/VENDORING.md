# Vendored BEHAVIOR runtime

This directory is tracked in the parent BEHAVIOR-VLAs repository. The original
NAS snapshot is preserved in commit `640e241`; its OSS archive checksum is in
[.source_provenance.json](.source_provenance.json). That file describes the
historical snapshot, not the current package version.

The runtime is upgraded to OmniGibson **3.9.3**, using the official
[`v3.9.3-post1` release](https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.3-post1),
commit `bd049de3119acdcdf2334fe9e1ebe060fa20c108`. There is no upstream `v3.9.3`
Git tag. [UPSTREAM.json](UPSTREAM.json) records the exact source and local changes.

## Scope

Included: OmniGibson, BDDL and JoyLo/GELLO runtime source, OmniGibson tests,
package build metadata, setup scripts, licenses, and the documentation logo
referenced by the runtime. This is a runtime subset, not a complete upstream
checkout. The documentation website, asset pipeline, knowledgebase website,
and eval-jobqueue are omitted. Use the parent project's
`scripts/setup_eval_env.sh` to install the included runtime on a prepared Linux
GPU host; do not enable omitted components through the upstream setup script.

Simulator assets, model weights, datasets, local environment keys and outputs
are not committed. BEHAVIOR 3.9.3 still requires `behavior-1k-assets-3.9.0` and
`omnigibson-robot-assets-3.8.2`, as specified by upstream `asset_utils.py`.
Isaac Sim 5.1 and Kit 107.3.1 remain the runtime dependencies.

## Evaluation entry points

- `python -m omnigibson.eval.eval` retains the project's sequential single
  environment evaluation, external subtask conditioning, selector, temporal
  voting, server identity checks and diagnostic result metadata. The
  `SingleEnvironmentAdapter` translates the 3.9.3 vectorized observation,
  action, task scope, goal and counter APIs for the existing queue and
  exact-state subtask evaluator.
- `python -m omnigibson.eval.batched_eval` runs the official 3.9.3 batched
  evaluator. `batched_eval.py` and `batched_evaluator.py` are unchanged copies
  of upstream's `eval.py` and `evaluator.py`; their hashes are in UPSTREAM.json.
  The batch entry point exposes upstream's options, not the custom single
  environment selector options. `BatchedEvaluator` remains importable from
  `omnigibson.eval.evaluator` for compatibility with upstream imports.

The websocket client retains project-specific subtask metadata while also
supporting upstream action chunk replay. Replay is disabled by default.

## Validation

The parent `scripts/test_behavior_version_compat.py` exercises CPU-only
observation/action conversion, per-environment predicates and task metrics,
shape-preserving counters, camera configuration, policy action shapes, server
identity/voting contracts and action chunk replay/reset. Parent evaluation and
submission regression tests are separate from simulator tests.

These checks do not start Isaac Sim. Closed-loop simulation and exact-state
restoration must additionally be smoke-tested in the prepared evaluation
conda environment on a GPU with RT cores, such as L20. H20 and macOS cannot
perform that validation.
