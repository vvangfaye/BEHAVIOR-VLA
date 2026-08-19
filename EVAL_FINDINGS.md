# Why the 50k-step baselines score ~0 in closed-loop evaluation

Audience: whoever is running training on the H20 node.
Status: both baselines fully evaluated closed-loop in OmniGibson on the L20 node.
Date: 2026-08-12

---

## TL;DR — the one thing to change

**Train longer. Nothing else comes close in expected value.**

Both models saw well under one pass over the data:

| model | steps | data actually seen |
| --- | --- | --- |
| π0.5 | 50,000 × batch 64 = 3.2M samples | **1.5% of one epoch** |
| GR00T N1.7 | 50,000 | **epoch 0.60** (trainer's own count) |

Dataset is 20,000 episodes / **210,916,774 frames** / 100 tasks / 30 fps.

For scale, the challenge's *own* provided GR00T baseline is `checkpoint-150000`, trained on a
**single** task. Per task, that is 150,000 steps against our ~500. Roughly **300x less
training per task**.

Target 150k–300k steps before drawing any conclusion about architecture, action
representation, or hyperparameters.

---

## 1. Closed-loop results (the actual benchmark metric)

100 tasks, `public_test` split, receding-horizon control, `max_steps = 1.5x` mean human
demo length. Every rollout ran to timeout.

| | rollouts | tasks | success | mean q_score | nonzero rollouts | tasks with any score |
| --- | --- | --- | --- | --- | --- | --- |
| π0.5 | 475 | 99 | **0** | **0.0045** | 14 | 8 |
| GR00T | 491 | 100 | **0** | **0.0008** | 3 | 3 |

Paired on identical (task, instance) pairs, n=472: π0.5 0.0046, GR00T 0.0008. Success is 0
for both. The nonzero scores are scattered partial goal conditions, not repeatable ability —
the two models overlap on only one task.

---

## 2. What the models did and did not learn

`normalized_agent_distance` is end-effector / base travel relative to the human demo:

| | base | left arm | right arm | arm/base |
| --- | --- | --- | --- | --- |
| π0.5 | **1.370** | 0.307 | 0.333 | 0.23 |
| GR00T | **2.303** | 0.116 | 0.155 | 0.06 |

Both **over-drive the base** (1.4x and 2.3x the human distance — wandering) and
**barely move the arms** (a third and an eighth of human arm motion). Rollout videos confirm
it: the robot drives around the room while the wrist camera views stay essentially fixed, and
target objects are never approached.

No arm motion means no grasping, no transport, no placement. `q_score ≈ 0` follows directly.

---

## 3. The proof: arm joints are worse than doing nothing

Open-loop action-prediction error, 200 samples over 100 tasks, raw joint units, scored over
the first 16 chunk steps (the window both models cover). `hold` = repeat the current joint
configuration for the whole chunk, i.e. freeze the arms.

| group | dims | hold | π0.5 | vs hold | GR00T | vs hold |
| --- | --- | --- | --- | --- | --- | --- |
| base | 3 | 0.0600 | 0.0281 | 2.1x better | 0.0251 | 2.4x better |
| torso | 4 | 0.0042 | 0.0030 | 1.4x better | 0.0071 | **WORSE** |
| left_arm | 7 | 0.0123 | 0.0138 | **WORSE** | 0.0199 | **WORSE** |
| left_gripper | 1 | 0.9779 | 0.0392 | 25.0x better | 0.0337 | 29.0x better |
| right_arm | 7 | 0.0110 | 0.0135 | **WORSE** | 0.0208 | **WORSE** |
| right_gripper | 1 | 0.9806 | 0.0352 | 27.8x better | 0.0316 | 31.0x better |

**All 14 arm dimensions are worse than freezing the arms, for both models.**

Where the aggregate advantage over `hold` comes from (dimension-weighted):

```
π0.5    grippers +96.3%    arms  -1.4%    base +4.9%
GR00T   grippers +101.5%   arms  -6.5%    base +5.6%   torso -0.6%
```

Two of 23 dimensions carry the entire result.

### Error growth across the chunk

```
step     hold     π0.5    GR00T      (mean |error| over the 14 arm dims)
   0   0.0039   0.0044   0.0105
   3   0.0072   0.0086   0.0138
   6   0.0100   0.0119   0.0178
   9   0.0130   0.0151   0.0222
  12   0.0164   0.0190   0.0271
  15   0.0195   0.0224   0.0313
```

`hold` grows because the demonstrated arm leaves its starting pose; that growth rate *is* the
true arm motion. A policy that tracked arm trajectories would stay flat and pull away from
`hold`. Instead both models grow at the same rate, parallel and above it — their output is
equivalent to "stay at the current pose" plus noise. GR00T is already 2.7x worse than `hold`
at step 0. Extending π0.5 to its full 32 steps continues the same trend (0.0368 vs 0.0314).

---

## 4. Do not use aggregate open-loop L1 as a progress signal

The earlier headline "L1 0.0157 vs hold 0.1008, 6.4x better" is **not evidence of
manipulation ability**. It is dominated by the two gripper dimensions, where freezing is
catastrophically wrong (0.98, near full scale) because the gripper command is effectively
binary. Meanwhile a whole action chunk of true arm motion is only about **0.0123 rad (0.7°)**,
so the arm term barely registers in the aggregate.

The original 16-step horizon made this invisible: at step 16 the arm curves are 0.0210 vs
0.0213, within 1.4% of each other.

**Report per joint group, always against the `hold` baseline.** `scripts/eval_openloop.py`
now does this, plus the per-chunk-position curve:

```bash
./scripts/eval_openloop.py --model pi05  --samples-per-task 2 --horizon 32 \
    --out eval_results/openloop_pi05_h32.json
./scripts/eval_openloop.py --model groot --samples-per-task 2 --horizon 32 \
    --groot-ckpt <checkpoint-N> --out eval_results/openloop_groot_h32.json
```

Changes made to that script: `--horizon` (default 32, hard cap 40 because `pick_samples`
reserves 40 frames of episode tail), `--dataset`, `per_position_l1` in the JSON, and a
`report()` that prints the per-group and per-position tables. Its hardcoded dataset path
`/var/tmp/behavior-1k/2026-challenge-demos` no longer exists and now defaults to
`dataset_views/pi05/2026-challenge-demos`.

**The check that matters: does `left_arm` / `right_arm` beat `hold`?** Today it does not.
Until it does, closed-loop success will stay at zero no matter what the aggregate L1 says.

---

## 5. Recommended actions, in priority order

1. **Train to 150k–300k steps.** This is the only change justified by the evidence. Everything
   below is secondary.
2. **Gate on the right metric.** Track per-group open-loop L1 against `hold` during training.
   The moment arm joints beat `hold` is the moment closed-loop success becomes possible.
3. **Only if arms still lose to `hold` at 150k+**, revisit the action representation. Arm
   actions are learned relative to current state, which makes "output zero" a near-optimal
   degenerate solution: per-chunk arm displacement is ~0.7°, so sitting at that solution costs
   almost nothing in loss. Options would be absolute joint targets, or up-weighting the 14 arm
   dimensions in the loss. Note the official baseline reaches usable behaviour with the
   *current* representation, so this is an amplifier of undertraining, not the root cause.
4. **Simulator instability** cost 25 π0.5 and 9 GR00T rollouts (~2-5%): PhysX blows up with a
   non-normalised gripper finger quaternion (`AssertionError` in `rigid_prim.py`) plus
   `Illegal BroadPhaseUpdateData`, during assisted-grasp raycasting. Does not affect the
   conclusions, but it caps achievable coverage.

---

## 6. Evaluation environment is validated — reuse it, do not rebuild blindly

`scripts/setup_eval_env.sh` was missing four things that each break evaluation. All are fixed
in the script now:

| missing | symptom |
| --- | --- |
| `joylo` (gello) | `omnigibson.eval.evaluator` imports `gello.utils.og_teleop_cfg`; upstream `setup.sh` rejects `--eval` without `--joylo` |
| Vulkan loader | no `libvulkan.so.1` anywhere in the container; Kit cannot start |
| `botocore` | `omni.replicator.core` S3 backend needs it; OmniGibson's kit config never enables the `pip_archive` extension that normally supplies it |
| `libGLU.so.1` | `libneuray.so` fails to load, which cascades into every RTX raytracing shader failing and then a segfault during scene load |
| `osqp` | `isaacsim.robot.wheeled_robots` (HolonomicController) fails to start; benign here but it should load |

The script's own Vulkan probe also segfaulted on newer loaders (fixed 824-byte ctypes buffer
for `VkPhysicalDeviceProperties`); it now only checks instance creation and device enumeration.

`scripts/eval_b1k.sh` fixes: pass the eval env's graphics environment explicitly
(`LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, `__EGL_VENDOR_LIBRARY_FILENAMES`,
`OMNI_KIT_ACCEPT_EULA`) because it invokes the interpreter directly rather than sourcing the
activation hook; `EVAL_ENV` default aligned with `ENV_ROOT`; `activate_env()` wraps
conda-pack's activate script which aborts under `set -u`; per-port server logs;
`INSTANCE_INDICES` for instance subsets.

`scripts/eval_sweep.sh` is new: shards the task list round-robin across N workers, each owning
two GPUs and its own port. Note the real bottleneck is **CPU, not GPU** — each Isaac Sim
process consumes ~10 of the 15 cores, so 4 workers already oversubscribe 3x while GPUs sit at
0-45% and 6-15 GB of 46 GB. Adding workers makes it slower. Measured throughput is ~172k sim
steps/hour for both models, i.e. ~42 h for 100 tasks x 5 instances.

### Two source fixes that affect training-adjacent code

- **`openpi/src/openpi/configs/robots/b1k.py`**: robot name was `robot`, but BEHAVIOR-1K 3.9.0
  names the R1Pro `robot_r1` and namespaces all observations under it. Serving raised
  `KeyError: 'robot::proprio'` on the first step. Renamed the robot and its three camera
  `obs_key`s. `dataset_key` (the training path) is untouched, so trained checkpoints are
  unaffected.
- **`Isaac-GR00T/gr00t/eval/eval_b1k_wrapper.py`**: `act_receeding_horizon` did
  `{k: v[indices] for k, v in input_batch.items()}`, but `process_input` returns a mapping of
  modality name to a *dict* of entries, so indexing the top-level values with an index array
  raised `TypeError: unhashable type: 'numpy.ndarray'`. That control mode was unusable. Added
  `_select_batch_elements`, which descends one level (arrays indexed on axis 0, language's
  inner list indexed by position).

### GR00T checkpoint was edited — read this before moving it

`nvidia/Cosmos-Reason2-2B` is a gated HF repo and unreachable here, and the local cache is
missing `processor_config.json` (which the checkpoint happens to contain). A merged local copy
was assembled at:

```
/mnt/workspace/wangfaye/cache/local_models/nvidia/Cosmos-Reason2-2B
```

The path deliberately contains `nvidia/Cosmos-Reason2` to satisfy the string check in
`gr00t_n1d7.py:491`, while being a local directory so transformers short-circuits
`_is_local or is_base_mistral(...)` and never hits the network.

In `outputs/gr00t_b1k_full_20260804_225513/checkpoint-50000`, **`model_name` in both
`config.json` and `processor_config.json` now points at that absolute path** (originals kept as
`*.orig`). This is not portable: on another host, either recreate that directory or restore
the `.orig` files and supply HF access.

---

## 7. Corrections to earlier claims — do not propagate these

Three things stated earlier in this investigation were wrong:

1. **"GR00T is ~6x slower per step than π0.5."** Wrong. That was measured on one small task
   where policy inference dominates. Measured over the real task mix, throughput is
   **identical** (171.6k vs 173.3k sim steps/hour, ratio 1.01x) because CPU-bound physics
   dominates and inference is a small fraction. The 10.6-day ETA derived from it was nonsense.
2. **"GR00T replans every 40 steps."** Wrong. Its modality config sets
   `delta_indices=list(range(16))`, so it emits **16** actions; the checkpoint's
   `action_horizon=40` is only model capacity. Both models replan every 16 steps, so the
   closed-loop protocols were already aligned.
3. **"π0.5's sweep took ~19 h."** It took **42.4 h** (475 rollouts). The 19 h figure was a
   guess, never measured.

Also note: GR00T's default `temporal_ensemble` control mode infers at *every* simulator step,
versus every 16 for receding horizon. The sweep reported here uses receding horizon for both
models. If you compare against numbers produced with `temporal_ensemble`, they are not the
same protocol.

---

## 8. Artifacts

```
eval_results/pi05/<task>/json/*.json      475 closed-loop rollouts + videos/
eval_results/groot/<task>/json/*.json     491 closed-loop rollouts + videos/
eval_results/openloop_pi05_h32.json       200-sample open-loop, horizon 32
eval_results/openloop_groot_h32.json      200-sample open-loop, horizon 32 (16 scored)
eval_results/{pi05,groot}/sweep_logs/     per-worker sweep logs
```

Checkpoints evaluated:

```
openpi/outputs/checkpoints/pi05_b1k/pi05_b1k_full_20260803_162620/49999
Isaac-GR00T/outputs/gr00t_b1k_full_20260804_225513/checkpoint-50000
```

## 9. What was not verified

No ablation was run to prove that more training fixes this — that needs a 150k-step checkpoint
to compare against. The attribution rests on three things: the hard data-coverage numbers, the
official baseline's training scale, and the observed split between easy channels (learned) and
the hardest channel (not learned), which is what a severe compute shortfall looks like.
