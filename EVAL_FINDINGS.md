# Why the B1K baselines score ~0 in closed-loop evaluation, and what more training bought

Audience: whoever is running training on the H20 node.
Status: π0.5 evaluated closed-loop at both 50k and 270k steps, GR00T at 50k, on the L20 node.
Date: 2026-08-12, updated 2026-08-23 with the 270k results.

---

## TL;DR — keep training; it is working, but it has not crossed the line yet

"Train longer" was the recommendation in the first version of this document. It has now been
tested, and it holds: π0.5 at **270k steps** beats the same model at **50k steps** on the
identical 100 tasks x 5 instances.

| paired on identical (task, instance), n=471 | 50k | 270k |
| --- | --- | --- |
| task success | 0 | **0** |
| mean `q_score` | 0.0046 | **0.0150** (3.3x) |
| rollouts scoring above zero | 14 | **31** |
| best single rollout | 0.333 | **0.500** |
| tasks scoring above zero | 8 | **16** |

So keep going — the next checkpoints are worth evaluating, and 270k is clearly not the ceiling.
But note what has *not* changed: **still zero full task completions in 488 rollouts**. Partial
credit is rising; nothing finishes.

Two things to take from this beyond "keep training":

- **Open-loop L1 is a weak predictor of closed-loop gains.** Arm-joint error improved only
  ~12% between the two checkpoints, while closed-loop mean `q_score` tripled. Do not read a
  small open-loop delta as "not learning much"; the earlier version of this document made
  exactly that mistake and under-called the gain.
- **The behaviour pattern did not change.** Measured as agent path length / human path length
  (medians, paired rollouts): base 0.99x -> 0.81x, left arm 3.56x -> 3.07x, right arm
  3.25x -> 3.00x. The base covers roughly the human's ground distance while the arms sweep
  three times the human path — jitter, not reaching, since `agent_distance` is cumulative path
  length rather than net displacement. Training shrinks both, i.e. the robot moves *less*, but
  the arms are still 3x too busy.

For scale on how undertrained these still are: the dataset is 20,000 episodes /
**210,916,774 frames** / 100 tasks / 30 fps. At 270k steps x batch 64 that is 17.3M samples,
about **8% of one epoch**. The challenge's own provided GR00T baseline is `checkpoint-150000`
on a **single** task.

---

## 1. Closed-loop results (the actual benchmark metric)

100 tasks, `public_test` split, receding-horizon control, `max_steps = 1.5x` mean human
demo length. Every rollout ran to timeout.

| | rollouts | tasks | success | mean q_score | nonzero rollouts | tasks with any score |
| --- | --- | --- | --- | --- | --- | --- |
| π0.5 50k | 475 | 99 | **0** | 0.0045 | 14 | 8 |
| π0.5 **270k** | 488 | 99 | **0** | **0.0145** | **31** | **16** |
| GR00T 50k | 491 | 100 | **0** | 0.0008 | 3 | 3 |

Paired on identical (task, instance) pairs, n=471: π0.5 50k 0.0046 vs π0.5 270k 0.0150. Success
is 0 everywhere.

At 50k the nonzero scores were scattered single instances — not repeatable ability. At 270k that
is partly no longer true: `make_rose_centerpieces` scores on all three evaluated instances and
`organizing_art_supplies` on two, which is a real if narrow capability.

Where the 270k gain came from — 11 tasks went from zero to scoring, 3 regressed:

| task | 50k | 270k |
| --- | --- | --- |
| `thawing_frozen_food` | 0.000 | **0.244** |
| `collecting_childrens_toys` | 0.000 | **0.171** |
| `make_rose_centerpieces` | 0.000 | **0.150** |
| `installing_a_scanner` | 0.000 | **0.100** |
| `organizing_art_supplies` | 0.000 | **0.080** |
| `putting_away_toys` | 0.000 | **0.075** |
| `collecting_aluminum_cans` | 0.000 | **0.067** |
| `laying_tile_floors` | 0.000 | **0.050** |
| `polishing_shoes` | 0.000 | **0.033** |
| `setting_the_fire` | 0.000 | **0.025** |
| `clean_up_your_desk` | 0.000 | **0.018** |
| `picking_up_toys` | 0.033 | 0.067 |
| `make_gift_bags_for_baby_showers` | 0.033 | 0.067 |
| `tidying_bedroom` | 0.133 | 0.133 |
| `putting_shoes_on_rack` | 0.100 | 0.100 |
| `setting_mousetraps` | 0.033 | 0.033 |
| `putting_away_Halloween_decorations` | 0.057 | **0.000** |
| `setup_a_bar_for_a_cocktail_party` | 0.020 | **0.000** |
| `getting_organized_for_work` | 0.020 | **0.000** |

The gains cluster on **collect-and-deposit** tasks (`collecting_*`, `putting_away_*`,
`organizing_*`, `picking_up_*`): pick an object, drop it in a container, repeat. Tasks needing a
long ordered sequence of distinct manipulations — `can_meat` (open cabinet, take out two hinged
jars, open, fill, close, replace, close cabinet) — still score zero.

---

## 2. What the models did and did not learn

`normalized_agent_distance = human_distance / agent_distance` (`agent_metric.py:61`) — an
efficiency ratio where **higher is better** and 1.0 means human parity. `agent_distance` itself
is the sum of per-step position deltas, i.e. cumulative **path length** in the world frame, not
net displacement. Earlier versions of this document read the ratio upside down; the table below
is inverted into the more legible agent/human multiple (medians over rollouts, so
lower is better and 1.0 is parity):

| | base | left arm | right arm | arm/base |
| --- | --- | --- | --- | --- |
| π0.5 50k | 0.99 | **3.56** | **3.25** | 3.4 |
| π0.5 270k | 0.81 | **3.07** | **3.00** | 3.7 |
| GR00T 50k | 0.48 | **8.88** | **6.69** | 16 |

So the base is *not* wandering: it covers about as much ground as the human (π0.5) or half as
much (GR00T). The arms are the anomaly — 3x the human path length for π0.5, 9x for GR00T.
Because this is path length rather than displacement, a large multiple means the end effector is
being shaken back and forth, not that it is reaching further. Rollout videos match: the wrist
cameras twitch continuously while target objects are never actually approached or grasped.

Two caveats on these numbers. Arm `agent_distance` is measured in the world frame, so it
includes whatever the base contributed. And the ratio is meaningless wherever the human barely
moved — `turning_on_radio` has a 0.4 m human path yet reports a 13x multiple.

No purposeful arm motion means no grasping, no transport, no placement. `q_score ≈ 0` follows
directly.

---

## 3. The proof: arm joints are worse than doing nothing

Open-loop action-prediction error, 200 samples over 100 tasks, raw joint units, scored over
the first 16 chunk steps (the window both models cover). `hold` = repeat the current joint
configuration for the whole chunk, i.e. freeze the arms.

| group | dims | hold | π0.5 50k | π0.5 **270k** | GR00T 50k |
| --- | --- | --- | --- | --- | --- |
| base | 3 | 0.0600 | 0.0281 | **0.0337** | 0.0251 |
| torso | 4 | 0.0042 | 0.0030 | **0.0051** | 0.0071 |
| left_arm | 7 | 0.0191 | 0.0217 (worse) | **0.0192** (ties) | 0.0199 (worse) |
| left_gripper | 1 | 0.9780 | 0.0459 | **0.0186** | 0.0337 |
| right_arm | 7 | 0.0188 | 0.0223 (worse) | **0.0198** (ties) | 0.0208 (worse) |
| right_gripper | 1 | 0.9808 | 0.0419 | **0.0298** | 0.0316 |

(50k and GR00T columns are the 16-step common window; the 270k column is horizon 32, so base and
torso are not directly comparable across columns — the arm and gripper verdicts are.)

**At 50k, all 14 arm dimensions were worse than freezing the arms. At 270k they merely tie it**
(0.0192 vs 0.0191, 0.0198 vs 0.0188). Still not better. Yet that modest change came with a 3.3x
closed-loop gain, which is why this metric should not be used alone to judge progress.

Where the aggregate advantage over `hold` comes from (dimension-weighted):

```
π0.5 50k    grippers +96.3%    arms  -1.4%    base +4.9%
π0.5 270k   grippers +96.1%    arms  -0.4%    base +3.8%
GR00T 50k   grippers +101.5%   arms  -6.5%    base +5.6%   torso -0.6%
```

Two of 23 dimensions still carry the entire result at 270k.

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

**Track `left_arm` / `right_arm` against `hold`, but do not gate on it.** An earlier version of
this document claimed closed-loop success would stay at zero until arm joints beat `hold`. The
270k run falsified that: the arms only *tie* `hold` there, and closed-loop mean `q_score` still
tripled with eleven tasks going from zero to scoring. Treat the per-group L1 as a cheap sanity
signal, not a gate.

---

## 5. Recommended actions, in priority order

1. **Keep training past 270k.** This is now evidence-backed rather than a guess: 50k -> 270k
   gave 3.3x mean `q_score`, 14 -> 31 scoring rollouts, and 8 -> 16 scoring tasks. 270k is still
   only ~8% of one epoch, so there is a lot of headroom. Save checkpoints at a regular interval
   so the gain-per-step curve can be measured.
2. **Gate on closed-loop `q_score`, not open-loop L1.** Open-loop barely moved (arms ~12%) while
   closed-loop tripled, so open-loop cannot resolve progress at this scale. A cheap proxy that
   does work: run the sweep restricted to the ~16 tasks that currently score, at 1-2 instances
   each. That is 16-32 rollouts, a few hours, instead of 42 hours for the full 500.
3. **Watch arm path length come down toward 1x.** With the metric read correctly, the base is
   already near human parity (0.99x -> 0.81x of human ground path) while the arms sweep ~3x the
   human path. That excess is jitter, not reaching. The signature of real improvement is the arm
   multiple falling toward 1.0 while `q_score` rises; the base falling below ~0.5x would instead
   mean it has stopped navigating at all.
4. **Revisit the action representation only once gain-per-step flattens.** Arm actions are
   learned relative to current state, which makes "output zero" a near-optimal degenerate
   solution: per-chunk arm displacement is ~0.7°, so sitting there costs almost nothing in loss.
   Options would be absolute joint targets, or up-weighting the 14 arm dimensions. The official
   baseline reaches usable behaviour with the *current* representation, and training is still
   paying off, so this is not the thing to change yet.
5. **Simulator instability** cost 25 rollouts at π0.5 50k, 12 at π0.5 270k and 9 at GR00T 50k
   (~2-5% each): PhysX blows up with a non-normalised gripper finger quaternion
   (`AssertionError` in `rigid_prim.py`) plus `Illegal BroadPhaseUpdateData`, during
   assisted-grasp raycasting. Does not affect the
   conclusions, but it caps achievable coverage.

---

## 6. Evaluation environment is validated — reuse it, do not rebuild blindly

`scripts/setup_eval_env.sh` was missing five things that each break evaluation. All are fixed
in the script now, and the script has since been verified to rebuild a working environment
from nothing after a pod restart:

| missing | symptom |
| --- | --- |
| `joylo` (gello) | `omnigibson.eval.evaluator` imports `gello.utils.og_teleop_cfg`; upstream `setup.sh` rejects `--eval` without `--joylo` |
| Vulkan loader | no `libvulkan.so.1` anywhere in the container; Kit cannot start |
| `botocore` | `omni.replicator.core` S3 backend needs it; OmniGibson's kit config never enables the `pip_archive` extension that normally supplies it |
| `libGLU.so.1` | `libneuray.so` fails to load, so the MDL-SDK fails, `carb.scenerenderer-rtx` cannot create its neuray database scope, every `rtx/raytracing` shader file then fails to load, and the process segfaults during scene load |
| `osqp` | `isaacsim.robot.wheeled_robots` (HolonomicController) fails to start; benign here but it should load |

`libGLU` is worth calling out: for two weeks it was only ever installed by hand, so the first
from-scratch rebuild silently lost it and reproduced the segfault. Anything fixed interactively
has to land in the script, or it does not survive the next `/dev/shm` wipe.

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

## 7. Closed-loop evaluation depends on the host GPU driver

Isaac Sim 5.1 needs a 580-series driver on this fleet. Known-good: **580.105.08** and
**580.82.07**. Known-broken: **535.161.08**, on which a scene load deadlocks. First failure in
the log, before the downstream noise:

```
[Error] [carb.graphics-vulkan.plugin] vkCreateRayTracingPipelinesKHR failed.
[Error] [omni.kit.app.plugin]         ASSERTION FAILED ... rtx.psodb/PsoRaytracing.cpp(1494)
[Error] [carb.graphics-vulkan.plugin] No Collection/Library can be nullptr with eCollection as compileType.
[Error] [gpu.foundation.plugin]       Failed to create the raytracing pipeline.
```

The symptom is a **hang, not a crash**: Kit reaches `app ready`, never reaches
`Imported scene`, stops writing to the log, and the simulator process spins on ~25 cores with
GPU utilisation at 0%. Two runs were lost this way, one of them for 29 hours.

Note that `isaacsim/kit/driver-requirements.json` is stale — it is manifest version 1 and
claims a 535.129.03 minimum with 535.161.07 recommended, which 535.161.08 satisfies. Do not
trust it; go by the versions above.

Ruled out as causes, so do not spend time on them again:

| hypothesis | finding |
| --- | --- |
| wrong Vulkan ICD library | `libGLX_nvidia` enumerates **0** devices; `libEGL_nvidia` is correct on both 535 and 580 |
| missing Vulkan extensions | all 8 GPUs advertise `VK_KHR_ray_tracing_pipeline`, `VK_KHR_pipeline_library`, `VK_EXT_graphics_pipeline_library`, `VK_KHR_acceleration_structure` |
| missing loader / libGLU | both present; Vulkan enumerates 8 devices |
| mixed driver versions | kernel module and every userspace lib agreed |
| stale RTX shader cache | **wrong, and clearing it is harmful** — see below |

**Never clear the RTX shader cache.** The `omni.hydra.rtx.shadercache.vulkan` wheel ships
~302 MB of precompiled pipelines and Kit does **not** rebuild them. Removing the cache turns
the hang into `Shader caches are missing from the application, which are required for RTX
renderer` followed by a segfault. `setup_eval_env.sh` only warns on a driver change now; it
deliberately does not touch the cache.

The environments live on `/dev/shm` and do not survive a pod restart, and a driver change
implies one. **Restore from the packed tarballs rather than rebuilding:**

```
/data/oss_bucket_0/wangfaye/conda_env/{b1k-pi05,b1k-eval,b1k-groot}.tar.gz
```

```bash
mkdir -p /dev/shm/wangfaye/conda_envs/b1k-eval
tar xzf /data/oss_bucket_0/wangfaye/conda_env/b1k-eval.tar.gz \
    -C /dev/shm/wangfaye/conda_envs/b1k-eval
source /dev/shm/wangfaye/conda_envs/b1k-eval/bin/activate && conda-unpack
./scripts/setup_eval_env.sh   # idempotent: adds anything newer, rewrites GPU manifests
```

Rebuilding from scratch instead cost over 40 minutes and hit two transient
`SSL_ERROR_SYSCALL` failures cloning the `lerobot` git dependency from GitHub. Refresh the
tarballs with `./scripts/pack_envs.sh` after any environment change — but not while a sweep is
running, since gzip competes for the CPU the simulator needs.

Assets are separate and do need re-extracting with `setup_eval_assets.sh` (~22 min); the zips
are cached on NAS, so nothing is re-downloaded. Checkpoints, Miniforge and the Cosmos-Reason2
local mirror are all on NAS and unaffected.

---

## 8. Corrections to earlier claims — do not propagate these

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

## 9. Artifacts

```
eval_results/pi05/<task>/json/*.json      475 closed-loop rollouts + videos/
eval_results/groot/<task>/json/*.json     491 closed-loop rollouts + videos/
eval_results/openloop_pi05_h32.json       200-sample open-loop, horizon 32
eval_results/openloop_groot_h32.json      200-sample open-loop, horizon 32 (16 scored)
eval_results/{pi05,groot}/sweep_logs/     per-worker sweep logs
```

Checkpoints evaluated:

```
openpi/outputs/checkpoints/pi05_b1k/pi05_b1k_full_20260803_162620/{49999,270000}
Isaac-GR00T/outputs/gr00t_b1k_full_20260804_225513/checkpoint-50000
```

## 10. What is established and what is still open

Established by measurement:

- More training helps, and substantially. 50k -> 270k on identical (task, instance) pairs gave
  3.3x mean `q_score`, 14 -> 31 scoring rollouts, 8 -> 16 scoring tasks. This is the ablation the
  first version of this document said was missing.
- Aggregate open-loop L1 is dominated by the two gripper dimensions and is a poor proxy for
  closed-loop ability.
- The failure is not a pipeline or configuration bug: proprio layout, delta transform, camera
  keys, prompts and observation namespacing were all verified against the simulator.

Still open:

- **Whether the gain keeps scaling.** Two points on the curve cannot show whether it is linear,
  saturating, or about to plateau. Checkpoints between 270k and whatever comes next would answer
  this, and it is the single most useful measurement to take next.
- **Whether this recipe can complete a task at all.** 963 rollouts across three checkpoints have
  produced zero full successes. Partial credit rising to 0.5 on `installing_a_scanner` is
  encouraging but not the same as finishing.
- **Whether shrinking motion is convergence or collapse.** With the metric read correctly, both
  base and arm path length *fall* with training (base 0.99x -> 0.81x, arms 3.56x -> 3.07x of the
  human path). Less jitter is the expected shape of progress, but a model drifting toward
  outputting nothing would look identical on this metric. Per-task motion vs `q_score` would
  separate the two.

---

## 11. GalaxeaVLA G0.5, step 280000: scores exactly zero, and it is a data-pipeline bug

Added 2026-08-27. Unlike sections 1-10, this failure **is** a configuration bug, and it is on the
training side. Do not spend more GPU time on this checkpoint.

### The measurement

Paired on the 459 `(task, instance)` keys all three runs share:

| | G0.5 280k | π0.5 50k | π0.5 270k |
| --- | --- | --- | --- |
| mean `q_score` | **0.0000** | 0.0047 | 0.0154 |
| rollouts above zero | **0** | 14 | 31 |
| tasks above zero | **0** | 8 | 16 |
| median base displacement | **0.007 m** | — | 19.48 m |
| rollouts with base < 0.1 m | **459 / 459** | — | 0 / 459 |

Max base displacement across all 459 G0.5 rollouts is 0.059 m. The robot never moves. This is not
a training-steps deficit: π0.5 at 50k, far less trained, still scores.

### Root cause

`action_op_mask` is not carried through the key rename that produces `lower_body`, so the group is
classified as no-op and dropped from **every** training target.

1. `base_processor.py:386-389` runs `action_filter -> action_state_transform -> normalizer -> merger`.
2. `BaseActionFilter.forward` (`action_filter.py:44-54`) builds `action_op_mask` from the *raw*
   `shape_meta` keys. For b1k those are `base_qvel, trunk_qpos, left_arm, left_gripper, right_arm,
   right_gripper` — there is no `lower_body`.
3. `BehaviorPerKeyTransform.forward` (`relative_action.py:656`) merges `action` `trunk_qpos +
   base_qvel` into `lower_body` and deletes the originals, but left `action_op_mask` untouched.
   `RotationTransform` (`rotation.py:79-103`) does synchronise the mask, so this was an omission,
   not a design choice.
4. `GroupedPaddingMerger._align_per_raw_key` (`action_state_merger.py:712-717`) finds no
   `lower_body` key in the mask dict and substitutes `zeros(7)` — an all-inactive mask.
5. `_noop_from_op_mask` (`vq_base.py:356-362`) marks a group no-op when no dim is active, and b1k
   sets `dropout_noop_parts: true` (`configs/task/b1k.yaml:87`). So `<lower_body_i>` is absent from
   every GT token sequence.

The AR head therefore never learned to emit a base command. Its decoded output is exactly 36
tokens — `<left_control_0> <right_control_0> <left_control_1> <right_control_1>` plus 32 action
tokens — with no `lower_body` and no gripper markers, in 464,595 of 464,660 inferences (99.99%).
Generation is not truncated: `_get_action_generation_max_new_tokens()` returns >= 300. The markers
exist in the vocabulary (`group_markers(14)` at server startup includes `<lower_body_0..3>`).
`serve_b1k.py:249-252` then holds the absent group — trunk from state, base velocity zero — which
is exactly the 0.007 m median.

Note `action_dim_is_pad` is computed from the `action` dict, which *does* contain `lower_body`, so
the pad mask was always correct. Only the op mask was broken.

### Verified

Replaying `BaseActionFilter -> BehaviorPerKeyTransform -> GroupedPaddingMerger` on a synthetic b1k
sample:

```
pre-fix   lower_body dims[20:27] op_active=False all_pad=False -> noop=True
post-fix  lower_body dims[20:27] op_active=True  all_pad=False -> noop=False
```

All four other groups are unaffected in both runs.

### Fix

`relative_action.py`: `BehaviorPerKeyTransform.forward` now mirrors the merge onto
`action_op_mask`. Scope is b1k only — this transform is referenced nowhere else — and it is a no-op
at inference, where there is no GT action and hence no op mask.

This does not repair the existing checkpoint. Step 280000 was trained against targets that never
contained the base group, so it needs retraining (or at minimum a long finetune, which now faces a
target-distribution shift). A quick way to confirm the fix end-to-end before committing GPU hours
is to dump one real training batch and check that `<lower_body_0>` appears in the serialised
labels.

### Also worth a look: the grippers

The grippers are dropped by a *different* rule and the fix above does not touch them.
`vq_base.py:394-399` applies a value-based no-op test only to keys matching
`rule_based_key_patterns: [gripper]`: if the binarised signal is constant across the chunk, the
group is dropped. A b1k gripper command rarely changes within 32 steps, so this fires on nearly
every sample and the model learned to omit gripper markers too. Holding the last command is
defensible semantics, but combined with the immobile base it means the policy can neither navigate
nor grasp. Whether that threshold rule is appropriate for a 32-step chunk at 30 fps is a separate
question worth settling.
