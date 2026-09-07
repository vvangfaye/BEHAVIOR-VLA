# π0.5 BEHAVIOR 评测记录

整理自 2026-08-12 的历史调查，含 2026-08-23 的 270k 结果。以下数字和环境结论来自历史记录，未在当前机器重跑；旧绝对路径仅用于追溯。

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
about **8% of one epoch**.

---

## 1. Closed-loop results (the actual benchmark metric)

100 tasks, `public_test` split, receding-horizon control, `max_steps = 1.5x` mean human
demo length. Every rollout ran to timeout.

| | rollouts | tasks | success | mean q_score | nonzero rollouts | tasks with any score |
| --- | --- | --- | --- | --- | --- | --- |
| π0.5 50k | 475 | 99 | **0** | 0.0045 | 14 | 8 |
| π0.5 **270k** | 488 | 99 | **0** | **0.0145** | **31** | **16** |

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

### OpenPI observation compatibility fix

- **`openpi/src/openpi/configs/robots/b1k.py`**: robot name was `robot`, but BEHAVIOR-1K 3.9.0
  names the R1Pro `robot_r1` and namespaces all observations under it. Serving raised
  `KeyError: 'robot::proprio'` on the first step. Renamed the robot and its three camera
  `obs_key`s. `dataset_key` (the training path) is untouched, so trained checkpoints are
  unaffected.
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
/data/oss_bucket_0/wangfaye/conda_env/{b1k-pi05,b1k-eval}.tar.gz
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
are cached on NAS, so nothing is re-downloaded. Checkpoints and Miniforge were stored on NAS in that environment.

---

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
- **Whether this recipe can complete a task at all.** 963 rollouts across the two π0.5 checkpoints have
  produced zero full successes. Partial credit rising to 0.5 on `installing_a_scanner` is
  encouraging but not the same as finishing.
- **Whether shrinking motion is convergence or collapse.** With the metric read correctly, both
  base and arm path length *fall* with training (base 0.99x -> 0.81x, arms 3.56x -> 3.07x of the
  human path). Less jitter is the expected shape of progress, but a model drifting toward
  outputting nothing would look identical on this metric. Per-task motion vs `q_score` would
  separate the two.

---
