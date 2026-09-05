#!/usr/bin/env python
"""Open-loop action-prediction error for the B1K baselines.

WHAT THIS IS NOT: the BEHAVIOR benchmark score. That metric is task success
(``q_score``) from closed-loop rollouts on test instances 301-340, which needs a
GPU with RT cores (H20 has none). The released demos carry a single ``train``
split, so there is no held-out trajectory set either.

WHAT THIS IS: on frames drawn from the training distribution, how closely does
the policy reproduce the demonstrated action chunk, in raw (unnormalised) joint
space. Useful to confirm the model learned something non-trivial, to compare the
two models on identical samples, and to find which tasks/joints predict worst.

Two trivial baselines are reported for scale:
  hold  - repeat the current joint state for the whole chunk
  zero  - emit zeros

Usage:
  ./scripts/eval_openloop.py --model pi05  --samples-per-task 2 --out eval_results/openloop_pi05.json
  ./scripts/eval_openloop.py --model groot --samples-per-task 2 --out eval_results/openloop_groot.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

DATASET = "/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/dataset_views/pi05/2026-challenge-demos"
# π0.5 predicts 32 steps, GR00T 40; 32 is the longest window both cover. A short
# window is not diagnostic: over 16 steps (~0.5 s at 30 Hz) the demonstrated arm
# joints move well under a degree, so freezing the arms scores about as well as
# tracking them. pick_samples keeps 40 frames of tail margin, so 40 is the cap.
EVAL_HORIZON = 32
# Raw R1Pro action layout (see Isaac-GR00T/examples/b1k/r1pro.py docstring).
ACTION_GROUPS = {
    "base": (0, 3),
    "torso": (3, 7),
    "left_arm": (7, 14),
    "left_gripper": (14, 15),
    "right_arm": (15, 22),
    "right_gripper": (22, 23),
}
# state indices matching each action group, for the "hold current state" baseline
STATE_FOR_ACTION = {
    "torso": (53, 57),
    "left_arm": (3, 10),
    "left_gripper": (24, 26),
    "right_arm": (28, 35),
    "right_gripper": (49, 51),
}


def pick_samples(ds_len: int, episodes_meta, per_task: int, seed: int):
    """Choose frame indices spread over all tasks, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    by_task = collections.defaultdict(list)
    for ep_idx, task_idx, start, end in episodes_meta:
        by_task[task_idx].append((ep_idx, start, end))
    picked = []
    for task_idx in sorted(by_task):
        eps = by_task[task_idx]
        for _ in range(per_task):
            ep_idx, start, end = eps[rng.integers(len(eps))]
            # keep away from the episode tail so the action chunk is unpadded
            hi = max(start + 1, end - 40)
            picked.append((task_idx, ep_idx, int(rng.integers(start, hi))))
    return picked


def load_episode_meta():
    import pyarrow.parquet as pq

    files = sorted(pathlib.Path(f"{DATASET}/meta/episodes").glob("chunk-*/*.parquet"))
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "task_index", "dataset_from_index", "dataset_to_index"])
        d = t.to_pydict()
        rows.extend(
            zip(d["episode_index"], d["task_index"], d["dataset_from_index"], d["dataset_to_index"])
        )
    return rows


def task_names():
    out = {}
    with open(f"{DATASET}/meta/tasks.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["task_index"]] = r["task_name"]
    return out


def summarise(errs: dict[str, list[np.ndarray]]) -> dict:
    """errs[name] = list of (horizon, dim) absolute-error arrays."""
    out = {}
    for name, chunks in errs.items():
        if not chunks:
            continue
        a = np.concatenate([c.reshape(-1, c.shape[-1]) for c in chunks], axis=0)
        out[name] = {
            "l1": float(np.abs(a).mean()),
            "rmse": float(np.sqrt((a**2).mean())),
            "groups": {
                g: float(np.abs(a[:, s:e]).mean()) for g, (s, e) in ACTION_GROUPS.items()
            },
        }
    return out


def main() -> None:
    # build_pi05/build_groot close over these, so they are set from argv below.
    global DATASET, EVAL_HORIZON
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pi05", "groot"])
    ap.add_argument("--samples-per-task", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pi05-ckpt", default="/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/openpi/"
                                           "outputs/checkpoints/pi05_b1k/pi05_b1k_full_20260803_162620/49999")
    ap.add_argument("--subtask-prompts", default="",
                    help="sub-task segment parquet. Feeds the ground-truth sub-task of the scored "
                         "frame, i.e. teacher forcing, so numbers come out optimistic. Leave unset "
                         "to measure deployment, where serving binds one whole-episode instruction "
                         "and no sub-task planner exists.")
    ap.add_argument("--groot-ckpt", default="")
    ap.add_argument("--limit", type=int, default=0, help="stop after N samples (smoke test)")
    ap.add_argument("--horizon", type=int, default=EVAL_HORIZON,
                    help="action-chunk steps to score; π0.5 predicts 32, GR00T 40")
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()

    # build_pi05/build_groot close over these, so set them before building.
    DATASET = args.dataset
    if args.horizon > 40:
        sys.exit("--horizon cannot exceed 40: pick_samples only reserves 40 frames of episode tail")
    EVAL_HORIZON = args.horizon

    names = task_names()
    meta = load_episode_meta()
    samples = pick_samples(0, meta, args.samples_per_task, args.seed)
    if args.limit:
        samples = samples[: args.limit]
    print(f"evaluating {len(samples)} samples over {len(set(s[0] for s in samples))} tasks", flush=True)

    predict, get_sample = build_model(args)

    per_task_err = collections.defaultdict(list)
    base_hold, base_zero, model_err = [], [], []
    # Per-position-in-chunk sums, so error growth across the chunk is visible.
    pos_sum = {"model": np.zeros((EVAL_HORIZON, 23)), "hold": np.zeros((EVAL_HORIZON, 23))}
    pos_n = np.zeros(EVAL_HORIZON)
    t0 = time.time()
    for n, (task_idx, ep_idx, frame) in enumerate(samples):
        item = get_sample(ep_idx, frame)
        if item is None:
            continue
        gt = item["gt_action"]          # (horizon, 23) raw
        valid = ~item["is_pad"]
        if valid.sum() == 0:
            continue
        pred = predict(item)            # (horizon, 23) raw
        h = min(len(gt), len(pred))
        gt, pred, valid = gt[:h], pred[:h], valid[:h]

        err = np.abs(pred - gt)[valid]
        model_err.append(err)
        per_task_err[names[task_idx]].append(err)

        hold = np.tile(item["hold_action"], (h, 1))
        base_hold.append(np.abs(hold - gt)[valid])
        base_zero.append(np.abs(np.zeros_like(gt) - gt)[valid])

        idx = np.where(valid)[0]
        pos_sum["model"][idx] += np.abs(pred - gt)[idx]
        pos_sum["hold"][idx] += np.abs(hold - gt)[idx]
        pos_n[idx] += 1

        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(samples)}  {time.time()-t0:.0f}s", flush=True)

    pos_curve = {
        k: (v[pos_n > 0] / pos_n[pos_n > 0][:, None]).tolist() for k, v in pos_sum.items()
    }
    result = {
        "model": args.model,
        "dataset": DATASET,
        "horizon": EVAL_HORIZON,
        "samples": len(model_err),
        "note": "training-distribution fit, NOT the benchmark q_score (needs closed-loop rollouts)",
        "overall": summarise({"model": model_err, "baseline_hold": base_hold, "baseline_zero": base_zero}),
        "per_position_l1": pos_curve,
        "per_task_l1": {
            t: float(np.concatenate([c.reshape(-1, c.shape[-1]) for c in v]).__abs__().mean())
            for t, v in sorted(per_task_err.items())
        },
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))
    report(result)
    worst = sorted(result["per_task_l1"].items(), key=lambda kv: -kv[1])[:10]
    print("\nworst tasks by L1:")
    for t, v in worst:
        print(f"  {v:.4f}  {t}")
    print(f"\nwrote {out}")


def report(result: dict) -> None:
    """Print the per-group and per-position view that the aggregate L1 hides."""
    m = result["overall"]["model"]["groups"]
    h = result["overall"]["baseline_hold"]["groups"]
    dims = {g: e - s for g, (s, e) in ACTION_GROUPS.items()}
    ndim = sum(dims.values())

    print(f"\n=== per group vs 'hold current joints' baseline (horizon {result['horizon']}) ===")
    print(f"{'group':16s} {'dims':>4s} {'model':>9s} {'hold':>9s} {'verdict':>26s}")
    for g in ACTION_GROUPS:
        verdict = "WORSE than frozen" if m[g] > h[g] else f"{h[g] / m[g]:.1f}x better than frozen"
        print(f"{g:16s} {dims[g]:4d} {m[g]:9.4f} {h[g]:9.4f} {verdict:>26s}")

    total = sum((h[g] - m[g]) * dims[g] for g in ACTION_GROUPS) / ndim
    print(f"\naggregate advantage over frozen arms: {total:.5f}   (share by group)")
    for g in ACTION_GROUPS:
        c = (h[g] - m[g]) * dims[g] / ndim
        print(f"  {g:16s} {c / total * 100:+7.1f}%")

    mc = np.asarray(result["per_position_l1"]["model"])
    hc = np.asarray(result["per_position_l1"]["hold"])
    print("\n=== error growth across the chunk (arm joints vs grippers) ===")
    print(f"{'step':>5s} {'arms_model':>11s} {'arms_hold':>11s} {'grip_model':>11s} {'grip_hold':>11s}")
    arms = list(range(*ACTION_GROUPS["left_arm"])) + list(range(*ACTION_GROUPS["right_arm"]))
    grip = list(range(*ACTION_GROUPS["left_gripper"])) + list(range(*ACTION_GROUPS["right_gripper"]))
    for i in range(len(mc)):
        if i % 4 == 0 or i == len(mc) - 1:
            print(f"{i:5d} {mc[i, arms].mean():11.4f} {hc[i, arms].mean():11.4f} "
                  f"{mc[i, grip].mean():11.4f} {hc[i, grip].mean():11.4f}")


def build_model(args):
    if args.model == "pi05":
        return build_pi05(args)
    return build_groot(args)


def build_pi05(args):
    import dataclasses

    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config
    import openpi.training.data_loader as _dl
    import openpi.transforms as _tf

    cfg = _config.get_config("pi05_b1k")
    base = dataclasses.replace(cfg.data.base_config, dataset_root=DATASET)
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, repo_id="behavior-1k/2026-challenge-demos", base_config=base, frame_stride=1,
            subtask_prompts_path=(args.subtask_prompts or None),
            subtask_whole_task_prob=0.0,
        ),
    )
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = _dl.create_b1k_dataset(dc, cfg.model.action_horizon)
    repack = _tf.compose(dc.repack_transforms.inputs)
    policy = _policy_config.create_trained_policy(cfg, args.pi05_ckpt)

    # dataset index == global frame index, so frame ids can be used directly
    def get_sample(ep_idx, frame):
        s = ds[frame]
        packed = repack(s)
        return {
            "policy_input": {k: np.asarray(v) if hasattr(v, "shape") else v for k, v in packed.items()},
            "gt_action": np.asarray(s["action"], dtype=np.float64)[:EVAL_HORIZON],
            "is_pad": np.asarray(s["action_is_pad"], dtype=bool)[:EVAL_HORIZON],
            "hold_action": hold_from_state(np.asarray(s["observation.state"], dtype=np.float64)),
        }

    def predict(item):
        res = policy.infer(item["policy_input"])
        return np.asarray(res["actions"], dtype=np.float64)

    return predict, get_sample


def hold_from_state(state: np.ndarray) -> np.ndarray:
    """Baseline: keep the current joint configuration, zero base velocity."""
    a = np.zeros(23)
    for g, (as_, ae) in ACTION_GROUPS.items():
        if g in STATE_FOR_ACTION:
            ss, se = STATE_FOR_ACTION[g]
            a[as_:ae] = state[ss:se][: ae - as_]
    return a


def build_groot(args):
    if not args.groot_ckpt:
        sys.exit("--groot-ckpt is required for --model groot")
    import glob
    import importlib.util
    import json

    import pyarrow.parquet as pq
    import torch
    from torchcodec.decoders import VideoDecoder

    spec = importlib.util.spec_from_file_location(
        "r1pro", "/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/Isaac-GR00T/examples/b1k/r1pro.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r1pro"] = mod
    spec.loader.exec_module(mod)
    cfgd = mod.b1k_r1pro_config
    mc = {k: v for k, v in cfgd.items() if k in ("video", "state", "action", "language")}
    action_keys = cfgd["action"].modality_keys
    state_keys = cfgd["state"].modality_keys
    video_keys = cfgd["video"].modality_keys
    # modality key (head/left_wrist/right_wrist) -> dataset column (observation.rgb.*).
    # The mapping lives in the dataset's meta/modality.json video section.
    import json as _json

    _modality = _json.load(open(f"{DATASET}/meta/modality.json"))
    video_cols = {k: _modality["video"][k]["original_key"] for k in video_keys}

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    state_span = {k: (_modality["state"][k]["start"], _modality["state"][k]["end"]) for k in state_keys}

    # transformers 4.5x calls huggingface_hub.model_info() over the network inside
    # PreTrainedTokenizerBase._patch_mistral_regex to detect Mistral tokenizers,
    # which raises under HF_HUB_OFFLINE. The GR00T backbone tokenizer is Qwen3VL,
    # not Mistral, so no-op'ing the patch is correct and keeps loading offline.
    import transformers.tokenization_utils_base as _tub

    _tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(
        lambda cls, tokenizer, *a, **k: tokenizer
    )

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.groot_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    fps = json.load(open(f"{DATASET}/meta/info.json"))["fps"]
    task_desc = {}  # task_index -> language prompt
    with open(f"{DATASET}/meta/tasks.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            task_desc[r["task_index"]] = r["task"]

    # Per-episode locations: data parquet (chunk,file), each video file
    # (chunk,file,from_ts), global start row, and task prompt. A sample then needs
    # only a few parquet rows and 3 decoded frames, not a full-episode decode.
    ep_meta, ep_prompt = {}, {}
    for f in sorted(glob.glob(f"{DATASET}/meta/episodes/chunk-*/*.parquet")):
        d = pq.read_table(f).to_pydict()
        for i, ep_idx in enumerate(d["episode_index"]):
            m = {
                "start": d["dataset_from_index"][i],
                "data": (d["data/chunk_index"][i], d["data/file_index"][i]),
            }
            for vk, col in video_cols.items():
                m[vk] = (
                    d[f"videos/{col}/chunk_index"][i],
                    d[f"videos/{col}/file_index"][i],
                    d[f"videos/{col}/from_timestamp"][i],
                )
            ep_meta[ep_idx] = m
            ep_prompt[ep_idx] = task_desc[d["task_index"][i]]

    ep_start = {e: m["start"] for e, m in ep_meta.items()}
    decoders: dict[str, VideoDecoder] = {}
    tables: dict = {}
    ep_rows: dict = {}

    def _episode_arrays(ep_idx):
        if ep_idx not in ep_rows:
            c, fi = ep_meta[ep_idx]["data"]
            if (c, fi) not in tables:
                path = f"{DATASET}/data/chunk-{c:03d}/file-{fi:03d}.parquet"
                tables[(c, fi)] = pq.read_table(
                    path, columns=["observation.state", "action", "episode_index"]
                ).to_pydict()
            t = tables[(c, fi)]
            rows = [j for j, e in enumerate(t["episode_index"]) if e == ep_idx]
            state = np.asarray([t["observation.state"][j] for j in rows], dtype=np.float64)
            action = np.asarray([t["action"][j] for j in rows], dtype=np.float64)
            ep_rows[ep_idx] = (state, action)
        return ep_rows[ep_idx]

    def _frame(ep_idx, vk, local):
        c, fi, ts0 = ep_meta[ep_idx][vk]
        path = f"{DATASET}/videos/{video_cols[vk]}/chunk-{c:03d}/file-{fi:03d}.mp4"
        dec = decoders.get(path) or decoders.setdefault(path, VideoDecoder(path, seek_mode="approximate"))
        img = dec.get_frame_played_at(ts0 + local / fps).data.numpy()  # (C, H, W)
        return np.transpose(img, (1, 2, 0))  # (H, W, C)

    def get_sample(ep_idx, frame):
        state_arr, action_arr = _episode_arrays(ep_idx)
        n = len(action_arr)
        local = frame - ep_start[ep_idx]
        if local < 0 or local + EVAL_HORIZON > n:
            return None
        gt = action_arr[local : local + EVAL_HORIZON]  # (H, 23), same source as π0.5
        s61 = state_arr[local]
        state = {k: s61[a:b].astype(np.float32)[None, None, :] for k, (a, b) in state_span.items()}
        video = {k: _frame(ep_idx, k, local)[None, None, ...] for k in video_keys}
        return {
            "policy_input": {
                "video": video,
                "state": state,
                "language": {"annotation.human.task_description": [[ep_prompt[ep_idx]]]},
            },
            "gt_action": gt,
            "is_pad": np.zeros(EVAL_HORIZON, dtype=bool),
            "hold_action": hold_from_state(s61),  # identical baseline to π0.5
        }

    def predict(item):
        act, _ = policy.get_action(item["policy_input"])
        chunk = np.concatenate([np.asarray(act[k]) for k in action_keys], axis=-1)  # (B, T, 23)
        return chunk[0].astype(np.float64)

    return predict, get_sample


if __name__ == "__main__":
    main()
