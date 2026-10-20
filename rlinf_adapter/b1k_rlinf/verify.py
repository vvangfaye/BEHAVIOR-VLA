"""Run a real exported checkpoint on an existing observation, without simulation."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .factory import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--export-reference-inputs", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    started = time.monotonic()
    cfg = OmegaConf.create(
        {
            "model_path": str(args.checkpoint.resolve()),
            "pi05": True,
            "precision": "fp32",
            "action_dim": 23,
            "num_action_chunks": 16,
            "num_steps": 10,
            "openpi": {
                "model_action_dim": 32,
                "action_horizon": 32,
                "max_token_len": 512,
                "train_expert_only": True,
                "noise_method": "flow_sde",
                "noise_level": 0.4,
                "ignore_last": True,
                "joint_logprob": False,
                "add_value_head": True,
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "config_name": "pi05_behavior",
            },
            "b1k_joint": {
                "enabled": True,
                "norm_stats_path": str(args.checkpoint / "norm_stats.json"),
                "tokenizer_path": str(args.tokenizer),
                "state_layout": "checkpoint61",
            },
        }
    )
    print("Loading converted checkpoint", flush=True)
    model = build_model(cfg, torch.float32).to(args.device).eval()
    load_seconds = time.monotonic() - started
    with np.load(args.sample, allow_pickle=False) as data:
        env = {
            "states": data["observation.state"][None].copy(),
            "main_images": data["observation.rgb.zed_link_camera_0"][None].copy(),
            "wrist_images": np.stack(
                [
                    data["observation.rgb.left_realsense_link_camera_0"],
                    data["observation.rgb.right_realsense_link_camera_0"],
                ]
            )[None].copy(),
            "task_descriptions": [str(data["prompt"].item())],
        }
    print("Generating subtask and Flow-SDE rollout", flush=True)
    initial_noise = torch.from_numpy(
        np.random.default_rng(42).standard_normal((1, 32, 32)).astype(np.float32)
    ).to(args.device)
    actions, result = model.predict_action_batch(env, mode="train", noise=initial_noise)
    print("Recomputing stored transition probabilities", flush=True)
    with torch.no_grad():
        replay = model.default_forward(result["forward_inputs"])
        if args.export_reference_inputs is not None:
            obs = model._observation_from_forward_inputs(result["forward_inputs"])
            ode_actions = model.sample_actions(obs, num_steps=10, noise=initial_noise)
            packed = {
                "state": obs.state.cpu().numpy(),
                "tokenized_prompt": obs.tokenized_prompt.cpu().numpy(),
                "tokenized_prompt_mask": obs.tokenized_prompt_mask.cpu().numpy(),
                "token_ar_mask": obs.token_ar_mask.cpu().numpy(),
                "noise": initial_noise.cpu().numpy(),
                "torch_actions": ode_actions.cpu().numpy(),
            }
            for key, value in obs.images.items():
                packed[f"image__{key}"] = value.cpu().numpy()
            args.export_reference_inputs.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.export_reference_inputs, **packed)
    max_logprob_diff = float((replay["logprobs"] - result["prev_logprobs"]).abs().max())
    if not torch.isfinite(actions).all() or not torch.isfinite(replay["logprobs"]).all():
        raise RuntimeError("Non-finite action or transition probability")
    torch.testing.assert_close(replay["logprobs"], result["prev_logprobs"], atol=1e-5, rtol=1e-5)
    n = int(result["forward_inputs"]["subtask_lengths"][0])
    ids = result["forward_inputs"]["subtask_tokens"][0, :n].cpu().tolist()
    record = {
        "device": args.device,
        "checkpoint": str(args.checkpoint.resolve()),
        "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "scope": "One real observation; no simulator, optimizer update or cross-backend parity claim",
        "generated_subtask": model.joint_tokenizer.processor.decode(ids),
        "subtask_tokens": ids,
        "action_shape": list(actions.shape),
        "max_logprob_recompute_error": max_logprob_diff,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "load_seconds": load_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
