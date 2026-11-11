#!/usr/bin/env python3
"""Validate every real cached sample through checkpoint input preparation.

The model factory is intercepted explicitly. No checkpoint weights are restored,
no actions are generated, and no simulator is run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from eval_cached_probe import VerifiedRawCache, eval_openloop, file_sha256


def array_summary(value):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("Input preparation produced nonfinite values")
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "minimum": float(value.min()), "maximum": float(value.max())}


def run_preflight(cache_manifest, checkpoint, norm_asset_id, max_token_len=512):
    import jax
    import jax.numpy as jnp
    import openpi.policies.policy_config as policy_config
    from openpi import transforms
    from openpi.configs import ROBOT_REGISTRY
    from openpi.models.model import Observation
    from openpi.shared import subtask_contract
    from openpi.training.subtask_data import SubtaskFromSegments

    cache = VerifiedRawCache(cache_manifest)
    args = SimpleNamespace(subtask_conditioning="provided", max_token_len=max_token_len,
                           pi05_ckpt=str(checkpoint), norm_asset_id=norm_asset_id, memory_cache_dir=None,
                           horizon=cache.probe["model_horizon_H"], dataset=str(cache.paths["dataset_root"]),
                           repo_id=cache.manifest["repo_id"], subtask_prompts=cache.paths["subtask_prompts"],
                           action_noise_seed=cache.probe["pairing_contract"]["action_noise_seed"])
    ids = eval_openloop.read_episode_ids(cache.paths["episodes_file"])
    metadata = eval_openloop.filter_episode_meta(eval_openloop.load_episode_meta(args.dataset), ids)
    labels = SubtaskFromSegments.from_parquet(args.subtask_prompts, episodes_index=ids)
    samples = eval_openloop.replay_samples(cache.paths["sample_keys_file"], metadata, labels,
                                          names=eval_openloop.task_names(args.dataset), repo_id=args.repo_id,
                                          horizon=args.horizon, action_noise_seed=args.action_noise_seed)
    last_raw = {}

    def sample_provider(sample):
        raw = cache(sample)
        last_raw.clear()
        last_raw.update(raw)
        return raw

    with mock.patch.object(policy_config, "create_trained_policy", return_value=None) as factory:
        _, get_sample = eval_openloop.build_pi05(args, ids, sample_provider=sample_provider)
    if factory.call_count != 1:
        raise RuntimeError("Expected exactly one intercepted policy construction")
    cfg = factory.call_args.args[0]
    norm = factory.call_args.kwargs["norm_stats"]
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    stored_mode = subtask_contract.load(Path(checkpoint) / "assets")
    if stored_mode is not None and stored_mode != cfg.model.subtask_action_conditioning:
        raise ValueError("This preflight configuration disagrees with the checkpoint's subtask mode")
    tokenizers = [item for item in data_config.model_transforms.inputs if isinstance(item, transforms.TokenizeSubtaskPrompt)]
    if len(tokenizers) != 1:
        raise ValueError("Expected exactly one real subtask tokenizer transform")
    tokenizer = tokenizers[0]
    pre_token = transforms.compose([
        transforms.InjectDefaultPrompt(None), *data_config.data_transforms.inputs,
        transforms.Normalize(norm, use_quantiles=data_config.use_quantile_norm),
        *[item for item in data_config.model_transforms.inputs
          if not isinstance(item, transforms.TokenizeSubtaskPrompt | transforms.PadStatesAndActions)],
    ])
    pad = transforms.PadStatesAndActions(cfg.model.action_dim)
    restore_actions = transforms.compose([
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])
    camera_keys = [camera.dataset_key for camera in ROBOT_REGISTRY[cfg.data.robot_config_name].observations.values()]
    records = []
    for sample in samples:
        item = get_sample(sample)
        if np.asarray(last_raw["observation.state"]).shape != (61,):
            raise ValueError("Raw R1Pro state is not 61-dimensional")
        if np.asarray(last_raw["action"]).shape != (args.horizon, 23):
            raise ValueError("Raw R1Pro action target has the wrong shape")
        raw_images = {}
        for key in camera_keys:
            image = np.asarray(last_raw[key])
            if image.ndim != 3 or image.shape[0] != 3 or image.dtype != np.float32:
                raise ValueError(f"Unexpected cached RGB layout: {key} {image.shape} {image.dtype}")
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"Cached RGB is outside float [0, 1]: {key}")
            raw_images[key] = array_summary(image)
        before_token = pre_token(copy.deepcopy(item["policy_input"]))
        if before_token["state"].shape != (23,) or before_token["actions"].shape != (args.horizon, 23):
            raise ValueError("Robot transforms did not produce 23-dimensional state/actions")
        prepared = tokenizer(before_token)
        label_tokens = prepared["tokenized_prompt"][prepared["token_loss_mask"]]
        state = before_token["state"] if tokenizer.discrete_state_input else None
        prepared.update(tokenizer.action_fields(label_tokens, state))
        prepared = pad(prepared)
        if prepared["state"].shape != (32,) or prepared["actions"].shape != (args.horizon, 32):
            raise ValueError("Model state/action padding has unexpected shape")
        if np.any(prepared["state"][23:] != 0) or np.any(prepared["actions"][:, 23:] != 0):
            raise ValueError("Model padded coordinates are not zero")
        restored = restore_actions({"state": prepared["state"].copy(), "actions": prepared["actions"].copy()})["actions"]
        np.testing.assert_allclose(restored, item["gt_action"], atol=2e-5, rtol=2e-5,
                                   err_msg="The checkpoint's normalization/action transforms do not round-trip recorded targets")
        expected_mask = np.arange(args.horizon) < sample.segment_end - sample.frame_index
        if not np.array_equal(item["valid"], expected_mask):
            raise ValueError("Action target mask differs from the frozen current-label prefix")
        for field in ("policy_input_sha256", "ground_truth_sha256", "action_noise_sha256"):
            if not isinstance(item.get(field), str) or len(item[field]) != 64:
                raise ValueError(f"Missing sample integrity hash: {field}")
        if item["action_noise"].shape != (32, 32) or item["action_noise"].dtype != np.float32:
            raise ValueError("Explicit initial action noise has the wrong shape/dtype")
        if not len(label_tokens) or prepared["tokenized_prompt"].shape != (max_token_len,):
            raise ValueError("Subtask tokenization has empty targets or the wrong context length")
        # Exercise the actual Observation parser as Policy.infer would, without
        # passing it to any model. It converts uint8 HWC images to float [-1, 1].
        batched = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], prepared)
        observation = Observation.from_dict(batched)
        model_images = {key: array_summary(value) for key, value in observation.images.items()}
        if any(value["shape"] != [1, 224, 224, 3] or value["minimum"] < -1 or value["maximum"] > 1
               for value in model_images.values()):
            raise ValueError("Model images have invalid shape or range")
        records.append({"episode_index": sample.episode_index, "frame_index": sample.frame_index,
                        "bucket": sample.probe_metadata["bucket"], "valid_action_steps": int(item["valid"].sum()),
                        "raw_images": raw_images, "raw_state": array_summary(last_raw["observation.state"]),
                        "normalized_state": array_summary(before_token["state"]),
                        "normalized_actions": array_summary(before_token["actions"]), "model_images": model_images,
                        "tokenized_prompt_valid_tokens": int(prepared["tokenized_prompt_mask"].sum()),
                        "subtask_target_tokens": len(label_tokens),
                        "action_roundtrip_max_abs_error": float(np.max(np.abs(restored - item["gt_action"]))),
                        **{field: item[field] for field in ("policy_input_sha256", "ground_truth_sha256", "action_noise_sha256")}})
    return {"schema_version": 1, "kind": "real_cached_input_preflight", "success": True,
            "cache_manifest": str(cache.manifest_path), "cache_manifest_sha256": cache.manifest_sha256,
            "checkpoint": str(Path(checkpoint).resolve()), "normalization_asset_id": args._resolved_norm_asset_id,
            "normalization_sha256": file_sha256(Path(args._norm_stats_path)),
            "samples": len(records), "raw_cache_samples_verified": len(cache.receipts),
            "tasks": len({sample.task_index for sample in samples}), "model_factory_intercepted": True,
            "model_restored": False, "inference_executed": False, "simulator_run": False,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite preflight evidence")
    result = run_preflight(args.cache_manifest, args.checkpoint, args.norm_asset_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
