#!/usr/bin/env python3
"""Pair native pi05 CFM evaluation with its ten-step Euler endpoint, locally.

Uses an existing verified raw-boundary cache and local JAX checkpoint. No model
code is changed and no downloads are allowed. This measures the CFM objective at
deployment bfloat16 weights, without training image augmentation; it does not
reconstruct historical training loss or decompose a loss into gauge components.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import urlparse
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("flow_probe_cached_eval", HERE.parent / "eval_cached_probe.py")
assert _SPEC is not None and _SPEC.loader is not None
cached = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cached
_SPEC.loader.exec_module(cached)
ev = cached.eval_openloop
MAX_SAMPLES = 3
REPEAT_RTOL = 1e-5
REPEAT_ATOL = 1e-6


def local_asset(url, *, cache_root: Path, force_download=False, **_kwargs) -> Path:
    """Resolve only already present files; never call the original downloader."""
    if force_download:
        raise ValueError("Downloads are disabled, including force_download")
    parsed = urlparse(str(url))
    if not parsed.scheme:
        path = Path(url).expanduser().resolve()
    else:
        path = (cache_root / parsed.netloc / parsed.path.strip("/")).resolve()
        if not path.is_relative_to(cache_root):
            raise ValueError("Cached asset path escapes OPENPI_DATA_HOME")
    if not path.exists():
        raise FileNotFoundError(f"Required local asset is missing; downloads are disabled: {path}")
    return path


@contextlib.contextmanager
def offline_assets():
    from openpi.shared import download

    cache_root = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
    with mock.patch.object(download, "maybe_download", side_effect=lambda url, **kw: local_asset(
        url, cache_root=cache_root, **kw
    )), mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
        yield


def cfm_draw(random, root_rng, action_shape):
    """Exactly pi0.py's split order, default RNG dtypes, and time distribution."""
    _preprocess_rng, noise_rng, time_rng = random.split(root_rng, 3)
    noise = random.normal(noise_rng, action_shape)
    time = random.beta(time_rng, 1.5, 1, action_shape[:-2]) * 0.999 + 0.001
    return noise, time


def numeric(value):
    array = np.asarray(value)
    values = array.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic output")
    return {"shape": list(array.shape), "dtype": str(array.dtype), "values": values.tolist()}


def grouped_error(prediction, target, valid):
    """Keep every physical coordinate separate, including base rotation/velocity."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if error.shape != (len(valid), 23) or not valid.any():
        raise ValueError("Endpoint error requires matching H x 23 arrays and a nonempty mask")
    result = {"valid_steps": int(valid.sum()), "groups": {}}
    for name, (start, end) in ev.ACTION_GROUPS.items():
        selected = error[valid, start:end]
        result["groups"][name] = {"coordinates": list(range(start, end)),
                                   "mse_per_coordinate": (selected ** 2).mean(axis=0).tolist(),
                                   "mae_per_coordinate": np.abs(selected).mean(axis=0).tolist()}
    return result


def repeat_checks(first, second):
    """Compare the two actual forwards, retaining failures in the receipt."""
    fields = {"native_returned_loss": (
        first["cfm"]["native_returned_loss_including_weighted_language_ce"],
        second["cfm"]["native_returned_loss_including_weighted_language_ce"]),
        "endpoint_actions": (first["endpoint"]["actions"], second["endpoint"]["actions"])}
    if first["cfm"]["metrics"].keys() != second["cfm"]["metrics"].keys():
        raise ValueError("Repeated native CFM metrics changed keys")
    fields.update({f"cfm_metric/{key}": (value, second["cfm"]["metrics"][key])
                   for key, value in first["cfm"]["metrics"].items()})
    checks = {}
    for name, (reference, repeated) in fields.items():
        left, right = np.asarray(reference["values"]), np.asarray(repeated["values"])
        same_layout = reference["shape"] == repeated["shape"] and reference["dtype"] == repeated["dtype"]
        checks[name] = {"same_shape_and_dtype": same_layout,
                        "max_abs_difference": float(np.max(np.abs(left - right))) if same_layout else None,
                        "within_tolerance": bool(same_layout and np.allclose(
                            left, right, rtol=REPEAT_RTOL, atol=REPEAT_ATOL))}
    identities = {
        "same_transformed_training_sample": first["transformed_training_sample_sha256"] == second["transformed_training_sample_sha256"],
        "same_root_key_data": first["rng"]["root_key_data"] == second["rng"]["root_key_data"],
        "same_noise": first["rng"]["noise_sha256"] == second["rng"]["noise_sha256"],
        "same_time": first["rng"]["time_sha256"] == second["rng"]["time_sha256"],
    }
    return {"rtol": REPEAT_RTOL, "atol": REPEAT_ATOL, "identities": identities,
            "checks": checks, "all_passed": all(identities.values()) and all(
                check["within_tolerance"] for check in checks.values()),
            "extra_cfm_forwards": 1, "extra_ten_step_euler_calls": 1,
            "checkpoint_reloaded": False}


def prepare_targets(raw, policy_input, data_config, policy, transform_dataset):
    """Use the real training transform and independently check the policy path."""
    if "action" not in raw or "actions" not in policy_input:
        raise ValueError("Training targets are missing: raw action and packed actions are required")
    prepared = transform_dataset([copy.deepcopy(raw)], data_config)[0]
    if "actions" not in prepared:
        raise ValueError("Training transform dropped actions")
    # This is SubtaskPolicy's existing provided-label preparation, not a new
    # normalizer/tokenizer. Check its arrays against the canonical training path.
    served = policy._observation_transform(copy.deepcopy(policy_input))
    served = policy._pad(policy._tokenizer_transform(served))
    if ev.input_content_sha256(prepared) != ev.input_content_sha256(served):
        raise ValueError("Training and provided-subtask policy transforms disagree")
    actions = np.asarray(prepared["actions"])
    expected = (policy._model.action_horizon, policy._model.action_dim)
    if actions.shape != expected or not np.isfinite(actions).all():
        raise ValueError(f"Invalid transformed training target: {actions.shape}, expected {expected}")
    if expected[1] != 32 or np.any(actions[:, 23:] != 0):
        raise ValueError("Expected the original 23 coordinates plus nine zero-padded target coordinates")
    # Round-trip the target with the very same output transform used by infer.
    recovered = policy._output_transform({"state": np.array(prepared["state"], copy=True),
                                           "actions": np.array(actions, copy=True)})["actions"]
    raw_actions = np.asarray(raw["action"])
    if raw_actions.shape != (expected[0], 23) or not np.allclose(recovered, raw_actions, rtol=1e-5, atol=1e-5):
        raise ValueError("Transformed training targets do not round-trip to the recorded expert actions")
    return prepared, float(np.max(np.abs(np.asarray(recovered) - raw_actions)))


def evaluate_sample(*, policy, raw, item, data_config, transform_dataset, batch_inputs,
                    compute_cfm, root_rng, root_key_data, noise, time):
    before = ev.input_content_sha256(raw)
    prepared, roundtrip_error = prepare_targets(raw, item["policy_input"], data_config, policy, transform_dataset)
    observation, actions = batch_inputs(copy.deepcopy(prepared))
    changed = copy.deepcopy(prepared)
    changed["actions"] = np.asarray(changed["actions"]) + np.float32(1.0)
    changed_observation, _changed_actions = batch_inputs(changed)
    observation_hash = ev.input_content_sha256(observation.to_dict())
    changed_observation_hash = ev.input_content_sha256(changed_observation.to_dict())
    target_changed = not np.array_equal(np.asarray(actions), np.asarray(_changed_actions))
    if not target_changed or observation_hash != changed_observation_hash:
        raise ValueError("Changing the transformed target leaked into the model Observation")
    expected = (1, policy._model.action_horizon, policy._model.action_dim)
    if tuple(actions.shape) != expected or str(actions.dtype) != "float32":
        raise ValueError("CFM targets must be batched float32 [1, H, 32]")
    noise_host, time_host = np.asarray(noise), np.asarray(time)
    if noise_host.shape != expected or noise_host.dtype != np.float32 or time_host.dtype != np.float32:
        raise ValueError("CFM noise/time must use the model's default float32 RNG convention")
    if time_host.shape != (1,) or not np.all((time_host >= 0.001) & (time_host <= 1)):
        raise ValueError("Invalid CFM Beta time draw")

    # The native API samples the exact same noise/time internally from root_rng.
    loss, metrics = compute_cfm(root_rng, observation, actions, train=False)
    if "action_loss" not in metrics or "subtask_loss" not in metrics:
        raise ValueError("Expected native joint CFM action_loss and subtask_loss metrics")
    if np.asarray(loss).shape != (1, expected[1]):
        raise ValueError("Unexpected native CFM loss shape")
    joint_expected = (float(np.asarray(metrics["action_loss"]))
                      + policy._model.subtask_loss_weight * float(np.asarray(metrics["subtask_loss"])))
    joint_residual = float(np.asarray(loss).mean()) - joint_expected
    if not np.isclose(float(np.asarray(loss).mean()), joint_expected, rtol=REPEAT_RTOL, atol=REPEAT_ATOL):
        raise ValueError("Native joint loss disagrees with action_loss + weighted language CE")
    inference = policy.infer(copy.deepcopy(item["policy_input"]), noise=noise_host[0].copy())
    if inference.get("subtask_source") != "provided":
        raise ValueError("Endpoint must use the same provided subtask as CFM")
    endpoint = np.asarray(inference["actions"])
    target = np.asarray(raw["action"])
    if endpoint.shape != target.shape or endpoint.shape != (expected[1], 23):
        raise ValueError("Endpoint must cover the complete H x 23 expert target")
    if not np.isfinite(endpoint).all() or ev.input_content_sha256(raw) != before:
        raise ValueError("Nonfinite endpoint or mutated raw cache arrays")
    valid = np.asarray(item["valid"], dtype=bool)
    prefix = valid & (np.arange(len(valid)) < 16)
    return {
        "raw_sample_sha256": before,
        "policy_input_sha256": item["policy_input_sha256"],
        "transformed_training_sample_sha256": ev.input_content_sha256(prepared),
        "target_leakage_control": {"changed_normalized_action_target": target_changed,
                                   "observation_sha256": observation_hash,
                                   "changed_target_observation_sha256": changed_observation_hash,
                                   "observation_identical": True, "extra_inference_calls": 0},
        "training_target": {**numeric(actions), "sha256": ev.input_content_sha256(np.asarray(actions)),
                            "physical_roundtrip_max_abs_error": roundtrip_error},
        "rng": {"root_key_data": np.asarray(root_key_data).tolist(),
                "noise_shape": list(noise_host.shape), "noise_dtype": str(noise_host.dtype),
                "noise_sha256": ev.input_content_sha256(noise_host),
                "time": numeric(time_host), "time_sha256": ev.input_content_sha256(time_host)},
        "cfm": {"metrics": {name: numeric(value) for name, value in metrics.items()},
                "native_returned_loss_including_weighted_language_ce": numeric(loss),
                "joint_loss_identity_residual": joint_residual,
                "valid_23_coordinate_cfm": "not_measured; native action_loss includes all H x 32 coordinates"},
        "endpoint": {"actions": numeric(endpoint), "ground_truth": numeric(target),
                     "actions_sha256": ev.input_content_sha256(endpoint), "valid_mask": valid.tolist(),
                     "gt_error_valid_horizon": grouped_error(endpoint, target, valid),
                     "gt_error_valid_prefix16": grouped_error(endpoint, target, prefix)},
    }


def selected_samples(cache, limit):
    from openpi.training.subtask_data import SubtaskFromSegments

    episodes = ev.read_episode_ids(cache.paths["episodes_file"])
    metadata = ev.filter_episode_meta(ev.load_episode_meta(cache.paths["dataset_root"]), episodes)
    names = ev.task_names(cache.paths["dataset_root"])
    segments = SubtaskFromSegments.from_parquet(cache.paths["subtask_prompts"], episodes_index=episodes)
    samples = ev.replay_samples(cache.paths["sample_keys_file"], metadata, segments, names=names,
                                repo_id=cache.manifest["repo_id"], horizon=cache.probe["model_horizon_H"],
                                action_noise_seed=cache.probe["pairing_contract"]["action_noise_seed"])
    return samples[:limit], episodes


def run(args):
    # Importing the model dependencies does not instantiate or load a model.
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import data_loader

    if jax.config.x64_enabled:
        raise ValueError("Set JAX_ENABLE_X64=0 to preserve the evaluated policy's float32 RNG convention")
    cache = cached.VerifiedRawCache(args.cache_manifest)
    samples, episodes = selected_samples(cache, args.limit)
    build_args = SimpleNamespace(pi05_ckpt=str(args.checkpoint), subtask_conditioning="provided",
                                 max_token_len=args.max_token_len, horizon=cache.probe["model_horizon_H"],
                                 dataset=str(cache.paths["dataset_root"]), memory_cache_dir=None,
                                 repo_id=cache.manifest["repo_id"], subtask_prompts=cache.paths["subtask_prompts"],
                                 norm_asset_id=args.norm_asset_id, action_noise_seed=None)
    captured, raws = {}, {}
    original_create = policy_config.create_trained_policy

    def capture_policy(cfg, checkpoint, **kwargs):
        if captured:
            raise ValueError("Expected exactly one policy construction")
        policy = original_create(cfg, checkpoint, sample_kwargs={"num_steps": 10}, **kwargs)
        if policy._is_pytorch_model or not policy._model.train_subtask_prediction:
            raise ValueError("This probe requires the local JAX joint pi05 policy")
        # The factory can resolve a conditioning mode from checkpoint assets.
        effective_model = dataclasses.replace(cfg.model,
            subtask_action_conditioning=policy._model.subtask_action_conditioning)
        data = cfg.data.create(cfg.assets_dirs, effective_model)
        data = dataclasses.replace(data, norm_stats=kwargs["norm_stats"])
        captured.update(policy=policy, data=data, model_config=effective_model)
        return policy

    def provide(sample):
        raw = cache(sample)
        raws[(sample.episode_index, sample.frame_index)] = copy.deepcopy(raw)
        return raw

    with mock.patch.object(policy_config, "create_trained_policy", side_effect=capture_policy):
        _predict, get_sample = ev.build_pi05(build_args, episodes, sample_provider=provide)
    policy = captured["policy"]
    compute = nnx_utils.module_jit(policy._model.compute_loss_with_metrics, static_argnames=("train",))

    def batch_inputs(prepared):
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], prepared)
        return model_lib.Observation.from_dict(batched), batched["actions"]

    results = []
    for sample in samples:
        item = get_sample(sample)
        raw = raws.pop((sample.episode_index, sample.frame_index))
        root = jax.random.fold_in(jax.random.fold_in(jax.random.key(args.rng_seed), sample.episode_index),
                                  sample.frame_index)
        noise, time = cfm_draw(jax.random, root, (1, policy._model.action_horizon, policy._model.action_dim))
        call_args = dict(policy=policy, raw=raw, item=item, data_config=captured["data"],
            transform_dataset=data_loader.transform_dataset, batch_inputs=batch_inputs, compute_cfm=compute,
            root_rng=root, root_key_data=jax.random.key_data(root), noise=noise, time=time)
        result = evaluate_sample(**call_args)
        if not results:
            result["same_model_repeat"] = repeat_checks(result, evaluate_sample(**call_args))
        result.update(sample_id=cache.records[(sample.episode_index, sample.frame_index)]["sample_id"],
                      episode_index=sample.episode_index, frame_index=sample.frame_index,
                      task_index=sample.task_index, subtask=sample.subtask)
        results.append(result)
        print(f"completed {len(results)}/{len(samples)} episode={sample.episode_index} frame={sample.frame_index}", flush=True)
    param_dtypes = Counter(str(value.dtype) for value in jax.tree.leaves(nnx.state(policy._model, nnx.Param)))
    if set(param_dtypes) != {"bfloat16"}:
        raise ValueError(f"Expected deployment bfloat16 parameters, got {dict(param_dtypes)}")
    return {
        "schema_version": 1, "kind": "local_pi05_native_cfm_and_euler_endpoint", "status": "smoke_complete",
        "scope": "Deployment bfloat16 weights, train=False, provided subtask; not historical training loss, "
                 "closed-loop performance, a gauge decomposition, or an objective-novelty claim.",
        "checkpoint": str(args.checkpoint), "normalization_asset_id": build_args._resolved_norm_asset_id,
        "normalization_sha256": cached.file_sha256(build_args._norm_stats_path),
        "use_quantile_norm": build_args._use_quantile_norm,
        "action_conditioning": policy._model.subtask_action_conditioning,
        "max_token_len": args.max_token_len, "action_horizon": policy._model.action_horizon,
        "model_action_dim": policy._model.action_dim, "parameter_dtype_counts": dict(param_dtypes),
        "jax_version": jax.__version__, "numpy_version": np.__version__, "backend": jax.default_backend(),
        "protocol": {"selection": "first requested 1..3 samples in the frozen manifest order",
                     "rng_seed": args.rng_seed, "rng": "fold_in(fold_in(key(seed), episode_index), frame_index)",
                     "time_distribution": "0.001 + 0.999 * Beta(1.5, 1)",
                     "shared_noise": "CFM internal split(root,3)[1] is the Euler initial noise",
                     "existing_pcg64_eval_noise_used": False, "train_image_augmentation": False,
                     "euler_steps": 10, "euler_dt": -0.1, "euler_start_time": 1.0,
                     "cfm_mask": "native unmasked H x 32, including nine padded target coordinates",
                     "language_ce": "recorded separately once in native metrics; native returned loss includes it",
                     "same_model_repeat_parity": {"scope": "first canonical sample only, same loaded model/root/noise",
                                                  "rtol": REPEAT_RTOL, "atol": REPEAT_ATOL},
                     "endpoint_error": "each physical coordinate separately; no mixed-unit scalar",
                     "downloads_allowed": False},
        "raw_cache": {"manifest_path": str(cache.manifest_path), "manifest_sha256": cache.manifest_sha256,
                      "read_receipts": list(cache.receipts.values())},
        "samples": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--limit", type=int, default=1, choices=range(1, MAX_SAMPLES + 1))
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--max-token-len", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}")
    if not (args.checkpoint / "params").is_dir() or not (args.checkpoint / "assets").is_dir():
        parser.error("--checkpoint must be an existing local checkpoint with params/ and assets/")
    if (args.checkpoint / "model.safetensors").exists():
        parser.error("This diagnostic requires the actual JAX checkpoint, not a Torch conversion")
    if not 0 <= args.rng_seed < 2 ** 32 or args.max_token_len < 1:
        parser.error("Require uint32 --rng-seed and positive --max-token-len")
    with offline_assets():
        result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
