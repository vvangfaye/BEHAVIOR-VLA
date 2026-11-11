"""Native free subtask generation on the existing 36 expert-observation cache."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cpus", required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, [int(x) for x in args.cpus.split(",")])
    sys.path.insert(0, str(HERE.parent / "flow_endpoint_screen_20261020"))
    import flow_endpoint_screen as flow
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.policies import policy_config

    destination = HERE / f"{args.name}.json"
    if destination.exists():
        raise ValueError("Refusing to overwrite completed results")
    cache = flow.cached.VerifiedRawCache(HERE.parent / "pilot_three_tasks_cache_v1/cache_manifest.json")
    samples, episodes = flow.selected_samples(cache, 36, "prefix")
    assert len(samples) == 36
    reference_path = HERE.parent / "calibration_screen_20261020/retrieved" / f"{args.name}.json"
    reference = json.loads(reference_path.read_text())
    reference_rows = {(r["episode_index"], r["frame_index"]): r for r in reference["sample_keys"]}
    names = flow.ev.task_names(cache.paths["dataset_root"])
    captured = {}
    original = policy_config.create_trained_policy

    def create(cfg, checkpoint, **kwargs):
        assert not captured
        policy = original(cfg, checkpoint, sample_kwargs={"num_steps": 10},
                          subtask_interval=1, subtask_temperature=0.0, **kwargs)
        assert not policy._is_pytorch_model
        captured.update(policy=policy, config=dataclasses.asdict(dataclasses.replace(
            cfg.model, subtask_action_conditioning=policy._model.subtask_action_conditioning)))
        return policy

    build_args = SimpleNamespace(pi05_ckpt=str(args.checkpoint), subtask_conditioning="provided",
                                 max_token_len=512, horizon=32,
                                 dataset=str(cache.paths["dataset_root"]), memory_cache_dir=None,
                                 repo_id=cache.manifest["repo_id"],
                                 subtask_prompts=cache.paths["subtask_prompts"],
                                 norm_asset_id="b1k_full100_joint_subtask_v1_drop_episode_s1",
                                 action_noise_seed=None)
    output, repeat = [], None
    with flow.offline_assets():
        print(f"Loading {args.name} on CPU", flush=True)
        with mock.patch.object(policy_config, "create_trained_policy", side_effect=create):
            _, get_sample = flow.ev.build_pi05(build_args, episodes, sample_provider=cache)
        policy = captured["policy"]
        assert digest(build_args._norm_stats_path) == reference["artifacts"]["norm_stats"]["sha256"]
        for index, sample in enumerate(samples):
            item = get_sample(sample)
            ref = reference_rows[(sample.episode_index, sample.frame_index)]
            assert item["policy_input_sha256"] == ref["policy_input_sha256"]
            raw = copy.deepcopy(item["policy_input"])
            raw.pop("subtask", None)
            raw.pop("actions", None)
            assert "subtask" not in raw and "actions" not in raw
            inputs = policy._observation_transform(copy.deepcopy(raw))
            assert "subtask" not in inputs and "actions" not in inputs
            state = inputs["state"] if policy._tokenizer_transform.discrete_state_input else None
            limit = policy._tokenizer_transform.max_subtask_len
            arrays = policy._tokenizer.tokenize_subtask_prefix(inputs["prompt"], state,
                                                               max_subtask_len=limit)
            packed = policy._pad(policy._pack(inputs, arrays))
            prefix = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], packed))
            key = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261020),
                                                       sample.episode_index), sample.frame_index)

            def generate():
                generated = jax.device_get(policy._generate(key, prefix, max_decoding_steps=limit,
                    temperature=0.0, eos_token_id=policy._tokenizer.eos_id))
                valid = bool(generated["valid_prefix"][0]) and bool(generated["has_eos"][0])
                tokens = generated["tokens"][0, :int(generated["lengths"][0])]
                text = policy._normalize_subtask(policy._tokenizer.decode_subtask(tokens)) if valid else None
                return text, np.asarray(tokens), valid

            label, tokens, valid = generate()
            if index == 0:
                again, again_tokens, again_valid = generate()
                repeat = {"text_equal": label == again, "tokens_equal": bool(np.array_equal(tokens, again_tokens)),
                          "validity_equal": valid == again_valid}
                assert all(repeat.values())
            row = {"task": names[sample.task_index], "episode_index": sample.episode_index,
                   "frame_index": sample.frame_index, "frames_to_annotation_end": sample.segment_end - sample.frame_index,
                   "annotation_label": sample.subtask, "generated_label": label, "valid_generation": valid,
                   "tokens": tokens.tolist(), "provided_policy_input_sha256": item["policy_input_sha256"],
                   "free_generation_input_sha256": flow.ev.input_content_sha256(raw),
                   "processed_prefix_sha256": flow.ev.input_content_sha256(prefix.to_dict())}
            output.append(row)
            print(json.dumps({"sample": index + 1, "task": row["task"], "frame": sample.frame_index,
                              "reference": sample.subtask, "generated": label}), flush=True)
        dtypes = sorted({str(p.dtype) for p in jax.tree.leaves(nnx.state(policy._model, nnx.Param))})
        assert dtypes == ["bfloat16"]
    result = {"status": "complete", "kind": "free_subtask_generation_on_existing_expert_cache",
              "checkpoint_name": args.name, "checkpoint_path": str(args.checkpoint), "backend": jax.default_backend(),
              "model_config": captured["config"], "parameter_dtypes": dtypes,
              "normalization_sha256": digest(build_args._norm_stats_path),
              "cache_manifest_sha256": cache.manifest_sha256,
              "raw_reference_sha256": digest(reference_path), "same_model_repeat": repeat,
              "generation_calls": len(output) + 1, "action_sampling_calls": 0, "samples": output,
              "scope": "36 training-seen expert states; no simulator, new observation collection or training. Annotation disagreement is not automatically a stage-selection error, especially at annotation boundaries."}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete", "samples": len(output), "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
