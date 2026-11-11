"""One backend, two full closed-loop episodes; common explicit initial noise."""

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
from common import Simulator, env_observation, label_text, torch_model, write


def jax_policy(plan):
    import jax
    import jax.numpy as jnp
    from b1k_rlinf.export import read_params
    from openpi.policies import policy_config
    from openpi.training import config

    cfg = config.get_config("pi05_b1k")
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(
            cfg.model,
            dtype="float32",
            train_subtask_prediction=True,
            max_token_len=512,
            max_subtask_len=64,
            subtask_loss_chunk_size=64,
        ),
        data=dataclasses.replace(
            cfg.data,
            repo_id="b1k_full100_joint_subtask_v1_drop_episode_s1",
            robot_config_name="b1k/R1Pro",
        ),
    )
    original = policy_config._model.restore_params

    def restore(path, dtype=None):
        params = read_params(Path(path).parent)
        return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)

    policy_config._model.restore_params = restore
    try:
        return policy_config.create_trained_policy(
            cfg,
            plan["original"],
            subtask_interval=1,
            subtask_temperature=0.0,
            sample_kwargs={"num_steps": 10},
        )
    finally:
        policy_config._model.restore_params = original


def infer_jax(policy, raw, noise):
    data = {"observation/state": raw["state"], "prompt": raw["prompt"]}
    for i, image in enumerate(raw["images"]):
        data[f"observation/image_{i}"] = image
    result = policy.infer(data, noise=noise)
    return np.asarray(result["actions"][:16], dtype=np.float32), result["subtask"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--backend", choices=["jax", "torch"], required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan["output"]) / args.backend
    root.mkdir(exist_ok=args.backend == "jax")
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    if args.backend == "jax":
        model = jax_policy(plan)
        import jax
        assert jax.default_backend() == "gpu"
        write(root / "device.json", {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()], "precision": "float32", "matmul_precision": str(jax.config.jax_default_matmul_precision)})
    else:
        import torch

        model = torch_model(plan)
    from b1k_rlinf.io import resize_image

    records = []
    for instance in (plan.get("baseline_instances", plan["eval_instances"]) if args.backend == "jax" else plan["eval_instances"]):
        out = root / str(instance)
        sim = Simulator(plan, out, instance, "public_test", plan["eval_steps"])
        noise_rng = np.random.default_rng(plan["seed"])
        trace = []
        try:
            for index in range((plan["eval_steps"] + 15) // 16):
                raw = sim.current["observation"]
                noise = noise_rng.standard_normal((1, 32, 32)).astype(np.float32)
                write(out / "inference.json", {"request": index, "state": "generating", "steps": sim.current["status"]["steps"]})
                if args.backend == "jax":
                    actions, text = infer_jax(model, raw, noise)
                else:
                    import torch

                    actions, result = model.predict_action_batch(
                        env_observation(raw),
                        mode="eval",
                        noise=torch.from_numpy(noise).cuda(),
                    )
                    text = label_text(model, result)
                    actions = actions[0].cpu().numpy()
                assert actions.shape == (16, 23) and np.isfinite(actions).all()
                write(out / "inference.json", {"request": index, "state": "generated", "steps": sim.current["status"]["steps"], "subtask": text})
                remaining = plan["eval_steps"] - sim.current["status"]["steps"]
                trace.append(
                    {"request": index, "subtask": text, "actions": actions.tolist()}
                )
                # Keep evenly spaced real observations for same-input GPU comparison.
                if args.backend == "jax" and index % 20 == 0 and index < 160:
                    np.savez_compressed(
                        out / f"probe_{index:04d}.npz",
                        state=raw["state"],
                        prompt=raw["prompt"],
                        subtask=text,
                        images=np.stack([resize_image(x) for x in raw["images"]]),
                        noise=noise,
                        actions=actions,
                    )
                sim.step(actions[:remaining], text)
                write(out / "progress.json", sim.current["status"])
                if sim.current["done"]:
                    break
            record = {
                "instance": instance,
                "backend": args.backend,
                **sim.current["status"],
            }
            write(out / "result.json", record)
            write(out / "actions.json", trace)
            records.append(record)
        finally:
            sim.close()
    if args.backend == "jax":
        for instance in plan["eval_instances"]:
            if instance not in plan["baseline_instances"]:
                records.append(json.loads((root / str(instance) / "result.json").read_text()))
        records.sort(key=lambda record: record["instance"])
    write(root / "summary.json", records)


if __name__ == "__main__":
    main()
