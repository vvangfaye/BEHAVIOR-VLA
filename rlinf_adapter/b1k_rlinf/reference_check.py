"""Compare adapted ODE actions to the original JAX model on identical inputs."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--rtol", type=float, default=0.003)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_source.resolve()))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import nnx_utils

    from .export import read_params

    started = time.monotonic()
    print("Restoring original JAX parameters", flush=True)
    params = read_params(args.checkpoint)
    params = jax.tree.map(jnp.asarray, params)
    config = Pi0Config(
        pi05=True,
        dtype="float32",
        action_horizon=32,
        max_token_len=512,
        train_subtask_prediction=True,
        subtask_action_conditioning="task_and_subtask",
    )
    model = config.load(params)
    with np.load(args.inputs, allow_pickle=False) as z:
        images = {key[len("image__") :]: jnp.asarray(z[key]) for key in z.files if key.startswith("image__")}
        obs = Observation.from_dict(
            {
                "image": images,
                "image_mask": {key: jnp.ones((1,), dtype=bool) for key in images},
                "state": jnp.asarray(z["state"]),
                "tokenized_prompt": jnp.asarray(z["tokenized_prompt"]),
                "tokenized_prompt_mask": jnp.asarray(z["tokenized_prompt_mask"]),
                "token_ar_mask": jnp.asarray(z["token_ar_mask"], dtype=jnp.int32),
            }
        )
        noise = jnp.asarray(z["noise"])
        expected = z["torch_actions"].copy()
    print("Running the original JAX ODE sampler", flush=True)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    actual = np.asarray(sample(jax.random.key(0), obs, noise=noise, num_steps=10))
    error = np.abs(actual - expected)
    passed = bool(np.allclose(actual, expected, atol=args.atol, rtol=args.rtol))
    result = {
        "passed": passed,
        "scope": "Original JAX vs adapted Torch FP32 ODE; identical observations, generated subtask tokens and initial noise",
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "atol": args.atol,
        "rtol": args.rtol,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
