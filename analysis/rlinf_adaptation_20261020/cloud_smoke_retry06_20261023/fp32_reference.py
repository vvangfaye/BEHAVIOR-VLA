"""Generate FP32 JAX reference actions on the 12 already-captured observations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from common import write
from rollout import infer_jax, jax_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    output = Path(plan["output"])
    policy = jax_policy(plan)
    probes = sorted(Path(plan["reference_probes_root"]).glob("*/probe_*.npz"))
    assert len(probes) == 12
    started = time.monotonic()
    for index, source in enumerate(probes):
        assert hashlib.sha256(source.read_bytes()).hexdigest() == plan["reference_probe_sha256"][str(source.relative_to(plan["reference_probes_root"]))]
        with np.load(source, allow_pickle=False) as data:
            raw = {"state": data["state"], "images": list(data["images"]), "prompt": str(data["prompt"])}
            actions, label = infer_jax(policy, raw, data["noise"])
            target = output / "precheck_reference" / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, state=raw["state"], prompt=raw["prompt"], subtask=label, images=np.stack(raw["images"]), noise=data["noise"], actions=actions)
        write(output / "reference_progress.json", {"completed": index+1, "total": len(probes), "elapsed_seconds": time.monotonic()-started, "precision": "float32"})
    print(json.dumps({"completed": len(probes), "elapsed_seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
