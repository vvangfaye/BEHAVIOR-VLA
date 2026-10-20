"""Export existing JAX weights and record the joint-policy input contract."""

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

UPSTREAM_COMMIT = "91a302ec50ddacd1fdabc5859b85e3661d768236"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    contract = json.loads((source / "assets/subtask_conditioning.json").read_text())
    if contract.get("action_conditioning") != "task_and_subtask":
        raise ValueError("Source checkpoint is not task_and_subtask")
    if not (source / "params/_METADATA").is_file():
        raise ValueError("Expected a JAX checkpoint with params/_METADATA")
    norm = args.norm_stats.resolve()
    record = {
        "source_checkpoint": str(source),
        "source_format": "jax_orbax",
        "upstream_commit": UPSTREAM_COMMIT,
        "action_conditioning": "task_and_subtask",
        "action_horizon": 32,
        "model_action_dim": 32,
        "env_action_dim": 23,
        "execute_horizon": 16,
        "max_token_len": 512,
        "max_subtask_len": 64,
        "norm_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
        "source_params_metadata_sha256": hashlib.sha256(
            (source / "params/_METADATA").read_bytes()
        ).hexdigest(),
        "new_optimizer": True,
    }
    if args.inspect_only:
        print(json.dumps(record, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite export directory {args.output}")
    from .export import export_checkpoint

    staging = args.output.with_name(args.output.name + ".partial-" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    export_checkpoint(source, norm, staging)
    shutil.copyfile(source / "assets/subtask_conditioning.json", staging / "subtask_conditioning.json")
    (staging / "b1k_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
    staging.rename(args.output)
    print(json.dumps({"exported": str(args.output.resolve()), **record}, indent=2))


if __name__ == "__main__":
    main()
