"""Launch a finite RLinf run with the joint-policy extension in every worker."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .io import CheckpointIO
from .prepare import UPSTREAM_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("config", "eval", "smoke", "train"), default="config")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    root = args.rlinf_root.resolve()
    checkpoint = args.checkpoint.resolve()
    contract = json.loads((checkpoint / "b1k_checkpoint.json").read_text())
    if contract.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Checkpoint was not prepared against the pinned RLinf version")
    io = CheckpointIO(checkpoint / "norm_stats.json")
    if io.sha256 != contract.get("norm_sha256"):
        raise ValueError("Checkpoint normalization checksum mismatch")
    if not (checkpoint / "model.safetensors").is_file() or not args.tokenizer.is_file():
        raise FileNotFoundError("Expected exported weights and a local SentencePiece tokenizer")
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        {
            "RLINF_ROOT": str(root),
            "RLINF_EXT_MODULE": "b1k_rlinf",
            "B1K_RL_CHECKPOINT": str(checkpoint),
            "B1K_TOKENIZER": str(args.tokenizer.resolve()),
            "B1K_RL_OUTPUT": str(args.output.resolve()),
            "EMBODIED_PATH": str(root / "examples/embodiment"),
            "PYTHONPATH": os.pathsep.join([str(package_root), str(root), env.get("PYTHONPATH", "")]),
        }
    )
    if args.mode == "config":
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        os.environ.update(env)
        with initialize_config_dir(config_dir=str(package_root / "config"), version_base="1.1"):
            cfg = compose(config_name="behavior_joint_pi05", overrides=args.overrides)
            print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    entry = root / "examples/embodiment/train_embodied_agent.py"
    command = [
        sys.executable,
        str(entry),
        "--config-path",
        str(package_root / "config"),
        "--config-name",
        "behavior_joint_pi05",
    ]
    if args.mode == "eval":
        command.append("runner.only_eval=True")
    elif args.mode == "smoke":
        command.extend(
            [
                "runner.max_epochs=1",
                "runner.save_interval=1",
                "runner.val_check_interval=1",
                "env.train.max_episode_steps=256",
                "env.train.max_steps_per_rollout_epoch=256",
                "env.eval.max_episode_steps=256",
                "env.eval.max_steps_per_rollout_epoch=256",
            ]
        )
    command.extend(args.overrides)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
