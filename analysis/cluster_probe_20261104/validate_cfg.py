#!/usr/bin/env python3
"""Pre-submit check of an RLinf config + overrides (run with the RLinf venv; called by submit_rlinf.py).

Composes the Hydra config exactly as the job will and checks the divisibility constraints RLinf asserts at runtime:
  env workers divide total_num_envs (train and eval), group_size divides total_num_envs,
  per-actor-rank rollout samples divisible by the per-rank batch, per-rank batch divisible by the micro batch.
Usage: validate_cfg.py <config_name> <n_gpus> <train|eval> [overrides...]   (exit 1 on violation)
"""
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = "/mnt/data1/faye/code/libero_rlinf_20261103/configs"
os.environ.setdefault("EMBODIED_PATH", "/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/examples/embodiment")


def n_gpus_of(spec, total):
    spec = str(spec).strip()
    if spec == "all":
        return total
    n = 0
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            n += int(b) - int(a) + 1
        else:
            n += 1
    return n


def main():
    cfg_name, total, entry, overrides = sys.argv[1], int(sys.argv[2]), sys.argv[3], [o for o in sys.argv[4:] if o]
    overrides = [o for o in overrides if not o.startswith("runner.ckpt_path")]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = compose(config_name=cfg_name, overrides=overrides)
    place = {}
    for comps, spec in OmegaConf.to_container(cfg.cluster.component_placement).items():
        for c in comps.split(","):
            place[c.strip()] = n_gpus_of(spec, total)
    errors, info = [], {}
    n_env = place.get("env", total)
    for split in (["train", "eval"] if entry == "train" else ["eval"]):
        e = cfg.env.get(split)
        if e is None or e.get("total_num_envs") is None:
            continue
        envs = int(e.total_num_envs)
        if envs % n_env:
            errors.append(f"env.{split}.total_num_envs={envs} not divisible by {n_env} env workers")
        g = int(e.get("group_size", 1) or 1)
        if envs % g:
            errors.append(f"env.{split}.total_num_envs={envs} not divisible by group_size={g}")
    if entry == "train":
        n_act = place.get("actor", total)
        t = cfg.env.train
        chunks = int(cfg.actor.model.num_action_chunks)
        steps = int(t.max_steps_per_rollout_epoch) // chunks
        samples = int(t.total_num_envs) * int(t.rollout_epoch) * steps
        per_rank = samples // n_act
        gb, mb = int(cfg.actor.global_batch_size), int(cfg.actor.micro_batch_size)
        bpr = gb // n_act
        info = dict(actor_ranks=n_act, env_workers=n_env, rollout_samples=samples, per_rank=per_rank,
                    batch_per_rank=bpr, micro_batch=mb, optimizer_steps_per_epoch=(per_rank // bpr) if bpr else None)
        if samples % n_act:
            errors.append(f"rollout samples {samples} not divisible by {n_act} actor ranks")
        if gb % n_act:
            errors.append(f"global_batch_size {gb} not divisible by {n_act} actor ranks")
        if bpr == 0 or per_rank % bpr:
            errors.append(f"per-rank rollout samples {per_rank} not divisible by per-rank batch {bpr}")
        if bpr and bpr % mb:
            errors.append(f"per-rank batch {bpr} not divisible by micro_batch_size {mb}")
    print({"ok": not errors, "errors": errors, **info})
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
