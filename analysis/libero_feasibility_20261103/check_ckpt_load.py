#!/usr/bin/env python3
"""CPU check that an RL checkpoint loads strictly into the eval model built from an eval config.

Usage (RLinf venv, PYTHONPATH=RLinf, EMBODIED_PATH set):
  CUDA_VISIBLE_DEVICES= python check_ckpt_load.py <config_name> <full_weights.pt> [hydra overrides...]
Prints missing/unexpected keys and the max |delta| of the trainable action-expert weights versus the SFT start.
"""
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models import get_model

CONFIG_DIR = "/mnt/data1/faye/code/libero_rlinf_20261103/configs"


def main():
    name, ckpt, overrides = sys.argv[1], sys.argv[2], sys.argv[3:]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = compose(config_name=name, overrides=overrides)
    model_cfg = OmegaConf.merge(cfg.rollout.model, {})
    model = get_model(model_cfg)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sd = torch.load(ckpt, map_location="cpu")
    result = model.load_state_dict(sd, strict=False)
    print("missing keys:", len(result.missing_keys), result.missing_keys[:5])
    print("unexpected keys:", len(result.unexpected_keys), result.unexpected_keys[:5])
    changed = 0
    max_delta = 0.0
    for k, v in model.state_dict().items():
        if k in before and before[k].shape == v.shape and v.dtype.is_floating_point:
            d = (v.float() - before[k].float()).abs().max().item()
            if d > 0:
                changed += 1
                max_delta = max(max_delta, d)
    print(f"tensors changed vs SFT start: {changed}, max |delta| {max_delta:.3e}")
    ok = not result.missing_keys and not result.unexpected_keys
    print("STRICT LOAD OK" if ok else "STRICT LOAD WOULD FAIL")


if __name__ == "__main__":
    main()
