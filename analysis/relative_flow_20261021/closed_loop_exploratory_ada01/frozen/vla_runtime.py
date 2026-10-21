"""Load the pinned Pi0.5 backend without distributed worker registration."""
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get('RLINF_ROOT', ROOT / 'analysis/rlinf_adaptation_20261020/upstream'))
ADAPTER = Path(os.environ.get('B1K_ADAPTER_ROOT', ROOT / 'BEHAVIOR-VLAs/rlinf_adapter'))
sys.path[:0] = [str(ADAPTER), str(UPSTREAM)]
for name in ('rlinf.models', 'rlinf.models.embodiment'):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(UPSTREAM.joinpath(*name.split('.')))]
        sys.modules[name] = package

import torch
from omegaconf import OmegaConf

from b1k_rlinf.factory import build_model


def load_base(checkpoint, tokenizer, *, device='cuda', precision='bf16'):
    checkpoint = Path(checkpoint)
    if precision not in ('bf16', 'fp32'):
        raise ValueError('Expected bf16 or fp32')
    cfg = OmegaConf.create({
        'model_path': str(checkpoint), 'pi05': True, 'precision': precision,
        'action_dim': 23, 'num_action_chunks': 16, 'num_steps': 10,
        'openpi': {'model_action_dim': 32, 'action_horizon': 32, 'max_token_len': 512,
                   'train_expert_only': True, 'noise_method': 'flow_sde', 'noise_level': .4,
                   'ignore_last': True, 'joint_logprob': False, 'add_value_head': False,
                   'config_name': 'pi05_behavior'},
        'b1k_joint': {'enabled': True, 'norm_stats_path': str(checkpoint / 'norm_stats.json'),
                      'tokenizer_path': str(tokenizer), 'state_layout': 'checkpoint61'}})
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float32
    base = build_model(cfg, dtype).to(device).eval()
    # Keep master training weights and Adam moments in fp32, even when the
    # frozen backbone and autocast activations use bf16.
    for parameter in base.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return base
