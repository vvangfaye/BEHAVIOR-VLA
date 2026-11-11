#!/usr/bin/env python3
"""CPU check of the RLinf ESPO model path on the real pi0.5 LIBERO checkpoint (synthetic observation).

1. ode_executed_actions(noise) equals RLinf's own flow-ODE recursion (sample_mean_var 'flow_ode', same noise).
2. expand_observation / KV repeat give per-copy results identical to the unexpanded batch.
3. A backward pass of a feature loss reaches action-expert parameters and none of the frozen VLM parameters.
Run (RLinf venv): CUDA_VISIBLE_DEVICES= PYTHONPATH=<RLinf> EMBODIED_PATH=<RLinf>/examples/embodiment python test_espo_rlinf_cpu.py
"""
import time

import torch
from hydra import compose, initialize_config_dir

from rlinf.models import get_model
from rlinf.models.embodiment.openpi import espo
from rlinf.models.embodiment.openpi.modules.model import IMAGE_KEYS, Observation, preprocess_observation
from rlinf.models.embodiment.openpi.sampling import rl_sampler

torch.manual_seed(0)
with initialize_config_dir(config_dir="/mnt/data1/faye/code/libero_rlinf_20261103/configs", version_base="1.1"):
    cfg = compose(config_name="libero_10_ppo_pi05_local2gpu_all")
t0 = time.time()
model = get_model(cfg.actor.model)
model.eval()
print(f"model loaded in {time.time() - t0:.0f}s; action_chunk={model.action_chunk} env_dim={model.action_env_dim} "
      f"horizon={model.action_horizon} model_dim={model.model_action_dim} steps={model.num_steps}")
B, M = 2, 2
images = {k: torch.rand(B, 3, 224, 224) * 2 - 1 for k in IMAGE_KEYS}
masks = {k: torch.ones(B, dtype=torch.bool) for k in IMAGE_KEYS}
tok = torch.randint(0, 1000, (B, int(cfg.actor.model.openpi.get("max_token_len", 200))))
tok_mask = torch.zeros_like(tok, dtype=torch.bool); tok_mask[:, :40] = True
obs = preprocess_observation(Observation(images=images, image_masks=masks, state=torch.zeros(B, 32),
                                         tokenized_prompt=tok, tokenized_prompt_mask=tok_mask), train=False)
with torch.no_grad():
    t0 = time.time()
    prefix_out, prefix_mask, kv = model.build_prefix_cache(obs)
    print(f"prefix built in {time.time() - t0:.0f}s: prefix_out {tuple(prefix_out.shape)} kv layers {len(kv)} k {tuple(kv[0][0].shape)}")
    noise = torch.randn(B, model.action_horizon, model.model_action_dim)
    u = espo.ode_executed_actions(model, obs, kv, prefix_mask, noise)
    # reference recursion written independently of espo.py
    x = noise.clone(); ts = rl_sampler.get_timesteps(model.num_steps, "cpu")
    for i in range(model.num_steps):
        v = model.velocity_from_suffix(model.run_suffix(obs, x, torch.full((B,), float(ts[i])), kv, prefix_mask)).float()
        x, _ = rl_sampler.sample_mean_var(x, v, torch.full((B,), i), noise_method="flow_ode", noise_level=0.0, num_steps=model.num_steps)
    ref = x[:, :model.action_chunk, :model.action_env_dim].reshape(B, -1)
    print("1. ODE sampler matches reference recursion: max|diff| =", (u - ref).abs().max().item())
    obs_m = espo.expand_observation(obs, M); kv_m = espo._repeat(kv, M); pm = prefix_mask.repeat_interleave(M, 0)
    u_m = espo.ode_executed_actions(model, obs_m, kv_m, pm, noise.repeat_interleave(M, 0))
    print("2. expanded batch reproduces per-state samples: max|diff| =", (u_m.view(B, M, -1) - u[:, None]).abs().max().item())
    z = espo.pooled_prefix(prefix_out, prefix_mask)
q = espo.ReturnPredictor(z.shape[1], u.shape[1])
for p in q.parameters():
    p.requires_grad_(False)
for p in model.parameters():
    p.grad = None
u_g = espo.ode_executed_actions(model, obs_m, kv_m, pm, torch.randn(B * M, model.action_horizon, model.model_action_dim))
phi = q.features(z.repeat_interleave(M, 0), u_g).view(B, M, -1).mean(1)
loss = (phi ** 2).sum()
t0 = time.time(); loss.backward(); print(f"backward through {model.num_steps} ODE steps x {B * M} samples: {time.time() - t0:.0f}s")
with_grad = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
print(f"3. params with nonzero grad: {len(with_grad)} (trainable {len(trainable)}); frozen params with grad: {len(frozen_with_grad)}")
print("   sample grad params:", with_grad[:3], "...", with_grad[-2:])
print("phi shape", tuple(phi.shape))
