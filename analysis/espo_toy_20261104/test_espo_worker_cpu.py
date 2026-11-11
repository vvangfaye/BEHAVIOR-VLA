#!/usr/bin/env python3
"""CPU end-to-end check of the RLinf ESPO worker code (_espo_prepare + _train_micro_batch_espo) on the real pi0.5
LIBERO model, with forward_inputs produced by the model's own train-mode rollout sampler on synthetic observations.
Checks: targets/noise shapes, prefix-cache slots, finite ESPO + critic losses, gradients on the action expert and the
value head only, and that one SGD step on the ESPO loss moves the model features toward the targets.
"""
import contextlib
import time
import types

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.openpi.actor_prefix_cache import ActorPrefixStore
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

torch.manual_seed(0)
with initialize_config_dir(config_dir="/mnt/data1/faye/code/libero_rlinf_20261103/configs", version_base="1.1"):
    cfg = compose(config_name="libero_10_ppo_pi05_local2gpu_all",
                  overrides=["algorithm.loss_type=espo", "+algorithm.espo.model_samples=2", "+algorithm.espo.eta=3.0",
                             "+algorithm.espo.q_steps=50", "actor.micro_batch_size=2", "+algorithm.espo.prox_coef=1.0"] + __import__("sys").argv[1:])
model = get_model(cfg.actor.model)
model.train()
N = 4
env_obs = {
    "main_images": torch.randint(0, 255, (N, 256, 256, 3), dtype=torch.uint8),
    "wrist_images": torch.randint(0, 255, (N, 256, 256, 3), dtype=torch.uint8),
    "states": torch.randn(N, 8),
    "task_descriptions": ["put both the alphabet soup and the tomato sauce in the basket"] * N,
}
t0 = time.time()
with torch.no_grad():
    actions, result = model.predict_action_batch(env_obs, mode="train", compute_values=True)
print(f"rollout sampler on CPU: {time.time() - t0:.0f}s; forward_inputs keys: {sorted(result['forward_inputs'].keys())}")
fi = result["forward_inputs"]
rb = {
    "forward_inputs": fi,
    "prev_logprobs": result["prev_logprobs"],
    "prev_values": result["prev_values"].reshape(N, 1),
    "advantages": torch.tensor([[1.0], [-1.0], [0.5], [0.0]]),
    "returns": torch.tensor([[1.0], [0.0], [0.8], [0.3]]),
    "loss_mask": torch.tensor([[True], [True], [True], [False]]),
    "loss_mask_sum": torch.tensor([[200.0], [480.0], [300.0], [480.0]]),
}
fake = types.SimpleNamespace(cfg=cfg, model=model, rollout_batch=rb, device=torch.device("cpu"),
                             amp_context=contextlib.nullcontext(), _rank=0, version=0, gradient_accumulation=1, optimizer_steps=100, critic_warmup_steps=0,
                             grad_scaler=types.SimpleNamespace(scale=lambda x: x))
fake.before_micro_batch = lambda m, is_last_micro_batch: contextlib.nullcontext()
for name in ["_espo_policy_module", "_espo_slots", "_espo_prepare", "_train_micro_batch_espo"]:
    setattr(fake, name, types.MethodType(getattr(EmbodiedFSDPActor, name), fake))
store = ActorPrefixStore(pin=False, verify_every=1)
model._actor_prefix_store = store
t0 = time.time()
stats = fake._espo_prepare(store, rollout_size=N, batch_size_per_rank=N)
print(f"prepare: {time.time() - t0:.0f}s stats {stats}")
print("targets", tuple(rb["espo_target"].shape), "noise", tuple(rb["espo_noise"].shape), "store", store.stats)
assert rb["espo_target"].shape[0] == N and rb["espo_noise"].shape[:2] == (N, 2)
assert torch.isfinite(rb["espo_target"]).all()
# one micro-batch (first 2 samples) through the ESPO loss, slot (0, 0) as in the update loop
store.slot = (0, 0)
mb = {k: (v[:2] if torch.is_tensor(v) else {kk: vv[:2] for kk, vv in v.items()}) for k, v in rb.items()}
for p in model.parameters():
    p.grad = None
metrics = {}
t0 = time.time()
fake._train_micro_batch_espo(mb, metrics, is_last=True)
print(f"train micro-batch: {time.time() - t0:.0f}s metrics { {k: v for k, v in metrics.items() if not k.startswith('critic/explained')} }")
named = dict(model.named_parameters())
g = [n for n, p in named.items() if p.grad is not None and p.grad.abs().sum() > 0]
print(f"params with grad: {len(g)}; value_head: {sum('value_head' in n for n in g)}; "
      f"frozen with grad: {sum(1 for n in g if not named[n].requires_grad)}")
# features move toward targets after a small SGD step on the action expert
with torch.no_grad():
    out0 = model(forward_inputs=mb["forward_inputs"], espo_noise=mb["espo_noise"], espo_feature_fn=fake._espo_q.features)
    d0 = ((out0["espo_phi"] - mb["espo_target"]) ** 2).sum(-1)[:2]
    for n, p in named.items():
        if p.grad is not None and "value_head" not in n:
            p -= 3e-3 * p.grad / (p.grad.norm() + 1e-12) * p.norm().clamp(min=1.0) * 0.01
    out1 = model(forward_inputs=mb["forward_inputs"], espo_noise=mb["espo_noise"], espo_feature_fn=fake._espo_q.features)
    d1 = ((out1["espo_phi"] - mb["espo_target"]) ** 2).sum(-1)[:2]
print("distance to targets before/after a small step:", d0.tolist(), "->", d1.tolist())
print("store stats", store.end_training_call())
