import json, os, sys, tempfile, types
sys.path.insert(0, os.path.abspath("ar_test/RLinf"))
import torch
from omegaconf import OmegaConf
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
d2 = tempfile.mkdtemp(); torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(4, 4))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
cfg = OmegaConf.create({"algorithm": {"accept_rollback": {"enable": True, "reference": "best", "delta_task": 0.15, "delta_mean": 0.08,
                        "z": 1.64, "alpha": 0.2, "min_n": 8, "lr_decay": 0.5, "wait_s": 3}}, "env": {"train": {"task_stats_dir": d2, "total_num_envs": 32, "rollout_epoch": 4}}})
act = types.SimpleNamespace(cfg=cfg, model=model, optimizer=opt, lr_scheduler=sched, _rank=0, device=torch.device("cpu"))
for n in ("_ar_check", "_ar_read_new"):
    setattr(act, n, types.MethodType(getattr(EmbodiedFSDPActor, n), act))
act._ar_drop, act._ar_to_cpu = EmbodiedFSDPActor._ar_drop, EmbodiedFSDPActor._ar_to_cpu
def write(rates):
    with open(f"{d2}/env_1.jsonl", "a") as f:
        for t, (n, p) in rates.items():
            k = round(n * p)
            for i in range(n): f.write(json.dumps({"task": t, "s": int(i < k)}) + "\n")
def train_step():
    x = torch.randn(8, 4); loss = model(x).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
base = {t: (13, 0.4) for t in range(10)}; base[9] = (11, 0.27)
train_step()                                   # optimizer has moments + step before the first snapshot
write(base); m = act._ar_check(); assert m["ar/accepted"] == 1
snap_params = [p.clone() for p in act._ar_state["snap"][0]]
snap_state = {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} for k, v in act._ar_state["snap"][1]["state"].items()}
live_params0 = [p.detach().clone() for p in model.parameters()]
for _ in range(3): train_step()                # live params / moments / step move on
for p, s in zip(model.parameters(), snap_params): assert not torch.equal(p.detach(), s)
for k, v in act._ar_state["snap"][1]["state"].items():   # snapshot unchanged by the live updates (independent copy)
    for kk, vv in v.items(): assert torch.equal(vv, snap_state[k][kk]), (k, kk)
# rule unchanged: single-task drop 0.23 accepted (no anchor move: rate below the anchor), real collapse rolled back
r = dict(base); r[8] = (13, 0.4 - 0.23); write(r); m = act._ar_check(); assert m["ar/accepted"] == 1
opt.param_groups[0]["lr"] = 8e-4
r2 = dict(base); r2[3] = (13, 0.0); write(r2); m = act._ar_check()
assert m["ar/accepted"] == 0 and abs(m["ar/lr"] - 4e-4) < 1e-12, m
for p, s in zip(model.parameters(), live_params0): assert torch.equal(p.detach(), s)   # params back at the anchor
for i, p in enumerate(model.parameters()):
    st = opt.state[p]
    assert torch.equal(st["exp_avg"], snap_state[i]["exp_avg"]) and torch.equal(st["exp_avg_sq"], snap_state[i]["exp_avg_sq"])
    assert st["step"].data_ptr() != act._ar_state["snap"][1]["state"][i]["step"].data_ptr()   # no aliasing
step_before = act._ar_state["snap"][1]["state"][0]["step"].clone()
train_step(); train_step()                     # Adam increments 'step' in place: the snapshot must not change
assert torch.equal(act._ar_state["snap"][1]["state"][0]["step"], step_before)
print("v2.2 ok: rule unchanged (accept 0.23 drop, roll back collapse, lr 8e-4 -> 4e-4), snapshot independent, exact restore, no step aliasing")
