import json, os, sys, tempfile, types
sys.path.insert(0, os.path.abspath("ar_test/RLinf"))
import torch
from omegaconf import OmegaConf
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor

def make(remeasure):
    d2 = tempfile.mkdtemp(); torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-6); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    cfg = OmegaConf.create({"algorithm": {"accept_rollback": {"enable": True, "reference": "best", "delta_task": 0.15, "delta_mean": 0.08,
                            "z": 1.64, "alpha": 0.2, "min_n": 8, "lr_decay": 0.5, "wait_s": 3, "remeasure": remeasure}},
                           "env": {"train": {"task_stats_dir": d2, "total_num_envs": 32, "rollout_epoch": 4}}})
    act = types.SimpleNamespace(cfg=cfg, model=model, optimizer=opt, lr_scheduler=sched, _rank=0, device=torch.device("cpu"))
    for n in ("_ar_check", "_ar_read_new"):
        setattr(act, n, types.MethodType(getattr(EmbodiedFSDPActor, n), act))
    act._ar_drop, act._ar_to_cpu = EmbodiedFSDPActor._ar_drop, EmbodiedFSDPActor._ar_to_cpu
    def write(tasks):  # tasks: {t: (n, s)}
        with open(f"{d2}/env_1.jsonl", "a") as f:
            for t, (n, s) in tasks.items():
                for i in range(n): f.write(json.dumps({"task": t, "s": int(i < s)}) + "\n")
    return act, write

# per-task (n, s) taken from the seed-3 log: 10:56 anchor, 11:09 candidate, 11:15 + 11:22 re-measurements of the anchor policy
A = {0: (13, 6), 1: (14, 8), 2: (16, 10), 3: (15, 14), 4: (12, 6), 5: (12, 2), 6: (12, 8), 7: (11, 2), 8: (12, 1), 9: (11, 1)}
C = dict(A); C[2] = (15, 2)                      # candidate after the update: task 2 collapses (real drop vs anchor)
R1 = {0: (13, 5), 1: (14, 7), 2: (16, 4), 3: (15, 15), 4: (15, 6), 5: (11, 0), 6: (12, 8), 7: (10, 0), 8: (11, 1), 9: (11, 1)}
R2 = dict(R1); R2[2] = (8, 2)
for rem in (False, True):
    act, write = make(rem)
    base = {t: (13, 5) for t in range(10)}; base[9] = (11, 3)
    write(base); m = act._ar_check(); assert m["ar/accepted"] == 1
    write(A); m = act._ar_check(); assert m["ar/accepted"] == 1          # new best -> anchor A (lucky task 2: 10/16)
    write(C); m = act._ar_check(); assert m["ar/accepted"] == 0          # rollback 1 (task 2 10/16 -> 2/15)
    write(R1); m1 = act._ar_check()                                      # re-measurement of the restored anchor policy
    write(R2); m2 = act._ar_check()
    ref2 = act._ar_state["ref"][2]
    print(f"remeasure={rem}: after rollback, same-policy rollout accepted={m1['ar/accepted']:.0f} (remeasure flag {m1['ar/remeasure']:.0f}), "
          f"lr {m1['ar/lr']:.2e}; next accepted={m2['ar/accepted']:.0f}; anchor task-2 reference now {ref2[1]}/{ref2[0]}, rollbacks {m2['ar/rollbacks']:.0f}")
    if rem:
        assert m1["ar/accepted"] == 1 and m1["ar/remeasure"] == 1 and abs(m1["ar/lr"] - 1.25e-6) < 1e-15
        assert ref2 == (32, 14)                                           # 10/16 pooled with 4/16
    else:
        assert m1["ar/accepted"] == 0                                     # v2.1 behaviour: the same policy is rolled back again
print("v2.3 ok")
