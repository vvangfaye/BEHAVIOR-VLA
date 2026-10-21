"""Synthetic sampler-tangent score fitting; reward is used only in evaluation."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


def main():
    plan_path = ROOT / "tangent_plan.json"
    plan = json.loads(plan_path.read_text())
    output = ROOT / "tangent_result.json"
    if output.exists():
        raise RuntimeError("Existing result; no automatic rerun")
    started = time.monotonic()
    torch.set_num_threads(4)
    delta = plan["tangent_half_step"]
    phi = math.exp(-0.5) / math.sqrt(2*math.pi)
    success0 = 0.5*math.erfc(1/math.sqrt(2))
    exact_gradient = -0.5*phi
    grids = [np.linspace(-9, 1, 25001), np.linspace(1, 9, 25001)]
    density = [np.exp(-x*x/2)/math.sqrt(2*math.pi) for x in grids]
    true_score = [x-1.5*(x*x-1) for x in grids]
    results = []
    for seed in plan["seeds"]:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed+1000)
        source = torch.randn((plan["training_samples_per_seed"], 1), generator=generator)
        action = source.clone()
        plus = delta + math.exp(-1.5*delta)*source
        minus = -delta + math.exp(1.5*delta)*source
        tangent = (plus-minus)/(2*delta)
        width = plan["hidden_width"]
        model = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width),
                              nn.SiLU(), nn.Linear(width, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"],
                                      weight_decay=plan["weight_decay"])
        for update in range(plan["updates"]):
            if time.monotonic()-started > plan["max_cpu_seconds"]:
                raise TimeoutError("Fixed CPU limit reached")
            index = torch.randint(len(action), (plan["batch_size"],), generator=generator)
            x = action[index].clone().requires_grad_(True)
            score = model(x)
            derivative = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
            loss = (0.5*score.square()-derivative*tangent[index]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite weak-score objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            estimated_scores = [model(torch.from_numpy(x[:, None]).float()).squeeze(1).numpy().astype(np.float64)
                                for x in grids]
        estimated_gradient = float(sum(np.trapz(a*p*s, x) for a, p, s, x in
                                        zip([-success0, 1-success0], density, estimated_scores, grids)))
        score_mse = float(sum(np.trapz(p*(s-t)**2, x) for p, s, t, x in
                             zip(density, estimated_scores, true_score, grids)))
        relative_error = abs(estimated_gradient-exact_gradient)/abs(exact_gradient)
        theta_new = .01*estimated_gradient
        result = dict(seed=seed, estimated_gradient=estimated_gradient,
                      relative_gradient_error=relative_error, score_mse=score_mse,
                      update_parameter=theta_new,
                      actual_success_after_update=.5*math.erfc((1-theta_new)/(math.exp(-1.5*theta_new)*math.sqrt(2))),
                      passed=bool(estimated_gradient<0 and relative_error<.25))
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started, exact_reward_gradient=exact_gradient,
                  actual_success_before_update=success0, results=results,
                  synthetic_gate_passed=all(row["passed"] for row in results),
                  limitations=plan["limits"], new_gpu_jobs=0,
                  new_simulator_episodes=0, goal_complete=False)
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == "__main__":
    main()
