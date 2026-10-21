"""Learn local action density ratios from samples, with analytic held-out checks."""
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
    plan_path = ROOT / "synthetic_plan.json"
    plan = json.loads(plan_path.read_text())
    output = ROOT / "synthetic_result.json"
    if output.exists():
        raise RuntimeError("Existing result; no implicit rerun")
    started = time.monotonic()
    torch.set_num_threads(4)
    delta = plan["parameter_half_step"]
    phi = math.exp(-0.5) / math.sqrt(2 * math.pi)
    success0 = 0.5 * math.erfc(1 / math.sqrt(2))

    def success(theta):
        return 0.5 * math.erfc((1 - theta) / (math.exp(-1.5 * theta) * math.sqrt(2)))

    def log_density(x, theta):
        return 1.5 * theta - 0.5 * (x - theta)**2 * math.exp(3 * theta)

    # Split at the reward discontinuity; rewards never enter classifier training.
    grids = [np.linspace(-9, 1, 25001), np.linspace(1, 9, 25001)]
    densities = [np.exp(-x*x/2) / math.sqrt(2*math.pi) for x in grids]
    exact_scores = [(log_density(x, delta) - log_density(x, -delta)) / (2*delta)
                    for x in grids]
    advantages = [-success0, 1-success0]

    def integrate_gradient(scores):
        return float(sum(np.trapz(a*p*s, x) for a, p, s, x in
                         zip(advantages, densities, scores, grids)))

    exact_fd_gradient = integrate_gradient(exact_scores)
    results = []
    for seed in plan["seeds"]:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 1000)

        def data(n):
            labels = (torch.arange(n) % 2).float().reshape(-1, 1)
            theta = delta * (2*labels - 1)
            actions = theta + torch.exp(-1.5*theta) * torch.randn((n, 1), generator=generator)
            return actions, labels

        train_x, train_y = data(plan["training_samples_per_seed"])
        test_x, test_y = data(plan["test_samples_per_seed"])
        width = plan["hidden_width"]
        model = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width),
                              nn.SiLU(), nn.Linear(width, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan["learning_rate"],
                                      weight_decay=plan["weight_decay"])
        for update in range(plan["updates"]):
            if time.monotonic() - started > plan["max_cpu_seconds"]:
                raise TimeoutError("Fixed synthetic CPU time cap reached")
            index = torch.randint(len(train_x), (plan["batch_size"],), generator=generator)
            loss = nn.functional.binary_cross_entropy_with_logits(model(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            test_loss = float(nn.functional.binary_cross_entropy_with_logits(model(test_x), test_y))
            score_estimates = [model(torch.from_numpy(x[:, None]).float()).squeeze(1).numpy().astype(np.float64)
                               / (2*delta) for x in grids]
        estimate = integrate_gradient(score_estimates)
        ratio_error = float(sum(np.trapz(p*(s-e)**2, x) for p, s, e, x in
                                zip(densities, score_estimates, exact_scores, grids)))
        relative_error = abs(estimate - exact_fd_gradient) / abs(exact_fd_gradient)
        theta_new = 0.01 * estimate
        result = dict(seed=seed, test_binary_cross_entropy=test_loss,
                      estimated_gradient=estimate, exact_fd_gradient=exact_fd_gradient,
                      relative_gradient_error=relative_error, score_mse=ratio_error,
                      update_parameter=theta_new, actual_success_after_update=success(theta_new),
                      passed=bool(estimate < 0 and relative_error < .25))
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(
        complete=True,
        observed_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,
        analytic_true_gradient=-0.5*phi,
        exact_finite_difference_score_gradient=exact_fd_gradient,
        finite_difference_bias=exact_fd_gradient+0.5*phi,
        original_cfm_surrogate_gradient=phi*math.pi/8,
        actual_success_before_update=success0,
        results=results,
        synthetic_gate_passed=all(row["passed"] for row in results),
        limitations=plan["scope_limits"],
        new_gpu_jobs=0,
        new_simulator_episodes=0,
        goal_complete=False,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
