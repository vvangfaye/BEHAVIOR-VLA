"""Test projected weak score fitting with latent ambiguity and nuisance dimensions."""
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
VARIANCE0 = 1.16


def marginal_score(x):
    mean = np.zeros(x.shape[1]); mean[:2] = [2, 1]
    residual = np.stack([x-mean, x+mean], axis=1)
    energy = residual[:, :, 0]**2/VARIANCE0 + residual[:, :, 1]**2
    logits = -0.5*energy
    logits -= logits.max(1, keepdims=True)
    weights = np.exp(logits); weights /= weights.sum(1, keepdims=True)
    r0, r1 = residual[:, :, 0], residual[:, :, 1]
    component = np.stack([r0/VARIANCE0, r0*r0/VARIANCE0-1,
                          r0*r1/VARIANCE0], axis=-1)
    return (weights[:, :, None]*component).sum(1)


def events(x):
    directions = np.asarray([[1., 0.], [1/math.sqrt(2), 1/math.sqrt(2)],
                             [1/math.sqrt(2), -1/math.sqrt(2)]])
    thresholds = [3., 2., 1.]
    rewards, means, gradients = [], [], []
    for w, threshold in zip(directions, thresholds):
        variance = VARIANCE0*w[0]**2+w[1]**2
        std = math.sqrt(variance)
        derivative_mean = np.asarray([w[0], 0., 0.])
        derivative_variance = np.asarray([0., 2*VARIANCE0*w[0]**2, 2*w[0]*w[1]])
        value, gradient = 0., np.zeros(3)
        for mode in [-1, 1]:
            z = (threshold-mode*(2*w[0]+w[1]))/std
            phi = math.exp(-z*z/2)/math.sqrt(2*math.pi)
            value += .25*math.erfc(z/math.sqrt(2))
            gradient += .5*phi*(derivative_mean/std+z*derivative_variance/(2*variance))
        rewards.append((x[:, :2]@w > threshold).astype(np.float64))
        means.append(value); gradients.append(gradient)
    return np.stack(rewards, 1)-np.asarray(means), np.stack(gradients)


def generate(n, dimension, generator):
    z = torch.randn((n, dimension+1), generator=generator)
    mode = (2*torch.randint(2, (n,), generator=generator)-1).float()
    raw = z[:, 0]+.4*z[:, -1]
    x = z[:, :dimension].clone()
    x[:, 0] = 2*mode+raw
    x[:, 1] += mode
    # Derivatives of first output wrt shift, log scale, and cross loading.
    tangent = torch.stack([torch.ones(n), raw, z[:, 1]], 1)
    return x, tangent


def main():
    plan_path = ROOT/'multimodal_plan.json'
    plan = json.loads(plan_path.read_text())
    output = ROOT/'multimodal_result.json'
    if output.exists():
        raise RuntimeError('Existing result; refusing automatic rerun')
    torch.set_num_threads(4)
    started = time.monotonic()
    rows = []
    for dimension in plan['dimensions']:
        for seed in plan['seeds']:
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed+1000+dimension)
            train_x, tangent = generate(plan['training_samples'], dimension, generator)
            test_x, _ = generate(plan['test_samples'], dimension, generator)
            width = plan['hidden_width']
            model = nn.Sequential(nn.Linear(dimension, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                          weight_decay=plan['weight_decay'])
            for update in range(plan['updates']):
                if time.monotonic()-started > plan['max_cpu_seconds']:
                    raise TimeoutError('Fixed CPU time cap reached')
                index = torch.randint(len(train_x), (plan['batch_size'],), generator=generator)
                x = train_x[index].clone().requires_grad_(True)
                v = (2*torch.randint(2, (len(index), 3), generator=generator)-1).float()
                projected = (model(x)*v).sum(1)
                derivative = torch.autograd.grad(projected.sum(), x, create_graph=True)[0][:, 0]
                projected_tangent = (tangent[index]*v).sum(1)
                loss = (.5*projected.square()-derivative*projected_tangent).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite projected weak-score objective')
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            with torch.inference_mode():
                prediction = torch.cat([model(x) for x in test_x.split(10000)]).numpy().astype(np.float64)
            truth = marginal_score(test_x.numpy().astype(np.float64))
            advantages, gradient = events(test_x.numpy().astype(np.float64))
            estimate = advantages.T@prediction/len(prediction)
            oracle_mc = advantages.T@truth/len(prediction)
            relative_error = float(np.linalg.norm(estimate-gradient)/np.linalg.norm(gradient))
            cosines = (estimate*gradient).sum(1)/(np.linalg.norm(estimate,axis=1)*np.linalg.norm(gradient,axis=1))
            score_error = float(np.square(prediction-truth).mean()/np.square(truth).mean())
            row = dict(dimension=dimension, seed=seed,
                       relative_gradient_error=relative_error,
                       event_gradient_cosines=cosines.tolist(), relative_score_mse=score_error,
                       analytic_event_gradients=gradient.tolist(),
                       learned_event_gradients=estimate.tolist(),
                       oracle_mc_relative_error=float(np.linalg.norm(oracle_mc-gradient)/np.linalg.norm(gradient)),
                       passed=bool(relative_error<.25 and np.all(cosines>.9) and score_error<.25))
            rows.append(row); print(json.dumps(row), flush=True)
    report = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started, results=rows,
                  synthetic_gate_passed=all(row['passed'] for row in rows),
                  scope='Known mixture marginal, discarded latent mode and noise, nuisance coordinates; no VLA or environment outcomes.',
                  new_gpu_jobs=0, new_simulator_episodes=0, goal_complete=False)
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
