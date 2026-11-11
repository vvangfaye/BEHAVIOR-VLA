"""CPU algorithm checks on a small continuous Gaussian latent chain.

Checks actual MH/MALA implementation, actor gradients, terminal prefix
marginalization and finite-compute variance. This is not a VLA experiment.
"""
import json
from pathlib import Path

import torch

from posterior import GaussianChainPosterior, normal_logprob


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent
GEN = torch.Generator().manual_seed(383529)
BATCH, DIM, STEPS = 8192, 4, 3
A0 = torch.randn(STEPS, DIM, DIM, generator=GEN) * 0.12 + 0.45 * torch.eye(DIM)
A1 = torch.randn(STEPS, DIM, DIM, generator=GEN) * 0.18
OFFSET = torch.randn(STEPS, DIM, generator=GEN) * 0.25
SCALE = torch.tensor([0.65, 0.55, 0.45])
THETA = torch.tensor(0.17, requires_grad=True)
MASK = torch.tensor([True, True, False, False])


def transition(x, step, theta=THETA):
    mean = x @ (A0[step] + theta * A1[step]).T + theta * OFFSET[step]
    return mean, torch.ones_like(mean) * SCALE[step]


def marginal_parameters(theta):
    mean, covariance = torch.zeros(DIM), torch.eye(DIM)
    for k in range(STEPS):
        a = A0[k] + theta * A1[k]
        mean = a @ mean + theta * OFFSET[k]
        covariance = a @ covariance @ a.T + SCALE[k]**2 * torch.eye(DIM)
    return mean[MASK], covariance[MASK][:, MASK]


def score(path, observed):
    total = torch.zeros(path.shape[0])
    for k in range(STEPS):
        mean, sigma = transition(path[:, k], k)
        derivative = path[:, k] @ A1[k].T + OFFSET[k]
        if k == STEPS - 1:
            total += ((observed - mean[:, MASK]) * derivative[:, MASK]
                      / sigma[:, MASK]**2).sum(-1)
        else:
            total += ((path[:, k + 1] - mean) * derivative / sigma**2).sum(-1)
    return total.detach()


def main():
    engine = GaussianChainPosterior(transition, STEPS, MASK)
    with torch.no_grad():
        chain = [torch.randn(BATCH, DIM, generator=GEN)]
        for k in range(STEPS):
            mean, sigma = transition(chain[-1], k)
            chain.append(mean + sigma * torch.randn(BATCH, DIM, generator=GEN))
        observed = chain[-1][:, MASK].clone()
        latent = torch.stack(chain[:-1], dim=1)
    original, observed_before = latent.clone(), observed.clone()

    local_errors, balance_errors = [], []
    for site in range(STEPS):
        x = latent[:16, site]
        y = x + 0.2 * torch.randn(x.shape, generator=GEN)
        changed = latent[:16].clone()
        changed[:, site] = y
        delta_joint = engine.log_joint(changed, observed[:16]) - engine.log_joint(latent[:16], observed[:16])
        delta_local = (engine.conditional_logprob(y, latent[:16], observed[:16], site)
                       - engine.conditional_logprob(x, latent[:16], observed[:16], site))
        local_errors.append(float((delta_joint - delta_local).abs().max().detach()))
        lp_x, grad_x = engine.node_value_grad(x, latent[:16], observed[:16], site)
        lp_y, grad_y = engine.node_value_grad(y, latent[:16], observed[:16], site)
        step = 0.3
        sigma = torch.full_like(x, step)
        q_yx = normal_logprob(y, x + step**2 * grad_x / 2, sigma).sum(-1)
        q_xy = normal_logprob(x, y + step**2 * grad_y / 2, sigma).sum(-1)
        log_ratio = lp_y + q_xy - lp_x - q_yx
        forward = lp_x + q_yx + torch.minimum(log_ratio, torch.zeros_like(log_ratio))
        reverse = lp_y + q_xy + torch.minimum(-log_ratio, torch.zeros_like(log_ratio))
        balance_errors.append(float((forward - reverse).abs().max()))
    assert max(local_errors) < 1e-10 and max(balance_errors) < 1e-10

    auto, = torch.autograd.grad(engine.log_joint(latent[:16], observed[:16]).sum(), THETA)
    analytic = score(latent[:16], observed[:16]).sum()
    gradient_error = float((auto - analytic).abs())
    assert gradient_error < 1e-10

    mean, covariance = marginal_parameters(THETA)
    mean_prime, covariance_prime = torch.autograd.functional.jvp(
        marginal_parameters, THETA, torch.ones_like(THETA))[1]
    inverse = covariance.detach().inverse()
    residual = observed - mean.detach()
    v = residual @ inverse
    exact_score = (-0.5 * torch.trace(inverse @ covariance_prime)
                   + v @ mean_prime + 0.5 * (v * (v @ covariance_prime)).sum(-1)).detach()
    reward = (torch.cos(observed[:, 0]) + 0.3 * torch.sin(observed[:, 1])).detach()
    exact_return = (torch.exp(-covariance[0, 0] / 2) * torch.cos(mean[0])
                    + 0.3 * torch.exp(-covariance[1, 1] / 2) * torch.sin(mean[1]))
    true_gradient, = torch.autograd.grad(exact_return, THETA)

    def summarize(samples, count):
        g = reward * samples
        error = float(g.mean() - true_gradient)
        standard_error = float(g.std(unbiased=True) / BATCH**0.5)
        return {'posterior_path_count': count, 'mean_gradient': float(g.mean()),
                'standard_error': standard_error, 'gradient_error': error,
                'error_in_standard_errors': error / standard_error,
                'gradient_variance': float(g.var(unbiased=True)),
                'conditional_score_mse': float((samples - exact_score).square().mean())}

    accumulated = score(latent, observed)
    summaries = [summarize(accumulated, 1)]
    acceptance = []
    for m in range(1, 16):
        for site in range(STEPS):
            latent, accepted = engine.mala_step(latent, observed, site, 0.3, generator=GEN)
            acceptance.append(float(accepted.to(torch.float64).mean()))
        accumulated = accumulated + score(latent, observed)
        if m + 1 in (4, 8, 16):
            summaries.append(summarize(accumulated / (m + 1), m + 1))
    assert torch.equal(observed, observed_before)
    assert not torch.equal(latent, original)
    assert max(abs(r['error_in_standard_errors']) for r in summaries) < 6
    result = {
        'scope': 'Small continuous linear-Gaussian chain, CPU only; not VLA feasibility or robot success.',
        'batch': BATCH, 'latent_nodes': STEPS, 'node_dimension': DIM,
        'observed_endpoint_dimensions': int(MASK.sum()),
        'local_vs_full_log_density_change_max_errors': local_errors,
        'mala_detailed_balance_log_flux_max_errors': balance_errors,
        'joint_parameter_gradient_max_abs_error': gradient_error,
        'true_return_gradient': float(true_gradient.detach()),
        'exact_marginal_gradient_variance': float((reward * exact_score).var(unbiased=True)),
        'mean_acceptance': sum(acceptance) / len(acceptance),
        'results': summaries,
        'observed_actions_bitwise_unchanged': True,
        'limitations': ['A linear chain can mix much more easily than a VLA.',
                        'Acceptance is not a substitute for effective sample size.',
                        'No native action expert or environment was executed.'],
    }
    (ROOT / 'continuous_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
