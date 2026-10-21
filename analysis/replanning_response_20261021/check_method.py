"""Check the candidate objective, gradients, and a limit of local penalties.

CPU only. The synthetic system is chosen for exact analysis, not to model a
BEHAVIOR task or to estimate an empirical robot success rate.
"""
import json
from pathlib import Path
import torch
from response_chain import response_chain

torch.set_default_dtype(torch.float64)
torch.manual_seed(20261021)


def exact_moment_check():
    horizon, dimension = 4, 2
    matrices = (torch.randn(horizon, dimension, dimension) * .3).requires_grad_()
    biases = (torch.randn(horizon, dimension) * .1).requires_grad_()
    initial_mean = torch.tensor([.1, -.2])
    initial_covariance = torch.diag(torch.tensor([.03, .07]))
    covariances = torch.diag_embed(torch.full((horizon, dimension), .02))
    weights = torch.eye(dimension).repeat(horizon, 1, 1)
    loss, means, variances = response_chain(
        matrices, biases, initial_mean, initial_covariance, covariances, weights)

    # Enumerate a finite independent disturbance distribution exactly: each
    # variable is +/- its standard deviation. This avoids Monte Carlo error.
    count = (horizon + 1) * dimension
    integers = torch.arange(2 ** count)
    signs = 2 * ((integers[:, None] >> torch.arange(count)) & 1).double() - 1
    noise = signs.reshape(-1, horizon + 1, dimension)
    state = initial_mean + noise[:, 0] * initial_covariance.diag().sqrt()
    costs = []
    for j in range(horizon):
        state = state @ matrices[j].T + biases[j] + noise[:, j+1] * .02 ** .5
        costs.append(torch.einsum('bi,ij,bj->b', state, weights[j], state))
    direct_loss = torch.stack(costs).sum(0).mean()
    analytic_gradients = torch.autograd.grad(loss, (matrices, biases), retain_graph=True)
    direct_gradients = torch.autograd.grad(direct_loss, (matrices, biases))
    gradient_error = max(float((left-right).abs().max())
                         for left, right in zip(analytic_gradients, direct_gradients))
    loss_error = float((loss-direct_loss).abs())
    assert loss_error < 1e-12 and gradient_error < 1e-12
    return {"loss": float(loss), "enumerated_loss": float(direct_loss),
            "absolute_loss_error": loss_error, "max_gradient_error": gradient_error,
            "enumerated_paths": len(signs)}


def gains(theta):
    # x_{j+1}=u_j; policy gains alternate theta and 3-theta.
    return torch.stack([theta, 3-theta, theta, 3-theta]).reshape(4, 1, 1)


def composed_cost(theta):
    matrices = gains(theta)
    bias = torch.zeros(4, 1)
    initial_mean = torch.zeros(1)
    initial_covariance = torch.ones(1, 1) * .01 ** 2
    disturbances = torch.ones(4, 1, 1) * .001 ** 2
    weights = torch.tensor([0., 1., 0., 1.]).reshape(4, 1, 1)
    return response_chain(matrices, bias, initial_mean,
                          initial_covariance, disturbances, weights)[0]


def optimize(loss_function, initial):
    theta = torch.tensor(initial, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=.015)
    for _ in range(1800):
        optimizer.zero_grad()
        loss = loss_function(theta)
        loss.backward()
        optimizer.step()
    value = float(theta.detach())
    return {"theta": value, "goal_state_expected_squared_error":
            float(composed_cost(theta).detach()),
            "two_step_gain": value*(3-value),
            "nominal_action_mse": 0.0}


def nonseparability_check():
    # Two-step variance from an initial perturbation is k0^2*k1^2.
    # Its mixed derivative rules out a sum of independent functions of k0/k1.
    k0 = torch.tensor(1.1, requires_grad=True)
    k1 = torch.tensor(.8, requires_grad=True)
    loss = (k0*k1).square()
    first = torch.autograd.grad(loss, k0, create_graph=True)[0]
    mixed = torch.autograd.grad(first, k1)[0]
    assert abs(float(mixed)-4*1.1*.8) < 1e-12
    return {"mixed_derivative": float(mixed), "expected": 4*1.1*.8,
            "interpretation": "Not a sum of per-transition penalties; not a novelty proof."}


result = {
    "status": "mathematical implementation check only",
    "scope": "No VLA, simulator, physical cache, or robot evaluation was run.",
    "exact_moments_and_gradients": exact_moment_check(),
    "nonseparability": nonseparability_check(),
    "controlled_example": {
        "description": "Affine scalar, alternating phase gains, exact local model; goals after steps 2 and 4.",
        "initial": {"theta": 2.2, "cost": float(composed_cost(torch.tensor(2.2)))},
        "full_response": optimize(composed_cost, 2.2),
        "sum_of_local_squared_gains": optimize(lambda theta: gains(theta).square().sum(), 2.2),
        "limits": ["Only the specified local squared-gain baseline is compared.",
                   "Does not establish superiority to all stability-aware or model-based methods.",
                   "Nominal demonstration state/action are zero; BC cannot identify any feedback gain.",
                   "Exact affine dynamics avoid the unresolved nonlinear response-cache problem."]
    },
    "native_vla_feasibility": "Unverified: paired render restoration, adequate response rank, cache validity, sampler memory/cost."
}
destination = Path(__file__).with_name('check_result.json')
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
