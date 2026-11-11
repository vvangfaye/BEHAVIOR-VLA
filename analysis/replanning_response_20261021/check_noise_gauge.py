"""Apply the existing law-vs-pathwise counterexample to this candidate loss.

Not a new robotic phenomenon. The earlier source is
../training_boundary_audit_20261019/feedback_proposal_gate_20261020/
check_law_vs_pathwise_derivative.json. Here we check its consequence for the
implemented multi-replanning objective, not for a generic Jacobian penalty.
"""
import itertools
import json
from pathlib import Path
import torch
from response_chain import response_chain

torch.set_default_dtype(torch.float64)
sigma = .1
initial_variance = 1e-4
horizon = 4
# Exact first/second Gaussian moments, including the zero cross moment.
nodes = torch.tensor([[2 ** .5, 0], [-2 ** .5, 0],
                      [0, 2 ** .5], [0, -2 ** .5]])
weights = torch.tensor([0., 0., 0., 1.]).reshape(horizon, 1, 1)


def implemented_expected_cost(omega):
    values = []
    for indices in itertools.product(range(len(nodes)), repeat=horizon):
        z = nodes[list(indices)]
        # At nominal state x=0, g(x,z)=sigma*z1 and dg/dx=sigma*omega*z2.
        matrices = (sigma * omega * z[:, 1]).reshape(horizon, 1, 1)
        biases = (sigma * z[:, 0]).reshape(horizon, 1)
        value, _, _ = response_chain(
            matrices, biases, torch.zeros(1),
            torch.tensor([[initial_variance]]),
            torch.zeros(horizon, 1, 1), weights)
        values.append(value)
    return torch.stack(values).mean()


def analytic_affine_cost(omega):
    variance = torch.tensor(initial_variance)
    for _ in range(horizon):
        variance = (sigma * omega).square() * variance + sigma ** 2
    return variance


rows = []
for value in (0., 10., 20.):
    omega = torch.tensor(value, requires_grad=True)
    predicted = implemented_expected_cost(omega)
    expected_affine = analytic_affine_cost(omega)
    assert abs(float(predicted-expected_affine)) < 1e-12
    gradient = torch.autograd.grad(predicted, omega)[0]
    rows.append({
        'omega': value,
        'implemented_terminal_cost': float(predicted.detach()),
        'true_terminal_cost': sigma ** 2,
        'implemented_gradient_wrt_omega': float(gradient),
        'true_gradient_wrt_omega': 0.,
    })

result = {
    'status': 'counterexample_verified',
    'provenance': '../training_boundary_audit_20261019/feedback_proposal_gate_20261020/check_law_vs_pathwise_derivative.json',
    'plant': 'x_(t+1)=a_t',
    'policy': 'a_t=sigma*(cos(omega*x_t)*z1_t+sin(omega*x_t)*z2_t), fresh independent Gaussian noise at every step',
    'exact_policy_law': 'N(0,sigma^2), independent of both x and omega',
    'rows': rows,
    'small_perturbation_analysis': {
        'true_one_step_second_moment': 'sigma^2 for every initial variance',
        'linearized_one_step_second_moment': 'sigma^2 + sigma^2*omega^2*initial_variance',
        'omitted_term': 'The cross term between nominal random action and second state derivative cancels the added variance at leading order.',
        'consequence': 'Shrinking the state radius does not make the leading response-cost coefficient intrinsic to the policy law.'
    },
    'scope': [
        'Does not invalidate exact affine-policy moment propagation or the two-pass chain rule.',
        'Does invalidate treating this first-order sampled-policy response loss as an intrinsic measure of stochastic closed-loop performance.',
        'A different fixed noise convention is a valid modeling choice, but its derivative penalty is an additional inductive bias, not a behavior-only objective.',
        'No VLA, simulator, or new GPU job was run.'
    ]
}
Path(__file__).with_name('noise_gauge_result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
