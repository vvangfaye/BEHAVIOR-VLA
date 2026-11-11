"""Verify first-order, one-anchor-at-a-time gradients for response training.

This is a small neural-policy check, not a native VLA memory benchmark.
"""
import json
from pathlib import Path
import torch
from torch import nn
from response_chain import response_chain

torch.set_default_dtype(torch.float64)
torch.manual_seed(812)
T, D, A, O = 7, 3, 4, 5
radius = .02
policy = nn.Sequential(nn.Linear(O, 32), nn.Tanh(), nn.Linear(32, A))
nominal = torch.randn(T, O)
directions = torch.randn(T, D, O)
observations = torch.cat([nominal[:, None],
                          nominal[:, None] + radius * directions,
                          nominal[:, None] - radius * directions], dim=1)
physical_A = torch.randn(T, D, D) * .1
physical_B = torch.randn(T, D, A) * .1
physical_c = torch.randn(T, D) * .01
initial_mean = torch.zeros(D)
initial_cov = torch.eye(D) * .01
disturbance = torch.eye(D).repeat(T, 1, 1) * .001
weights = torch.eye(D).repeat(T, 1, 1)


def responses(outputs):
    u = outputs[..., 0, :]
    k = ((outputs[..., 1:1+D, :] - outputs[..., 1+D:, :]) / (2 * radius)).transpose(-1, -2)
    return u, k


def outer(u, k):
    matrices = physical_A + physical_B @ k
    biases = physical_c + (physical_B @ u[..., None]).squeeze(-1)
    return response_chain(matrices, biases, initial_mean, initial_cov,
                          disturbance, weights)[0]


policy.zero_grad()
outputs = policy(observations)
u_full, k_full = responses(outputs)
full_loss = outer(u_full, k_full)
full_loss.backward()
full_gradients = [p.grad.clone() for p in policy.parameters()]
del outputs, u_full, k_full, full_loss

# Pass 1 stores response values, not policy activation graphs.
with torch.no_grad():
    stored_outputs = torch.stack([policy(x) for x in observations])
    stored_u, stored_k = responses(stored_outputs)
u_leaf, k_leaf = stored_u.detach().requires_grad_(), stored_k.detach().requires_grad_()
loss = outer(u_leaf, k_leaf)
u_adjoint, k_adjoint = torch.autograd.grad(loss, (u_leaf, k_leaf))

# Pass 2 reconstructs and backpropagates only one anchor's policy graph.
# A stochastic native sampler must replay the same per-anchor random noise.
policy.zero_grad()
maximum_replay_error = 0.
for j in range(T):
    output = policy(observations[j])
    maximum_replay_error = max(maximum_replay_error,
                               float((output.detach()-stored_outputs[j]).abs().max()))
    u, k = responses(output)
    (u @ u_adjoint[j] + (k*k_adjoint[j]).sum()).backward()
replayed_gradients = [p.grad.clone() for p in policy.parameters()]
maximum_gradient_error = max(float((a-b).abs().max())
                             for a, b in zip(full_gradients, replayed_gradients))
assert maximum_replay_error == 0.
assert maximum_gradient_error < 1e-12

result = {
    "status": "passed",
    "scope": "CPU neural-policy chain-rule check; no VLA, simulator or GPU memory measurement",
    "anchors": T, "state_directions": D, "action_dimensions": A,
    "policy_parameters": sum(p.numel() for p in policy.parameters()),
    "loss": float(loss.detach()),
    "maximum_replay_output_error": maximum_replay_error,
    "maximum_gradient_error": maximum_gradient_error,
    "higher_order_parameter_autodiff_required": False,
    "native_requirements": [
        "Replay identical stochastic action noise and any other stochastic layer state.",
        "Keep parameters, policy buffers, and physical response cache unchanged between both passes.",
        "Use the same observation processing, action denormalization and executed prefix in both passes.",
        "Measure actual native sampler activation memory; this script does not establish that bound."
    ]
}
Path(__file__).with_name('two_pass_result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
