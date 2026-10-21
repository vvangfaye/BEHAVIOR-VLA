"""Structural and real-dimension backward checks, not policy validation."""
import json
from pathlib import Path
import torch
from head import CanonicalExecutionHead, two_sample_energy

torch.set_num_threads(4)
torch.manual_seed(20261021)

small = CanonicalExecutionHead(5, 3, 8).double()
context = torch.randn(1, 5, dtype=torch.float64)
noise = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
direct = small(context, noise)
gradient = torch.autograd.grad(small.potential(context, noise).sum(), noise, create_graph=True)[0]
error = float((direct-gradient).abs().max())
jacobian = torch.autograd.functional.jacobian(lambda z: small(context, z[None])[0], noise[0])
symmetry_error = float((jacobian-jacobian.T).abs().max())
minimum_eigenvalue = float(torch.linalg.eigvalsh(jacobian).min())
assert error < 1e-12 and symmetry_error < 1e-12 and minimum_eigenvalue > -1e-12

# The entire 16 x 23 prefix, with enough prototypes to permit full rank.
full = CanonicalExecutionHead(130, 16*23, 512, temperature=3.)
contexts = torch.randn(4, 130)
first = full(contexts, torch.randn(4, 16*23))
second = full(contexts, torch.randn(4, 16*23))
target = torch.randn_like(first).clamp(-1, 1)
loss = two_sample_energy(first, second, target)
loss.backward()
gradient_norm = sum(float(p.grad.square().sum()) for p in full.parameters() if p.grad is not None) ** .5
assert torch.isfinite(first).all() and (first.abs() <= 1+1e-6).all()
assert gradient_norm > 0 and all(torch.isfinite(p.grad).all() for p in full.parameters() if p.grad is not None)

result = {
    'status': 'structural_check_passed',
    'potential_gradient_output_max_error': error,
    'noise_jacobian_symmetry_max_error': symmetry_error,
    'noise_jacobian_minimum_eigenvalue': minimum_eigenvalue,
    'executed_dimension': 16*23,
    'prototypes': 512,
    'parameters': sum(p.numel() for p in full.parameters()),
    'parameter_gradient_norm': gradient_norm,
    'scope': 'CPU structural check with random contexts; no learned VLA, action-law fidelity, or closed-loop performance result.'
}
Path(__file__).with_name('head_check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
