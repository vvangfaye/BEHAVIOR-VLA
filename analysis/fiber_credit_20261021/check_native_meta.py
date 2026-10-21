"""One native calibration update with learned noise-contrast parameters.

Reuses the prior exact-reference numerical geometry. No actor optimizer,
rollout, reward label, cloud resource, or data collection is involved.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
import torch
from torch import nn
from torch.func import functional_call, jvp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'analysis/relative_flow_20261021'),
               str(ROOT/'analysis/action_support_20261021'),
               str(ROOT/'analysis/native_score_implicit_20261021')]
from adapter import NativeAdapter
from vla_data import VlaPilotData
from vla_runtime import load_base
from implicit_geometry import local_surrogate
from learned_contrast import contrast, directional_variance_objective


class NativeFunction(nn.Module):
    def __init__(self, adapter, prepared):
        super().__init__()
        self.adapter = adapter
        self.prepared = prepared

    def forward(self, noise):
        x = noise.reshape(1, 32, 32)
        for k in range(10):
            t = torch.full((1,), 1-k/10, device=x.device, dtype=x.dtype)
            x = x-self.adapter.velocity(self.prepared, x, t)/10
        return x[0, :16, :23].reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    start = time.monotonic()
    report = dict(complete=False, actor_training=False, calibration_training=True,
                  simulation=False, cloud=False, observations=1, device='cpu',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__).resolve(), HERE/'learned_contrast.py')})

    def record(stage, **kwargs):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-start,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2, **kwargs)
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(stage=stage, **kwargs)), flush=True)

    try:
        record('loading')
        cached_path = HERE/'native_cpu_01/geometry.pt'
        cached = torch.load(cached_path, map_location='cpu', weights_only=True)
        prior = json.loads((HERE/'native_cpu_01/result.json').read_text())
        assert prior['complete'] and prior['index'] == cached['index']
        plan = json.loads((ROOT/'analysis/action_support_20261021/local_vla_pilot_ada01/plan.json').read_text())
        base = load_base(plan['checkpoint'], plan['tokenizer'], device='cpu', precision='fp32')
        adapter = NativeAdapter(base).eval()
        data = VlaPilotData(plan['data_directory'], base)
        index = cached['index']
        obs, _, _ = data.batch([index])
        sampler = NativeFunction(adapter, adapter.prepare(obs))
        trainable = [(n, p) for n, p in sampler.named_parameters() if p.requires_grad]
        names, parameters = zip(*trainable)
        assert sum(p.numel() for p in parameters) == prior['intended_trainable_parameters']
        z, e, b, div_b = [cached[k] for k in ('z', 'probe', 'b', 'base_divergence')]
        c, div_c = cached['control'], cached['control_divergence']
        front = torch.tensor([t*32+a for t in range(16) for a in range(23)])
        tail = front+512
        # Shared amplitude and angle: a minimal learning check, not a claim
        # that this two-parameter family is adequate for native variance.
        phi = nn.Parameter(torch.tensor([1., math.pi/4], dtype=torch.float64))

        def corrected_field(function):
            feature = lambda noise:contrast(function, noise, front, tail,
                                            phi[1].to(noise.dtype).expand(len(front)))
            value, tangent = jvp(feature, (z,), (e.to(z.dtype),))
            return phi[0]*(((c.T@z.double()-div_c)*value.double()).sum()
                           -((c.T@e)*tangent.double()).sum())

        tick = time.monotonic()
        total = local_surrogate(sampler, z, b, div_b, e)+corrected_field(sampler)
        g = torch.autograd.grad(total, parameters, create_graph=False, allow_unused=True)
        g = tuple(torch.zeros_like(p) if value is None else value.detach()
                  for p, value in zip(parameters, g))
        assert all(torch.isfinite(v).all() for v in g)
        energy = sum(float(v.double().square().sum()) for v in g)
        record('actor_gradient', actor_gradient_seconds=time.monotonic()-tick,
               gradient_squared_norm=energy, intended_parameters=sum(p.numel() for p in parameters))
        del total
        gc.collect()

        def control_only(values):
            function = lambda noise:functional_call(sampler, dict(zip(names, values)), (noise,))
            return corrected_field(function)

        tick = time.monotonic()
        # A fixed scaling keeps Adam's numbers manageable. It is independent
        # of phi during this derivative and does not alter its direction.
        scale = max(energy, 1.)
        meta = directional_variance_objective(control_only, parameters, g)/scale
        meta_gradient = torch.autograd.grad(meta, phi)[0]
        assert torch.isfinite(meta_gradient).all() and meta_gradient.abs().max() > 0
        before = phi.detach().clone()
        optimizer = torch.optim.SGD([phi], lr=.01)
        phi.grad = meta_gradient.detach()
        optimizer.step()
        assert torch.isfinite(phi).all() and not torch.equal(before, phi.detach())
        record('complete', complete=True, meta_gradient_seconds=time.monotonic()-tick,
               meta_gradient=meta_gradient.tolist(), calibration_parameters_before=before.tolist(),
               calibration_parameters_after=phi.detach().tolist(), completed_calibration_updates=1,
               geometry_source_sha256=hashlib.sha256(cached_path.read_bytes()).hexdigest(),
               scope='One native contrast-calibration gradient/update using one cached observation/noise/probe. Not a variance estimate, held-out gain or actor-learning result.')
    except Exception as exc:
        record('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
