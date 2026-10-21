"""Bounded native check of the proposed conditional-null gradient control.

Uses one existing observation; no fitting, optimizer update, rollout or cloud job.
Keeps the numerical geometry so a later check need not rebuild its Jacobian.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import resource
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/relative_flow_20261021'))
sys.path.insert(0, str(ROOT / 'analysis/action_support_20261021'))
sys.path.insert(0, str(ROOT / 'analysis/native_score_implicit_20261021'))
from adapter import NativeAdapter
from vla_runtime import load_base
from vla_data import VlaPilotData
from implicit_geometry import geometry, local_surrogate
from null_transport import reference_parts, null_parts, null_surrogate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    start = time.monotonic()
    report = dict(complete=False, training=False, simulation=False, cloud=False,
                  device='cpu', precision='float32 model; float64 geometry',
                  euler_steps=10, executed_shape=[16, 23], calibration=False,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__).resolve(), HERE/'null_transport.py',
                                           ROOT/'analysis/native_score_implicit_20261021/implicit_geometry.py']})

    def record(stage, **kw):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-start,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2, **kw)
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(stage=stage, elapsed_seconds=report['elapsed_seconds'], **kw)), flush=True)

    try:
        plan = json.loads((ROOT/'analysis/action_support_20261021/local_vla_pilot_ada01/plan.json').read_text())
        record('loading')
        base = load_base(plan['checkpoint'], plan['tokenizer'], device='cpu', precision='fp32')
        adapter = NativeAdapter(base).eval()
        trainable = [(n, p) for n, p in adapter.named_parameters() if p.requires_grad]
        for p in base.parameters():
            p.requires_grad_(False)
        data = VlaPilotData(plan['data_directory'], base)
        index = int(data.train_ids[0])
        obs, _, _ = data.batch([index])
        prepared = adapter.prepare(obs)
        record('prepared', index=index, episode=int(data.episode[index]), frame=int(data.frame[index]),
               intended_trainable_parameters=sum(p.numel() for _, p in trainable))

        def function(z):
            x = z.reshape(1, 32, 32)
            for k in range(10):
                t = torch.full((1,), 1-k/10, dtype=x.dtype, device=x.device)
                x = x-adapter.velocity(prepared, x, t)/10
            return x[0, :16, :23].reshape(-1)

        z = torch.randn(1024)
        with torch.no_grad():
            reference = adapter.sample(obs, z.reshape(1, 32, 32), 10)[0, :16, :23].reshape(-1)
            err = float((reference-function(z)).abs().max())
        assert err == 0, err
        record('sampler_parity', maximum_sampler_difference=err)

        zgrad = z.clone().requires_grad_(True)
        output = function(zgrad)
        rows = []
        tick = time.monotonic()
        for first in range(0, 368, 8):
            last = min(368, first+8)
            block = torch.autograd.grad(output[first:last], zgrad,
                                        grad_outputs=torch.eye(last-first, dtype=output.dtype),
                                        is_grads_batched=True, retain_graph=True, create_graph=False)[0]
            rows.append(block.detach().double())
            if first == 0 or last % 64 == 0 or last == 368:
                record('jacobian', rows_completed=last, jacobian_seconds=time.monotonic()-tick)
        j = torch.cat(rows)
        del rows, output, zgrad
        gc.collect()
        b, m, s = geometry(j)
        error = float((j@b-torch.eye(368, dtype=j.dtype)).abs().max())
        assert error < 1e-5, error
        probe = (torch.randint(0, 2, (1024,))*2-1).double()
        tick = time.monotonic()
        parts = reference_parts(function, z, j, b, m, probe)
        assert torch.isfinite(parts['div_b']).all()

        # L injects physical tail coordinates; only N L is tangent. Raw tail
        # noise is not assumed independent of the executed prefix.
        front = torch.tensor([t*32+a for t in range(16) for a in range(23)])
        tail = front+16*32
        injection = torch.zeros(1024, 368, dtype=j.dtype)
        injection[tail, torch.arange(368)] = 1
        perm = torch.arange(1024)
        perm[front] = tail
        perm[tail] = front
        rotation = torch.eye(1024)[perm]
        c, d = null_parts(parts, injection)
        null_error = float((j@c).abs().max())
        assert null_error < 1e-5, null_error
        assert torch.isfinite(c).all() and torch.isfinite(d).all()
        record('control_geometry', right_inverse_error=error, null_projection_error=null_error,
               condition_number=float(s[0]/s[-1]), control_field_norm=float(c.norm()),
               control_divergence_norm=float(d.norm()), divergence_seconds=time.monotonic()-tick)
        torch.save(dict(z=z, j=j, b=b, inverse_gram=m, probe=probe,
                        control=c, control_divergence=d, base_divergence=parts['div_b'],
                        index=index, episode=int(data.episode[index]), frame=int(data.frame[index])),
                   args.output/'geometry.pt')

        for _, p in trainable:
            p.requires_grad_(True)
        parameters = [p for _, p in trainable]
        tick = time.monotonic()
        base_grad = torch.autograd.grad(local_surrogate(function, z, b, parts['div_b'], probe),
                                       parameters, allow_unused=True)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in base_grad)
        base_seconds = time.monotonic()-tick
        record('base_gradient', base_gradient_seconds=base_seconds)
        gc.collect()
        tick = time.monotonic()
        control_grad = torch.autograd.grad(null_surrogate(function, z, c, d, probe, rotation),
                                          parameters, allow_unused=True)
        assert all(g is None or bool(torch.isfinite(g).all()) for g in control_grad)
        control_seconds = time.monotonic()-tick
        gram = torch.zeros(2, 2, dtype=torch.float64)
        groups = {}
        nonzero = 0
        for (name, _), g0, g1 in zip(trainable, base_grad, control_grad):
            if g0 is None and g1 is None:
                continue
            aa = float(g0.double().square().sum()) if g0 is not None else 0.
            cc = float(g1.double().square().sum()) if g1 is not None else 0.
            ac = float((g0.double()*g1.double()).sum()) if g0 is not None and g1 is not None else 0.
            gg = torch.tensor([[aa, ac], [ac, cc]], dtype=torch.float64)
            gram += gg
            group = '.'.join(name.split('.')[:4])
            groups[group] = groups.get(group, torch.zeros_like(gram))+gg
            if g1 is not None:
                nonzero += int(g1.count_nonzero())
        assert gram[1, 1] > 0
        record('complete', complete=True, control_gradient_seconds=control_seconds,
               base_gradient_norm=float(gram[0, 0].sqrt()), control_gradient_norm=float(gram[1, 1].sqrt()),
               gradient_cosine=float(gram[0, 1]/(gram[0, 0]*gram[1, 1]).sqrt()),
               gram=gram.tolist(), group_grams={k: v.tolist() for k, v in groups.items()},
               control_nonzero_gradient_elements=nonzero,
               scope='One native conditional-null surrogate derivative. No fitted coefficients, optimizer update, reward-learning or variance-reduction claim.')
    except Exception as exc:
        record('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
