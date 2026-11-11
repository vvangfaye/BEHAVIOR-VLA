"""Fixed moment-matched Gaussian comparators for the native score screen.

These approximate the output distribution, not the VLA's flow-training loss.
They consume exactly the train/test split and tangents of the original screen.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fit_native_scores import witnesses

ROOT = Path(__file__).resolve().parent / 'native_score_ada01'


def gaussian_scores(x, h, test, *, diagonal, shrinkage):
    """Differentiate the Gaussian fitted moments, holding the ridge target fixed."""
    n, d = x.shape
    mean = x.mean(0)
    centered = x - mean
    dh = h.mean(0)
    hc = h - dh[None]
    covariance = centered.T @ centered / n
    covariance = (1 - shrinkage) * covariance + shrinkage * np.eye(d)
    if diagonal:
        covariance = np.diag(np.diag(covariance))
    inverse = np.linalg.solve(covariance, np.eye(d))
    q = (test - mean) @ inverse
    scores = []
    for j in range(h.shape[1]):
        dc = (centered.T @ hc[:, j] + hc[:, j].T @ centered) / n
        dc *= 1 - shrinkage
        if diagonal:
            dc = np.diag(np.diag(dc))
        score = q @ dh[j] + .5 * np.sum((q @ dc) * q, axis=1)
        score -= .5 * np.trace(inverse @ dc)
        scores.append(score)
    return np.stack(scores, axis=1)


def check_formula():
    """Compare analytic baseline scores with finite-differenced log densities."""
    rng = np.random.default_rng(109)
    d = 4
    z = np.concatenate([np.eye(d), -np.eye(d)]) * np.sqrt(d)
    bmat = rng.normal(size=(d, d)) + 2 * np.eye(d)
    mu = rng.normal(size=d)
    drift = rng.normal(size=d)
    amat = rng.normal(size=(d, d))
    x = z @ bmat.T + mu
    h = (z @ amat.T + drift)[:, None, :]
    test = rng.normal(size=(21, d))
    errors = {}
    for diagonal in [False, True]:
        def density(eta):
            source = x + eta * h[:, 0]
            centered = source - source.mean(0)
            cov = .9 * centered.T @ centered / len(source) + .1 * np.eye(d)
            if diagonal:
                cov = np.diag(np.diag(cov))
            residual = test - source.mean(0)
            return -.5 * np.linalg.slogdet(cov)[1] - .5 * np.sum(
                (residual @ np.linalg.inv(cov)) * residual, axis=1)
        exact = gaussian_scores(x, h, test, diagonal=diagonal, shrinkage=.1)[:, 0]
        numeric = (density(1e-5) - density(-1e-5)) / 2e-5
        error = float(np.max(np.abs(exact - numeric)))
        if error > 1e-6:
            raise AssertionError((diagonal, error))
        errors['diagonal' if diagonal else 'full'] = error
    return errors


def main():
    protocol = json.loads((ROOT / 'gaussian_baseline_protocol.json').read_text())
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_sha != protocol['script_sha256']:
        raise ValueError('Frozen baseline source changed')
    plan = json.loads((ROOT / 'plan.json').read_text())
    if hashlib.sha256((ROOT / 'plan.json').read_bytes()).hexdigest() != protocol['plan_sha256']:
        raise ValueError('Collection plan changed')
    if not (ROOT / 'collection_complete.json').exists():
        raise RuntimeError('Wait for original collection to finish')
    out = ROOT / 'gaussian_baseline_result.json'
    if out.exists():
        raise RuntimeError('Existing baseline result; refusing overwrite')
    torch.set_num_threads(4)
    rows = []
    for index in plan['context_indices']:
        path = ROOT / f'context_{index}.npz'
        receipt = json.loads((ROOT / f'context_{index}.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['file_sha256']:
            raise ValueError('Collection data hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            x, h = data['actions'].astype('float64'), data['tangents'].astype('float64')
        n = plan['train_samples_per_context']
        mean, std = x[:n].mean(0), x[:n].std(0)
        floor = .02 * np.median(std[std > 1e-8])
        std = np.maximum(std, floor)
        x, h = (x - mean) / std, h / std[None, None]
        vr = plan['verification']
        rng = np.random.default_rng(vr['witness_seed'])
        projection = rng.normal(size=(vr['random_projection_count'], x.shape[1]))
        projection /= np.linalg.norm(projection, axis=1, keepdims=True)
        train_f, _ = witnesses(x[:n], h[:n], projection, 0.)
        test_f, deriv = witnesses(x[n:], h[n:], projection, train_f.mean(0))
        ref = deriv.mean(0)
        ref_se = deriv.std(0, ddof=1) / np.sqrt(len(deriv))
        informative = np.abs(ref) > 2 * ref_se
        for name, diagonal in [('diagonal_gaussian', True), ('full_gaussian', False)]:
            scores = gaussian_scores(x[:n], h[:n], x[n:], diagonal=diagonal,
                                     shrinkage=protocol['covariance_shrinkage'])
            residual = test_f[:, :, None] * scores[:, None, :] - deriv
            delta = residual.mean(0)
            se = residual.std(0, ddof=1) / np.sqrt(len(residual))
            rows.append(dict(context=index, method=name,
                relative_informative_moment_error=float(np.linalg.norm(delta[informative]) /
                    max(np.linalg.norm(ref[informative]), 1e-12)),
                three_se_coverage=float((np.abs(delta) <= 3 * se + 1e-12).mean()),
                informative_moments=int(informative.sum())))
    out.write_text(json.dumps(dict(formula_check=check_formula(), rows=rows,
        script_sha256=source_sha, scope='Distribution approximation comparator only; '
        'no original gate change, no policy update or task-outcome evidence.'), indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
