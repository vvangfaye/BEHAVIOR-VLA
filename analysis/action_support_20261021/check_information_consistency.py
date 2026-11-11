"""Exact two-state countercheck; no robot, model, or empirical causal claims.

A future observation reveals U in {-1,+1}; the fixed future target is U.
Before revelation, the predictor is b; afterwards it is c*U.
Compare symmetric matching, conditional-moment matching, and detached-target
matching. The last has a semi-gradient, not the gradient of the symmetric loss.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def risk(b, c):
    return sum(((b-u)**2 + (c*u-u)**2) / 2 for u in (-1., 1.))


def symmetric_loss(b, c, weight):
    return risk(b, c) + weight * (b*b + c*c)


def gradient(b, c, weight, mode):
    # Conditional-moment penalty = (b-E[cU])^2 = b^2.
    # Detached-target matching produces the same expected semi-gradient.
    return (2*(1+weight)*b,
            2*(c-1) + (2*weight*c if mode == 'symmetric' else 0))


def main():
    rows = []
    fd_max_error = 0.
    for weight in (0., .1, 1., 3., 10.):
        for mode in ('symmetric', 'conditional_moment', 'detached_target'):
            b, c = 0., 1/(1+weight) if mode == 'symmetric' else 1.
            g = gradient(b, c, weight, mode)
            assert max(map(abs, g)) < 1e-12
            # Explicit iteration also checks the semi-gradient fixed point.
            bi, ci = .3, -.7
            lr = .4/(1+weight)
            for _ in range(10000):
                db, dc = gradient(bi, ci, weight, mode)
                bi, ci = bi-lr*db, ci-lr*dc
            assert abs(bi-b) < 1e-10 and abs(ci-c) < 1e-10
            rows.append(dict(weight=weight, mode=mode, before=b,
                             revealed_information_gain=c,
                             before_mse=1+b*b, after_mse=(c-1)**2,
                             total_mse=risk(b,c), stationary_gradient=list(g)))
        b, c = .3, -.7
        eps = 1e-6
        fd = ((symmetric_loss(b+eps,c,weight)-symmetric_loss(b-eps,c,weight))/(2*eps),
              (symmetric_loss(b,c+eps,weight)-symmetric_loss(b,c-eps,weight))/(2*eps))
        exact = gradient(b,c,weight,'symmetric')
        fd_max_error = max(fd_max_error, *(abs(x-y) for x,y in zip(fd,exact)))
    assert fd_max_error < 1e-7
    # A detached fully informed target is the original label in this example:
    # its expected supervised gradient adds no new information.
    for b in (-.7,0.,.4):
        label_gradient = sum(2*(b-u)/2 for u in (-1.,1.))
        teacher_gradient = sum(2*(b-1.*u)/2 for u in (-1.,1.))
        assert label_gradient == teacher_gradient
    out = dict(
        observed_at=datetime.now(timezone.utc).isoformat(),
        kind='exact_finite_information_revelation_countercheck',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        setup='U equiprobable +/-1; early prediction b; informed prediction cU; target U',
        rows=rows, finite_difference_max_error=fd_max_error,
        findings=[
            'Symmetric consistency suppresses the correct response to newly revealed information.',
            'Conditional moment matching preserves that response in this example.',
            'Ordinary detached-target matching has exactly the same expected semi-gradient here.',
            'A perfect privileged teacher supplies the existing label here; no extra information is obtained.'
        ],
        limits=[
            'This is a deliberately simple finite distribution, not measured VLA behavior.',
            'There is no physical rollout or new robot result.',
            'General tower identities require nested histories and the same future target/random variable.',
            'Single-frame observations are generally not nested information sets.',
            'Expert on-trajectory identities do not certify the induced deployed state distribution.',
            'This does not reproduce or refute the empirical results of PFD or ChunkFlow.',
            'A learned approximate teacher may offer finite-sample inductive bias; this check does not exclude it.'
        ],
        decision='Do not promote conditional information consistency alone as a new training method.',
        goal_complete=False,
    )
    dest=Path(__file__).with_name('information_consistency_check.json')
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(output=str(dest),checks_passed=True,
                         finite_difference_max_error=fd_max_error,
                         symmetric_after_mse_at_weight1=rows[6]['after_mse'],
                         detached_after_mse_at_weight1=rows[8]['after_mse'])))


if __name__ == '__main__':
    main()
