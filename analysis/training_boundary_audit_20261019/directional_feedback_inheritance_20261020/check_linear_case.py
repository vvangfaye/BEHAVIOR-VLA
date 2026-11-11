"""Stress-test selective derivative inheritance against strong simple controls.

Constructive linear example only. It neither measures VLA derivatives nor
establishes a new algorithm. The soft full-Jacobian control is essential.
"""
import json
from pathlib import Path


def gains(theta):
    a,b=theta
    return a+2*b,2*a+b


def row(name,theta):
    gx,gy=gains(theta)
    # x_{t+1}=1.5*x_t+u_x; y_{t+1}=1.5*y_t+u_y.
    # All demonstrations have y=0, and target u_x=-2*x.
    return {'name':name,'parameters':theta,'x_feedback_gain':gx,'y_feedback_gain':gy,
            'unit_variance_demonstration_mse':(gx-2)**2,
            'x_closed_loop_multiplier':1.5-gx,'y_closed_loop_multiplier':1.5-gy,
            'stable':abs(1.5-gx)<1 and abs(1.5-gy)<1}


def from_gains(x,y):
    return [(2*y-x)/3,(2*x-y)/3]


def main():
    parent=[1.,0.]
    # Euclidean-gradient descent on (theta1+2*theta2-2)^2 from parent.
    sft=[1.2,.4]
    results=[row('parent',parent),row('ordinary_sft_minimum_change',sft),
             row('selective_y_derivative_inheritance',from_gains(2.,2.))]
    for alpha in (0.,.25,.5,.625,.75,1.):
        theta=[a+alpha*(b-a) for a,b in zip(parent,sft)]
        results.append(row('parameter_interpolation_'+str(alpha),theta))
    # Strong baseline: standard all-direction soft derivative matching.
    # Objective (gx-2)^2 + lambda*((gx-1)^2 + (gy-2)^2).
    for coefficient in (.001,.01,.1,1.,10.):
        gx=(2+coefficient)/(1+coefficient)
        results.append(row('full_derivative_matching_'+str(coefficient),from_gains(gx,2.)))
    assert results[0]['stable'] and not results[1]['stable']
    assert results[2]['stable'] and results[2]['unit_variance_demonstration_mse']<1e-28
    simple=next(r for r in results if r['name']=='full_derivative_matching_0.001')
    assert simple['stable'] and simple['unit_variance_demonstration_mse']<1e-6
    report={'status':'complete','kind':'constructive_linear_control_and_strong_baseline_audit',
            'state_support':'Demonstrations have covariance diag(1,0); projector is exact in this toy.',
            'results':results,
            'findings':[
                'Data fitting can change an unobserved feedback direction through shared parameters.',
                'Selective derivative inheritance can preserve stability while changing observed behavior.',
                'Global parameter interpolation trades off adaptation and stability in this example.',
                'However, standard soft full-Jacobian matching attains arbitrarily small fit error and the same stability. The toy does not establish a decisive advantage for a novel projected loss.'
            ],
            'limitations':[
                'No real VLA, demonstrations or simulator were used.',
                'The prior y-controller was chosen stable; inheriting an unstable prior would retain its error.',
                'Real local covariance does not automatically identify causal feedback or sufficient data support.',
                'No novelty or publication readiness is inferred.'
            ]}
    out=Path(__file__).with_suffix('.json')
    with out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'complete','selective_stable':results[2]['stable'],
                      'full_jacobian_baseline_stable':simple['stable'],
                      'full_jacobian_baseline_mse':simple['unit_variance_demonstration_mse']}))


if __name__=='__main__':main()
