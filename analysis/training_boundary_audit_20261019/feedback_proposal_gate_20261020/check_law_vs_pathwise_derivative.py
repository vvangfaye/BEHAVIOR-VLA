"""Analytic distinction: derivative of a law vs. law of a derivative.

This does not dispute the validity of a specified random-function model.
It checks what that model identifies if used to regularize a Markov policy
which draws fresh independent action noise at each query.
"""
import json
import math
from pathlib import Path


def mmd_zero_mean_diagonal_gaussians(first, second, bandwidth_squared=1.0):
    def kernel_expectation(variances):
        return math.prod(1 + variance/bandwidth_squared for variance in variances)**-0.5
    return (kernel_expectation([2*x for x in first])
            + kernel_expectation([2*x for x in second])
            - 2*kernel_expectation([x+y for x,y in zip(first,second)]))


def main():
    # Z_1,Z_2 iid N(0,1).
    # g_0(x,Z) = Z_1; g_1(x,Z) = cos(x) Z_1 + sin(x) Z_2.
    # Both action laws are N(0,1) for every x. With fresh noise per query,
    # they therefore induce identical Markov action kernels in any plant.
    output_mmd = mmd_zero_mean_diagonal_gaussians([1.0],[1.0])
    # Joint laws of (output,input derivative): diag(1,0) vs diag(1,1).
    joint_mmd = mmd_zero_mean_diagonal_gaussians([1.0,0.0],[1.0,1.0])
    for x in (-2.0,0.0,0.7,3.0):
        c,s=math.cos(x),math.sin(x)
        assert math.isclose(c*c+s*s,1.0,abs_tol=1e-15)
        assert math.isclose(c*(-s)+s*c,0.0,abs_tol=1e-15)
    assert abs(output_mmd)<1e-15 and joint_mmd>0.09
    result={
        'kind':'analytic_identifiability_check_not_a_robot_experiment',
        'policy_0':'g_0(x,z)=z_1',
        'policy_1':'g_1(x,z)=cos(x)*z_1+sin(x)*z_2',
        'conditional_output_law_both':'N(0,1) for every x',
        'joint_output_derivative_covariances':[[1,0],[1,1]],
        'gaussian_rbf_mmd_squared_of_output_laws':output_mmd,
        'gaussian_rbf_mmd_squared_of_joint_output_derivative_laws':joint_mmd,
        'conditional_law_finite_difference':'Zero for both, for every nonzero finite input offset.',
        'finding':'Matching the full distribution of sample-wise input derivatives can still penalize equivalent fresh-noise Markov policies. Matching distributions does not by itself remove a noise-parameterization dependency if the objects being matched contain sample-wise derivatives.',
        'important_scope':['Fresh independent policy noise per query is essential; fixed latent plans across time define a different controller with potentially different behavior.',
                           'A specified random-function derivative is mathematically valid. This result concerns identifying an intrinsic property of the conditional action kernel from it.',
                           'No claim of an error in Distributional Sobolev RL: its specified reparameterized world/value functions are different modeling objects.'],
        'candidate_target':'For bounded physical outcome features phi, estimate D_mu(s,delta)=E[phi(Y)|do(s+delta)]-E[phi(Y)|do(s)] using independently sampled action noise. This target is invariant under output-law-preserving noise reparameterizations.',
        'not_yet_established':['An effective and novel VLA training loss based on this target.',
                               'A task-relevant choice of outcome features and perturbations.',
                               'Benefit over statewise policy-distribution distillation or existing stability methods.',
                               'A practical simulator-to-policy gradient route or held-out closed-loop gain.'],
        'primary_comparison':{'title':'Distributional value gradients for stochastic environments',
                              'url':'https://arxiv.org/html/2601.20071',
                              'sections_read':['Abstract','Appendix A definitions','Appendix B.1 and B.2']},
        'new_gpu_jobs':0,'new_neural_inference_calls':0,
    }
    with Path(__file__).with_suffix('.json').open('x')as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k]for k in ('gaussian_rbf_mmd_squared_of_output_laws','gaussian_rbf_mmd_squared_of_joint_output_derivative_laws','finding')},indent=2))


if __name__=='__main__':
    main()
