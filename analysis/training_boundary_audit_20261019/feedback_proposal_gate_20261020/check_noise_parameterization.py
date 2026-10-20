"""Analytic Gaussian counterexample: sampler sensitivity is not law sensitivity.

No VLA, training or simulator is used. Each policy call draws fresh isotropic
Gaussian noise. State-dependent rotations preserve its conditional law.
"""
from pathlib import Path
import json
import math

def main():
    gain, sigma, delta = 0.5, 0.1, 0.01
    # Plant: s_next = s + a, s in R^2.
    # Policy: a = -gain*s + sigma*R(omega*s[0])*z, z ~ N(0,I_2).
    # The action and state-transition laws are identical for every omega.
    # Evaluate s=0 and s'=(delta,0), reusing the same z for both samplers.
    distribution_w2_squared = ((1.0-gain)*delta)**2
    rows = []
    for angle in [0.0, math.pi/2, math.pi]:
        omega = angle/delta
        common_noise_squared = distribution_w2_squared + 4*sigma**2*(1-math.cos(angle))
        rows.append({
            'rotation_angle_between_conditions': angle,
            'omega': omega,
            'common_noise_expected_squared_successor_distance': common_noise_squared,
            'conditional_successor_W2_squared': distribution_w2_squared,
            'common_noise_to_W2_squared_ratio': common_noise_squared/distribution_w2_squared,
            'common_noise_RMS_distance_divided_by_initial_distance': math.sqrt(common_noise_squared)/delta,
            'W2_distance_divided_by_initial_distance': 1.0-gain,
            'expected_squared_derivative_along_first_state_coordinate': (1-gain)**2+2*sigma**2*omega**2,
        })
    assert math.isclose(rows[-1]['common_noise_to_W2_squared_ratio'], 3201)
    assert all(math.isclose(row['conditional_successor_W2_squared'],distribution_w2_squared) for row in rows)
    result = {
        'type':'analytic_counterexample_not_VLA_evidence',
        'assumptions':['Fresh independent isotropic Gaussian noise at every policy call.',
                       'Exact arithmetic and exact conditional samplers; no finite diffusion-solver claim.',
                       'Euclidean physical state metric; plant is a two-dimensional integrator.'],
        'policy':'a(s,z)=-0.5*s+0.1*R(omega*s[0])*z',
        'conditional_action_law':'Normal(-0.5*s,0.01*I_2), independent of omega',
        'conditional_successor_law':'Normal(0.5*s,0.01*I_2), independent of omega',
        'rows':rows,
        'proved':'At identical conditional policy and physical transition laws, a same-noise sensitivity score can vary by orders of magnitude solely through noise parameterization.',
        'not_proved':['SFT in the project mainly changes noise parameterization.',
                      'Common random numbers are invalid for estimating a paired treatment effect.',
                      'Action-distribution similarity guarantees task success or identical behavior under persistent latent memory.',
                      'Wasserstein regularization or distributional contraction is itself novel.'],
        'interpretation':'Same-noise sensitivity describes the chosen coupling. It must not automatically be identified with a property of the marginal policy or physical robustness.',
        'next_real_data_requirement':'Use independent repeated samples and within-checkpoint split-sample controls to test whether between-checkpoint action-distribution changes exceed sampling variability. Fixed-subtask tests cover only the action head under that supplied conditioning.',
    }
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'invariant_successor_W2':math.sqrt(distribution_w2_squared),'same_noise_squared_distance_ratio':rows[-1]['common_noise_to_W2_squared_ratio'],'VLA_evidence':False},ensure_ascii=False))

if __name__=='__main__':
    main()
