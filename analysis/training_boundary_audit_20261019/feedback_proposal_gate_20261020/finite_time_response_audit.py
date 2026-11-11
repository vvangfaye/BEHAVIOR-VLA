"""Stress-test a proposed transient-amplification training objective.

Analytic linear-system examples and existing simulator-probe evidence only.
"""
import json
from pathlib import Path

import numpy as np


def peak_raw_gain(a, horizon=16):
    values = [float(np.linalg.norm(np.linalg.matrix_power(a, k), 2)) for k in range(1, horizon+1)]
    return {'gain': max(values), 'step': values.index(max(values))+1}


def main():
    a = np.array([[.7, 3.], [0., .7]])
    rows = []
    original_gain = peak_raw_gain(a)
    for scale in [1., .001, 100.]:
        s = np.diag([scale, 1.])
        inverse = np.linalg.inv(s)
        changed = s @ a @ inverse
        recovered = [inverse @ np.linalg.matrix_power(changed, k) @ s for k in range(1, 17)]
        assert all(np.allclose(p, np.linalg.matrix_power(a, k)) for k, p in enumerate(recovered, 1))
        physical_gain = max(float(np.linalg.norm(p, 2)) for p in recovered)
        assert abs(physical_gain-original_gain['gain']) < 1e-10
        rows.append({'coordinate_scale': scale, 'eigenvalues': np.linalg.eigvals(changed).tolist(),
                     'naive_coordinate_norm_peak': peak_raw_gain(changed),
                     'gain_in_original_physical_metric': physical_gain})
    assert rows[0]['naive_coordinate_norm_peak']['gain'] > 1
    assert rows[1]['naive_coordinate_norm_peak']['gain'] < 1
    assert rows[2]['naive_coordinate_norm_peak']['gain'] > 100
    result = {
        'status': 'complete', 'kind': 'analytic_objective_audit_not_a_VLA_experiment',
        'coordinate_change_examples': rows,
        'established': [
            'Asymptotic stability alone does not rule out large finite-time gain.',
            'A raw state/latent Euclidean gain can be made small or large by an invertible coordinate change without changing physical trajectories.',
            'The gain is unchanged when the metric or input/output maps are transformed consistently.'
        ],
        'prior_art': [
            {'title': 'Stable-BC: Controlling Covariate Shift with Stable Behavior Cloning',
             'url': 'https://arxiv.org/html/2408.06246',
             'scope': 'Model-based loss uses the real parts of local error-dynamics eigenvalues; do not attribute a general finite-time gain guarantee.'},
            {'title': 'Input-Side Variance Suppression under Non-Normal Transient Amplification in Continuous-Control Reinforcement Learning',
             'url': 'https://arxiv.org/abs/2604.17744',
             'scope': 'Studies non-normal amplification in continuous-control RL; transient amplification is not an unreported robot-learning mechanism.'},
            {'title': 'Controlling Transient Amplification Improves Long-horizon Rollouts',
             'url': 'https://arxiv.org/abs/2605.08856',
             'scope': 'Regularizes normality defects and cross-step Jacobian commutators in autoregressive neural simulators; not a direct study of VLA post-training.'},
            {'title': 'Denoising-based Contractive Imitation Learning',
             'url': 'https://arxiv.org/abs/2503.15918',
             'scope': 'Uses a dynamics model and joint state-action denoiser to encourage contraction without new expert demonstrations.'}
        ],
        'simulator_measurement_caution': {
            'existing_receipt': '../snapshot_probe_20261020/restore_repeatability_analysis.json',
            'fact': 'The original-to-restored comparison differs after stepping despite nearly identical recorded initial states; all three restored-to-restored recorded traces match.',
            'consequence': 'Do not divide original-to-restored trajectory differences by tiny recorded pose differences and call it a policy amplification estimate. Hidden solver state can differ, and the probe used no VLA.'
        },
        'requirements_for_a_meaningful_candidate': [
            'Use task-relevant physical input/output variables and a fixed justified metric, not an arbitrary learned latent norm.',
            'Apply controlled perturbations after the same restoration procedure in both branches.',
            'Distinguish perturbation magnitude, closed-loop transport and actual task-failure margins.',
            'Establish that the measured effect changes with finetuning and predicts independent deployment outcomes.',
            'Demonstrate a contribution beyond prior contraction, transient-gain regularization and task-sensitive policy optimization.'
        ],
        'decision': 'Not yet a novel, validated training proposal. Retain finite-time response as a possible diagnostic, with metric and restoration controls.',
        'goal_complete': False,
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'coordinate_scale_vs_raw_peak': [[r['coordinate_scale'], r['naive_coordinate_norm_peak']['gain']] for r in rows],
                      'physical_peak_unchanged': original_gain['gain']}))


if __name__ == '__main__':
    main()
