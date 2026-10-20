#!/usr/bin/env python3
"""Strictly pair native-loss screens; score physical coordinates separately.

The energy estimate uses off-diagonal pairs (a U statistic). Its Monte Carlo
unbiasedness is conditional on the single recorded target, not a statement
about task generalization, closed-loop reward, or the joint 23-D distribution.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def at(value, path):
    for name in path.split('.'):
        value = value[name]
    return value


def numeric(record, shape):
    require(record['shape'] == shape, 'Unexpected numeric shape')
    def check(value):
        if isinstance(value, list):
            for child in value:
                check(child)
        else:
            require(isinstance(value, (int, float)) and math.isfinite(value), 'Nonfinite numeric')
    check(record['values'])
    return record['values']


def distance(left, right):
    require(len(left) == len(right) and len(left) > 0, 'Distance layout mismatch')
    return math.sqrt(sum((a-b)**2 for a, b in zip(left, right)) / len(left))


def energy(draws, target):
    count = len(draws)
    require(count >= 2, 'Need independent endpoint draws')
    first = statistics.mean(distance(draw, target) for draw in draws)
    # Each unordered pair occurs twice in the ordered sum, canceling 1/2.
    spread = sum(distance(draws[i], draws[j]) for i in range(count)
                 for j in range(i)) / (count * (count - 1))
    return {'score': first-spread, 'mean_distance_to_target': first,
            'half_expected_pair_distance_u': spread}


def paired_summary(left, right):
    require(len(left) == len(right) and len(left) >= 2, 'Invalid paired sample count')
    delta = [b-a for a, b in zip(left, right)]
    return {'baseline_mean': statistics.mean(left), 'candidate_mean': statistics.mean(right),
            'candidate_minus_baseline_mean': statistics.mean(delta),
            'paired_delta_sd': statistics.stdev(delta),
            'paired_delta_monte_carlo_sem': statistics.stdev(delta)/math.sqrt(len(delta)),
            'paired_deltas': delta, 'draw_count': len(delta)}


def compare(base, candidate):
    same_top = ['kind', 'status', 'normalization_asset_id', 'normalization_sha256',
                'use_quantile_norm', 'action_conditioning', 'anchor_count', 'draws_per_anchor',
                'effective_model_config', 'max_token_len', 'action_horizon', 'model_action_dim',
                'parameter_dtype_counts', 'jax_version', 'numpy_version', 'backend', 'protocol',
                'raw_cache.manifest_sha256']
    for path in same_top:
        require(at(base, path) == at(candidate, path), 'Protocol mismatch: '+path)
    require(base['status'] == 'screen_complete', 'Incomplete input screen')
    require(base['draws_per_anchor'] == 8, 'Frozen protocol requires eight draws')
    require(base['anchor_count'] == 3, 'Frozen protocol requires three anchors')
    require(base['action_horizon'] == 32 and base['model_action_dim'] == 32, 'Wrong model layout')
    def indexed(document):
        result = {}
        for row in document['samples']:
            key = (row['episode_index'], row['frame_index'], row['draw_index'])
            require(key not in result, 'Duplicate sample')
            result[key] = row
        require(len(result) == 24, 'Wrong sample count')
        require(document['samples'][0]['same_model_repeat']['all_passed'], 'Repeat control failed')
        return result
    left, right = indexed(base), indexed(candidate)
    require(left.keys() == right.keys(), 'Pairing keys differ')
    same_row = ['sample_id', 'task_index', 'subtask', 'raw_sample_sha256', 'policy_input_sha256',
                'transformed_training_sample_sha256', 'target_leakage_control',
                'training_target', 'rng', 'endpoint.ground_truth', 'endpoint.valid_mask']
    for key in left:
        for path in same_row:
            require(at(left[key], path) == at(right[key], path), f'Pair mismatch {key}: {path}')
    anchors = sorted({key[:2] for key in left})
    results = []
    for anchor in anchors:
        rows = [[side[(*anchor, draw)] for draw in range(8)] for side in (left, right)]
        for row in rows[0][1:]:
            for path in ('raw_sample_sha256', 'training_target', 'endpoint.ground_truth', 'endpoint.valid_mask'):
                require(at(row, path) == at(rows[0][0], path), 'Within-anchor target changed: '+path)
        entry = {'episode_index': anchor[0], 'frame_index': anchor[1],
                 'task_index': rows[0][0]['task_index'], 'native_metrics': {}, 'endpoint': {}}
        for name in ('action_loss', 'subtask_loss'):
            values = [[numeric(row['cfm']['metrics'][name], []) for row in group] for group in rows]
            entry['native_metrics'][name] = paired_summary(*values)
        target = numeric(rows[0][0]['endpoint']['ground_truth'], [32, 23])
        valid = rows[0][0]['endpoint']['valid_mask']
        require(len(valid) == 32 and all(isinstance(v, bool) for v in valid), 'Invalid validity mask')
        outputs = [[numeric(row['endpoint']['actions'], [32, 23]) for row in group] for group in rows]
        for mask_name, selected in (
            ('valid_prefix16', [t for t in range(16) if valid[t]]),
            ('valid_horizon', [t for t in range(32) if valid[t]])):
            require(selected, 'Empty mask')
            coordinates = []
            for j in range(23):
                truth = [target[t][j] for t in selected]
                distributions = [[[draw[t][j] for t in selected] for draw in group] for group in outputs]
                scores = [energy(draws, truth) for draws in distributions]
                mse = [[sum((a-b)**2 for a,b in zip(draw, truth))/len(truth)
                        for draw in draws] for draws in distributions]
                coordinates.append({'coordinate': j, 'baseline_energy': scores[0],
                    'candidate_energy': scores[1],
                    'energy_delta': scores[1]['score']-scores[0]['score'],
                    'sample_mse': paired_summary(*mse)})
            entry['endpoint'][mask_name] = {'valid_steps': len(selected), 'coordinates': coordinates}
        results.append(entry)
    return {'kind': 'strict_paired_native_cfm_endpoint_screen', 'status': 'complete',
            'checks': {'same_protocol_and_inputs': True, 'repeat_controls_passed': True,
                       'anchors': len(results), 'draws_per_anchor': 8},
            'interpretation': 'Negative deltas favor the candidate. Native CFM includes all 32x32 coordinates. '
              'Endpoint energy is a separate marginal score per physical coordinate trajectory. '
              'SEM measures only paired Monte Carlo uncertainty at each fixed observation. '
              'No population CI, joint action-distribution score, closed-loop claim, or causal decomposition.',
            'anchors': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    result['input_sha256'] = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                             for key, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
