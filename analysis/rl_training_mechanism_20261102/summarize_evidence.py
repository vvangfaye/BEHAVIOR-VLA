"""Local, reproducible summaries of the read-only rollout/buffer audit."""
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMPARISON = HERE.parent / 'rl_scheme_comparison_20261101'


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    audit = json.loads((HERE / 'result.json').read_text())
    comparisons = json.loads((COMPARISON / 'comparisons.json').read_text())
    evaluation = {}
    for variant, result in comparisons.items():
        pairs = result['pairs']
        gains = sum(p['success'] and not p['baseline_success'] for p in pairs)
        losses = sum(p['baseline_success'] and not p['success'] for p in pairs)
        discordant = gains + losses
        pvalue = min(1., 2 * sum(math.comb(discordant, i)
                               for i in range(min(gains, losses) + 1)) / 2 ** discordant)
        by_instance = collections.defaultdict(list)
        by_seed = collections.defaultdict(list)
        for p in pairs:
            by_instance[p['instance']].append(int(p['success']) - int(p['baseline_success']))
            by_seed[p['seed']].append(p)
        # A diagnostic interval over the fixed evaluated instances, not training randomness.
        # Gamma=1 remains provisional until all 40 paired cases are available.
        deltas = np.array([np.mean(v) for v in by_instance.values()])
        rng = np.random.default_rng(602411)
        bootstrap = rng.choice(deltas, size=(20000, len(deltas))).mean(axis=1)
        evaluation[variant] = {
            'completed': len(pairs), 'complete': result['complete'],
            'successes': sum(p['success'] for p in pairs),
            'baseline_successes': sum(p['baseline_success'] for p in pairs),
            'mean_score': result['mean_score'], 'gains': gains, 'losses': losses,
            'mcnemar_exact_two_sided_p_unclustered': pvalue,
            'instance_cluster_bootstrap_delta_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'by_seed': {str(s): {'n': len(rows), 'successes': sum(p['success'] for p in rows),
                                'baseline_successes': sum(p['baseline_success'] for p in rows),
                                'mean_score': float(np.mean([p['score'] for p in rows]))}
                        for s, rows in by_seed.items()},
        }
    save('evaluation_statistics.json', evaluation)

    eval_goals = {}
    for variant in ['original', 'control', 'balanced_ppo', 'gamma1_ppo', 'group_outcome']:
        pairs = comparisons['balanced_ppo' if variant == 'original' else variant]['pairs']
        counts = collections.Counter()
        for p in pairs:
            summary = json.loads((COMPARISON / 'evidence' / variant /
                                  f"i{p['instance']}_s{p['seed']}" / 'goal_summary.json').read_text())
            g = summary['final']['goals']
            ever = summary['ever_achieved']
            bottles = int(g['bottle_1_on_table']) + int(g['bottle_2_on_table'])
            counts[f"final_bottles{bottles}_closed{int(g['fridge_closed'])}"] += 1
            counts['never_achieved_any_bottle'] += not (ever['bottle_1_on_table'] or ever['bottle_2_on_table'])
            counts['ever_achieved_each_bottle_not_necessarily_simultaneous'] += (
                ever['bottle_1_on_table'] and ever['bottle_2_on_table'])
        eval_goals[variant] = dict(counts)
    save('evaluation_goals.json', eval_goals)

    updates = {}
    for variant in comparisons:
        if variant == 'control':
            folder = HERE.parent / 'rlinf_adaptation_20261020/multitask_validation_20261029/water/retrieved/train1_r0_a4'
        else:
            folder = COMPARISON / 'jobs' / ('t-' + variant.replace('_', '-') + '-a4') / 'retrieved'
        result = json.loads((folder / 'training_updates/train1/result.json').read_text())
        resume = json.loads((folder / 'checkpoints/round_01/resume.json').read_text())
        norms = [r['gradient_norm_before_clip'] for r in result['update_steps']]
        updates[variant] = {
            'source': str(folder), 'optimizer_steps': result['optimizer_steps'],
            'early_stopped': result['early_stopped'], 'episodes': result['episodes'],
            'chunks': result['chunks'], 'final_batch_kl': result['final_batch_kl'],
            'gradient_norm_min_median_max': [min(norms), float(np.median(norms)), max(norms)],
            'all_updates_accepted': all(r['accepted'] for r in result['update_steps']),
            'state': resume['state'],
        }
    save('update_statistics.json', updates)

    trajectory_rows = []
    for e in audit['episodes']:
        row = {k: e[k] for k in ['episode', 'instance', 'seed', 'success', 'score', 'chunks', 'task_reward_sum']}
        for variant in ['control', 'balanced_ppo', 'gamma1_ppo']:
            for key in ['start_return', 'start_value', 'mean_adv', 'positive_adv_fraction', 'weight']:
                row[variant + '_' + key] = e['variants'][variant][key]
        trajectory_rows.append(row)
    with (HERE / 'trajectory_summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    derived = {}
    for variant in ['control', 'balanced_ppo', 'gamma1_ppo']:
        success = [e for e in audit['episodes'] if e['success']]
        failure = [e for e in audit['episodes'] if not e['success']]
        pairs = audit['initial_return_pair_ranking'][variant]['pairs']
        derived[variant] = {
            'success_negative_episode_mean_advantages': sum(e['variants'][variant]['mean_adv'] < 0 for e in success),
            'failure_positive_episode_mean_advantages': sum(e['variants'][variant]['mean_adv'] > 0 for e in failure),
            'mixed_pairs_success_higher_mean_advantages': sum(
                p['success_weighted_mean_adv'] > p['failure_mean_adv'] for p in pairs),
            'mixed_pairs_success_higher_start_returns': sum(p['correct_return_ranking'] for p in pairs),
        }
    save('derived_mechanisms.json', derived)
    print(json.dumps({'evaluation': evaluation, 'derived': derived}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
