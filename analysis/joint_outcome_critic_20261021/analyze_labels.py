"""Label feasibility and task-only held-out prediction; no image or policy claims."""
import collections
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def extract(path):
    d = json.loads(path.read_text())
    r = d['result']
    top = r['final_goal_state']['top_level']
    yes, no = set(top['satisfied']), set(top['unsatisfied'])
    assert not yes & no
    assert yes | no == set(range(len(yes) + len(no)))
    bits = [int(i in yes) for i in range(len(yes) + len(no))]
    assert bool(bits) and bool(all(bits)) == d['success'] == top['success']
    return {'task': d['task'], 'instance': d['instance_id'], 'bits': bits,
            'success': int(d['success']), 'q': r['q_score']['final'],
            'truncated': r.get('truncated'), 'steps': r['steps']}


def predictions(train, m):
    # Fixed Beta(1/2,1/2) priors. AR conditions on the requested all-one prefix.
    n = len(train)
    direct = (sum(all(y) for y in train) + .5) / (n + 1)
    independent = math.prod((sum(y[j] for y in train) + .5) / (n + 1) for j in range(m))
    ar = 1.
    prefix_n = []
    for j in range(m):
        selected = [y for y in train if all(y[:j])]
        prefix_n.append(len(selected))
        ar *= (sum(y[j] for y in selected) + .5) / (len(selected) + 1)
    return {'scalar': direct, 'independent_product': independent, 'autoregressive_joint': ar}, prefix_n


def main():
    receipt = json.loads((HERE / 'read_receipt.json').read_text())
    rows = [extract(HERE / f['file']) for f in receipt['files']]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['task']].append(row)
    tasks = []
    heldout = []
    for task, rs in sorted(groups.items()):
        m = len(rs[0]['bits'])
        assert all(len(r['bits']) == m for r in rs)
        p = [sum(r['bits'][j] for r in rs)/len(rs) for j in range(m)]
        actual = sum(r['success'] for r in rs)/len(rs)
        patterns = collections.Counter(''.join(map(str,r['bits'])) for r in rs)
        for i, row in enumerate(rs):
            estimates, prefix_n = predictions([r['bits'] for k, r in enumerate(rs) if k != i], m)
            heldout.append({'task': task, 'instance': row['instance'], 'success': row['success'],
                            'predictions': estimates, 'ar_prefix_training_counts': prefix_n})
        tasks.append({'task': task, 'n': len(rs), 'clauses': m, 'success_rate': actual,
                      'clause_rates': p, 'product_of_marginals': math.prod(p),
                      'patterns': dict(patterns),
                      'mixed_failure_episodes': sum(0 < sum(r['bits']) < m for r in rs),
                      'q_one_but_failure': sum(r['q'] == 1 and not r['success'] for r in rs)})
    scores = {}
    for method in ('scalar', 'independent_product', 'autoregressive_joint'):
        scores[method] = {
            'brier': sum((r['predictions'][method]-r['success'])**2 for r in heldout)/len(heldout),
            'log_loss': sum(-(r['success']*math.log(max(1e-12,r['predictions'][method])) +
                            (1-r['success'])*math.log(max(1e-12,1-r['predictions'][method])))
                            for r in heldout)/len(heldout)}
    output = {'scope': '90 episode initial-state task-only baselines; no observations, no trained VLA, no policy-improvement evidence',
              'episodes': len(rows), 'tasks': tasks, 'successes': sum(r['success'] for r in rows),
              'mixed_failure_episodes': sum(0 < sum(r['bits']) < len(r['bits']) for r in rows),
              'loo_task_only_scores': scores, 'heldout_predictions': heldout,
              'limits': ['10 examples per task; label correlations are descriptive, not population estimates',
                         'Entire stacking_wood task excluded by predeclared byte limit',
                         'All failures truncated: deadline vs infrastructure cause must be checked before training',
                         'No scalar-critic deficiency inferred from a marginal-product error',
                         'Autoregressive all-one-prefix counts can shrink, so more clause labels do not guarantee more success supervision']}
    (HERE/'label_analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    (HERE/'labels.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('heldout_predictions','limits')}, indent=2))


if __name__ == '__main__':
    main()
