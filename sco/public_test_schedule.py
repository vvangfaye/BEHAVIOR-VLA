"""Choose unsubmitted frozen shards in task rounds without using model scores."""
import argparse
import json
from pathlib import Path


def pending_order(plan, submitted_indices):
    assert plan['checkpoint_step'] == 1688977 and plan['total_rollouts'] == 2000
    assert plan['production_rollouts'] == 1999 and plan['reuse_pilot_regardless_of_model_score'] is True
    shards = plan['shards']
    assert [row['index'] for row in shards] == list(range(len(shards)))
    tasks = list(dict.fromkeys(row['task'] for row in shards))
    assert len(tasks) == 100
    keys = [(row['task'], i) for row in shards for i in row['instance_ids']]
    assert len(keys) == len(set(keys)) == 1999
    pilot = plan['pilot_official_result']
    assert pilot == dict(task='turning_on_radio', instance_id=301, rollout_id=0)
    assert (pilot['task'], pilot['instance_id']) not in keys
    assert set(keys) | {(pilot['task'], pilot['instance_id'])} == {
        (task, i) for task in tasks for i in range(301, 321)}
    assert submitted_indices <= set(range(len(shards)))
    # All frozen cases remain scheduled. Only their admission order changes.
    task_rank = {task: n for n, task in enumerate(tasks)}
    rounds = {}
    ranked = []
    for shard in shards:
        assert shard['expected_rollouts'] == len(shard['instance_ids'])
        task = shard['task']
        round_index = rounds.get(task, 0)
        rounds[task] = round_index + 1
        ranked.append((round_index, task_rank[task], shard['index']))
    return [index for _, _, index in sorted(ranked) if index not in submitted_indices]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.evidence / 'production_plan.json').read_text())
    submitted = set()
    for path in (args.evidence / 'production_submissions').glob('a*.json'):
        row = json.loads(path.read_text())
        index = row['index']
        assert path.stem == 'a' + str(index).zfill(3) and index not in submitted
        submitted.add(index)  # Ambiguous, failed, and running receipts also prevent duplicates.
    order = pending_order(plan, submitted)
    print(json.dumps(dict(read_only=True, scheduling_only=True, model_scores_not_used=True,
        pending_shards=len(order), next_index=order[0] if order else None,
        first_indices=order[:20], admission_requires_fresh_capacity=True)))
