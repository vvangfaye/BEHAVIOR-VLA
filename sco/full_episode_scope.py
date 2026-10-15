"""Exact scope validation for the independently authorized final-test round."""
import hashlib
import json
from pathlib import Path


def pilot_gate_waived(root, plan=None):
    """Explicit user waiver for this frozen new round; never a passed audit."""
    root = Path(root).resolve()
    campaign = root.parent
    if plan is None and (root / 'plan.json').exists():
        plan = json.loads((root / 'plan.json').read_text())
    if plan and plan.get('paired_evaluation_only'):
        from paired_checkpoint_evaluation import evaluation_gate_allowed
        return evaluation_gate_allowed(root, plan)
    if plan and plan.get('original_repeat_collection'):
        from original_full_episode_repeat import gate_allowed
        return gate_allowed(root, plan)
    path = campaign / 'pilot_acceptance_waiver.json'
    if not path.exists():
        return False
    waiver = json.loads(path.read_text())
    plan = plan or json.loads((root / 'plan.json').read_text())
    validate_final1000(plan)
    raw = (campaign / 'authorization.json').read_bytes()
    auth = json.loads(raw)
    assert auth['kind'] == 'native_full_episode_independent_final1000'
    assert auth['roots'] == [str(root)] and auth['fingerprints'] == [plan['fingerprint']]
    assert waiver['kind'] == 'user_waived_pilot_production_gate'
    assert waiver['authorization_sha256'] == hashlib.sha256(raw).hexdigest()
    assert waiver['root'] == str(root) and waiver['plan_fingerprint'] == plan['fingerprint']
    assert waiver['deadline_epoch'] == auth['deadline_epoch']
    assert waiver['total_episodes'] == 1000 and waiver['max_pending_tasks'] == 5
    assert waiver['allow_production_without_pilot_acceptance'] is True
    assert waiver['artifact_acceptance_complete'] is False
    return True

def validate_final1000(plan):
    assert plan['expected_episodes'] == 1000
    assert plan['collection_scope'] == 'final1000_instances_301_310'
    if plan.get('original_repeat_collection'):
        from original_full_episode_repeat import validate_original_repeat
        validate_original_repeat(plan)
    elif plan.get('trained_model_export'):
        from trained_full_episode_collection import validate_model_plan
        validate_model_plan(plan)
    else:
        assert plan['sampling_round'] == 3
        assert plan['checkpoint_identity']['checkpoint_step'] == 1688977
    assert plan['official_benchmark'] is False
    assert plan['automatic_failure_recovery'] is False
    tasks = set(plan['task_prompts'])
    assert len(tasks) == 100 and 'turning_on_radio' in tasks
    assert len(plan['shards']) == 201
    assert plan['shards'][0] == dict(index=0, task='turning_on_radio', instance_ids=[301], pilot=True)
    pairs = []
    for index, shard in enumerate(plan['shards']):
        assert shard['index'] == index and shard['task'] in tasks
        assert 1 <= len(shard['instance_ids']) <= 5
        assert shard['pilot'] is (index == 0)
        assert all(type(i) is int and 301 <= i <= 310 for i in shard['instance_ids'])
        pairs.extend((shard['task'], i) for i in shard['instance_ids'])
    expected = {(task, i) for task in tasks for i in range(301, 311)}
    assert len(pairs) == len(set(pairs)) == 1000 and set(pairs) == expected
    assert set(plan['policy_seeds']) == {f'{task}:{i}' for task, i in expected}
    seeds = list(plan['policy_seeds'].values())
    assert all(type(s) is int and 0 <= s < 2**32 for s in seeds)
    assert len(set(seeds)) == 1000
    assert plan['maximum_archive_parts'] == 8192 and plan['archive_part_bytes'] == 8 * 1024**2
    return plan
