"""Freeze two additional 2000-instance rounds from the accepted first-round bytes."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import tarfile

import full_episode_collection as collection
from full_episode_acceptance import validate_gate
from restore_full_episode_collection import read_plan

REPO = Path(__file__).resolve().parents[1]
EXTRAS = ('scripts/full_episode_repeat_seeding.py',
          'openpi/scripts/b1k/serve_full_episode_repeat.py')


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Frozen source anchor is missing or ambiguous: ' + before[:80])
    return text.replace(before, after, 1)


def revised_sources(parent):
    parent = Path(parent)
    plan = read_plan(parent)
    archive = (parent / 'overlay.tar.gz').read_bytes()
    assert collection.digest(archive) == plan['overlay_sha256']
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as stream:
        for member in stream:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            if member.name == '_FULL_EPISODE_OVERLAY.json':
                continue
            body = stream.extractfile(member).read()
            assert collection.digest(body) == plan['overlay_files_sha256'][member.name]
            files[member.name] = body
    assert set(files) == set(plan['overlay_files_sha256'])
    shell = files['scripts/eval_b1k.sh'].decode()
    shell = replace_once(shell, 'python scripts/b1k/serve_b1k.py',
                         'python scripts/b1k/serve_full_episode_repeat.py')
    files['scripts/eval_b1k.sh'] = shell.encode()
    driver = files['sco/run_full_episode_collection.py'].decode()
    driver = replace_once(driver,
        "        all_public_instances_training_seen=True, release_uploaded_scratch=not payload['pilot'])",
        "        all_public_instances_training_seen=True, release_uploaded_scratch=not payload['pilot'],\n"
        "        sampling_round=payload['sampling_round'], policy_seeds=payload['policy_seeds'],\n"
        "        parent_plan_fingerprint=payload['parent_plan_fingerprint'])")
    driver = replace_once(driver,
        "        assert receipt['result'] == {k: v for k, v in result.items()",
        "        from full_episode_repeat_seeding import seed_spec\n"
        "        assert result['repeat_sampling'] == dict(seed_spec(manifest, payload['task'], instance),\n"
        "                                                      server_seed_acknowledged=True)\n"
        "        assert receipt['result'] == {k: v for k, v in result.items()")
    files['sco/run_full_episode_collection.py'] = driver.encode()
    queue = files['scripts/eval_b1k_queue.py'].decode()
    queue = replace_once(queue, '                trajectory = NativeFullEpisodeTrajectory(trajectory_path, identity)',
        '                from full_episode_repeat_seeding import begin_episode\n'
        '                identity["repeat_sampling"] = begin_episode(evaluator, manifest, args.task_name, instance_id)\n'
        '                trajectory = NativeFullEpisodeTrajectory(trajectory_path, identity)')
    queue = replace_once(queue, '                result = trajectory.finish(result, capture_observation(evaluator.obs))',
        '                result["repeat_sampling"] = evaluator.policy.proof()\n'
        '                result = trajectory.finish(result, capture_observation(evaluator.obs))')
    files['scripts/eval_b1k_queue.py'] = queue.encode()
    raw = files['scripts/full_episode_trajectory.py'].decode()
    raw = replace_once(raw, '    result.update(task_success=success, success_source=',
        '    from full_episode_repeat_seeding import validate_result_seed\n'
        '    validate_result_seed(result, metadata)\n'
        '    result.update(task_success=success, success_source=')
    files['scripts/full_episode_trajectory.py'] = raw.encode()
    for name in EXTRAS:
        files[name] = (REPO / name).read_bytes()
    for name, body in files.items():
        if name.endswith('.py'):
            compile(body, name, 'exec')
    return files


def policy_seeds(parent_plan):
    used, result = set(), {}
    for round_id in (1, 2):
        result[round_id] = {}
        for task in sorted(parent_plan['task_prompts']):
            for instance in range(301, 321):
                raw = f"native-full-repeat-v1:{parent_plan['fingerprint']}:{round_id}:{task}:{instance}"
                seed = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:4], 'big')
                while seed in used:
                    seed = (seed + 1) % 2**32
                used.add(seed)
                result[round_id][f'{task}:{instance}'] = seed
    assert len(used) == 4000
    return result


def freeze(parent, campaign):
    parent, campaign = Path(parent).resolve(), Path(campaign).resolve()
    original = read_plan(parent)
    validate_gate(parent, original)
    assert original['expected_episodes'] == 2000 and not original.get('sampling_round')
    campaign.mkdir(parents=True, exist_ok=False)
    files = revised_sources(parent)
    seeds = policy_seeds(original)
    roots = [parent]
    for round_id in (1, 2):
        root = campaign / f'round-{round_id}'
        root.mkdir()
        source = root / 'overlay_sources'
        for name, body in files.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        hashes = {name: collection.digest(body) for name, body in files.items()}
        with tarfile.open(root / 'overlay.tar.gz', 'w:gz') as stream:
            for name, body in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size = len(body)
                stream.addfile(member, io.BytesIO(body))
            body = json.dumps(hashes, sort_keys=True).encode()
            member = tarfile.TarInfo('_FULL_EPISODE_OVERLAY.json')
            member.size = len(body)
            stream.addfile(member, io.BytesIO(body))
        plan = dict(original, run_id=f'native-full2000-1688977-20261012-repeat{round_id}',
                    sampling_round=round_id, parent_plan_fingerprint=original['fingerprint'],
                    policy_seeds=seeds[round_id], policy_seed_protocol='native_full_episode_seed_ack_v1',
                    overlay_files_sha256=hashes,
                    overlay_sha256=collection.digest((root / 'overlay.tar.gz').read_bytes()))
        plan.pop('fingerprint')
        plan['fingerprint'] = collection.digest(json.dumps(plan, sort_keys=True).encode())
        collection.save(root / 'plan.json', plan)
        collection.save(root / 'held_out_instances.json', plan['held_out_instances'])
        (root / 'submissions').mkdir()
        roots.append(root)
    deadline = json.loads((parent / 'controller_authorization.json').read_text())['deadline_epoch']
    authorization = dict(kind='native_full_episode_three_round_campaign',
        total_episodes=6000, additional_episodes=4000, rounds=[0, 1, 2],
        roots=[str(root) for root in roots],
        fingerprints=[read_plan(root)['fingerprint'] for root in roots],
        deadline_epoch=deadline, max_active_gpus=None,
        admission='fresh_complete_idle_worker_capacity_shared_across_all_rounds',
        automatic_failure_recovery=False, checkpoint_step=1688977,
        all_public_instances_training_seen=True, validation_instances=original['held_out_instances'])
    collection.save(campaign / 'authorization.json', authorization)
    for root in roots[1:]:
        plan = read_plan(root)
        collection.save(root / 'controller_authorization.json', dict(
            kind='native_full_episode_finite_controller', plan_fingerprint=plan['fingerprint'],
            expected_episodes=2000, deadline_epoch=deadline, max_active_gpus=None,
            admission='fresh_complete_idle_worker_capacity', automatic_failure_recovery=False))
    collection.save(campaign / 'HANDOFF.json', dict(state='frozen_waiting_for_seeded_pilot',
        authorization='authorization.json', expected_episodes=6000, additional_episodes=4000,
        pilot_required_per_additional_round=True, original_first_round_retained=True,
        controller_required='watch_full_episode_repeats.py', capacity='capacity.json',
        deadline_epoch=deadline, ready_for_training=False))
    return authorization


def validate_seed_gate(root):
    root = Path(root)
    plan = read_plan(root)
    gate = validate_gate(root, plan)
    if plan.get('sampling_round'):
        result = json.loads((root / 'pilot_completion.json').read_text())['object_checkpoints'][0]['result']
        assert result.get('repeat_sampling') == dict(protocol='native_full_episode_seed_ack_v1',
            sampling_round=plan['sampling_round'], policy_seed=plan['policy_seeds']['turning_on_radio:301'],
            parent_plan_fingerprint=plan['parent_plan_fingerprint'], server_seed_acknowledged=True)
    return gate


@contextmanager
def submission_context(root, all_roots):
    root = Path(root).resolve()
    source = root / 'overlay_sources'
    previous_repo, previous_gate = collection.REPO, collection.require_capacity

    def shared_gate(observation, own_ledger, cli, env):
        merged = dict(own_ledger)
        for other in all_roots:
            if Path(other).resolve() == root:
                continue
            for path in (Path(other) / 'submissions').glob('*.json'):
                row = json.loads(path.read_text())
                assert row['name'] not in merged
                merged[row['name']] = row
        previous_gate(observation, merged, cli, env)

    if source.exists():
        plan = read_plan(root)
        for name, expected in plan['overlay_files_sha256'].items():
            assert collection.digest((source / name).read_bytes()) == expected
        collection.REPO = source
    collection.require_capacity = shared_gate
    try:
        yield
    finally:
        collection.REPO, collection.require_capacity = previous_repo, previous_gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.parent, args.campaign)
    print(json.dumps(dict(total_episodes=result['total_episodes'], additional_episodes=4000,
                          roots=result['roots'], deadline_epoch=result['deadline_epoch'])))
