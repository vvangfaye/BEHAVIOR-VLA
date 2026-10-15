"""Inventory and restore verified native episode objects; never mark training ready."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile

BUCKET = '260010165'
BLOCK = 8 * 1024**2
JSON_LIMIT = 256 * 1024**2


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def private_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), 'Refusing a symlink output')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def not_found(error):
    response = getattr(error, 'response', {})
    return (response.get('Error', {}).get('Code') in ('NoSuchKey', 'NotFound', '404')
            or response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 404)


def get_json(client, key, *, optional=False):
    try:
        response = client.get_object(Bucket=BUCKET, Key=key)
    except Exception as error:
        if optional and not_found(error):
            return None
        raise RuntimeError('Collection object read failed: ' + type(error).__name__) from None
    body = response['Body']
    try:
        raw = body.read(JSON_LIMIT + 1)
        require(len(raw) <= JSON_LIMIT, 'Collection JSON exceeds its bounded size')
        return json.loads(raw)
    finally:
        body.close()


def read_plan(root):
    plan = json.loads((Path(root) / 'plan.json').read_text())
    expected = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != 'fingerprint'},
                                        sort_keys=True).encode()).hexdigest()
    require(plan.get('kind') == 'native_full_episode_collection_plan'
            and plan.get('fingerprint') == expected, 'Frozen collection plan identity differs')
    require(re.fullmatch('[a-z0-9][a-z0-9-]{0,60}', plan.get('run_id', '')) is not None,
            'Invalid collection run ID')
    pairs = set()
    for index, shard in enumerate(plan['shards']):
        task = shard.get('task')
        require(shard.get('index') == index and isinstance(task, str)
                and re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_-]{0,200}', task) is not None,
                'Invalid frozen shard identity')
        instances = shard.get('instance_ids')
        require(isinstance(instances, list) and instances
                and all(type(i) is int and 301 <= i <= 320 for i in instances),
                'Invalid frozen public instance scope')
        for instance in instances:
            require((task, instance) not in pairs, 'Duplicate frozen episode identity')
            pairs.add((task, instance))
    require(len(pairs) == plan.get('expected_episodes'), 'Frozen episode count differs')
    if plan.get('expected_episodes') == 1000:
        from full_episode_scope import validate_final1000
        validate_final1000(plan)
    elif plan.get('paired_evaluation_only'):
        from paired_checkpoint_evaluation import validate_plan
        validate_plan(plan)
    require(isinstance(plan.get('checkpoint_identity'), dict) and plan['checkpoint_identity'],
            'Frozen checkpoint identity is missing')
    return plan


def prefix_for(plan):
    return 'behavior1k/full-episode-collection/' + plan['run_id'] + '/' + plan['fingerprint'][:12]


def accepted_source(root, plan):
    path = Path(root) / 'pilot_accepted.json'
    if not path.exists():
        return None
    from full_episode_acceptance import validate_gate
    gate = validate_gate(root, plan)
    require(gate.get('plan_fingerprint') == plan['fingerprint'], 'Pilot gate belongs to another plan')
    source = gate.get('source_manifest_sha256')
    require(is_sha(source), 'Pilot gate lacks its exact deployed source identity')
    require(all(gate.get(key) is True for key in ('full_episode_raw_verified',
                'real_codec_reader_verified', 'native_frame_action_alignment_verified')),
            'Pilot gate has not accepted the real trajectory and codec')
    return source


def validate_episode(receipt, plan, shard, source, *, maximum_parts=512, generation=0):
    require(isinstance(receipt, dict), 'Invalid episode receipt')
    task, instance, rollout = receipt.get('task'), receipt.get('instance_id'), receipt.get('rollout_id')
    require(task == shard['task'] and type(instance) is int and instance in shard['instance_ids']
            and type(rollout) is int and rollout == 0, 'Episode receipt is outside the frozen shard')
    require(receipt.get('kind') == 'native_full_episode_object_checkpoint'
            and receipt.get('run_id') == plan['run_id'] + f"-a{shard['index']:04d}"
            and receipt.get('source_manifest_sha256') == source, 'Episode run or source differs')
    require(isinstance(receipt.get('attempt_id'), str) and receipt['attempt_id']
            and all(is_sha(receipt.get(key)) for key in ('trajectory_sha256', 'content_sha256', 'subtask_sha256')),
            'Episode receipt lacks its attempt or immutable hashes')
    result = receipt.get('result')
    require(isinstance(result, dict) and result.get('task') == task
            and type(result.get('instance_id')) is int and result['instance_id'] == instance
            and type(result.get('rollout_id')) is int and result['rollout_id'] == 0
            and type(result.get('steps')) is int and result['steps'] > 0
            and type(result.get('success')) is bool and receipt.get('success') is result['success']
            and type(receipt.get('steps')) is int and receipt['steps'] == result['steps'],
            'Episode result differs from its receipt')
    require(type(result.get('terminated')) is bool and type(result.get('truncated')) is bool
            and (result['terminated'] or result['truncated']), 'Episode has no final environment boundary')
    require(result.get('official_benchmark') is False, 'Collection result is marked as official evaluation')
    require(isinstance(result.get('final_goal_state'), dict)
            and result['final_goal_state'].get('success') is result['success'], 'Environment goal result differs')
    condition = result.get('conditioning', {})
    require(condition.get('mode') == 'whole_task' and condition.get('oracle') is False
            and condition.get('privileged') is False and result.get('task_success') is result['success']
            and result.get('success_source') == 'environment_task_and_final_goal',
            'Episode result lacks native whole-task environment success provenance')
    if result['success']:
        require(result['terminated'], 'Successful episode must contain environment termination')
    archive = receipt.get('archive')
    archive_key = prefix_for(plan) + f'/episodes/{task}/{instance}/attempt-1' + (f'-g{generation}' if generation else '') + '.tar'
    require(isinstance(archive, dict) and archive.get('object_key') == archive_key
            and type(archive.get('bytes')) is int and archive['bytes'] > 0
            and archive['bytes'] <= 64 * 1024**3
            and is_sha(archive.get('sha256')) and isinstance(archive.get('etag'), str) and archive['etag']
            and type(archive.get('parts')) is int and 1 <= archive['parts'] <= maximum_parts
            and archive.get('part_content_md5_verified') is True, 'Episode archive identity is invalid')
    return task, instance, rollout


def inventory(root, client):
    root = Path(root)
    plan = read_plan(root)
    source = accepted_source(root, plan)
    source_accepted = source is not None
    completions, pending = [], []
    for path in sorted((root / 'submissions').glob('*.json')):
        submission = json.loads(path.read_text())
        index = submission.get('index')
        require(type(index) is int and 0 <= index < len(plan['shards'])
                and path.name == f'{index:04d}.json', 'Submission shard identity differs')
        shard = plan['shards'][index]
        from full_episode_collection import submission_generation
        generation = submission_generation(root, submission)
        require(submission.get('plan_fingerprint') == plan['fingerprint']
                and submission.get('task') == shard['task']
                and submission.get('instance_ids') == shard['instance_ids']
                and submission.get('name') == f"behavior-full-ep-{plan['fingerprint'][:8]}-a{index:04d}-g{generation}",
                'Submission belongs to another frozen scope')
        completion_key = prefix_for(plan) + f'/shards/{index:04d}' + (f'-g{generation}' if submission.get('failure_recovery') else '') + '.json'
        if not submission.get('completion_key'):
            pending.append(index)
            continue
        require(submission['completion_key'] == completion_key, 'Submission completion key differs')
        completion = get_json(client, completion_key, optional=True)
        if completion is None:
            pending.append(index)
            continue
        require(isinstance(submission.get('uid'), str) and submission['uid'], 'Completed shard lacks its exact UID')
        require(isinstance(completion, dict) and completion.get('kind') == 'native_full_episode_collection_complete'
                and completion.get('complete') is True and completion.get('official_benchmark') is False
                and completion.get('ready_for_training') is False
                and completion.get('run_id') == plan['run_id'] + f'-a{index:04d}'
                and completion.get('checkpoint') == plan['checkpoint_identity']
                and is_sha(completion.get('source_manifest_sha256')), 'Completed shard run/checkpoint/source differs')
        completions.append((index, submission, completion))
    if source is None:
        pilot = next((value for i, _, value in completions if plan['shards'][i].get('pilot')), None)
        require(pilot is not None or not completions, 'Unaccepted collection has no completed pilot source')
        source = None if pilot is None else pilot['source_manifest_sha256']
    rows, identities = [], set()
    for index, submission, completion in completions:
        shard = plan['shards'][index]
        shard_source, maximum_parts = source, plan.get('maximum_archive_parts', 512)
        if submission.get('transport_revision_sha256'):
            from full_episode_transport_revision import validate
            revision = validate(root)
            require(submission['transport_revision_sha256'] == revision['revision_sha256']
                    and submission.get('deployed_source_sha256') == revision['source_manifest_sha256']
                    and (submission.get('generation', 0) == 0 or submission.get('failure_recovery')), 'Unaccepted transport revision')
            shard_source, maximum_parts = revision['source_manifest_sha256'], revision['maximum_parts']
        require(completion['source_manifest_sha256'] == shard_source, 'Completed shard deployed source differs')
        recovery = submission.get('failure_recovery')
        expected_instances = recovery['missing_instance_ids'] if recovery else shard['instance_ids']
        receipts = completion.get('object_checkpoints')
        require(isinstance(receipts, list) and len(receipts) == len(expected_instances)
                and type(completion.get('episodes')) is int and completion['episodes'] == len(receipts),
                'Completed shard episode coverage differs')
        shard_pairs = set()
        published = [(r, shard_source, submission.get('generation', 0) if recovery else 0, maximum_parts) for r in receipts]
        if recovery:
            for v in recovery['preserved_episodes']:
                old_receipt = get_json(client, v['receipt_key'])
                require(hashlib.sha256(json.dumps(old_receipt, sort_keys=True).encode()).hexdigest() == v['receipt_sha256'], 'Preserved receipt changed')
                published.append((old_receipt, v['source_manifest_sha256'], 0, plan.get('maximum_archive_parts', 512)))
        for receipt, episode_source, episode_generation, episode_parts in published:
            identity = validate_episode(receipt, plan, shard, episode_source, maximum_parts=episode_parts, generation=episode_generation)
            require(identity not in identities, 'Duplicate completed episode identity')
            identities.add(identity)
            shard_pairs.add(identity)
            task, instance, rollout = identity
            receipt_key = prefix_for(plan) + f'/episodes/{task}/{instance}/attempt-1' + (f'-g{episode_generation}' if episode_generation else '') + '.json'
            require(get_json(client, receipt_key) == receipt, 'Published object receipt differs from shard completion')
            rows.append(dict(shard_index=index, execution_uid=submission['uid'], task=task,
                instance_id=instance, rollout_id=rollout, success=receipt['success'], steps=receipt['steps'],
                raw_sha256=receipt['trajectory_sha256'], object_key=receipt['archive']['object_key'],
                receipt_key=receipt_key, receipt=receipt,
                source_manifest_sha256=episode_source,
                transport_revision_sha256=submission.get('transport_revision_sha256')))
        require(shard_pairs == {(shard['task'], i, 0) for i in shard['instance_ids']},
                'Completed shard is missing expected episode pairs')
        require(type(completion.get('frames')) is int
                and completion['frames'] == sum(row['steps'] for row in receipts)
                and type(completion.get('successful_episodes')) is int
                and completion['successful_episodes'] == sum(row['success'] for row in receipts),
                'Completed shard aggregate counts differ')
    return dict(kind='native_full_episode_object_inventory', plan_fingerprint=plan['fingerprint'],
        run_id=plan['run_id'], checkpoint=plan['checkpoint_identity'], source_manifest_sha256=source,
        deployed_source_sha256s=sorted({row['source_manifest_sha256'] for row in rows}),
        source_accepted=source_accepted, expected_episodes=plan['expected_episodes'],
        completed_shards=len(completions), pending_submitted_shards=pending,
        episodes=len(rows), successful_episodes=sum(row['success'] for row in rows),
        frames=sum(row['steps'] for row in rows), episode_rows=rows,
        official_benchmark=False, ready_for_training=False)


def restore_episode(row, output, client):
    """Download and extract only two fixed ordinary members, then rename atomically."""
    output = Path(output)
    require(isinstance(row.get('task'), str)
            and re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_-]{0,200}', row['task']) is not None
            and type(row.get('shard_index')) is int and row['shard_index'] >= 0
            and type(row.get('instance_id')) is int and 301 <= row['instance_id'] <= 320
            and type(row.get('rollout_id')) is int and row['rollout_id'] == 0,
            'Unsafe restored episode identity')
    target = output / f"a{row['shard_index']:04d}-{row['task']}-{row['instance_id']}-r{row['rollout_id']}"
    def restored_record():
        return dict(shard_index=row['shard_index'], task=row['task'], instance_id=row['instance_id'],
            rollout_id=row['rollout_id'], success=row['success'], steps=row['steps'],
            raw_sha256=row['raw_sha256'], object_key=row['object_key'], path=str(target.resolve() / 'episode.hdf5'),
            source_manifest_sha256=row['receipt']['source_manifest_sha256'],
            transport_revision_sha256=row.get('transport_revision_sha256'))
    require(not target.is_symlink(), 'Refusing a symlink episode target')
    if target.exists():
        require(target.is_dir() and {p.name for p in target.iterdir()}
                == {'episode.hdf5', 'result.json', 'object_receipt.json'}
                and all(p.is_file() and not p.is_symlink() for p in target.iterdir()),
                'Previously restored episode layout differs')
        require(json.loads((target / 'object_receipt.json').read_text()) == row['receipt']
                and json.loads((target / 'result.json').read_text()) == row['receipt']['result']
                and sha256(target / 'episode.hdf5') == row['raw_sha256'],
                'Previously restored episode bytes or receipt differ')
        return restored_record()
    temporary = Path(tempfile.mkdtemp(prefix='.episode-', dir=output))
    try:
        expected = row['receipt']['archive']
        try:
            head = client.head_object(Bucket=BUCKET, Key=row['object_key'])
        except Exception as error:
            raise RuntimeError('Archive HEAD failed: ' + type(error).__name__) from None
        require(head.get('ContentLength') == expected['bytes'] and head.get('ETag') == expected['etag'],
                'Stored archive HEAD differs from its receipt')
        try:
            response = client.get_object(Bucket=BUCKET, Key=row['object_key'])
        except Exception as error:
            raise RuntimeError('Archive read failed: ' + type(error).__name__) from None
        archive_path = temporary / 'download.tar'
        body, count, value = response['Body'], 0, hashlib.sha256()
        try:
            require(response.get('ContentLength') == expected['bytes'] and response.get('ETag') == expected['etag'],
                    'Stored archive GET identity differs from its receipt')
            with archive_path.open('xb') as out:
                archive_path.chmod(0o600)
                for block in iter(lambda: body.read(BLOCK), b''):
                    count += len(block)
                    require(count <= expected['bytes'], 'Archive exceeds its frozen byte count')
                    value.update(block)
                    out.write(block)
        finally:
            body.close()
        require(count == expected['bytes'] and value.hexdigest() == expected['sha256'],
                'Archive size or SHA256 differs from its receipt')
        with tarfile.open(archive_path, 'r:') as archive:
            members = archive.getmembers()
            require(len(members) == 2 and {m.name for m in members} == {'episode.hdf5', 'result.json'}
                    and all(m.isfile() for m in members), 'Unsafe or unexpected archive members')
            for member in members:
                require(member.size <= expected['bytes'] and member.size > 0,
                        'Archive member exceeds its frozen byte count')
                if member.name == 'result.json':
                    require(member.size <= JSON_LIMIT, 'Episode result JSON exceeds its bounded size')
                incoming = archive.extractfile(member)
                require(incoming is not None, 'Missing ordinary archive member')
                with incoming, (temporary / member.name).open('xb') as out:
                    os.chmod(out.name, 0o600)
                    shutil.copyfileobj(incoming, out, BLOCK)
                    out.flush()
                    os.fsync(out.fileno())
        raw_path = temporary / 'episode.hdf5'
        require(sha256(raw_path) == row['raw_sha256'], 'Raw episode SHA256 differs from its receipt')
        require(json.loads((temporary / 'result.json').read_text()) == row['receipt']['result'],
                'Archived episode result differs from its published receipt')
        private_json(temporary / 'object_receipt.json', row['receipt'])
        archive_path.unlink()
        require(not target.exists() and not target.is_symlink(), 'Episode restore target appeared during restore')
        temporary.rename(target)
        return restored_record()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def restore(root, output, client, *, include_failures=False):
    output = Path(output)
    require(not output.is_symlink(), 'Refusing a symlink restore root')
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    output.chmod(0o700)
    with (output / '.restore.lock').open('a') as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = inventory(root, client)
        require(report['source_accepted'], 'Restore requires the accepted pilot source gate')
        private_json(output / 'inventory.json', report)
        records = []
        for row in report['episode_rows']:
            if (row['success'] and row['receipt']['result']['terminated']
                    and not row['receipt']['result']['truncated']) or include_failures:
                records.append(restore_episode(row, output, client))
                private_json(output / 'restored_paths.json', dict(kind='native_full_episode_raw_restore',
                    plan_fingerprint=report['plan_fingerprint'], source_manifest_sha256=report['source_manifest_sha256'],
                    include_failures=include_failures, episodes=records, ready_for_training=False))
        result = dict(kind='native_full_episode_raw_restore', plan_fingerprint=report['plan_fingerprint'],
            source_manifest_sha256=report['source_manifest_sha256'], include_failures=include_failures,
            episodes=records, ready_for_training=False)
        private_json(output / 'restored_paths.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('inventory', 'restore'):
        child = subparsers.add_parser(command)
        child.add_argument('--root', type=Path, required=True)
        child.add_argument('--output', type=Path, required=True)
        if command == 'restore':
            child.add_argument('--include-failures', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, '/private/tmp')
    from behavior_aoss_client import client
    if args.command == 'inventory':
        report = inventory(args.root, client())
        require(not args.output.is_symlink(), 'Refusing a symlink inventory root')
        args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
        args.output.chmod(0o700)
        private_json(args.output / 'inventory.json', report)
        print(json.dumps({key: report[key] for key in ('episodes', 'successful_episodes', 'completed_shards',
                                                       'source_accepted', 'ready_for_training')}))
    else:
        report = restore(args.root, args.output, client(), include_failures=args.include_failures)
        print(json.dumps(dict(restored_episodes=len(report['episodes']), ready_for_training=False,
                              paths_file=str(args.output / 'restored_paths.json'))))


if __name__ == '__main__':
    main()
