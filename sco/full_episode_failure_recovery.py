"""Finite, explicitly authorized recovery of the user's frozen 78 failed shards.

Keep all previous generations and completed episode objects. No automatic retry
of a recovery failure and no scope/budget extension.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from full_episode_collection import digest, save, submission_generation
from submit_full_collection_shard import describe, environment


def load_authorization(campaign):
    campaign = Path(campaign).resolve()
    raw = (campaign / 'authorization.json').read_bytes()
    original = json.loads(raw)
    auth = json.loads((campaign / 'failure_recovery_authorization.json').read_text())
    assert auth['kind'] in ('user_failure78_recovery', 'user_final1000_failure5_recovery')
    assert auth['original_authorization_sha256'] == digest(raw)
    assert auth['roots'] == original['roots'] and auth['deadline_epoch'] == original['deadline_epoch']
    assert auth['max_additional_attempts_per_shard'] == 1 and auth['max_pending_tasks'] == 5
    expected = 5 if auth['kind'] == 'user_final1000_failure5_recovery' else 78
    assert len(auth['failures']) == expected and len({v['previous_uid'] for v in auth['failures']}) == expected
    if expected == 5:
        assert original['kind'] == 'native_full_episode_independent_final1000'
        assert original['total_episodes'] == 1000
        assert {v['index'] for v in auth['failures']} == {1, 24, 36, 78, 84}
        assert auth['target_episodes'] == 1000 and auth['preserve_all_existing_episodes'] is True
    return auth


def validate_preparation(root, previous, recovery):
    root = Path(root).resolve()
    auth = load_authorization(recovery['campaign'])
    assert str(root) in auth['roots'] and time.time() < auth['deadline_epoch']
    assert not any((Path(p) / 'STOP').exists() for p in [recovery['campaign']] + auth['roots'])
    item = next(v for v in auth['failures'] if v['previous_uid'] == previous['uid'])
    assert item['root'] == str(root) and item['index'] == previous['index']
    assert item['eligible'] is True
    if auth['kind'] == 'user_failure78_recovery':
        assert item['category'] == 'multipart_put_URLError'
    else:
        assert item['category'] in ('abrupt_mid_episode_worker_exit', 'abrupt_initialization_worker_exit')
        assert recovery['kind'] == auth['kind']
    assert previous.get('failure_recovery') is None
    assert recovery['previous_uid'] == previous['uid']
    assert recovery['generation'] == previous.get('generation', 0) + 1
    assert digest((root / recovery['history']).read_bytes()) == recovery['history_sha256']
    assert json.loads((root / recovery['history']).read_text()) == previous
    proof_raw = (root / recovery['evidence']).read_bytes()
    assert digest(proof_raw) == recovery['evidence_sha256']
    proof = json.loads(proof_raw)
    assert proof['verified'] is True and proof['old_writer_stopped'] is True
    assert proof['uid'] == previous['uid'] and proof['state'] == 'FAILED'
    assert proof['completion_present'] is False
    assert recovery['preserved_episodes'] == proof['preserved_episodes']
    assert recovery['missing_instance_ids'] == proof['missing_instance_ids']
    preserved = [v['instance_id'] for v in recovery['preserved_episodes']]
    missing = recovery['missing_instance_ids']
    assert len(set(preserved + missing)) == len(previous['instance_ids'])
    assert sorted(preserved + missing) == sorted(previous['instance_ids'])
    assert set(preserved).isdisjoint(missing)
    submission_generation(root, previous)
    return proof


def validate_record(root, record):
    recovery = record['failure_recovery']
    previous = json.loads((root / recovery['history']).read_text())
    # Past results remain readable after the original deadline; admission alone
    # checks the current time. Validate the same immutable proof without renewal.
    auth = load_authorization(recovery['campaign'])
    assert str(root.resolve()) in auth['roots']
    item = next(v for v in auth['failures'] if v['previous_uid'] == previous['uid'])
    assert item['eligible'] and item['root'] == str(root.resolve())
    assert digest((root / recovery['history']).read_bytes()) == recovery['history_sha256']
    assert digest((root / recovery['evidence']).read_bytes()) == recovery['evidence_sha256']
    proof = json.loads((root / recovery['evidence']).read_text())
    assert proof['uid'] == previous['uid'] and proof['state'] == 'FAILED' and proof['old_writer_stopped']
    assert proof['verified'] and not proof['completion_present']
    assert recovery['missing_instance_ids'] == proof['missing_instance_ids']
    assert recovery['preserved_episodes'] == proof['preserved_episodes']
    assert record['generation'] == previous.get('generation', 0) + 1 == recovery['generation']
    assert all(record[k] == previous[k] for k in ('index','task','instance_ids','plan_fingerprint'))
    assert record['name'] == previous['name'].rsplit('-g',1)[0] + '-g' + str(record['generation'])
    preserved = [v['instance_id'] for v in recovery['preserved_episodes']]
    assert len(set(preserved + recovery['missing_instance_ids'])) == len(previous['instance_ids'])
    assert sorted(preserved + recovery['missing_instance_ids']) == sorted(previous['instance_ids'])
    submission_generation(root, previous)


def prepare(campaign, root, index, cli):
    campaign, root = Path(campaign).resolve(), Path(root).resolve()
    auth = load_authorization(campaign)
    previous_raw = (root / f'submissions/{index:04d}.json').read_bytes()
    previous = json.loads(previous_raw)
    item = next(v for v in auth['failures'] if v['previous_uid'] == previous['uid'])
    assert item['eligible'] and item['root'] == str(root) and item['index'] == index
    audit_raw = (campaign / auth['audit_file']).read_bytes()
    assert digest(audit_raw) == auth['audit_sha256']
    proof = next(v for v in json.loads(audit_raw)['failures'] if v['uid'] == previous['uid'])
    assert proof['original_receipt_sha256'] == digest(previous_raw)
    job = describe(cli, environment(), previous['name'])
    assert job['uid'] == previous['uid'] and job['state'] == 'FAILED'
    workers = subprocess.check_output([cli,'acp','jobs','get-workers',previous['name'],
        '--workspace-name=share-space'],env=environment(),text=True,timeout=45)
    assert previous['uid'] in workers and 'Running' not in workers
    assert any(v in workers for v in ('Failed','Deleted','Stopped','Succeeded'))
    history = root / 'submission_history' / f'{index:04d}'
    history.mkdir(parents=True,exist_ok=True)
    old_path = history / f"g{previous.get('generation',0)}.json"
    if old_path.exists():
        assert old_path.read_bytes() == previous_raw
    else:
        old_path.write_bytes(previous_raw)
    generation = previous.get('generation',0) + 1
    label = 'failure5' if auth['kind'] == 'user_final1000_failure5_recovery' else 'failure78'
    proof_path = history / f'{label}-g{generation}-evidence.json'
    if proof_path.exists():
        assert json.loads(proof_path.read_text()) == proof
    else:
        save(proof_path,proof)
    recovery = dict(kind=auth['kind'],campaign=str(campaign),generation=generation,
        previous_uid=previous['uid'],history=str(old_path.relative_to(root)),history_sha256=digest(previous_raw),
        evidence=str(proof_path.relative_to(root)),evidence_sha256=digest(proof_path.read_bytes()),
        missing_instance_ids=proof['missing_instance_ids'],preserved_episodes=proof['preserved_episodes'])
    validate_preparation(root,previous,recovery)
    return previous,recovery


def choose_recovery(campaign, cli, storage):
    """One transport-recovery canary first; stop on its failure, never retry it."""
    campaign = Path(campaign)
    path = campaign / 'failure_recovery_authorization.json'
    if not path.exists():
        return None
    auth = load_authorization(campaign)
    eligible = [v for v in auth['failures'] if v['eligible']]
    if not eligible:
        return None
    if auth['kind'] == 'user_final1000_failure5_recovery':
        # Explicit finite repair authorization, with the user's production
        # acceptance waiver still recorded separately from real acceptance.
        for item in eligible:
            root = Path(item['root'])
            current = json.loads((root / f"submissions/{item['index']:04d}.json").read_text())
            if current.get('failure_recovery'):
                validate_record(root, current)
                continue
            assert current['uid'] == item['previous_uid']
            return root, item['index']
        return None
    canary = eligible[0]
    root = Path(canary['root'])
    current = json.loads((root / f"submissions/{canary['index']:04d}.json").read_text())
    if current.get('failure_recovery'):
        gate = campaign / 'failure_recovery_canary_accepted.json'
        if not gate.exists():
            job = describe(cli,environment(),current['name'])
            assert job['uid'] == current['uid']
            if job['state'] in ('FAILED','STOPPED','DELETED','CANCELLED'):
                save(campaign/'failure_recovery_status.json',dict(state='canary_failed_no_further_recovery',uid=current['uid'],observed_at=time.time()))
                return None
            if job['state'] != 'SUCCEEDED':
                return None
            from restore_full_episode_collection import inventory,restore_episode
            result = inventory(root,storage)
            rows = [v for v in result['episode_rows'] if v['shard_index'] == canary['index']]
            assert len(rows) == len(current['instance_ids'])
            # Verify full archive SHA/content alignment on the recovered canary,
            # not just the platform terminal state.
            output=campaign/'failure_recovery_canary_verified';output.mkdir(exist_ok=True)
            recovered = next(v for v in rows if v['instance_id'] in current['failure_recovery']['missing_instance_ids'])
            verified=restore_episode(recovered,output,storage)
            save(gate,dict(uid=current['uid'],observed_at=time.time(),verified_episode=verified,
                full_archive_sha_verified=True,all_episode_receipts_verified=True,ready_for_training=False))
    for item in eligible:
        r=Path(item['root']);v=json.loads((r/f"submissions/{item['index']:04d}.json").read_text())
        if v.get('failure_recovery'):
            continue
        assert v['uid'] == item['previous_uid'], 'Changed recovery identity requires inspection'
        return r,item['index']
    return None


def published_coverage(plan, records, objects):
    """Count exact published JSON/archive pairs, preserving old generations."""
    from restore_full_episode_collection import prefix_for
    prefix = prefix_for(plan)
    covered, missing = [], []
    identities = set()
    for record in records:
        repair = record.get('failure_recovery')
        preserved = {v['instance_id']: v for v in repair['preserved_episodes']} if repair else {}
        for instance in record['instance_ids']:
            identity = (record['task'], instance)
            assert identity not in identities
            identities.add(identity)
            if instance in preserved:
                key = preserved[instance]['receipt_key']
            else:
                if repair:
                    assert instance in repair['missing_instance_ids']
                suffix = f"-g{record['generation']}" if repair else ''
                key = prefix + f"/episodes/{record['task']}/{instance}/attempt-1{suffix}.json"
            archive = key[:-5] + '.tar'
            pair = dict(task=record['task'], instance_id=instance, receipt_key=key, archive_key=archive)
            if (0 < objects.get(key, 0) <= 256 * 1024**2
                    and 0 < objects.get(archive, 0) <= 64 * 1024**3):
                covered.append(pair)
            else:
                missing.append(pair)
    return covered, missing


def monitor_final1000_coverage(campaign, storage):
    """Read-only completion monitoring after finite initial/recovery admission."""
    from restore_full_episode_collection import read_plan, prefix_for
    campaign = Path(campaign)
    auth = load_authorization(campaign)
    assert auth['kind'] == 'user_final1000_failure5_recovery'
    root = Path(auth['roots'][0])
    plan = read_plan(root)
    records = [json.loads(p.read_text()) for p in sorted((root / 'submissions').glob('*.json'))]
    assert len(records) == len(plan['shards']) == 201
    for record in records:
        assert record.get('uid') and record['plan_fingerprint'] == plan['fingerprint']
        assert record['instance_ids'] == plan['shards'][record['index']]['instance_ids']
        submission_generation(root, record)
    objects = {}
    for page in storage.get_paginator('list_objects_v2').paginate(
            Bucket='260010165', Prefix=prefix_for(plan) + '/episodes/'):
        objects.update({v['Key']: v['Size'] for v in page.get('Contents', [])})
    covered, missing = published_coverage(plan, records, objects)
    assert len(covered) + len(missing) == 1000
    progress = dict(observed_at=time.time(), target_episodes=1000,
        published_json_archive_pairs=len(covered), missing_pairs=missing,
        all_target_pairs_published=len(covered) == 1000,
        whole_archive_readback_accepted=False, ready_for_training=False)
    save(campaign / 'full_episode_target_progress.json', progress)
    return progress
