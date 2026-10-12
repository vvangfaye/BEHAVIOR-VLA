import copy
import hashlib
import io
import json
import stat
import tarfile

import pytest

import restore_full_episode_collection as restore


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_tar(raw, result, *, name='episode.hdf5', link=False):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as archive:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = '/tmp/escaped-episode'
            archive.addfile(member)
        else:
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        data = json.dumps(result).encode()
        member = tarfile.TarInfo('result.json')
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return out.getvalue()


class MissingObject(Exception):
    response = {'Error': {'Code': 'NoSuchKey'}}


class MemoryObjects:
    def __init__(self, objects):
        self.objects = objects
        self.get_keys = []

    def head_object(self, *, Bucket, Key):
        assert Bucket == restore.BUCKET
        if Key not in self.objects:
            raise MissingObject()
        return {'ContentLength': len(self.objects[Key]), 'ETag': '"stored-1"'}

    def get_object(self, *, Bucket, Key):
        self.get_keys.append(Key)
        return dict(self.head_object(Bucket=Bucket, Key=Key), Body=io.BytesIO(self.objects[Key]))


@pytest.fixture
def collection(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    (root / 'submissions').mkdir(parents=True)
    source = 'a' * 64
    plan = dict(kind='native_full_episode_collection_plan', run_id='fixture-native', expected_episodes=2,
                checkpoint_identity={'checkpoint_step': 1688977, 'checkpoint_source': '/immutable/ckpt'},
                shards=[dict(index=0, task='turning_on_radio', instance_ids=[301], pilot=True),
                        dict(index=1, task='turning_on_radio', instance_ids=[302], pilot=False)])
    plan['fingerprint'] = sha(json.dumps(plan, sort_keys=True).encode())
    (root / 'plan.json').write_text(json.dumps(plan))
    (root / 'pilot_accepted.json').write_text(json.dumps(dict(plan_fingerprint=plan['fingerprint'],
        source_manifest_sha256=source, full_episode_raw_verified=True, real_codec_reader_verified=True,
        native_frame_action_alignment_verified=True)))
    # Actual gate evidence verification is covered by test_full_episode_acceptance;
    # this fixture isolates object restore contracts from that separate verifier.
    monkeypatch.setattr('full_episode_acceptance.validate_gate',
        lambda root, plan: json.loads((root / 'pilot_accepted.json').read_text()))
    objects = {}
    for index, shard in enumerate(plan['shards']):
        instance, success = shard['instance_ids'][0], index == 0
        result = dict(task=shard['task'], instance_id=instance, rollout_id=0, success=success, steps=3,
                      terminated=success, truncated=not success, final_goal_state={'success': success},
                      official_benchmark=False, task_success=success,
                      success_source='environment_task_and_final_goal',
                      conditioning={'mode': 'whole_task', 'oracle': False, 'privileged': False})
        raw = b'fake HDF5 bytes for sdk-only integrity check ' + str(instance).encode()
        archive_bytes = make_tar(raw, result)
        object_prefix = restore.prefix_for(plan) + f"/episodes/{shard['task']}/{instance}/attempt-1"
        receipt = dict(kind='native_full_episode_object_checkpoint', run_id=plan['run_id'] + f'-a{index:04d}',
            source_manifest_sha256=source, task=shard['task'], instance_id=instance, rollout_id=0,
            success=success, steps=3, attempt_id='attempt-1/rollout-0', trajectory_sha256=sha(raw),
            content_sha256='b' * 64, subtask_sha256='c' * 64, result=result,
            archive=dict(object_key=object_prefix + '.tar', bytes=len(archive_bytes), sha256=sha(archive_bytes),
                         etag='"stored-1"', parts=1, part_content_md5_verified=True))
        completion_key = restore.prefix_for(plan) + f'/shards/{index:04d}.json'
        completion = dict(kind='native_full_episode_collection_complete', complete=True,
            official_benchmark=False, ready_for_training=False, checkpoint=plan['checkpoint_identity'],
            run_id=receipt['run_id'], source_manifest_sha256=source, episodes=1,
            successful_episodes=int(success), frames=3, object_checkpoints=[receipt])
        submission = dict(index=index, name=f"behavior-full-ep-{plan['fingerprint'][:8]}-a{index:04d}-g0",
            plan_fingerprint=plan['fingerprint'], task=shard['task'], instance_ids=[instance],
            uid=f'exact-uid-{index}', completion_key=completion_key)
        (root / 'submissions' / f'{index:04d}.json').write_text(json.dumps(submission))
        objects[object_prefix + '.tar'] = archive_bytes
        objects[object_prefix + '.json'] = json.dumps(receipt).encode()
        objects[completion_key] = json.dumps(completion).encode()
    return root, plan, MemoryObjects(objects)


def test_default_restores_only_success_and_keeps_private_verified_paths(collection, tmp_path):
    root, _, client = collection
    output = tmp_path / 'restored'
    report = restore.restore(root, output, client)
    assert len(report['episodes']) == 1 and report['episodes'][0]['success'] is True
    assert report['ready_for_training'] is False
    assert sum(key.endswith('.tar') for key in client.get_keys) == 1
    raw = output / 'a0000-turning_on_radio-301-r0' / 'episode.hdf5'
    assert restore.sha256(raw) == report['episodes'][0]['raw_sha256']
    assert stat.S_IMODE(raw.stat().st_mode) == 0o600
    assert stat.S_IMODE((output / 'restored_paths.json').stat().st_mode) == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert not list(output.glob('.episode-*'))


def test_accepted_transport_source_keeps_original_pilot_and_per_episode_identity(collection, monkeypatch):
    root, plan, client = collection
    revision = dict(revision_sha256='d' * 64, source_manifest_sha256='e' * 64, maximum_parts=8192)
    monkeypatch.setattr('full_episode_transport_revision.validate', lambda _: revision)
    path = root / 'submissions/0001.json'
    submission = json.loads(path.read_text())
    submission.update(transport_revision_sha256=revision['revision_sha256'],
                      deployed_source_sha256=revision['source_manifest_sha256'])
    path.write_text(json.dumps(submission))
    key = restore.prefix_for(plan) + '/shards/0001.json'
    completion = json.loads(client.objects[key])
    completion['source_manifest_sha256'] = revision['source_manifest_sha256']
    receipt = completion['object_checkpoints'][0]
    receipt['source_manifest_sha256'] = revision['source_manifest_sha256']
    receipt['archive']['parts'] = 600
    client.objects[key] = json.dumps(completion).encode()
    receipt_key = restore.prefix_for(plan) + '/episodes/turning_on_radio/302/attempt-1.json'
    client.objects[receipt_key] = json.dumps(receipt).encode()
    report = restore.inventory(root, client)
    assert report['source_manifest_sha256'] == 'a' * 64
    assert report['deployed_source_sha256s'] == ['a' * 64, 'e' * 64]
    assert report['episode_rows'][1]['transport_revision_sha256'] == 'd' * 64
    assert report['successful_episodes'] == 1 and not report['ready_for_training']
    submission['transport_revision_sha256'] = 'f' * 64
    path.write_text(json.dumps(submission))
    with pytest.raises(ValueError, match='Unaccepted transport'):
        restore.inventory(root, client)


def test_explicit_failures_are_raw_only_and_need_pilot_source_acceptance(collection, tmp_path):
    root, _, client = collection
    report = restore.restore(root, tmp_path / 'all-raw', client, include_failures=True)
    assert len(report['episodes']) == 2 and report['ready_for_training'] is False
    (root / 'pilot_accepted.json').unlink()
    assert restore.inventory(root, client)['source_accepted'] is False
    with pytest.raises(ValueError, match='accepted pilot'):
        restore.restore(root, tmp_path / 'unaccepted', client)


@pytest.mark.parametrize('change', ['checkpoint', 'source', 'pair', 'receipt'])
def test_completed_objects_must_match_frozen_scope_and_exact_receipt(collection, change):
    root, plan, client = collection
    key = restore.prefix_for(plan) + '/shards/0000.json'
    completion = json.loads(client.objects[key])
    if change == 'checkpoint':
        completion['checkpoint']['checkpoint_step'] = 1375000
    elif change == 'source':
        completion['source_manifest_sha256'] = 'e' * 64
    elif change == 'pair':
        completion['object_checkpoints'][0]['instance_id'] = 320
    else:
        completion['object_checkpoints'][0]['subtask_sha256'] = 'e' * 64
    client.objects[key] = json.dumps(completion).encode()
    with pytest.raises(ValueError):
        restore.inventory(root, client)


@pytest.mark.parametrize('change', ['sha', 'raw_sha', 'escape', 'link', 'result'])
def test_failed_download_validation_never_publishes_partial_episode(collection, tmp_path, change):
    root, _, client = collection
    row = copy.deepcopy(restore.inventory(root, client)['episode_rows'][0])
    if change == 'sha':
        row['receipt']['archive']['sha256'] = '0' * 64
    elif change == 'raw_sha':
        row['raw_sha256'] = '0' * 64
    else:
        result = row['receipt']['result'].copy()
        if change == 'result':
            result['steps'] = 100
        raw = b'fake HDF5 bytes for sdk-only integrity check 301'
        data = make_tar(raw, result, name='../episode.hdf5' if change == 'escape' else 'episode.hdf5',
                        link=change == 'link')
        client.objects[row['object_key']] = data
        row['receipt']['archive'].update(bytes=len(data), sha256=sha(data))
    output = tmp_path / 'restored'
    output.mkdir()
    with pytest.raises(ValueError):
        restore.restore_episode(row, output, client)
    assert list(output.iterdir()) == []
    assert not (tmp_path / 'episode.hdf5').exists()


def test_frozen_plan_tampering_fails_before_cloud_read(collection):
    root, _, client = collection
    plan = json.loads((root / 'plan.json').read_text())
    plan['run_id'] = 'another-run'
    (root / 'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='plan identity'):
        restore.inventory(root, client)
    assert client.get_keys == []


def test_incremental_restore_reuses_only_identical_existing_raw(collection, tmp_path):
    root, _, client = collection
    output = tmp_path / 'incremental'
    first = restore.restore(root, output, client)
    archive_reads = sum(key.endswith('.tar') for key in client.get_keys)
    assert restore.restore(root, output, client) == first
    assert sum(key.endswith('.tar') for key in client.get_keys) == archive_reads
    raw = output / 'a0000-turning_on_radio-301-r0' / 'episode.hdf5'
    raw.write_bytes(b'corrupted after initial restore')
    with pytest.raises(ValueError, match='Previously restored'):
        restore.restore(root, output, client)


def test_recovered_shard_merges_preserved_episode_without_overwriting_it(collection, monkeypatch):
    root, plan, client = collection
    old_prefix=restore.prefix_for(plan)
    old=json.loads(client.objects[old_prefix+'/episodes/turning_on_radio/302/attempt-1.json'])
    plan['shards'][1]['instance_ids']=[302,303];plan['expected_episodes']=3
    plan.pop('fingerprint');plan['fingerprint']=sha(json.dumps(plan,sort_keys=True).encode())
    (root/'plan.json').write_text(json.dumps(plan));new_prefix=restore.prefix_for(plan)
    converted={}
    for key,body in client.objects.items():
        if key.endswith('.json'):body=body.replace(old_prefix.encode(),new_prefix.encode())
        converted[key.replace(old_prefix,new_prefix)]=body
    client.objects=converted
    for path in (root/'submissions').glob('*.json'):
        v=json.loads(path.read_text());v['plan_fingerprint']=plan['fingerprint'];v['name']=f"behavior-full-ep-{plan['fingerprint'][:8]}-a{v['index']:04d}-g0";v['completion_key']=v['completion_key'].replace(old_prefix,new_prefix);path.write_text(json.dumps(v))
    gate=json.loads((root/'pilot_accepted.json').read_text());gate['plan_fingerprint']=plan['fingerprint'];(root/'pilot_accepted.json').write_text(json.dumps(gate))
    old=json.loads(client.objects[new_prefix+'/episodes/turning_on_radio/302/attempt-1.json'])
    preserved_bytes=client.objects[new_prefix+'/episodes/turning_on_radio/302/attempt-1.json']
    revision=dict(revision_sha256='d'*64,source_manifest_sha256='e'*64,maximum_parts=8192)
    monkeypatch.setattr('full_episode_transport_revision.validate',lambda _:revision)
    monkeypatch.setattr('full_episode_collection.submission_generation',lambda r,v:v.get('generation',0))
    new=copy.deepcopy(old);new['instance_id']=303;new['result']['instance_id']=303;new['source_manifest_sha256']=revision['source_manifest_sha256'];new['archive']['object_key']=new_prefix+'/episodes/turning_on_radio/303/attempt-1-g1.tar'
    new_key=new_prefix+'/episodes/turning_on_radio/303/attempt-1-g1.json';client.objects[new_key]=json.dumps(new).encode()
    completion=dict(kind='native_full_episode_collection_complete',complete=True,official_benchmark=False,ready_for_training=False,checkpoint=plan['checkpoint_identity'],run_id=new['run_id'],source_manifest_sha256=revision['source_manifest_sha256'],episodes=1,successful_episodes=0,frames=3,object_checkpoints=[new])
    key=new_prefix+'/shards/0001-g1.json';client.objects[key]=json.dumps(completion).encode()
    path=root/'submissions/0001.json';v=json.loads(path.read_text());v.update(name=v['name'][:-1]+'1',generation=1,instance_ids=[302,303],completion_key=key,transport_revision_sha256=revision['revision_sha256'],deployed_source_sha256=revision['source_manifest_sha256'],failure_recovery=dict(missing_instance_ids=[303],preserved_episodes=[dict(instance_id=302,receipt_key=new_prefix+'/episodes/turning_on_radio/302/attempt-1.json',receipt_sha256=sha(json.dumps(old,sort_keys=True).encode()),source_manifest_sha256=old['source_manifest_sha256'])]));path.write_text(json.dumps(v))
    result=restore.inventory(root,client)
    assert result['episodes']==3 and result['frames']==9
    assert len({(r['task'],r['instance_id']) for r in result['episode_rows']})==3
    assert client.objects[new_prefix+'/episodes/turning_on_radio/302/attempt-1.json']==preserved_bytes
    # A changed historical receipt is rejected rather than silently substituted.
    old['steps']=4;client.objects[new_prefix+'/episodes/turning_on_radio/302/attempt-1.json']=json.dumps(old).encode()
    with pytest.raises(ValueError,match='Preserved receipt changed'):restore.inventory(root,client)
