"""Finite scope, source freezing and storage behavior, without cloud resources."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / 'scripts')]
from full_episode_collection import build_plan


def test_five_queued_slots_shared_with_pending_and_invisible_creations(monkeypatch):
    import full_episode_collection as collection
    observation = dict(admission_mode='bounded_spot_queue',
        queue_authorization='user_20261012_max_five_queued', workspace='share-space',
        resource_pool=collection.POOL, quota_type='spot', max_pending_tasks=5,
        observed_at=100, deadline_epoch=10000, pending_requests={'queued': 1})
    monkeypatch.setattr(collection.time, 'time', lambda: 1000)
    monkeypatch.setattr(collection, 'pending_requests', lambda *args: {'queued': 1, 'starting': 1})
    ledger = {'old-running': dict(uid='old', reserved_at=1, created_at=2),
              'new-running': dict(uid='new', reserved_at=101, created_at=102),
              'unknown': dict(reserved_at=90)}
    collection.require_capacity(observation, ledger, None, None)
    ledger['fifth'] = dict(uid='fifth', reserved_at=101, created_at=102)
    with pytest.raises(collection.CapacityUnavailable, match='Five pending'):
        collection.require_capacity(observation, ledger, None, None)
    # Refreshing the real queue observation releases earlier running/finished
    # reservations, without granting a second budget for the same observation.
    observation.update(observed_at=999, pending_requests={})
    collection.require_capacity(observation, ledger, None, None)
    observation['deadline_epoch'] = 999
    with pytest.raises(AssertionError):
        collection.require_capacity(observation, ledger, None, None)


def test_campaign_queue_excludes_other_submitters_but_shares_all_rounds(monkeypatch):
    import full_episode_collection as collection
    observation = dict(admission_mode='bounded_spot_queue',
        queue_authorization='user_20261012_max_five_queued', workspace='share-space',
        resource_pool=collection.POOL, quota_type='spot', max_pending_tasks=5,
        observed_at=100, deadline_epoch=10000, queue_scope='authorized_campaign_only',
        pending_requests={'other-batch': 5, 'round0': 1})
    monkeypatch.setattr(collection.time, 'time', lambda: 1000)
    monkeypatch.setattr(collection, 'pending_requests',
                        lambda *args: {'other-batch': 5, 'round0': 1, 'round1': 1})
    ledger = {name: dict(uid=name, reserved_at=1, created_at=2)
              for name in ('round0', 'round1')}
    ledger['round2-unknown'] = dict(reserved_at=90)
    ledger['round2-new'] = dict(uid='new', reserved_at=101, created_at=102)
    collection.require_capacity(observation, ledger, None, None)
    ledger['fifth'] = dict(uid='five', reserved_at=101, created_at=102)
    with pytest.raises(collection.CapacityUnavailable):
        collection.require_capacity(observation, ledger, None, None)


def test_campaign_refresh_does_not_release_invisible_creation_without_uid_proof(tmp_path, monkeypatch):
    import refresh_full_episode_queue as queue
    root = tmp_path / 'round0'
    (root / 'submissions').mkdir(parents=True)
    auth = dict(roots=[str(root)], deadline_epoch=10000)
    policy = dict(roots=auth['roots'], original_deadline_epoch=10000,
                  queue_authorization='user_20261012_max_five_queued', max_pending_tasks=5,
                  queue_scope='authorized_campaign_only', controller_poll_seconds=45)
    (tmp_path / 'authorization.json').write_text(json.dumps(auth))
    (tmp_path / 'queue_policy.json').write_text(json.dumps(policy))
    (tmp_path / 'queue_admission.json').write_text(json.dumps(dict(observed_at=100,
                                                        pending_requests={'external': 5})))
    (root / 'submissions/0000.json').write_text(json.dumps(dict(name='own', uid='uid',
                                                               created_at=101)))
    monkeypatch.setattr(queue.time, 'time', lambda: 1000)
    monkeypatch.setattr(queue, 'pending_requests', lambda *args: {'external': 5})
    monkeypatch.setattr(queue, 'describe', lambda *args: dict(uid='uid', state='STARTING'))
    assert queue.refresh(tmp_path, None)['pending_requests'] == {'own': 1}
    monkeypatch.setattr(queue, 'describe', lambda *args: dict(uid='uid', state='RUNNING'))
    assert queue.refresh(tmp_path, None)['pending_requests'] == {}
    (tmp_path / 'queue_admission.json').write_text(json.dumps(dict(observed_at=100,
                                                        pending_requests={'own': 1})))
    monkeypatch.setattr(queue, 'describe', lambda *args: dict(uid='wrong', state='RUNNING'))
    with pytest.raises(AssertionError):
        queue.refresh(tmp_path, None)


def test_observed_batch_outlives_120_seconds_but_never_reuses_slots(monkeypatch):
    import full_episode_collection as collection
    observation = dict(workspace='share-space', resource_pool='computing-cluster-5090-01g',
        quota_type='spot', gpus_per_job=1, vcpus_per_job=14, memory_gib_per_job=60,
        free_gpus=11, free_vcpus=163, free_memory_gib=659, observed_at=100,
        admission_mode='finite_observed_batch', worker_budget=10, deadline_epoch=10000,
        batch_authorization='user_20261012_every20min_fill_observed_capacity')
    monkeypatch.setattr(collection.time, 'time', lambda: 1000)
    monkeypatch.setattr(collection, 'pending_requests', lambda *args: {'pending': 1})
    ledger = {str(i): dict(uid=str(i), reserved_at=101, created_at=102, state='SUCCEEDED')
              for i in range(8)}
    collection.require_capacity(observation, ledger, None, None)
    ledger['ninth'] = dict(uid='9', reserved_at=101, created_at=102)
    with pytest.raises(collection.CapacityUnavailable):
        collection.require_capacity(observation, ledger, None, None)
    observation.pop('admission_mode')
    with pytest.raises(collection.CapacityUnavailable, match='Fresh capacity'):
        collection.require_capacity(observation, {}, None, None)


def test_bootstrap_retries_dns_without_starting_driver_twice(monkeypatch):
    import io
    import urllib.error
    import full_episode_collection as collection
    calls, delays, runs = [], [], []
    body = b'{"scope": "fixture"}'
    def get(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise urllib.error.URLError('Temporary failure in name resolution')
        return io.BytesIO(body)
    monkeypatch.setattr(collection.urllib.request, 'urlopen', get)
    monkeypatch.setattr(collection.time, 'sleep', delays.append)
    driver = b'def run(payload):\n    runs.append(payload)\n'
    exec(collection.build_startup('https://private.invalid/payload', collection.digest(body), driver),
         {'runs': runs})
    assert len(calls) == 3 and delays == [1, 2]
    assert runs == [{'scope': 'fixture'}]


def test_bootstrap_dns_retries_are_bounded_and_do_not_leak_url(monkeypatch, capsys):
    import urllib.error
    import full_episode_collection as collection
    calls, delays = [], []
    def get(*args, **kwargs):
        calls.append(args)
        raise urllib.error.URLError('secret-signed-url')
    monkeypatch.setattr(collection.urllib.request, 'urlopen', get)
    monkeypatch.setattr(collection.time, 'sleep', delays.append)
    with pytest.raises(RuntimeError, match='six bounded attempts'):
        collection.fetch_startup_payload('https://private.invalid', 'a' * 64)
    assert len(calls) == 6 and delays == [1, 2, 4, 8, 16]
    assert 'secret-signed-url' not in capsys.readouterr().out


def test_bootstrap_does_not_retry_sha_mismatch_or_forbidden(monkeypatch):
    import io
    import urllib.error
    import full_episode_collection as collection
    monkeypatch.setattr(collection.time, 'sleep', lambda *_: pytest.fail('must not retry'))
    monkeypatch.setattr(collection.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'wrong'))
    with pytest.raises(ValueError, match='SHA256'):
        collection.fetch_startup_payload('https://private.invalid', 'a' * 64)
    def forbidden(*args, **kwargs):
        raise urllib.error.HTTPError('https://private.invalid', 403, 'forbidden', {}, None)
    monkeypatch.setattr(collection.urllib.request, 'urlopen', forbidden)
    with pytest.raises(urllib.error.HTTPError):
        collection.fetch_startup_payload('https://private.invalid', 'a' * 64)


@pytest.mark.parametrize('existing_object', [False, True])
def test_startup_recovery_preserves_history_and_refuses_existing_data(tmp_path, monkeypatch, existing_object):
    import time
    import full_episode_collection as collection
    root = tmp_path
    (root / 'submissions').mkdir()
    previous = dict(name='behavior-full-ep-abc-a0001-g0', uid='old-uid', index=1,
        task='radio', instance_ids=[301], plan_fingerprint='abc', assignment_root='/afs/a0001',
        completion_key='completion')
    (root / 'submissions/0001.json').write_text(json.dumps(previous))
    (root / 'plan.json').write_text(json.dumps(dict(run_id='run', fingerprint='abc')))
    (root / 'production_0001_bootstrap_dns_evidence.json').write_text(json.dumps(dict(
        uid='old-uid', error='Temporary failure in name resolution', inference_started=False)))
    proof_path = root / 'afs.json'
    proof_path.write_text(json.dumps(dict(observed_at=time.time(), assignment_roots={'/afs/a0001': False},
        observer_job=dict(name='observer', uid='observer-uid'))))
    monkeypatch.setattr(collection, 'describe', lambda cli, env, name:
        dict(uid='observer-uid', state='RUNNING') if name == 'observer' else dict(uid='old-uid', state='FAILED'))
    monkeypatch.setattr(collection.subprocess, 'check_output', lambda *a, **k: 'old-uid Failed')
    class Missing(Exception):
        response = {'Error': {'Code': '404'}}
    class Client:
        def head_object(self, **kwargs):
            if existing_object:
                return {}
            raise Missing()
    created = []
    monkeypatch.setattr(collection, 'submit', lambda *a, **k: created.append(k) or k)
    if existing_object:
        with pytest.raises(ValueError, match='Existing episode'):
            collection.recover_startup(root, 1, 'capacity', 'sco', Client(), proof_path)
        assert not created and not (root / 'submission_history').exists()
    else:
        result = collection.recover_startup(root, 1, 'capacity', 'sco', Client(), proof_path)
        assert result['previous'] == previous
        assert json.loads((root / 'submission_history/0001/g0.json').read_text()) == previous
        record = dict(previous, name='behavior-full-ep-abc-a0001-g1', generation=1,
                      startup_recovery=result['recovery'])
        assert collection.submission_generation(root, record) == 1
        history = root / result['recovery']['history']
        history.write_text('{}')
        with pytest.raises(AssertionError):
            collection.submission_generation(root, record)


def ready():
    tasks = ['turning_on_radio'] + [f'task_{i:03d}' for i in range(99)]
    return dict(state='ready', checkpoint_step=1688977, checkpoint='/checkpoint', code='/base',
        source_manifest_sha256='a' * 64, scope=dict(tasks=tasks, checkpoint_files=[dict(relative_path='params/a', bytes=42)]))


def test_expired_capacity_after_object_preparation_never_creates_job(tmp_path, monkeypatch):
    import full_episode_collection as collection
    root = tmp_path
    (root / 'submissions').mkdir()
    (root / 'overlay.tar.gz').write_bytes(b'overlay')
    plan = dict(run_id='fixture', shards=[dict(index=0, pilot=True, task='radio', instance_ids=[301])],
        overlay_sha256=collection.digest(b'overlay'), overlay_files_sha256={
            'sco/run_full_episode_collection.py': collection.digest(
                (collection.REPO / 'sco/run_full_episode_collection.py').read_bytes())})
    plan['fingerprint'] = collection.digest(json.dumps(plan, sort_keys=True).encode())
    (root / 'plan.json').write_text(json.dumps(plan))
    capacity = root / 'capacity.json'; capacity.write_text('{}')
    monkeypatch.setattr(collection, 'describe', lambda *a: dict(
        uid='c14d0892eaa148d381c7f4424684c9d3', resource_pool={'name': collection.POOL}))
    monkeypatch.setattr(collection.subprocess, 'check_output', lambda *a, **k: '2.2 TB 1.3 TB')
    monkeypatch.setattr(collection.subprocess, 'run', lambda *a, **k: pytest.fail('create RPC must not run'))
    checks = []
    def admission(*args):
        checks.append(True)
        if len(checks) == 2:
            raise collection.CapacityUnavailable('observation expired')
    monkeypatch.setattr(collection, 'require_capacity', admission)
    class Client:
        def generate_presigned_url(self, *a, **k): return 'https://private.invalid'
        def put_object(self, **k): pass
        def create_multipart_upload(self, **k): return {'UploadId': 'fixture'}
    with pytest.raises(collection.CapacityUnavailable):
        collection.submit(root, 0, capacity, 'sco', Client())
    assert not (root / 'submissions/0000.json').exists()
    archived = list((root / 'unsubmitted_preparations').glob('*.json'))
    assert len(archived) == 1 and json.loads(archived[0].read_text())['state'] == 'preparing_objects'


def test_public_scope_is_exactly_once_and_validation_is_disjoint():
    source = ready()
    plan = build_plan(source, {task: task for task in source['scope']['tasks']},
                      [['turning_on_radio', 123]], run_id='full-ep-fixture')
    pairs = [(s['task'], i) for s in plan['shards'] for i in s['instance_ids']]
    assert len(pairs) == len(set(pairs)) == 2000
    assert plan['shards'][0] == dict(index=0, task='turning_on_radio', instance_ids=[301], pilot=True)
    assert max(len(s['instance_ids']) for s in plan['shards']) == 5
    assert len(plan['shards']) == 401
    assert plan['official_benchmark'] is False and plan['all_public_instances_training_seen'] is True
    assert plan['automatic_failure_recovery'] is False


@pytest.mark.parametrize('validation', [[], [['turning_on_radio', 301]], [['unknown', 99]]])
def test_validation_cannot_include_collected_public_instances(validation):
    source = ready()
    with pytest.raises(ValueError):
        build_plan(source, {task: task for task in source['scope']['tasks']}, validation, run_id='fixture')


def test_checkpoint_storage_receipt_does_not_reference_mutable_result(tmp_path, monkeypatch):
    import full_episode_checkpoint as checkpoint
    import full_episode_multipart
    import full_episode_trajectory
    path = tmp_path / 'episode.hdf5'; path.write_bytes(b'fixture raw bytes')
    result = dict(task='task', instance_id=301, rollout_id=0, steps=4, success=False)
    inspected = dict(result=dict(result), metadata=dict(attempt_id='first'), steps=4,
                     content_sha256='a' * 64, subtask_sha256='b' * 64)
    monkeypatch.setattr(full_episode_trajectory, 'inspect_full_episode_trajectory', lambda *a, **k: inspected)
    monkeypatch.setattr(checkpoint, 'request', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    written = []
    monkeypatch.setattr(checkpoint, 'upload_json', lambda value, urls: written.append(json.loads(json.dumps(value))))
    class Sink:
        def __init__(self, *a): self.body = bytearray()
        def write(self, body): self.body.extend(body); return len(body)
        def finish(self): return dict(object_key='raw.tar', bytes=len(self.body), sha256='c' * 64)
    monkeypatch.setattr(full_episode_multipart, 'MultipartArchive', Sink)
    manifest = dict(run_id='run', source_manifest_sha256='d' * 64,
                    object_storage={'task:301:0': dict(archive={}, receipt={'get': 'private-placeholder'})})
    receipt = checkpoint.checkpoint_episode(path, result, manifest, attempt_id='first')
    result['full_episode_object_checkpoint'] = receipt
    json.dumps(result)  # Adding the receipt must not create a circular result.
    assert receipt == written[0] and 'full_episode_object_checkpoint' not in receipt['result']


def test_existing_storage_receipt_prevents_duplicate_writer(tmp_path, monkeypatch):
    import full_episode_checkpoint as checkpoint
    import full_episode_trajectory
    result = dict(task='task', instance_id=301, rollout_id=0)
    monkeypatch.setattr(full_episode_trajectory, 'inspect_full_episode_trajectory',
                        lambda *a, **k: dict(result=result, metadata=dict(attempt_id='first')))
    monkeypatch.setattr(checkpoint, 'request', lambda *a, **k: (b'{}', {}))
    with pytest.raises(ValueError, match='already has an object receipt'):
        checkpoint.checkpoint_episode(tmp_path / 'raw', result, dict(object_storage={
            'task:301:0': dict(archive={}, receipt={'get': 'private-placeholder'})}), attempt_id='first')
