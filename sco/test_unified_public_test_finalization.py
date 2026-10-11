"""Offline integrity tests; synthetic data never represents model evaluation."""
from copy import deepcopy
import json
import time

import pytest

from unified_public_test_finalization import (
    QUEUE_SHA, digest, prepare, validate_recovery_chain, validate_terminal,
)


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + '\n').encode()
    path.write_bytes(raw)
    return digest(raw)


def initial():
    return dict(index=9, uid='uid-g0', name='job-g0', generation=0, state='submitted',
        assignment_root='/mnt/afs/frozen/a009', run_id='run-a009', source_manifest_sha256='source',
        driver_sha256='driver', checkpoint='/mnt/afs/checkpoint', code='/mnt/afs/code',
        task='task', instance_ids=[301], expected_rollouts=1)


def advance(root, old):
    new = dict(old, uid='uid-g%d' % (old['generation']+1), name='job-g%d' % (old['generation']+1),
               generation=old['generation']+1)
    row = dict(task='task', instance_id=301, state='pending', attempts=old['generation']+1, max_attempts=3)
    proof = dict(state='terminal_worker_queue_reconciled', old_uid=old['uid'], old_generation=old['generation'],
        old_writer_terminal=True, completed_artifacts_preserved=True, queue_source_sha256=QUEUE_SHA,
        assignment_root=old['assignment_root'], metadata=dict(run_id=old['run_id'], mode='public_test',
        num_rollouts=1, require_videos=False, checkpoint_step=1688977, source_manifest_sha256='source'),
        before_queue=[dict(row, state='running')], queue=[row],
        artifacts_before={'attempts/kept.json': dict(bytes=3,sha256='preserved')},
        artifacts_after={'attempts/kept.json': dict(bytes=3,sha256='preserved')},
        **{'status.json':dict(source_manifest_sha256='source')})
    proof_path = 'production_recoveries/g%d-proof.json' % new['generation']
    sha = put(root, proof_path, proof)
    receipt_path = 'production_recoveries/g%d.json' % new['generation']
    receipt = dict(new, old_submission=deepcopy(old), old_writer_terminal=True, old_worker_phase='Deleted',
        queue_untouched=False, queue_reconciliation_proof=proof_path, queue_reconciliation_proof_sha256=sha,
        attempt_counts_preserved=True)
    put(root, receipt_path, receipt)
    new['recovery_receipt'] = receipt_path
    return new


def test_incomplete_scope_stops_before_observation_or_writes(tmp_path):
    calls = []
    def forbidden(receipt):
        calls.append(receipt)
        raise AssertionError('network must not be reached')
    with pytest.raises(ValueError, match='Full scope incomplete'):
        prepare(tmp_path, '/new/standard.public.team.affiliation.20261011', '/new/archive.zip', '/new/audit.json',
                observer=forbidden, count_fn=lambda _:dict(actual_completed_rollouts=1999,accepted_rollouts=80))
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_full_recovery_chain_checked_even_with_current_uid(tmp_path):
    g1 = advance(tmp_path, initial())
    g2 = advance(tmp_path, g1)
    links = validate_recovery_chain(tmp_path, g2)
    assert [link['old_submission']['uid'] for link in links] == ['uid-g1','uid-g0']
    # A defect in the older generation cannot hide behind a correct newest receipt.
    path = tmp_path / g1['recovery_receipt']
    old = json.loads(path.read_text())
    old['old_submission']['source_manifest_sha256'] = 'other-checkpoint-source'
    put(tmp_path, g1['recovery_receipt'], old)
    with pytest.raises(ValueError, match='Recovery changed source_manifest_sha256'):
        validate_recovery_chain(tmp_path, g2)


@pytest.mark.parametrize('change,message', [
    ('bytes', 'proof SHA'), ('uid', 'proof identity'), ('attempts', 'attempt budget'),
    ('artifact', 'original artifact'), ('active', 'Old writer not terminal'),
])
def test_recovered_provenance_rejects_corruption(tmp_path, change, message):
    current = advance(tmp_path, initial())
    receipt_path = current['recovery_receipt']
    receipt = json.loads((tmp_path / receipt_path).read_text())
    proof_path = receipt['queue_reconciliation_proof']
    proof = json.loads((tmp_path / proof_path).read_text())
    if change == 'active':
        receipt['old_writer_terminal'] = False
    else:
        if change == 'uid': proof['old_uid'] = 'another-job'
        elif change == 'attempts': proof['queue'][0]['attempts'] = 0
        elif change == 'artifact': proof['artifacts_after']['attempts/kept.json']['sha256'] = 'changed'
        else: proof['irrelevant'] = 'bytes changed'
        sha = put(tmp_path, proof_path, proof)
        if change != 'bytes': receipt['queue_reconciliation_proof_sha256'] = sha
    put(tmp_path, receipt_path, receipt)
    with pytest.raises(ValueError, match=message):
        validate_recovery_chain(tmp_path, current)


def test_current_exact_uid_and_worker_must_be_terminal():
    receipt = initial()
    observed = dict(uid=receipt['uid'], name=receipt['name'], owner='260010165', state='SUCCEEDED',
                    worker_uid_present=True, worker_phases=['Completed'], observed_at=time.time())
    validate_terminal(receipt, observed, 'SUCCEEDED')
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_terminal(receipt, dict(observed,uid='another-uid'), 'SUCCEEDED')
    with pytest.raises(ValueError, match='not proven stopped'):
        validate_terminal(receipt, dict(observed,worker_phases=['Running']), 'SUCCEEDED')


@pytest.fixture
def mocked_execution(tmp_path, monkeypatch):
    """Synthetic control-flow fixture; substitutes both pinned helper modules."""
    import base64
    import builtins
    import unified_public_test_finalization as unified

    calls = []
    monkeypatch.setattr(builtins, '_unified_finalization_test_calls', calls, raising=False)
    sources = {
        'streaming_finalize_public_test.py': b'''import builtins
def run(payload):
    builtins._unified_finalization_test_calls.append('finalizer')
    return dict(state='complete', public_test_rollouts=2000)
''',
        'streaming_package_public_test.py': b'''import builtins
def package(destination, archive):
    builtins._unified_finalization_test_calls.append('package')
    return dict(roundtrip_verified=True, public_test_rollouts=2000)
''',
    }
    monkeypatch.setattr(unified, 'PINS', {name: digest(raw) for name, raw in sources.items()})
    body = dict(state='ready_for_one_unified_audit', original_count=2000, accepted_count=80,
        sources={name: base64.b64encode(raw).decode() for name, raw in sources.items()},
        outputs=dict(destination=str(tmp_path / 'standard.public.synthetic.synthetic.20261011'),
                     archive=str(tmp_path / 'synthetic.zip'), audit_report=str(tmp_path / 'audit.json')),
        payload=dict(assignments=[], code=str(tmp_path / 'frozen-code')))
    envelope = dict(bundle=body, sha256=digest(unified.canonical(body)))
    def audit(unused_body):
        calls.append('audit')
        return dict(state='unified_audit_verified', rollouts=2000)
    monkeypatch.setattr(unified, 'inspect_afs', audit)
    return unified, envelope, calls


def test_failed_unified_audit_never_calls_finalizer_or_package(mocked_execution, monkeypatch):
    unified, envelope, calls = mocked_execution
    def fail_audit(unused_body):
        calls.append('audit')
        raise ValueError('synthetic AFS audit failure')
    monkeypatch.setattr(unified, 'inspect_afs', fail_audit)
    with pytest.raises(ValueError, match='synthetic AFS audit failure'):
        unified.execute(envelope)
    assert calls == ['audit']
    outputs = {key: unified.Path(value) for key, value in envelope['bundle']['outputs'].items()}
    assert all(not path.exists() for path in outputs.values())
    assert unified.Path(str(outputs['audit_report']) + '.intent.json').exists()
    # An interrupted/failed first execution cannot silently re-audit or retry.
    with pytest.raises(FileExistsError):
        unified.execute(envelope)
    assert calls == ['audit']


def test_valid_unified_audit_orders_helpers_once_and_rejects_duplicate(mocked_execution):
    unified, envelope, calls = mocked_execution
    result = unified.execute(envelope)
    assert result['state'] == 'finalized_and_packaged'
    assert calls == ['audit', 'finalizer', 'package']
    with pytest.raises(ValueError, match='Preserve existing final outputs'):
        unified.execute(envelope)
    assert calls == ['audit', 'finalizer', 'package']
    # Simulate a lost synthetic audit report: the permanent intent independently
    # blocks a second writer even though the output-presence guard no longer does.
    unified.Path(envelope['bundle']['outputs']['audit_report']).unlink()
    with pytest.raises(FileExistsError):
        unified.execute(envelope)
    assert calls == ['audit', 'finalizer', 'package']
