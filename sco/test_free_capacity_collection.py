import json
from pathlib import Path

import pytest

from free_capacity_collection import available_slots, CapacityUnavailable, CapacityGate, GuardedSubprocess


def observation():
    return dict(workspace='share-space', resource_pool='computing-cluster-5090-01g', quota_type='spot',
                gpus_per_job=1, vcpus_per_job=14, memory_gib_per_job=60, observed_at=100,
                free_gpus=37, free_vcpus=528, free_memory_gib=2220)


def test_all_resources_limit_capacity_and_existing_pending_is_deducted():
    value = observation()
    assert available_slots(value, {}, {'pending': 3}, 110) == 34
    value['free_memory_gib'] = 120
    assert available_slots(value, {}, {}, 110) == 2
    value['free_vcpus'] = 14
    assert available_slots(value, {}, {}, 110) == 1


def test_new_creation_and_unknown_intent_keep_slots_without_double_counting():
    ledger = {'new': dict(reserved_at=101, created_at=102, uid='uid'),
              'unknown': dict(reserved_at=90, creation_attempted=True)}
    assert available_slots(observation(), ledger, {'new': 1, 'other': 2}, 110) == 33


def test_fresh_observation_releases_allocated_requests_but_not_unknowns():
    ledger = {'new': dict(reserved_at=101, created_at=102, uid='uid'),
              'unknown': dict(reserved_at=90, creation_attempted=True)}
    value = observation(); value['observed_at'] = 110
    assert available_slots(value, ledger, {}, 115) == 36
    assert available_slots(value, ledger, {'new': 1}, 115) == 35


@pytest.mark.parametrize('age', [-1, 121])
def test_future_or_stale_observation_never_admits(age):
    with pytest.raises(CapacityUnavailable):
        available_slots(observation(), {}, {}, 100 + age)


@pytest.mark.parametrize('key,value', [('quota_type', 'reserved'), ('resource_pool', 'debug-cluster-5090'),
                                      ('vcpus_per_job', 1), ('memory_gib_per_job', 1)])
def test_wrong_pool_quota_or_spec_fails_closed(key, value):
    data = observation();data[key] = value
    with pytest.raises(AssertionError):
        available_slots(data, {}, {}, 110)


def test_reservation_precedes_rpc_and_cannot_be_reused(tmp_path):
    capacity = tmp_path / 'capacity.json';capacity.write_text(json.dumps(observation()))
    marker = tmp_path / 'cancel';marker.write_text('cancelled')
    import hashlib
    policy = dict(kind='free_capacity_low50_extension_authorization', max_active_gpus=None,
                  indices=list(range(299, 335)), deadline_epoch=300, capacity_observation=str(capacity),
                  cancellation_marker_sha256={str(marker): hashlib.sha256(marker.read_bytes()).hexdigest()})
    path = tmp_path / 'authorization.json';path.write_text(json.dumps(policy))
    gate = CapacityGate(path, tmp_path, '', {}, query=lambda: {}, clock=lambda: 110)
    gate.reserve('job')
    restored = CapacityGate(path, tmp_path, '', {}, query=lambda: {}, clock=lambda: 110)
    with pytest.raises(AssertionError):restored.reserve('job')
    restored.before_copy(['sco', 'acp', 'jobs', 'copy', '--name=job'])
    assert json.loads(restored.ledger_path.read_text())['job']['creation_attempted']
    with pytest.raises(AssertionError):restored.before_copy(['sco', 'acp', 'jobs', 'copy', '--name=job'])
    marker.write_text('new cancellation')
    with pytest.raises(AssertionError):restored.check_authorization()


def test_copy_proxy_never_runs_rpc_when_admission_fails(monkeypatch):
    from types import SimpleNamespace
    def reject(command):raise CapacityUnavailable()
    def forbidden(*args, **kwargs):raise AssertionError('RPC must not execute')
    monkeypatch.setattr('free_capacity_collection.subprocess.run', forbidden)
    with pytest.raises(CapacityUnavailable):
        GuardedSubprocess(SimpleNamespace(before_copy=reject)).run(['sco', 'acp', 'jobs', 'copy'])
