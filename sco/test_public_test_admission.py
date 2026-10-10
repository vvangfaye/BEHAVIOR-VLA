from datetime import datetime, timezone, timedelta
import json
from types import SimpleNamespace

import pytest

import public_test_admission as admission
from public_test_admission import standard_counts, validate_standard_capacity


def job(uid, name='old-eval', state='RUNNING', gpu=1, replicas=0, total=1, quota='RESERVED'):
    return dict(uid=uid, name=name, state=state, ownership=dict(user_name='260010165'),
                scheduling=dict(quota_type=quota), roles=[dict(total_replicas=total,
                resource_spec=[dict(replicas=replicas, limits={'nvidia.com/gpu': gpu})])])


def capacity(jobs):
    return dict(standard_counts(jobs), fresh_query=True, quota_type='reserved', owner='260010165',
                observed_at=datetime.now(timezone.utc).isoformat(), standard_limit=4,
                latest_goal_standard_limit=1)


def test_zero_replica_placeholder_and_multiworker_counts_are_conservative():
    counts = standard_counts([job('a'), job('b', gpu=2, total=2), job('c', quota='SPOT')])
    assert counts['used_gpus'] == 5


def test_standard_reuse_after_previous_pilot_completes():
    jobs = [job(str(i)) for i in range(3)] + [job('pilot', name='behavior-public1688977-pilot', state='SUCCEEDED')]
    validate_standard_capacity(capacity(jobs), now=datetime.now(timezone.utc))


def test_four_cards_or_latest_pilot_running_prevents_another_standard_job():
    for jobs in ([job(str(i)) for i in range(4)],
                 [job('pilot', name='behavior-public1688977-pilot')]):
        with pytest.raises(AssertionError):
            validate_standard_capacity(capacity(jobs), now=datetime.now(timezone.utc))


def test_suspending_still_reserves_gpu_and_stale_admission_is_rejected():
    item = capacity([job('stopping', state='SUSPENDING')])
    assert item['used_gpus'] == 1
    item['observed_at'] = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    with pytest.raises(AssertionError):
        validate_standard_capacity(item, now=datetime.now(timezone.utc))


def test_repeated_pages_cannot_hide_occupied_cards():
    with pytest.raises(AssertionError):
        standard_counts([job('same'), job('same')])


def test_state_pagination_covers_all_active_jobs_and_forward_transitions(monkeypatch):
    queries = []
    def reply(command, **kwargs):
        state = next(x.split('=', 1)[1] for x in command if x.startswith('--state='))
        page = next(x.split('=', 1)[1] for x in command if x.startswith('--page-token='))
        queries.append((state, page))
        if state == 'STARTING' and page == '1':
            rows = [job(str(i), state='STARTING') for i in range(100)]
        elif state == 'STARTING':
            rows = [job('next', state='STARTING')]
        elif state == 'RUNNING':
            rows = [job('0')]  # Job transitioned after its STARTING snapshot.
        else:
            rows = []
        return SimpleNamespace(stdout=json.dumps(rows))
    monkeypatch.setattr(admission.subprocess, 'run', reply)
    item = admission.query_standard_capacity('fixture-cli', {})
    assert item['used_gpus'] == 101
    assert ('STARTING', '2') in queries and len(set(x[0] for x in queries)) == 10
