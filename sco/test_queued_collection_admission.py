from queued_collection_admission import pending_gpu_count
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
from bounded_collection_admission import GPUAllowanceFull
from queued_collection_admission import QueuedCollectionAdmission
from test_bounded_collection_admission import job


def gate_fixture(tmp_path, jobs):
    marker = tmp_path / 'USER_CANCEL_NONRUNNING.json'
    marker.write_text('{}')
    markers = {str(tmp_path): hashlib.sha256(marker.read_bytes()).hexdigest()}
    policy = dict(max_active_gpus=64, owner='owner', workspace='share-space',
                  campaign_fingerprint='frozen', deadline_epoch=10000,
                  cancelled_markers_sha256=markers)
    auth = tmp_path / 'authorization.json'
    auth.write_text(json.dumps(dict(kind='hard50_collection_pending5_authorization',
        max_active_gpus=None, max_pending_gpus=5, allowed_indices=[0, 1],
        campaign_fingerprint='frozen', deadline_epoch=10000, cancelled_markers_sha256=markers)))
    specs = [dict(index=i, directory=tmp_path, prefix=f'job-{i}',
             prepared={'plan': {'selected_groups': [{'rate': .25 if i < 2 else .75}]}})
             for i in range(3)]
    def invisible(*args):
        raise subprocess.CalledProcessError(65, ['redacted'])
    return QueuedCollectionAdmission(policy, tmp_path, specs, SimpleNamespace(describe=invisible),
        'unused', {}, authorization=auth, query=lambda: jobs, clock=lambda: 100)


def test_running_jobs_do_not_consume_queue_budget():
    assert pending_gpu_count(dict(active_job_states={'RUNNING': 100, 'QUEUEING': 3, 'STARTING': 1},
                                  unknown_or_reserved_jobs=[])) == 4


def test_unknown_submission_keeps_its_queue_slot():
    assert pending_gpu_count(dict(active_job_states={'RUNNING': 10, 'QUEUEING': 4},
                                  unknown_or_reserved_jobs=['uncertain-copy'])) == 5


def test_stopping_and_restarting_are_conservatively_charged():
    assert pending_gpu_count(dict(active_job_states={'RUNNING': 10, 'SUSPENDING': 2, 'RESTARTING': 1},
                                  unknown_or_reserved_jobs=[])) == 3


def test_other_batch_ignored_and_running_count_unlimited(tmp_path):
    gate = gate_fixture(tmp_path, [job('unrelated-batch', 'QUEUEING', gpu=100), job('job-2-g0')])
    gate.begin_pass()
    gate.reserve('job-0-g1')
    gate.before_copy(['sco', 'acp', 'jobs', 'copy', '--name=job-0-g1'])
    assert gate.last['pending_gpus'] == 1 and gate.last['running_gpus'] == 1


def test_fifth_slot_reserved_and_sixth_blocked_before_copy(tmp_path):
    jobs = [job(f'job-{i//2}-g{i%2}', 'QUEUEING') for i in range(4)]
    gate = gate_fixture(tmp_path, jobs)
    gate.begin_pass()
    gate.reserve('job-0-g2')
    gate.before_copy(['sco', 'acp', 'jobs', 'copy', '--name=job-0-g2'])
    with pytest.raises(GPUAllowanceFull):
        gate.reserve('job-1-g2')
    jobs.append(job('job-2-g0', 'STARTING'))
    with pytest.raises(GPUAllowanceFull):
        gate.before_copy(['sco', 'acp', 'jobs', 'copy', '--name=job-0-g2'])


def test_non_difficult_shard_and_changed_authorization_blocked(tmp_path):
    gate = gate_fixture(tmp_path, [])
    with pytest.raises(GPUAllowanceFull):
        gate.reserve('job-2-g1')
    gate.authorization_path.write_text('{}')
    with pytest.raises(AssertionError):
        gate.reserve('job-0-g1')


def test_five_minute_round_allows_multiple_and_restart_preserves_interval(tmp_path):
    gate = gate_fixture(tmp_path, [])
    gate.round_interval = 300
    gate.begin_pass()
    gate.reserve('job-0-g1')
    gate.before_copy(['sco', 'acp', 'jobs', 'copy', '--name=job-0-g1'])
    gate.reserve('job-1-g1')
    gate.before_copy(['sco', 'acp', 'jobs', 'copy', '--name=job-1-g1'])
    assert len(json.loads(gate.round_clock_path.read_text())['creation_attempts']) == 2
    restarted = gate_fixture(tmp_path, [])
    restarted.round_interval = 300
    restarted.begin_pass()
    with pytest.raises(GPUAllowanceFull):
        restarted.reserve('job-0-g2')
    assert 'job-0-g2' not in restarted.reservations
    restarted.clock = lambda: 400
    restarted.begin_pass()
    restarted.reserve('job-0-g2')
