"""Offline orchestration checks: ready gate, frozen identities, durable CPU intent."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

import video_eval_campaign as campaign
import video_eval_finish as finish


def minimal_plan():
    return dict(run_id='offline-campaign', object_prefix='offline/campaign',
                afs_output_root='/mnt/afs/offline/campaign')


def ready_payload():
    return dict(state='ready', source_manifest_sha256='a' * 64,
                checkpoint_ready_sha256='b' * 64, worker_sha256='c' * 64,
                code='/mnt/afs/offline/code')


def test_campaign_waits_for_download_then_freezes_inputs_before_controller(tmp_path, monkeypatch):
    (tmp_path / 'preparation_plan.json').write_text(json.dumps(minimal_plan()))
    events = []
    def prepare(root, path):
        events.append('prepare')
        return ready_payload() if events.count('prepare') == 2 else dict(state='waiting_download')
    def wait(root, seconds):
        assert seconds == 45
        events.append('wait45')
    def adapter(root, plan):
        events.append('adapter')
        saved = json.loads((root / 'plan.json').read_text())
        assert plan == saved
        assert saved['immutable_inputs']['worker_sha256'] == 'c' * 64
        return object()
    class FakeController:
        def __init__(self, root, adapter):
            events.append('controller')
        def run(self, *, once, poll_seconds):
            assert not once and poll_seconds == 45
            events.append('run')
            return dict(completion_state='stopped', verified_episodes=0, finished=False)
    monkeypatch.setattr(campaign, 'step', prepare)
    monkeypatch.setattr(campaign, 'wait_after_round', wait)
    monkeypatch.setattr(campaign, 'Adapter', adapter)
    monkeypatch.setattr(campaign, 'Controller', FakeController)
    monkeypatch.setattr(sys, 'argv', ['campaign', '--root', str(tmp_path)])
    campaign.main()
    assert events == ['prepare', 'wait45', 'prepare', 'adapter', 'controller', 'run']


def test_campaign_once_while_download_incomplete_does_not_construct_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, 'step', lambda *args: dict(state='waiting_download'))
    def forbidden(*args):
        raise AssertionError('GPU adapter must not be initialized before readiness')
    monkeypatch.setattr(campaign, 'Adapter', forbidden)
    monkeypatch.setattr(sys, 'argv', ['campaign', '--root', str(tmp_path), '--once'])
    campaign.main()
    assert not (tmp_path / 'plan.json').exists()


def test_campaign_refuses_changed_existing_plan(tmp_path, monkeypatch):
    (tmp_path / 'preparation_plan.json').write_text(json.dumps(minimal_plan()))
    (tmp_path / 'plan.json').write_text(json.dumps(dict(minimal_plan(), immutable_inputs={'wrong': True})))
    monkeypatch.setattr(campaign, 'step', lambda *args: ready_payload())
    monkeypatch.setattr(sys, 'argv', ['campaign', '--root', str(tmp_path), '--once'])
    with pytest.raises(AssertionError, match='Existing production plan changed'):
        campaign.main()


class MemoryClient:
    def __init__(self):
        self.puts = []
        self.final = {}
    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        return 'https://offline.invalid/object'
    def put_object(self, **kwargs):
        self.puts.append(kwargs['Key'])
    def get_object(self, **kwargs):
        return {'Body': io.BytesIO(json.dumps(self.final).encode())}


@pytest.fixture
def finish_setup(tmp_path, monkeypatch):
    (tmp_path / 'frozen').mkdir()
    (tmp_path / 'plan.json').write_text(json.dumps(minimal_plan()))
    (tmp_path / 'preparation_ready.json').write_text(json.dumps(ready_payload()))
    (tmp_path / 'SUBMISSION_README.md').write_text('Offline reproducibility fixture')
    monkeypatch.setattr(finish, 'summarize', lambda root: {'all_results_verified': True})
    monkeypatch.setattr(finish, 'make_finalization_payload', lambda *args: {'offline': True})
    # These tests isolate create/UID reconciliation after the independent image gate.
    monkeypatch.setattr(finish, 'image_packaging_readiness', lambda *args: {'state': 'ready'})
    client = MemoryClient()
    monkeypatch.setattr(finish, 'aoss_client', lambda: client)
    monkeypatch.setattr(finish, 'environment', lambda: {})
    job = dict(name='offline-campaign-package', uid='original-cpu-uid', state='QUEUEING',
               ownership={'user_name': '260010165'}, resource_pool={'name': 'share-cluster'})
    monkeypatch.setattr(finish, 'cli_json', lambda args: dict(job))
    return tmp_path, client, job


def test_final_cpu_unknown_create_outcome_is_reconciled_not_resubmitted(finish_setup, monkeypatch):
    root, client, job = finish_setup
    creates = []
    def create(args, **kwargs):
        creates.append(args)
        intent = json.loads((root / 'finalization_submission.json').read_text())
        assert intent['name'] == 'offline-campaign-package'
        assert intent['gpu_count'] == 0
        raise subprocess.TimeoutExpired('offline-copy', 90)
    monkeypatch.setattr(finish.subprocess, 'run', create)
    with pytest.raises(subprocess.TimeoutExpired):
        finish.step(root)
    assert len(creates) == 1
    for _ in range(2):
        assert finish.step(root)['state'] == 'finalizing'
    assert len(creates) == 1
    assert len(client.puts) == 1
    assert json.loads((root / 'finalization_submission.json').read_text())['uid'] == job['uid']
    assert '--worker-spec=n6ls.iu.i40.8c32g' in creates[0]


def test_final_cpu_failed_attempt_is_not_retried(finish_setup, monkeypatch):
    root, client, job = finish_setup
    creates = []
    def create(args, **kwargs):
        creates.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(finish.subprocess, 'run', create)
    job['state'] = 'FAILED'
    assert finish.step(root)['state'] == 'finalization_failed'
    assert finish.step(root)['state'] == 'finalization_failed'
    assert len(creates) == 1


def test_final_cpu_receipt_must_match_run_and_frozen_inputs(finish_setup, monkeypatch):
    root, client, job = finish_setup
    monkeypatch.setattr(finish.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    job['state'] = 'SUCCEEDED'
    client.final = dict(verified=True, completed=1000, run_id='different-run',
                        source_manifest_sha256='a' * 64, checkpoint_ready_sha256='b' * 64,
                        zip_full_readback_verified=True, direct_videos_uploaded=True, state='packaged')
    with pytest.raises(AssertionError):
        finish.step(root)
    assert not (root / 'FINAL_VERIFICATION.json').exists()
    client.final['run_id'] = 'offline-campaign'
    client.final['source_manifest_sha256'] = 'wrong'
    with pytest.raises(AssertionError):
        finish.step(root)
    assert not (root / 'FINAL_VERIFICATION.json').exists()
    client.final['source_manifest_sha256'] = 'a' * 64
    assert finish.step(root)['state'] == 'complete'
    assert (root / 'FINAL_VERIFICATION.json').is_file()


def test_finalization_gate_never_creates_cpu_job_for_partial_results(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, 'summarize', lambda root: dict(all_results_verified=False, completed=999))
    def forbidden():
        raise AssertionError('Must not access cloud before all 1000 are verified')
    monkeypatch.setattr(finish, 'aoss_client', forbidden)
    assert finish.step(tmp_path) == {'state': 'incomplete_results', 'completed': 999}
    assert not (tmp_path / 'finalization_submission.json').exists()
