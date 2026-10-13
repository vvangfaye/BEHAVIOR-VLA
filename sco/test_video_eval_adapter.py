"""Offline integration checks for paired pilot artifacts and bounded polling."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import cv2
import numpy as np
import pytest

import video_eval_adapter
from video_eval_adapter import Adapter
from video_eval_queue import Controller


def digest(data):
    return hashlib.sha256(data).hexdigest()


class MemoryObjects:
    def __init__(self):
        self.archive = b''
        self.receipt = b''
        self.downloads = 0

    def get_object(self, *, Bucket, Key):
        assert Bucket == '260010165'
        assert Key.endswith('/receipt.json')
        return {'Body': io.BytesIO(self.receipt)}

    def head_object(self, *, Bucket, Key):
        assert Bucket == '260010165'
        assert Key.endswith('/episode.tar.gz')
        return {'ContentLength': len(self.archive)}

    def download_file(self, bucket, key, destination):
        assert bucket == '260010165'
        assert key.endswith('/episode.tar.gz')
        self.downloads += 1
        Path(destination).write_bytes(self.archive)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'frozen').mkdir()
    source = b'# immutable offline worker fixture\n'
    (root / 'frozen/video_eval_worker.py').write_bytes(source)
    ready = dict(state='ready', checkpoint_step=2000000,
                 source_manifest_sha256='a' * 64, checkpoint_ready_sha256='b' * 64,
                 worker_sha256=digest(source))
    (root / 'preparation_ready.json').write_text(json.dumps(ready))
    plan = dict(object_prefix='offline/video-integration', pilot_python=sys.executable,
                immutable_inputs={key: ready[key] for key in (
                    'source_manifest_sha256', 'checkpoint_ready_sha256', 'worker_sha256')})
    client = MemoryObjects()
    monkeypatch.setattr(video_eval_adapter, 'aoss_client', lambda: client)
    adapter = Adapter(root, plan)
    record = dict(index=0, task='task_0', instance_id=301, name='offline-pilot-e0000')

    bundle = tmp_path / 'bundle'
    for directory in ('json', 'videos', 'config', 'metadata', 'wrapper'):
        (bundle / directory).mkdir(parents=True)
    result = dict(task=record['task'], instance_id=301, rollout_id=0, steps=6,
                  success=False, q_score={'final': .4}, terminated=False, truncated=True)
    (bundle / 'json/task_0301_0.json').write_text(json.dumps(result))
    (bundle / 'config/r1pro.yaml').write_text('task: offline-fixture\n')
    (bundle / 'wrapper/default_wrapper.py').write_text('# offline fixture\n')
    (bundle / 'metadata/RESULT_MANIFEST.json').write_text('{"rollout_count":1}\n')
    video = bundle / 'videos/task_0301_0.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (672, 448))
    assert writer.isOpened(), 'Offline OpenCV must support MP4 encoding'
    for index in range(result['steps']):
        frame = np.zeros((448, 672, 3), dtype=np.uint8)
        frame[:, :, index % 3] = 40 + index * 30
        writer.write(frame)
    writer.release()
    files = [dict(path=p.relative_to(bundle).as_posix(), bytes=p.stat().st_size,
                  sha256=digest(p.read_bytes())) for p in sorted(bundle.rglob('*')) if p.is_file()]
    report = dict(checkpoint_step=2000000, task=record['task'], instance_id=301,
                  run_id=record['name'], execution_name=record['name'],
                  source_manifest_sha256=ready['source_manifest_sha256'],
                  checkpoint_ready_sha256=ready['checkpoint_ready_sha256'],
                  state='artifacts_verified', verified=False, model_success=False,
                  q_score=.4, steps=result['steps'], files=files,
                  full_video_decode_verified=True, archive_readback_verified=True,
                  video_validation={'frames_equal_simulator_steps': True})
    # Mirror worker packaging: receipt paths are relative to bundle, while the
    # archive's one top-level directory is named "episode".
    (bundle / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(report))
    archive = tmp_path / 'artifact.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        output.add(bundle, arcname='episode', recursive=True)
    client.archive = archive.read_bytes()
    report.update(state='complete', verified=True,
                  archive={'bytes': len(client.archive), 'sha256': digest(client.archive)})
    client.receipt = json.dumps(report).encode()
    return root, adapter, client, record, report, ready, plan


def test_worker_archive_prefix_and_native_mp4_pass_full_pilot_verification(setup):
    root, adapter, client, record, report, _, _ = setup
    assert all(not row['path'].startswith('episode/') for row in report['files'])
    result = adapter.inspect(record)
    assert result['verified'] is True
    assert result['model_success'] is False  # retain unsuccessful model episodes too
    accepted = json.loads((root / 'pilot_accepted.json').read_text())
    assert accepted['full_decode'] == {'frames': 6}
    assert accepted['archive_sha256'] == report['archive']['sha256']
    assert accepted['canonical_result_kept_regardless_of_score'] is True
    assert (root / 'pilot_artifacts/episode/videos/task_0301_0.mp4').is_file()
    assert client.downloads == 1
    # A repeated offline verification may reuse only the same hashed archive.
    adapter.verify_pilot_archive(record, report, adapter.keys(record)['archive'])
    assert client.downloads == 1


def test_archive_digest_mismatch_does_not_accept_pilot(setup):
    root, adapter, client, record, report, _, _ = setup
    report['archive']['sha256'] = '0' * 64
    with pytest.raises(AssertionError):
        adapter.verify_pilot_archive(record, report, adapter.keys(record)['archive'])
    assert not (root / 'pilot_accepted.json').exists()


def test_archive_member_digest_mismatch_does_not_accept_pilot(setup):
    root, adapter, client, record, report, _, _ = setup
    report['files'][0]['sha256'] = '0' * 64
    with pytest.raises(AssertionError):
        adapter.verify_pilot_archive(record, report, adapter.keys(record)['archive'])
    assert not (root / 'pilot_accepted.json').exists()


def test_video_frame_mismatch_does_not_accept_pilot(setup):
    root, adapter, client, record, report, _, _ = setup
    report['steps'] += 1
    with pytest.raises(AssertionError, match='Independent pilot decode failed'):
        adapter.verify_pilot_archive(record, report, adapter.keys(record)['archive'])
    assert not (root / 'pilot_accepted.json').exists()


def test_mismatched_receipt_identity_rejected_before_pilot_acceptance(setup):
    root, adapter, client, record, report, _, _ = setup
    report['instance_id'] = 302
    client.receipt = json.dumps(report).encode()
    with pytest.raises(AssertionError, match='instance_id'):
        adapter.inspect(record)
    assert client.downloads == 0
    assert not (root / 'pilot_accepted.json').exists()


@pytest.mark.parametrize('field', ['source_manifest_sha256', 'checkpoint_ready_sha256', 'worker_sha256'])
def test_changed_immutable_input_rejected_on_restart(setup, field):
    root, adapter, client, record, report, ready, plan = setup
    changed = copy.deepcopy(plan)
    changed['immutable_inputs'][field] = 'e' * 64
    with pytest.raises(AssertionError, match=field):
        Adapter(root, changed)
    assert client.downloads == 0


def test_changed_worker_bytes_rejected_even_if_plan_not_changed(setup):
    root, adapter, client, record, report, ready, plan = setup
    (root / 'frozen/video_eval_worker.py').write_text('# different worker\n')
    with pytest.raises(AssertionError):
        Adapter(root, plan)
    assert client.downloads == 0


def test_round_query_error_keeps_slot_and_forbids_topup(tmp_path):
    plan = dict(checkpoint_step=2000000, run_id='offline-queue-test', episodes=[
        dict(index=t * 10 + i, task=f'task_{t}', instance_id=301 + i)
        for t in range(100) for i in range(10)])
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    adapter = Adapter.__new__(Adapter)
    adapter.root = tmp_path
    jobs, creates = {}, []
    fail_names = set()
    def create(record):
        creates.append(record['index'])
        job = dict(name=record['name'], uid=f"uid-{record['index']}", state='QUEUEING')
        jobs[record['name']] = job
        return job
    def describe(name):
        if name in fail_names:
            raise TimeoutError('private-placeholder')
        return jobs[name]
    adapter.create = create
    adapter._describe_live = describe
    adapter._receipt_live = lambda record: None
    adapter.inspect = lambda record: {'verified': True, 'model_success': False} if record['index'] == 0 else None
    controller = Controller(tmp_path, adapter)
    assert controller.run_once()['submitted_this_pass'] == [0]
    jobs[controller._name(0)]['state'] = 'SUCCEEDED'
    status = controller.run_once()
    assert status['submitted_this_pass'] == [1, 2, 3, 4, 5]
    assert status['pending_gpus'] == 5
    fail_names.add(controller._name(3))
    before = list(creates)
    status = controller.run_once()
    assert not status['query_complete']
    assert status['pending_gpus'] == 5  # four queued plus the one unknown
    assert status['states']['UNKNOWN'] == 1
    assert status['states']['QUEUEING'] == 4
    assert creates == before
    assert status['submitted_this_pass'] == []
    assert 'private-placeholder' not in (tmp_path / 'submissions/0003.json').read_text()
