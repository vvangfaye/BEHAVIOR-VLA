import hashlib
import json
from pathlib import Path

import pytest

import audit_public_test_pilot as audit


def pilot_fixture(root):
    relative = 'pi05/turning_on_radio/json/turning_on_radio_301_0.json'
    result_path = root / 'results' / relative
    result_path.parent.mkdir(parents=True)
    # A zero model score is valid and must never trigger resampling.
    result = dict(task='turning_on_radio', instance_id=301, rollout_id=0, steps=10,
                  success=False, q_score={'final': 0},
                  conditioning=dict(mode='whole_task', privileged=False, oracle=False),
                  subtask_events=[dict(source='generated')])
    result_path.write_text(json.dumps(result))
    video_path = root / 'results/video.mp4'
    video_path.write_bytes(b'synthetic-hash-fixture')
    frames = []
    for n in (0, 5, 9):
        path = root / f'pilot-frame-{n}.png'
        path.write_bytes(b'\x89PNG\r\n\x1a\nsynthetic-hash-fixture')
        frames.append(dict(frame=n, path=str(path), width=640, height=480))
    payload = dict(assignment_root=str(root), source_manifest_sha256='source',
                   driver_sha256='driver', uid='synthetic-pilot', run_id='run', report_put='https://unused.test')
    done = dict(state='complete', checkpoint_step=1688977, source_manifest_sha256='source',
                task='turning_on_radio', instance_ids=[301], rollout_count=1,
                video_validation=dict(frames=10, fps=30, decoded_frames=frames))
    (root / '_PILOT_SUCCESS').write_text(json.dumps(done))
    def artifact(path):
        return dict(path=str(path.relative_to(root / 'results')), size=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(metadata=dict(checkpoint_step=1688977, source_manifest_sha256='source', run_id='run'),
        rollout_count=1, rollouts=[dict(task='turning_on_radio', instance_id=301, rollout_id=0,
                                      json=artifact(result_path), video=artifact(video_path))])
    (root / 'RESULT_MANIFEST.json').write_text(json.dumps(manifest))
    return payload, result_path, video_path


def test_zero_score_is_valid_and_original_bytes_are_retained(tmp_path, monkeypatch):
    payload, result, _ = pilot_fixture(tmp_path)
    before = result.read_bytes()
    uploaded = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def upload(request, **kwargs):
        uploaded.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr(audit.urllib.request, 'urlopen', upload)
    audit.run(payload)
    assert result.read_bytes() == before
    assert uploaded[0]['q_score'] == 0 and uploaded[0]['model_success'] is False
    assert uploaded[0]['state'] == 'verified'


def test_tampered_video_cannot_pass_artifact_verification(tmp_path):
    payload, _, video = pilot_fixture(tmp_path)
    video.write_bytes(b'tampered')
    with pytest.raises(AssertionError):
        audit.run(payload)
