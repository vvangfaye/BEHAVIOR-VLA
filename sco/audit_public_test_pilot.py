"""Read and verify completed pilot artifacts without changing evaluation results."""
import base64
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def run(payload):
    root = Path(payload['assignment_root']).resolve()
    done = json.loads((root / '_PILOT_SUCCESS').read_text())
    assert done['state'] == 'complete' and done['checkpoint_step'] == 1688977
    assert done['source_manifest_sha256'] == payload['source_manifest_sha256']
    assert done['task'] == 'turning_on_radio' and done['instance_ids'] == [301]
    assert done['rollout_count'] == 1
    manifest_raw = (root / 'RESULT_MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest['metadata']['checkpoint_step'] == 1688977
    assert manifest['metadata']['source_manifest_sha256'] == payload['source_manifest_sha256']
    assert manifest['metadata']['run_id'] == payload['run_id']
    assert manifest['rollout_count'] == 1 and len(manifest['rollouts']) == 1
    row = manifest['rollouts'][0]
    assert (row['task'], row['instance_id'], row['rollout_id']) == ('turning_on_radio', 301, 0)
    checked = {}
    for kind in ('json', 'video'):
        item = row[kind]
        path = (root / 'results' / item['path']).resolve()
        assert root in path.parents and path.is_file()
        assert path.stat().st_size == item['size'] and digest(path) == item['sha256']
        checked[kind] = item
    raw = (root / 'results' / row['json']['path']).read_bytes()
    result = json.loads(raw)
    assert (result['task'], result['instance_id'], result['rollout_id']) == ('turning_on_radio', 301, 0)
    assert result['steps'] > 0 and isinstance(result['success'], bool)
    q = result['q_score']['final']
    assert isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) and 0 <= q <= 1
    assert result['conditioning']['mode'] == 'whole_task'
    assert result['conditioning']['privileged'] is False and result['conditioning']['oracle'] is False
    events = result['subtask_events']
    assert any(event['source'] == 'generated' for event in events)
    assert all(event['source'] in ('generated', 'cached') for event in events)
    video = done['video_validation']
    assert video['frames'] > 1 and video['fps'] > 0
    assert [x['frame'] for x in video['decoded_frames']] == [0, video['frames'] // 2, video['frames'] - 1]
    frames = []
    for item in video['decoded_frames']:
        path = Path(item['path']).resolve()
        assert path.parent == root and item['width'] > 0 and item['height'] > 0
        pixels = path.read_bytes()
        assert pixels.startswith(b'\x89PNG\r\n\x1a\n') and len(pixels) > 8
        frames.append(dict(name=path.name, sha256=hashlib.sha256(pixels).hexdigest(),
                           bytes=len(pixels), data_base64=base64.b64encode(pixels).decode()))
    report = dict(state='verified', checkpoint_step=1688977,
        source_manifest_sha256=payload['source_manifest_sha256'], driver_sha256=payload['driver_sha256'],
        pilot_uid=payload['uid'], completed_at=time.time(), full_artifact_sha256_verified=True,
        task='turning_on_radio', instance_ids=[301], whole_task_native_generated_subtasks=True,
        canonical_result_kept_regardless_of_score=True, gpu_count=0,
        model_steps=result['steps'], model_success=result['success'], q_score=q,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), checked_artifacts=checked,
        result_json_base64=base64.b64encode(raw).decode(), video_validation=video, frames=frames)
    request = urllib.request.Request(payload['report_put'], data=json.dumps(report).encode(), method='PUT')
    with urllib.request.urlopen(request, timeout=60):
        pass
    print('STATUS_JSON=' + json.dumps({k: v for k, v in report.items()
        if k not in ('result_json_base64', 'frames', 'video_validation')}), flush=True)
