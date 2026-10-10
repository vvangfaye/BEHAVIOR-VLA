"""Independently read back a completed production shard, preserving zero scores."""
import base64
import hashlib
import json
import math
from pathlib import Path
import urllib.request


def inspect(payload):
    root = Path(payload['assignment_root']).resolve()
    done_raw = (root / '_ASSIGNED_SUCCESS').read_bytes()
    done = json.loads(done_raw)
    assert done['state'] == 'complete' and done['checkpoint_step'] == 1688977
    assert done['source_manifest_sha256'] == payload['source_manifest_sha256']
    assert done['task'] == payload['task'] and done['instance_ids'] == payload['instance_ids']
    expected = {(payload['task'], i, 0) for i in payload['instance_ids']}
    assert len(expected) == len(payload['instance_ids']) == done['rollout_count']
    assert all(301 <= i <= 320 for i in payload['instance_ids'])
    manifest_raw = (root / 'RESULT_MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest['metadata']['checkpoint_step'] == 1688977
    assert manifest['metadata']['source_manifest_sha256'] == payload['source_manifest_sha256']
    assert manifest['metadata']['run_id'] == payload['run_id']
    keys = [(r['task'], r['instance_id'], r['rollout_id']) for r in manifest['rollouts']]
    assert len(keys) == len(expected) == manifest['rollout_count'] and set(keys) == expected
    results = []
    for row, key in zip(manifest['rollouts'], keys):
        path = (root / 'results' / row['json']['path']).resolve()
        assert root / 'results' in path.parents and path.is_file()
        raw = path.read_bytes()
        assert len(raw) == row['json']['size'] and hashlib.sha256(raw).hexdigest() == row['json']['sha256']
        result = json.loads(raw)
        assert (result['task'], result['instance_id'], result['rollout_id']) == key
        assert result['steps'] > 0 and isinstance(result['success'], bool)
        q = result['q_score']['final']
        assert isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) and 0 <= q <= 1
        assert result['conditioning']['mode'] == 'whole_task'
        assert result['conditioning']['privileged'] is False and result['conditioning']['oracle'] is False
        events = result['subtask_events']
        assert any(event['source'] == 'generated' for event in events)
        assert all(event['source'] in ('generated', 'cached') for event in events)
        assert math.isfinite(result['time']['normalized_time']) and result['time']['normalized_time'] > 0
        assert all(math.isfinite(result['normalized_agent_distance'][side]) for side in ('base', 'left', 'right'))
        results.append(dict(task=key[0], instance_id=key[1], rollout_id=key[2],
            model_steps=result['steps'], model_success=result['success'], q_score=q,
            json=row['json'], result_json_base64=base64.b64encode(raw).decode()))
    return dict(state='verified', checkpoint_step=1688977, gpu_count=0,
        source_manifest_sha256=payload['source_manifest_sha256'], driver_sha256=payload['driver_sha256'],
        job_uid=payload['uid'], task=payload['task'], instance_ids=payload['instance_ids'],
        run_id=payload['run_id'], rollout_count=len(results), full_artifact_sha256_verified=True,
        whole_task_native_generated_subtasks=True, canonical_results_kept_regardless_of_score=True,
        marker_sha256=hashlib.sha256(done_raw).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), results=results)


def run(payload):
    report = inspect(payload)
    request = urllib.request.Request(payload['report_put'], data=json.dumps(report).encode(), method='PUT')
    with urllib.request.urlopen(request, timeout=60):
        pass
    print('STATUS_JSON=' + json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
