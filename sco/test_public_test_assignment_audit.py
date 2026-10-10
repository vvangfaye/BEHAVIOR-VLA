import base64
import hashlib
import json
from pathlib import Path

import pytest

from audit_public_test_assignment import inspect
from public_test_artifacts import validate_report
from test_finalize_public_test import complete_fixture


def test_zero_score_is_verified_and_preserved(complete_fixture):
    """Synthetic files exercise audit fidelity, not model success."""
    assignment = complete_fixture['assignments'][0]
    root = Path(assignment['root'])
    manifest_path = root / 'RESULT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['metadata']['run_id'] = 'fixture-a000'
    row = manifest['rollouts'][0]
    result_path = root / 'results' / row['json']['path']
    result = json.loads(result_path.read_bytes())
    result['q_score']['final'] = 0.0
    raw = json.dumps(result).encode()
    result_path.write_bytes(raw)
    row['json'].update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    marker_path = root / '_ASSIGNED_SUCCESS'
    marker = json.loads(marker_path.read_text())
    marker['rollout_count'] = 20
    marker_path.write_text(json.dumps(marker))
    payload = dict(assignment_root=str(root), task=assignment['task'], instance_ids=assignment['instance_ids'],
        source_manifest_sha256=complete_fixture['source_manifest_sha256'], run_id='fixture-a000',
        uid='fixture', driver_sha256='fixture')
    report = inspect(payload)
    assert report['rollout_count'] == 20 and report['results'][0]['q_score'] == 0
    assert base64.b64decode(report['results'][0]['result_json_base64']) == raw
    submission = {**payload, 'uid': payload['uid']}
    originals = validate_report(report, submission)
    assert originals[result_path.name] == raw
    with pytest.raises(AssertionError, match='exact submitted shard'):
        validate_report({**report, 'source_manifest_sha256': 'another-checkpoint-source'}, submission)
    with pytest.raises(AssertionError):
        validate_report({**report, 'job_uid': 'another-job'}, submission)
    with pytest.raises(AssertionError):
        validate_report({**report, 'results': report['results'][:-1]}, submission)
    result_path.write_bytes(raw + b' ')
    with pytest.raises(AssertionError):
        inspect(payload)
