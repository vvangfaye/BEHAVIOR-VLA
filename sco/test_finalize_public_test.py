import hashlib
import json
from pathlib import Path

import pytest

from finalize_public_test import run


@pytest.fixture
def complete_fixture(tmp_path):
    """Synthetic result files test integrity checks, not simulator/model behavior."""
    code = tmp_path / 'code'
    official_name = 'BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/score_utils.py'
    official = (Path(__file__).resolve().parents[1] / official_name).read_bytes()
    (code / official_name).parent.mkdir(parents=True)
    (code / official_name).write_bytes(official)
    tasks = [f'task_{n}' for n in range(100)]
    scope = dict(checkpoint_step=1688977, expected_rollouts=2000, tasks=tasks)
    provenance = dict(scope=scope,files_sha256={official_name:hashlib.sha256(official).hexdigest()})
    raw = json.dumps(provenance).encode()
    (code / '_PUBLIC_TEST_SOURCE_READY').write_bytes(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    assignments = []
    for task in tasks:
        root = tmp_path / task
        ids = list(range(301,321))
        result_dir = root / 'results'
        result_dir.mkdir(parents=True)
        rows = []
        for i in ids:
            value = dict(task=task, instance_id=i, rollout_id=0, steps=100, success=i % 2 == 0,
                conditioning=dict(mode='whole_task',privileged=False,oracle=False),
                subtask_events=[dict(step=0,source='generated',subtask='fixture action')],
                q_score=dict(final=1.0 if i % 2 == 0 else 0.25),
                time=dict(normalized_time=1.0), normalized_agent_distance=dict(base=1.0,left=1.0,right=1.0))
            raw = json.dumps(value).encode()
            path = f'{task}_{i}_0.json'
            (result_dir / path).write_bytes(raw)
            rows.append(dict(task=task,instance_id=i,rollout_id=0,
                json=dict(path=path,size=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
        metadata = dict(checkpoint_step=1688977,source_manifest_sha256=source_hash)
        (root / 'RESULT_MANIFEST.json').write_text(json.dumps(dict(metadata=metadata,rollout_count=20,rollouts=rows)))
        (root / '_ASSIGNED_SUCCESS').write_text(json.dumps(dict(state='complete',task=task,instance_ids=ids,**metadata)))
        assignments.append(dict(root=str(root),pilot=False,task=task,instance_ids=ids))
    return dict(code=str(code),scope=scope,source_manifest_sha256=source_hash,assignments=assignments,
        destination=str(tmp_path/'standard.public.fixture.fixture.20261010'))


def test_full_results_use_official_scorer(complete_fixture):
    report = run(complete_fixture)
    assert report['scores']['overall_scores']['q_score'] == 0.625
    assert report['scores']['overall_scores']['task_sr'] == 0.5
    assert len(report['files']) == 2000


def test_missing_episode_cannot_become_implicit_zero(complete_fixture):
    complete_fixture['assignments'].pop()
    with pytest.raises(AssertionError, match='incomplete'):
        run(complete_fixture)
    assert not Path(complete_fixture['destination']).exists()


def test_different_checkpoint_rejected(complete_fixture):
    root = Path(complete_fixture['assignments'][0]['root'])
    path = root / '_ASSIGNED_SUCCESS'
    value = json.loads(path.read_text())
    value['checkpoint_step'] = 1375000
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        run(complete_fixture)
    assert not Path(complete_fixture['destination']).exists()


def test_duplicate_episode_rejected(complete_fixture):
    complete_fixture['assignments'].append(complete_fixture['assignments'][0])
    with pytest.raises(AssertionError, match='Duplicate'):
        run(complete_fixture)
    assert not Path(complete_fixture['destination']).exists()
