"""Strictly assemble complete whole-task test results and run the frozen official scorer.

Run only after every assigned shard and the fixed pilot have been accepted.
Missing test cases never become implicit zero-score entries.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path


def run(payload):
    scope = payload['scope']
    assert scope['checkpoint_step'] == 1688977 and scope['expected_rollouts'] == 2000
    code = Path(payload['code'])
    provenance_bytes = (code / '_PUBLIC_TEST_SOURCE_READY').read_bytes()
    assert hashlib.sha256(provenance_bytes).hexdigest() == payload['source_manifest_sha256']
    provenance = json.loads(provenance_bytes)
    assert provenance['scope'] == scope, 'Final scope must equal the frozen preparation scope'
    expected = {(task, i, 0) for task in scope['tasks'] for i in range(301, 321)}
    assert len(expected) == 2000
    records = {}
    for assignment in payload['assignments']:
        root = Path(assignment['root'])
        complete = json.loads((root / ('_PILOT_SUCCESS' if assignment['pilot'] else '_ASSIGNED_SUCCESS')).read_text())
        assert complete['state'] == 'complete' and complete['checkpoint_step'] == 1688977
        assert complete['source_manifest_sha256'] == payload['source_manifest_sha256']
        assert complete['instance_ids'] == assignment['instance_ids']
        assert complete['task'] == assignment['task']
        manifest = json.loads((root / 'RESULT_MANIFEST.json').read_text())
        assert manifest['metadata']['checkpoint_step'] == 1688977
        assert manifest['metadata']['source_manifest_sha256'] == payload['source_manifest_sha256']
        assert manifest['rollout_count'] == len(assignment['instance_ids'])
        assert {(row['task'],row['instance_id'],row['rollout_id']) for row in manifest['rollouts']} == {
            (assignment['task'],i,0) for i in assignment['instance_ids']}
        for row in manifest['rollouts']:
            key = (row['task'], row['instance_id'], row['rollout_id'])
            assert key in expected and key not in records, 'Duplicate or out-of-scope test episode'
            source = root / 'results' / row['json']['path']
            raw = source.read_bytes()
            assert len(raw) == row['json']['size'] and hashlib.sha256(raw).hexdigest() == row['json']['sha256']
            result = json.loads(raw)
            assert (result['task'], result['instance_id'], result['rollout_id']) == key
            assert result['steps'] > 0 and isinstance(result['success'], bool)
            conditioning = result['conditioning']
            assert conditioning['mode'] == 'whole_task' and conditioning['privileged'] is False
            assert conditioning['oracle'] is False
            assert any(event['source'] == 'generated' for event in result['subtask_events'])
            assert all(event['source'] in ('generated','cached') for event in result['subtask_events'])
            q = result['q_score']['final']
            assert isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) and 0 <= q <= 1
            assert math.isfinite(result['time']['normalized_time']) and result['time']['normalized_time'] > 0
            assert all(math.isfinite(result['normalized_agent_distance'][side]) for side in ('base','left','right'))
            records[key] = (source, row['json']['sha256'], q)
    assert set(records) == expected, f'Full test set is incomplete: {len(expected-set(records))} missing'
    destination = Path(payload['destination'])
    assert not destination.exists(), 'Preserve existing final packages; do not overwrite'
    track, split, team, affiliation, date = destination.name.split('.')
    assert track == 'standard' and split == 'public' and all((team, affiliation, date))
    (destination / 'json').mkdir(parents=True)
    for (task, instance, rollout), (source, expected_sha, _) in sorted(records.items()):
        target = destination / 'json' / f'{task}_{instance}_{rollout}.json'
        digest = hashlib.sha256()
        with source.open('rb') as input_stream, target.open('xb') as output_stream:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b''):
                digest.update(chunk)
                output_stream.write(chunk)
        assert digest.hexdigest() == expected_sha, 'Original result changed during final assembly'
    official_name = 'BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/score_utils.py'
    official = (code / official_name).read_bytes()
    assert hashlib.sha256(official).hexdigest() == provenance['files_sha256'][official_name]
    tree = ast.parse(official)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compute_final_q_score']
    assert len(functions) == 1
    # Load the unmodified official scoring function without initializing a simulator on the CPU worker.
    namespace = dict(os=os, json=json, TASK_NAMES_TO_INDICES={task:n for n,task in enumerate(scope['tasks'])},
        TEST_INSTANCE_IDS=list(range(301,341)), NUM_PUBLIC_TEST_INSTANCES=20,
        NUM_HIDDEN_TEST_INSTANCES=20, EVAL_TIMEOUT_MULTIPLIER=1.5)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(code/official_name),'exec'),namespace)
    score_dir = destination / 'scores'
    score_dir.mkdir()
    namespace['compute_final_q_score'](str(destination), str(score_dir), final_score_only=True)
    score_files = list(score_dir.glob('*.json'))
    assert len(score_files) == 1
    scores = json.loads(score_files[0].read_text())
    mean_q = sum(row[2] for row in records.values()) / 2000
    success_rate = sum(row[2] == 1 for row in records.values()) / 2000
    assert abs(scores['overall_scores']['q_score'] - mean_q) < 1e-12
    assert abs(scores['overall_scores']['task_sr'] - success_rate) < 1e-12
    report = dict(state='complete', checkpoint_step=1688977, public_test_rollouts=2000,
        tasks=100, source_manifest_sha256=payload['source_manifest_sha256'],
        official_scorer_sha256=hashlib.sha256(official).hexdigest(), scores=scores,
        files=[dict(task=k[0],instance_id=k[1],rollout_id=k[2],sha256=v[1]) for k,v in sorted(records.items())])
    (destination / 'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    (destination / '_SUCCESS').write_text(json.dumps(dict(checkpoint_step=1688977,rollouts=2000,q_score=mean_q,task_sr=success_rate))+'\n')
    return report
