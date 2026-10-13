"""Reference lineage admission tests; no model, simulator, or cloud calls."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
from build_codex_subtask_trial import DRIVER, verify_reference_readiness


def write_manifest(path, data):
    data = {k: v for k, v in data.items() if k != 'fingerprint'}
    data['fingerprint'] = hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(data))
    return data


def fixture(tmp_path):
    original = tmp_path/'original'
    parent = original/'demonstrations/7003/manifest.json'
    child = tmp_path/'phase/demonstrations/7003/manifest.json'
    parent.parent.mkdir(parents=True); child.parent.mkdir(parents=True)
    sample = dict(episode_index=7003, task='camera', task_instance_id=4, subtask='pick up camera')
    metadata = dict(source_kind='original_train', source_dataset=str(original),
                    episode_index=7003, task='camera', instance_id=4)
    for path in (parent.parent, child.parent):
        (path/'example.png').write_bytes(b'verified-source-image')
    example = dict(image='example.png', image_sha256=hashlib.sha256(b'verified-source-image').hexdigest())
    base = dict(kind='codex_collection_demonstrations_v1', metadata=metadata,
                required_subtask='pick up camera', examples=[example])
    source = write_manifest(parent, base)
    ready = dict(state='complete', original_reader_verified=True, raw_success_verified=True,
                 selected_episodes=[7003], demonstrations=[dict(task='camera', manifest=str(parent),
                    fingerprint=source['fingerprint'])])
    ready_path = original/'_ORIGINAL_DATA_READY.json'
    ready_path.write_text(json.dumps(ready))
    derived = {**base, 'reference_frame_policy': 'recorded_full_close_command_v1',
        'derived_from_demonstration': dict(manifest=str(parent), fingerprint=source['fingerprint'],
            manifest_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())}
    write_manifest(child, derived)
    return parent, child, sample, ready_path


def test_original_and_derived_references_keep_original_readiness_unchanged(tmp_path):
    parent, child, sample, ready = fixture(tmp_path)
    before = ready.read_bytes()
    assert verify_reference_readiness(parent, sample)['derived_reference'] is False
    report = verify_reference_readiness(child, sample)
    assert report['derived_reference'] is True and report['original_manifest'] == str(parent)
    assert ready.read_bytes() == before


@pytest.mark.parametrize('damage', ['source', 'parent_hash', 'parent_fingerprint', 'image',
                                  'missing_parent', 'not_ready', 'episode'])
def test_derived_reference_cannot_bypass_original_source_or_ready_gate(tmp_path, damage):
    parent, child, sample, ready = fixture(tmp_path)
    data = json.loads(child.read_text())
    if damage == 'source': data['metadata']['instance_id'] = 5
    elif damage == 'parent_hash': data['derived_from_demonstration']['manifest_sha256'] = '0'*64
    elif damage == 'parent_fingerprint': data['derived_from_demonstration']['fingerprint'] = '0'*64
    elif damage == 'image': (child.parent/'example.png').write_bytes(b'changed')
    elif damage == 'missing_parent': del data['derived_from_demonstration']
    elif damage == 'not_ready':
        row = json.loads(ready.read_text()); row['raw_success_verified'] = False
        ready.write_text(json.dumps(row))
    elif damage == 'episode': sample['episode_index'] = 7004
    write_manifest(child, data)
    with pytest.raises(ValueError):
        verify_reference_readiness(child, sample)


def test_frozen_driver_contains_the_executed_reference_validator(tmp_path):
    parent, child, sample, ready = fixture(tmp_path)
    module = ast.parse(DRIVER)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'verify_reference_readiness')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-validator>', 'exec'), namespace)
    assert namespace['verify_reference_readiness'](child, sample) == verify_reference_readiness(child, sample)
