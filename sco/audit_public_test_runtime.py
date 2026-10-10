"""CPU-only read audit of frozen code/dependencies and all public scene assets."""
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def run(payload):
    code=Path(payload['code']);runtime=Path(payload['runtime']);checkpoint=Path(payload['checkpoint'])
    raw=(code/'_PUBLIC_TEST_SOURCE_READY').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==payload['source_manifest_sha256']
    provenance=json.loads(raw);scope=provenance['scope']
    assert scope['expected_rollouts']==2000 and scope['checkpoint_step']==1688977
    raw_base=(code/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(raw_base).hexdigest()==provenance['base_source_sha256']
    base=json.loads(raw_base)
    checked=[]
    for name,expected in base['overlay_files_sha256'].items():
        if name not in provenance['files_sha256']:
            assert digest(code/name)==expected
            checked.append(name)
    for name,expected in provenance['files_sha256'].items():
        assert digest(code/name)==expected;checked.append(name)
    assert (runtime/'_RUNTIME_READY').is_file()
    graphics_bytes=(runtime/'graphics-runtime/_READY').read_bytes()
    assert hashlib.sha256(graphics_bytes).hexdigest()==base['graphics_runtime_manifest_sha256']
    graphics=json.loads(graphics_bytes)
    for name,expected in graphics['libraries_sha256'].items():assert digest(Path(graphics['library_path'])/name)==expected
    library=provenance['supplemental_library'];libroot=Path(library['root'])
    libbytes=(libroot/'_READY').read_bytes()
    assert hashlib.sha256(libbytes).hexdigest()==library['ready_manifest_sha256']
    lib=json.loads(libbytes)
    for name,expected in lib['libraries_sha256'].items():assert digest(Path(lib['library_path'])/name)==expected
    assert json.loads((checkpoint/'_READY').read_text())['source_manifest_sha256']==payload['source_manifest_sha256']
    assert json.loads((checkpoint/'assets/subtask_conditioning.json').read_text())['action_conditioning']=='task_and_subtask'
    challenge=runtime/'omnigibson-data/2026-challenge-task-instances'
    with (challenge/'metadata/B100_task_misc.csv').open() as source:
        tasks={row['Task'] for row in csv.DictReader(source)}
    assert tasks==set(scope['tasks']) and len(tasks)==100
    expected={(task,i) for task in tasks for i in range(301,321)}
    found={}
    pattern=re.compile(r'_task_(.+)_0_(\d+)_template-tro_state\.json$')
    for path in (challenge/'scene_test/public').rglob('*.json'):
        match=pattern.search(path.name)
        if match:
            key=(match.group(1),int(match.group(2)))
            if key in expected:
                assert key not in found and path.stat().st_size>0
                found[key]=str(path.relative_to(challenge))
    assert set(found)==expected, f'Missing {len(expected-set(found))} public scene states'
    stats={}
    for line in (challenge/'metadata/task.jsonl').read_text().splitlines():
        if line.strip():
            row=json.loads(line);stats[row['task_name']]=row
    assert tasks <= set(stats)
    assert all(stats[task]['length']>0 for task in tasks)
    report=dict(state='passed',checkpoint_step=1688977,source_manifest_sha256=payload['source_manifest_sha256'],
        public_tasks=100,public_scene_states=2000,all_public_states_present=True,
        task_timeout_metadata_complete=True,source_files_sha256_verified=len(checked),
        graphics_library_files_verified=len(graphics['libraries_sha256']),
        supplemental_library_files_verified=len(lib['libraries_sha256']),gpu_count=0,
        real_closed_loop_validated=False,completed_at=time.time(),scene_paths=sorted(found.values()))
    if payload.get('report_put'):
        request=urllib.request.Request(payload['report_put'],data=json.dumps(report).encode(),method='PUT')
        with urllib.request.urlopen(request,timeout=30):pass
    print('STATUS_JSON='+json.dumps({key:value for key,value in report.items() if key!='scene_paths'}),flush=True)
