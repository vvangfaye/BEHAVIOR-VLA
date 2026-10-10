"""Create and test a new source snapshot while preserving its parent and queues."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request


def run(payload):
    parent = Path(payload['parent_code_root'])
    code = Path(payload['code_root'])
    root = Path(payload['report_root'])
    root.mkdir(exist_ok=False, parents=True)
    log = (root / 'driver.log').open('x', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    parent_bytes = (parent / '_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(parent_bytes).hexdigest() == payload['parent_source_manifest_sha256']
    parent_source = json.loads(parent_bytes)
    status = {'state': 'running', 'gpu_count': 0, 'started_at': time.time(),
              'parent_source_manifest_sha256': payload['parent_source_manifest_sha256'],
              'code_root': str(code), 'run_root': payload['run_root']}

    def publish(stage, required=False):
        status.update(stage=stage, updated_at=time.time())
        (root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        for attempt in range(4 if required else 1):
            try:
                request = urllib.request.Request(payload['report_put'],
                    data=json.dumps(status).encode(), method='PUT')
                with urllib.request.urlopen(request, timeout=20):
                    pass
                return
            except Exception as error:
                status['last_report_upload_error_type'] = type(error).__name__
                if required:
                    time.sleep(min(2**attempt, 4))
        if required:
            raise RuntimeError('Bounded final source status upload failed')

    try:
        assert not code.exists(), 'A repair needs a fresh immutable source directory'
        assert not (parent / 'nebula/config.sh').exists()
        for relative, expected in parent_source['overlay_files_sha256'].items():
            assert hashlib.sha256((parent / relative).read_bytes()).hexdigest() == expected
        publish('copying_parent_source')
        shutil.copytree(parent, code, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        blob = base64.b64decode(payload['overlay'])
        assert hashlib.sha256(blob).hexdigest() == payload['overlay_sha256']
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
            for member in archive.getmembers():
                relative = Path(member.name)
                assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
                assert str(relative) in payload['overlay_files_sha256']
                assert relative.name != 'config.sh' and '.private.' not in str(relative)
                target = code / relative
                assert target.resolve().is_relative_to(code.resolve()), 'Overlay must stay inside the new source snapshot'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                target.chmod(member.mode & 0o777)
        source = dict(parent_source)
        source.update(code_root=str(code), created_at=time.time(),
            parent_code_root=str(parent),
            parent_source_manifest_sha256=payload['parent_source_manifest_sha256'],
            repair_reason=payload['repair_reason'],
            repair_overlay_sha256=payload['overlay_sha256'],
            repair_files_sha256=payload['overlay_files_sha256'])
        source['overlay_files_sha256'] = dict(parent_source['overlay_files_sha256'])
        source['overlay_files_sha256'].update(payload['overlay_files_sha256'])
        for relative, expected in source['overlay_files_sha256'].items():
            assert hashlib.sha256((code / relative).read_bytes()).hexdigest() == expected
        publish('real_serializer_cpu_regression')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
            OMNIGIBSON_NO_OMNIVERSE='1', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_DISABLE_TORCH_COMPILE='1',
            LD_LIBRARY_PATH=str(Path(payload['runtime_root']) / 'envs/b1k-eval/lib') +
            ':' + environment.get('LD_LIBRARY_PATH', ''))
        interpreter = Path(payload['runtime_root']) / 'envs/b1k-eval/bin/python'
        subprocess.run(['bash', '-n', str(code / 'scripts/eval_b1k.sh')], check=True)
        subprocess.run([str(interpreter), '-m', 'py_compile',
            str(code / 'sco/run_full100_assignment.py'),
            str(code / 'sco/run_subtask_validation.py')],
            cwd=code, env=environment, check=True)
        subprocess.run([str(interpreter), '-m', 'pytest',
            'scripts/test_materialized_state.py', 'scripts/test_recorded_native_state.py',
            *payload.get('additional_cpu_tests', []),
            '-q'], cwd=code, env=environment, check=True)
        if payload.get('source_role_probe_code'):
            probe = payload['source_role_probe_code'].replace('PLACEHOLDER_CODE_ROOT', str(code))
            result = subprocess.run([str(interpreter), '-c', probe],
                env=environment, check=True, capture_output=True, text=True, timeout=600)
            status['original_raw_source_role_probe'] = json.loads(result.stdout.strip().splitlines()[-1])
        if payload.get('native_validation_sample_ids'):
            native = "import json,sys;from pathlib import Path\n"
            native += "sys.path.insert(0," + repr(str(code / 'scripts')) + ")\n"
            native += "import eval_b1k_subtask_samples as r\nfrom recorded_native_state import decode_start,preview_source_role_bindings\n"
            native += "p=r.load_panel(" + repr(str(Path(payload['run_root']) / 'panel.json')) + ")\n"
            native += "ids=" + repr(payload['native_validation_sample_ids']) + "\n"
            native += "by_id={s['sample_id']:s for s in p['samples']};samples=[by_id[i] for i in ids]\n"
            native += "cache=r.load_episode_cache(r.raw_hdf5_path(" + repr(payload['raw_root']) + ",samples[0]),samples)\n"
            native += "for s in samples:\n"
            native += " proof=decode_start(cache,s);binding=preview_source_role_bindings(cache,proof,s['predicate'])\n"
            native += " print(json.dumps({'sample_id':s['sample_id'],'raw_sha256':cache['sha256'],'selected_state_sha256':proof['selected_state_sha256'],'binding':binding}))\n"
            subprocess.run([str(interpreter), '-c', native], env=environment, check=True)
            status['original_transition_native_source_decode_passed'] = True
        sys_path = str(code / 'sco')
        inspect = "import sys;sys.path.insert(0," + repr(sys_path) + ")\n"
        inspect += "import full100_shards as s\nfrom pathlib import Path\n"
        inspect += "p,a,m,q=s.load_assignment(" + repr(payload['run_root']) + "," + str(payload.get('repair_assignment_index', 3)) + ")\n"
        inspect += "r=Path(m['queue_db']).parent\nassert not s.runner.dynamic.supervisor_is_active(r)\n"
        if payload.get('completed_assignment_readonly_invariants'):
            inspect += "import hashlib,json\n"
            inspect += "names=['manifest.json','queue.sqlite3','_ASSIGNED_SUCCESS','result_manifest.json']\n"
            inspect += "before_files={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}\n"
            inspect += "assert all(j['state']=='succeeded' for j in q.snapshot(readonly=True)['jobs'])\n"
            inspect += "marker=json.loads((r/'_ASSIGNED_SUCCESS').read_text())\n"
            inspect += "assert marker['assignment_index']==a['index'] and marker['sample_ids']==a['sample_ids']\n"
            inspect += "assert marker['panel_fingerprint']==p['fingerprint'] and marker['complete_full100'] is False\n"
            inspect += "assert marker['result_manifest_sha256']==s.runner.dynamic._fingerprint(json.loads((r/'result_manifest.json').read_text()))\n"
        else:
            inspect += "assert not (r/'_ASSIGNED_SUCCESS').exists()\n"
        inspect += "assert not any(j['state']=='exhausted' for j in q.snapshot(readonly=True)['jobs'])\n"
        if payload.get('validate_ephemeral_runtime_contract'):
            inspect += "import os\nbefore=(r/'manifest.json').read_bytes()\n"
            inspect += "os.environ['SCO_EPHEMERAL_RUNTIME_ROOT']='/var/tmp/behavior-1k-runtime/cpu-storage-contract'\n"
            inspect += "spec=s.runner.worker_specs(m['gpus'],m['base_port'],tuple(m['tasks']))[0]\n"
            inspect += "env=s.runner.child_environment(m,spec,0)\n"
            inspect += "assert env['SCO_EPHEMERAL_RUNTIME_ROOT']==os.environ['SCO_EPHEMERAL_RUNTIME_ROOT']\n"
            inspect += "assert env['OUTPUT_ROOT']==m['results_root'] and env['PI05_EVAL_QUEUE_DB']==m['queue_db']\n"
            inspect += "assert (r/'manifest.json').read_bytes()==before\n"
        if payload.get('completed_assignment_readonly_invariants'):
            inspect += "assert before_files=={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names}\n"
        subprocess.run([str(interpreter), '-c', inspect], env=environment, check=True)
        if payload.get('validate_ephemeral_runtime_contract'):
            status['ephemeral_runtime_contract_checked_against_original_queue'] = True
        assert (parent / '_CODE_SNAPSHOT_READY').read_bytes() == parent_bytes
        for relative, expected in parent_source['overlay_files_sha256'].items():
            assert hashlib.sha256((parent / relative).read_bytes()).hexdigest() == expected
        source_bytes = (json.dumps(source, indent=2) + '\n').encode()
        (code / '_CODE_SNAPSHOT_READY').write_bytes(source_bytes)
        status.update(state='complete', completed_at=time.time(),
            source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
            real_serializer_regression_passed=True, parent_unchanged=True,
            original_queue_manifests_unchanged=True)
        publish('source_repair_ready', required=True)
    except BaseException as error:
        traceback.print_exc()
        status.update(state='failed', error_type=type(error).__name__)
        publish('failed')
        raise
