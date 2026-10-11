"""Collect and verify trainable subtask rollouts on one SenseCore GPU."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


def run(payload):
    runtime, code, prepared, root = (Path(payload[k]) for k in
        ('runtime_root', 'code_root', 'prepared_root', 'output_root'))
    recovery = payload.get('recovery')
    if recovery:
        assert recovery['old_platform_state'] in ('FAILED', 'SUSPENDED', 'SUCCEEDED')
        assert recovery['old_writer_stopped'] is True and recovery['old_uid']
        assert root.is_dir()
    else:
        root.mkdir(parents=True, exist_ok=False)
    writer_lock = (root / '.collection-writer.lock').open('a')
    fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if recovery and (root/'status.json').exists():
        history=root/'execution_history';history.mkdir(exist_ok=True)
        with (history/(recovery['old_uid']+'.json')).open('x') as saved:
            saved.write((root/'status.json').read_text())
    source_bytes = (code / '_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == payload['source_manifest_sha256']
    source = json.loads(source_bytes)
    for name, expected in source['overlay_files_sha256'].items():
        assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected
    assert (runtime / '_RUNTIME_READY').is_file()
    graphics_bytes = (runtime / 'graphics-runtime/_READY').read_bytes()
    assert hashlib.sha256(graphics_bytes).hexdigest() == source['graphics_runtime_manifest_sha256']
    graphics_ready = json.loads(graphics_bytes)
    for name, expected in graphics_ready['libraries_sha256'].items():
        assert hashlib.sha256((Path(graphics_ready['library_path']) / name).read_bytes()).hexdigest() == expected
    plan = json.loads((prepared / 'collection_plan.json').read_text())
    assert plan['fingerprint'] == payload['plan_fingerprint']
    ready = json.loads((runtime / 'checkpoint/_READY').read_text())
    assert ready['step'] == plan['checkpoint_identity']['checkpoint_step'] == 1375000
    assert ready['manifest_sha256'] == plan['checkpoint_identity']['checkpoint_manifest_sha256']
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total',
                                    '--format=csv,noheader'], text=True).strip().splitlines()
    assert len(rows) == 1 and any(n in rows[0] for n in ('5090', 'N12LP'))
    driver = rows[0].split(',')[2].strip()
    libraries = [p for parent in ('/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu',
                                  '/usr/lib64', '/lib/x86_64-linux-gnu', '/lib64')
                 for p in Path(parent).glob('libEGL_nvidia.so.' + driver)]
    assert libraries
    graphics = root / 'graphics' / payload['execution_name']; graphics.mkdir(parents=True)
    vk, egl = graphics / 'nvidia_icd.json', graphics / '10_nvidia.json'
    vk.write_text(json.dumps({'file_format_version':'1.0.0', 'ICD':{
        'library_path':str(libraries[0]), 'api_version':'1.3.242'}})+'\n')
    egl.write_text(json.dumps({'file_format_version':'1.0.0', 'ICD':{'library_path':str(libraries[0])}})+'\n')
    cache = Path('/var/tmp/behavior-1k-runtime') / payload['execution_name']
    cache.mkdir(parents=True, exist_ok=False)
    assert cache.stat().st_dev != runtime.stat().st_dev and shutil.disk_usage(cache).free >= 40*1024**3
    shared = payload.get('shared_results_root')
    results = Path(shared) if shared else root / 'results'
    results.mkdir(exist_ok=bool(recovery or shared), parents=True)
    if shared:
        assert payload.get('task_name')
        locks = results / 'locks'; locks.mkdir(exist_ok=True)
        task_lock = (locks / (payload['task_name']+'.lock')).open('a')
        fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('PI05_', 'SUBTASK_', 'EXTERNAL_SUBTASK', 'PROVIDED_SUBTASK', 'GOALKEEPER_', 'B1K_LEGACY')):
            env.pop(name)
    for name in ('PYTHONHOME', 'PYTHONPATH', 'MAX_STEPS', 'INSTANCE_INDICES'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_HEADLESS='1',
        OMNIGIBSON_DISABLE_TORCH_COMPILE='1', OPENPI_DATA_HOME=str(runtime / 'openpi-cache'),
        OMNIGIBSON_DATA_PATH=str(runtime / 'omnigibson-data'),
        PI05_ENV=str(runtime / 'envs/b1k-pi05'), EVAL_ENV=str(runtime / 'envs/b1k-eval'),
        PI05_CKPT=str(runtime / 'checkpoint'), PI05_REPO_ID='b1k_full100_joint_subtask_v1_drop_episode_s1',
        SUBTASK_PREDICTION='1', MAX_TOKEN_LEN='512', MAX_SUBTASK_LEN='64', EXECUTE_HORIZON='16',
        PI05_MEM_FRACTION='0.35', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        MODEL='pi05', SERVER_GPU='0', EVAL_GPU='0', PORT='8930', MODE='train',
        PROMPT_MODE='whole_task', NUM_ROLLOUTS='1', WRITE_VIDEO='1', OUTPUT_ROOT=str(results),
        PI05_SUBTASK_SAMPLE_PANEL=str(prepared / 'parent_panel.json'),
        PI05_SUBTASK_RAW_DATA_ROOT=payload['raw_data_root'],
        PI05_SUBTASK_COLLECTION_PLAN=str(prepared / 'collection_plan.json'),
        SUBTASK_SAMPLE_MAX_STEPS=str(plan['max_steps']), SCO_EPHEMERAL_RUNTIME_ROOT=str(cache),
        EVAL_RUNTIME_ROOT=str(cache / 'sim'), VK_ICD_FILENAMES=str(vk), __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
        LD_LIBRARY_PATH=str(runtime / 'graphics-runtime/usr/lib/x86_64-linux-gnu')+':'+
            str(runtime / 'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH',''))
    if 'recorded_actions_before_start_native_grasp_recovery_v1' in plan.get('source_protocol_by_sample', {}).values():
        env['B1K_LEGACY_HISTORY_RECOVERY'] = 'recorded_actions_before_start_native_grasp_recovery_v1'
    status = dict(state='running', gpu_rows=rows, source_manifest_sha256=payload['source_manifest_sha256'],
        plan_fingerprint=plan['fingerprint'], expected_trials=plan['expected_trials'], checkpoint_step=1375000,
        output_root=str(root), started_at=time.time(), official_benchmark=False)
    status['generation']=payload.get('generation',0)

    def publish():
        status['updated_at'] = time.time()
        (root / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
        print('STATUS_JSON='+json.dumps(status), flush=True)
        if payload.get('report_put'):
            request = urllib.request.Request(payload['report_put'], data=json.dumps(status).encode(), method='PUT')
            urllib.request.urlopen(request, timeout=30).close()

    publish()
    try:
        subprocess.run(['sha256sum','-c','SHA256SUMS.inference'],cwd=runtime/'checkpoint',check=True)
        sys.path.insert(0, str(code / 'scripts'))
        from subtask_improvement import load_plan
        from subtask_sample_panel import load_panel
        from collect_b1k_subtasks import validate_collection
        panel = load_panel(prepared / 'parent_panel.json')
        load_plan(prepared / 'collection_plan.json', panel)
        contract = dict(run_id=payload['execution_name'], kind=plan['kind'], official_benchmark=False,
            plan_fingerprint=plan['fingerprint'], parent_panel_fingerprint=panel['fingerprint'],
            checkpoint_identity=plan['checkpoint_identity'], max_steps=plan['max_steps'])
        if recovery:
            previous = json.loads((results / 'collection_contract.json').read_text())
            assert previous['plan_fingerprint'] == plan['fingerprint']
            assert previous['checkpoint_identity'] == plan['checkpoint_identity']
            partial = validate_collection(results, panel, plan, require_complete=False)
            preserved = set(partial['completed_trial_ids'])
            expected = {t['trial_id'] for t in plan['trials']}
            for directory in results.glob('pi05/*/trials/*'):
                if shared and directory.parents[1].name != payload['task_name']:
                    continue
                assert directory.is_dir() and directory.name in expected
                if directory.name not in preserved:
                    target = root / 'interrupted' / payload['execution_name'] / directory.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    directory.rename(target)
            env['PI05_COLLECTION_RESUME_COMPLETED'] = '1'
            status['preserved_completed_trial_ids'] = sorted(preserved)
        elif shared:
            previous = json.loads((results / 'collection_contract.json').read_text())
            assert previous['plan_fingerprint'] == plan['fingerprint']
            assert previous['checkpoint_identity'] == plan['checkpoint_identity']
        else:
            (results / 'collection_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
        executions = root / 'source_executions'; executions.mkdir(exist_ok=True)
        record = {k:v for k,v in payload.items() if k != 'report_put'}
        with (executions / (payload['execution_name']+'.json')).open('x') as stream:
            stream.write(json.dumps(record, indent=2)+'\n')
        tasks = [payload['task_name']] if shared else sorted({g['task'] for g in plan['selected_groups']})
        assert set(tasks) <= {g['task'] for g in plan['selected_groups']}
        with (root / (payload['execution_name']+'.driver.log')).open('x', buffering=1) as log:
            subprocess.run(['bash',str(code/'scripts/eval_b1k.sh'),*tasks],env=env,cwd=code,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        summary = validate_collection(results, panel, plan, require_complete=not shared)
        if shared:
            assigned = {t['trial_id'] for t in plan['trials'] if
                        next(s for s in panel['samples'] if s['sample_id']==t['sample_id'])['task'] in tasks}
            assert assigned <= set(summary['completed_trial_ids'])
            summary.update(assigned_trial_ids=sorted(assigned), shard_complete=True)
            (root / '_SHARD_SUCCESS').write_text(json.dumps(summary, indent=2)+'\n')
        else:
            (results / 'collection_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            (results / '_SUCCESS').write_text(json.dumps(summary, indent=2)+'\n')
        status.update(collection_summary=summary, state='collection_complete')
        publish()
        if summary['ready_for_export'] and not shared:
            dataset = root / 'exports' / payload['execution_name']; dataset.parent.mkdir(exist_ok=True)
            with (root / (payload['execution_name']+'.export.log')).open('x', buffering=1) as log:
                subprocess.run([str(runtime/'envs/b1k-eval/bin/python'),str(code/'scripts/export_subtask_collection.py'),
                    '--collection-root',str(results),'--panel',str(prepared/'parent_panel.json'),
                    '--collection-plan',str(prepared/'collection_plan.json'),
                    '--output',str(dataset),'--repo-id',payload['execution_name']],
                    env=dict(env,PYTHONPATH=str(code/'BEHAVIOR-1K')+':'+str(code/'BEHAVIOR-1K/OmniGibson')),
                    cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert (dataset/'_VERIFIED_EXPORT').is_file()
            status.update(export=json.loads((dataset/'export_report.json').read_text()), dataset_root=str(dataset),
                          trainable_dataset_verified=True)
        elif not shared:
            status.update(trainable_dataset_verified=False, zero_success=True)
        else:
            status.update(global_export_pending=True)
        status.update(state='complete', completed_at=time.time())
        publish()
    except BaseException as error:
        status.update(state='failed', error_type=type(error).__name__, error=str(error))
        publish()
        raise
