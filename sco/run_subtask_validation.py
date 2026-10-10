"""Run one-GPU real model validation through the existing strict evaluator."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request


def run(payload):
    root = Path(payload['runtime_root'])
    assert (root / '_RUNTIME_READY').is_file(), 'CPU restoration is incomplete'
    assert (root / 'graphics-runtime/_READY').is_file(), 'GL loader restoration is incomplete'
    graphics_bytes = (root / 'graphics-runtime/_READY').read_bytes()
    graphics_ready = json.loads(graphics_bytes)
    for name, expected in graphics_ready['libraries_sha256'].items():
        assert hashlib.sha256((Path(graphics_ready['library_path']) / name).read_bytes()).hexdigest() == expected
    raw = Path(payload['raw_root'])
    assert (raw / '_DOWNLOAD_COMPLETE').is_file(), 'Frozen raw panel is incomplete'
    output = Path(payload['output_root'])
    assert not output.exists(), 'Validation requires a fresh output root'
    output.mkdir(parents=True)
    log = (output / 'driver.log').open('x', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    code = Path(payload.get('code_root', root / 'code'))
    if payload.get('code_root'):
        source_bytes = (code / '_CODE_SNAPSHOT_READY').read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == payload['source_manifest_sha256']
        for relative, expected in json.loads(source_bytes)['overlay_files_sha256'].items():
            assert hashlib.sha256((code / relative).read_bytes()).hexdigest() == expected
    evaluation = root / 'envs/b1k-eval'
    policy = root / 'envs/b1k-pi05'
    status = {'state': 'preflight', 'gpu_count': 1, 'started_at': time.time(),
              'runtime_root': str(root), 'raw_root': str(raw)}
    ready = json.loads((root / 'checkpoint/_READY').read_text())
    status.update(checkpoint_step=ready['step'], checkpoint_manifest_sha256=ready['manifest_sha256'])
    status['graphics_runtime_manifest_sha256'] = hashlib.sha256(graphics_bytes).hexdigest()
    if payload.get('code_root'):
        status.update(code_root=str(code), source_manifest_sha256=payload['source_manifest_sha256'])
    status_path = output / 'job_status.json'

    def publish():
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        if payload.get('report_put'):
            report = dict(status)
            try:
                request = urllib.request.Request(payload['report_put'],
                    data=json.dumps(report).encode(), method='PUT')
                with urllib.request.urlopen(request, timeout=30):
                    pass
            except Exception:
                print('validation_report_upload_failed', flush=True)

    publish()

    def command(args, **kwargs):
        return subprocess.run([str(v) for v in args], check=True, **kwargs)

    try:
        rows = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip().splitlines()
        assert len(rows) == 1 and any(name in rows[0] for name in ('5090', 'N12LP')), rows
        print('allocated_gpu', rows, flush=True)
        command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                 '--format=csv,noheader,nounits'])
        driver = rows[0].split(',')[2].strip()
        candidates = [p for parent in ['/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu',
                      '/usr/lib64', '/lib/x86_64-linux-gnu', '/lib64']
                      for p in Path(parent).glob('libEGL_nvidia.so.' + driver)]
        assert candidates, 'Graphics driver libraries are missing'
        library = str(candidates[0])
        # Driver versions can differ between SCO nodes sharing the same AFS env.
        vulkan = output / 'graphics/nvidia_icd.json'
        egl = output / 'graphics/10_nvidia.json'
        vulkan.parent.mkdir(parents=True, exist_ok=True)
        egl.parent.mkdir(parents=True, exist_ok=True)
        vulkan.write_text(json.dumps({'file_format_version':'1.0.0',
            'ICD':{'library_path':library,'api_version':'1.3.242'}}) + '\n')
        egl.write_text(json.dumps({'file_format_version':'1.0.0',
            'ICD':{'library_path':library}}) + '\n')
        environment = os.environ.copy()
        environment.pop('PYTHONHOME', None)
        environment.pop('PYTHONPATH', None)
        assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,80}', payload['run_id'])
        cache_root = Path('/var/tmp/behavior-1k-runtime') / payload['run_id']
        cache_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        assert cache_root.stat().st_dev != root.stat().st_dev, 'Worker cache must be outside AFS'
        local_free_bytes = shutil.disk_usage(cache_root).free
        assert local_free_bytes >= 40 * 1024**3, 'Worker temporary disk lacks texture-cache capacity'
        environment['SCO_EPHEMERAL_RUNTIME_ROOT'] = str(cache_root)
        status.update(ephemeral_runtime_cache_root=str(cache_root),
                      ephemeral_disk_free_bytes_at_start=local_free_bytes,
                      ephemeral_cache_on_different_filesystem=True)
        environment.update(PYTHONNOUSERSITE='1', OMNI_KIT_ACCEPT_EULA='YES',
            OMNIGIBSON_DATA_PATH=str(root / 'omnigibson-data'),
            OMNIGIBSON_HEADLESS='1', OMNIGIBSON_DISABLE_TORCH_COMPILE='1',
            OPENPI_DATA_HOME=str(root / 'openpi-cache'),
            XLA_PYTHON_CLIENT_PREALLOCATE='false',
            CUDA_VISIBLE_DEVICES='0', EXECUTE_HORIZON='16', MAX_SUBTASK_LEN='64',
            VK_ICD_FILENAMES=str(vulkan), __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
            LD_LIBRARY_PATH=str(root / 'graphics-runtime/usr/lib/x86_64-linux-gnu') + ':' +
                str(evaluation / 'lib') + ':' + environment.get('LD_LIBRARY_PATH',''))
        command([policy / 'bin/python', '-c',
            'import jax,openpi,openpi_client;d=jax.devices();assert len(d)==1 and d[0].platform=="gpu",d;print("policy_devices",d)'], env=environment)
        command([evaluation / 'bin/python', code / 'nebula/check_eval_runtime.py',
                 '--cache-dir', output / 'warp-preflight'], env=environment)
        command(['sha256sum', '-c', 'SHA256SUMS.inference'], cwd=root / 'checkpoint')
        status.update(state='running', gpu=rows[0], driver_library=library)
        publish()
        panel = code / 'eval_results/subtask100_native_baseline_1375000_20261007_091214/panel.json'
        plan = code / 'eval_results/subtask_recorded_native_1375000_20261007_081519/validation_plan.json'
        if payload.get('validation_sample_ids'):
            # Freeze exact recorded row hashes before starting the simulator.
            plan = output / 'validation_plan.json'
            prepare = "import json,sys;from pathlib import Path\n"
            prepare += "sys.path.insert(0," + repr(str(code / 'scripts')) + ")\n"
            prepare += "import eval_b1k_subtask_samples as r\nfrom recorded_native_state import decode_start\n"
            prepare += "p=r.load_panel(" + repr(str(panel)) + ")\n"
            prepare += "ids=" + repr(payload['validation_sample_ids']) + "\n"
            prepare += "by_id={s['sample_id']:s for s in p['samples']};samples=[by_id[i] for i in ids]\n"
            prepare += "assert len({s['episode_index'] for s in samples})==1\n"
            prepare += "path=r.raw_hdf5_path(" + repr(str(raw.parent)) + ",samples[0])\n"
            prepare += "cache=r.load_episode_cache(path,samples)\n"
            prepare += "relative='task-%04d/episode_%08d.hdf5'%(samples[0]['task_index'],samples[0]['raw_episode_id'])\n"
            prepare += "plan={'schema_version':1,'kind':'subtask_state_validation',\n"
            prepare += " 'panel_fingerprint':p['fingerprint'],'sample_ids':ids,'max_steps':256,\n"
            prepare += " 'require_recorded_native_state':True,'checkpoint_step':1375000,\n"
            prepare += " 'checkpoint_manifest_sha256':" + repr(ready['manifest_sha256']) + ",\n"
            prepare += " 'raw_source_sha256':{relative:cache['sha256']},\n"
            prepare += " 'selected_state_sha256':{s['sample_id']:decode_start(cache,s)['selected_state_sha256'] for s in samples},\n"
            prepare += " 'state_protocol':'recorded_native_fields_v1','official_benchmark':False,\n"
            prepare += " 'production_difficulty_baseline':False}\n"
            prepare += "Path(" + repr(str(plan)) + ").write_text(json.dumps(plan,indent=2)+'\\n')\n"
            command([evaluation / 'bin/python', '-c', prepare], env=environment, cwd=code)
            status.update(validation_plan=str(plan), expected_samples=len(payload['validation_sample_ids']))
            publish()
        command([evaluation / 'bin/python', code / 'nebula/subtask_eval.py', 'run',
            '--panel', panel, '--inventory', panel.with_name('raw_inventory.json'),
            '--raw-source', raw, '--raw-root', output / 'raw-stage',
            '--run-root', output / 'local-results', '--run-id', payload['run_id'],
            '--persistent-output', output / 'validated-results',
            '--checkpoint', root / 'checkpoint', '--checkpoint-source', root / 'checkpoint',
            '--repo-id', 'b1k_full100_joint_subtask_v1_drop_episode_s1',
            '--eval-env', evaluation, '--pi05-env', policy,
            '--asset-path', root / 'omnigibson-data', '--workers', '1',
            '--validation-plan', plan, '--base-port', '8780', '--sample-max-steps', '256'],
            env=environment, cwd=code)
        summary = json.loads((output / 'validated-results/validation_summary.json').read_text())
        assert summary['complete'] is True and summary['eligible_rollouts'] > 0
        if payload.get('validation_sample_ids'):
            assert summary['samples'] == summary['eligible_rollouts'] == len(payload['validation_sample_ids'])
            assert summary['setup_excluded'] == 0
        status.update(state='complete', completed_at=time.time(), summary=summary)
        publish()
        (output / '_SUCCESS').write_text(json.dumps(status, indent=2) + '\n')
    except BaseException as error:
        import traceback
        traceback.print_exc()
        status.update(state='failed', error_type=type(error).__name__, updated_at=time.time())
        publish()
        raise
