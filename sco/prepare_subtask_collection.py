"""Stage a separate source snapshot and freeze a bounded native collection pilot."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def prepare_inputs(payload):
    code, output, runtime = (Path(payload[k]) for k in ('code_root', 'output_root', 'runtime_root'))
    sys.path.insert(0, str(code / 'scripts'))
    from accepted_subtask_collection import candidates, select_starts, freeze
    from subtask_sample_panel import load_panel
    import eval_b1k_subtask_samples as runner
    from recorded_native_state import decode_start, validate_baseline_result
    panel = load_panel(payload['panel_path'])
    metrics, acceptance = payload['metrics'], payload['acceptance']
    groups, deferred = candidates(metrics, acceptance, panel, threshold=payload['threshold'],
                                   min_eligible=payload['min_eligible'])
    pilot_groups = [g for g in groups if not payload.get('pilot_tasks') or g['task'] in payload['pilot_tasks']]
    chosen = select_starts(panel, pilot_groups, payload['held_out_instances'],
        max_groups=payload['max_groups'], starts_per_group=payload['starts_per_group'], seed=payload['seed'])
    assignments = {a['index']: a for a in metrics['assignments']}
    group_assignments = {(g['task'], g['subtask']): assignments[g['assignment_index']] for g in groups}
    proofs = {}
    for episode in sorted({s['episode_index'] for s in chosen}):
        samples = [s for s in chosen if s['episode_index'] == episode]
        cache = runner.load_episode_cache(runner.raw_hdf5_path(payload['raw_data_root'], samples[0]), samples)
        for sample in samples:
            decoded = decode_start(cache, sample)
            assignment = group_assignments[sample['task'], sample['subtask']]
            manifest = json.loads((Path(assignment['producing_run_root']) / 'manifest.json').read_text())
            if manifest['sample_max_steps'] != 2048:
                raise ValueError('Accepted baseline used a different action budget')
            result_path = Path(manifest['results_root']) / 'pi05' / sample['task'] / 'json' / (runner._artifact_stem(sample)+'.json')
            result = json.loads(result_path.read_text())
            provenance = {'sha256': cache['sha256'], 'selected_state_sha256': decoded['selected_state_sha256']}
            validate_baseline_result(result, sample, provenance, 2048)
            proofs[sample['sample_id']] = dict(raw_sha256=cache['sha256'],
                selected_state_sha256=decoded['selected_state_sha256'], baseline_result_sha256=sha(result_path),
                baseline_metric_eligible=result['metric_eligible'])
    ready = json.loads((runtime / 'checkpoint/_READY').read_text())
    identity = dict(checkpoint_step=ready['step'], checkpoint_manifest_sha256=ready['manifest_sha256'],
                    checkpoint_source=str(runtime / 'checkpoint'))
    plan = freeze(panel, groups, chosen, proofs, identity, payload['held_out_instances'],
        accepted_input_sha256=payload['accepted_input_sha256'], repeats=payload['repeats'],
        max_steps=payload['max_steps'], seed=payload['seed'])
    (output / 'parent_panel.json').write_text(json.dumps(panel, indent=2)+'\n')
    (output / 'collection_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    from subtask_improvement import load_plan
    load_plan(output / 'collection_plan.json', panel)
    (output / 'candidates.json').write_text(json.dumps(dict(eligible_native_groups=groups,
        deferred_groups=deferred, accepted_task_count=acceptance['accepted_task_count'],
        pending_task_count=acceptance['pending_task_count']), indent=2)+'\n')
    # The shared filesystem's free space is not the user's AFS quota.
    quota = Path('/mnt/afs/260010165/.collection-quota-probe')
    quota.write_text('bounded collection write check\n')
    quota.unlink()
    if shutil.disk_usage(output).free < plan['maximum_output_bytes'] + 16*1024**3:
        raise ValueError('AFS lacks collection output budget plus reserve')
    return dict(plan=plan, candidates=len(groups), deferred_groups=len(deferred),
                checkpoint_identity=identity, write_probe_passed=True,
                shared_disk_free_bytes=shutil.disk_usage(output).free,
                personal_quota_free_bytes=None, quota_must_be_checked_before_gpu_submission=True)


def run(payload):
    output, code, parent = (Path(payload[k]) for k in ('output_root', 'code_root', 'parent_code_root'))
    output.mkdir(parents=True, exist_ok=False)
    status = dict(state='preparing', gpu_count=0, started_at=time.time(), output_root=str(output),
                  code_root=str(code), raw_data_root=payload['raw_data_root'])

    def publish():
        status['updated_at'] = time.time()
        (output / 'preparation_status.json').write_text(json.dumps(status, indent=2)+'\n')
        print('STATUS_JSON='+json.dumps(status), flush=True)
        if payload.get('report_put'):
            request = urllib.request.Request(payload['report_put'], data=json.dumps(status).encode(), method='PUT')
            urllib.request.urlopen(request, timeout=30).close()

    try:
        parent_bytes = (parent / '_CODE_SNAPSHOT_READY').read_bytes()
        assert hashlib.sha256(parent_bytes).hexdigest() == payload['parent_source_manifest_sha256']
        source = json.loads(parent_bytes)
        for name, expected in source['overlay_files_sha256'].items():
            assert sha(parent / name) == expected
        assert not code.exists()
        shutil.copytree(parent, code, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        blob = base64.b64decode(payload['overlay'])
        assert hashlib.sha256(blob).hexdigest() == payload['overlay_sha256']
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
            for member in archive.getmembers():
                p = Path(member.name)
                assert member.isfile() and not p.is_absolute() and '..' not in p.parts
                assert member.name in payload['overlay_files_sha256'] and p.name != 'config.sh'
                target = code / p
                assert target.resolve().is_relative_to(code.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                target.chmod(member.mode & 0o777)
        source.update(parent_code_root=str(parent), parent_source_manifest_sha256=payload['parent_source_manifest_sha256'],
                      code_root=str(code), created_at=time.time(), migration_base_commit=payload['migration_base_commit'])
        source['overlay_files_sha256'].update(payload['overlay_files_sha256'])
        for name, expected in source['overlay_files_sha256'].items():
            assert sha(code / name) == expected
        source_bytes = (json.dumps(source, indent=2)+'\n').encode()
        (code / '_CODE_SNAPSHOT_READY').write_bytes(source_bytes)
        status['source_manifest_sha256'] = hashlib.sha256(source_bytes).hexdigest()
        publish()
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='', OMNIGIBSON_NO_OMNIVERSE='1',
            OMNIGIBSON_HEADLESS='1', PYTHONPATH=str(code / 'sco')+':'+str(code / 'scripts'),
            LD_LIBRARY_PATH=str(Path(payload['runtime_root']) / 'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH',''))
        clean_payload = {k:v for k,v in payload.items() if k not in ('overlay', 'report_put')}
        inputs = output / 'preparation_inputs.json'
        inputs.write_text(json.dumps(clean_payload, indent=2)+'\n')
        python = Path(payload['runtime_root']) / 'envs/b1k-eval/bin/python'
        subprocess.run([str(python), '-m', 'pytest', 'scripts/test_subtask_improvement.py',
            'sco/test_accepted_subtask_collection.py','sco/test_collection_resume.py', '-q'],
            cwd=code, env=env, check=True)
        script = 'import json;from pathlib import Path;from prepare_subtask_collection import prepare_inputs;p=Path('+repr(str(inputs))+');r=prepare_inputs(json.loads(p.read_text()));p.with_name("prepared.json").write_text(json.dumps(r,indent=2)+"\\n")'
        subprocess.run([str(python), '-c', script], cwd=code, env=env, check=True)
        status.update(json.loads((output / 'prepared.json').read_text()), state='complete', completed_at=time.time())
        publish()
    except BaseException as error:
        status.update(state='failed', error_type=type(error).__name__, error=str(error))
        publish()
        raise
