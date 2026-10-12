"""Run a separate native episode collection without touching benchmark attempts."""
import hashlib
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import zipfile


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def stage_source(payload):
    """Clone the verified immutable source, then bind only explicit overlays."""
    base, target = Path(payload['base_code']), Path(payload['code'])
    original = (base / '_PUBLIC_TEST_SOURCE_READY').read_bytes()
    assert hashlib.sha256(original).hexdigest() == payload['base_source_manifest_sha256']
    ready = target / '_FULL_EPISODE_SOURCE_READY'
    if ready.exists():
        record = json.loads(ready.read_text())
        assert record['overlay_sha256'] == payload['overlay_sha256']
        for name, expected in record['files_sha256'].items():
            assert sha(target / name) == expected
        return record['source_manifest_sha256']
    assert not target.exists(), 'Inspect interrupted source staging; never overwrite'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, target, symlinks=True)
    archive = target / 'full_episode_overlay.tar.gz'
    with urllib.request.urlopen(payload['overlay_get'], timeout=60) as response, archive.open('xb') as out:
        shutil.copyfileobj(response, out)
    assert sha(archive) == payload['overlay_sha256']
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            name = Path(member.name)
            assert not name.is_absolute() and '..' not in name.parts and member.isfile()
            destination = target / name
            assert not destination.is_symlink()
        stream.extractall(target)
    provenance = json.loads(original)
    files = json.loads((target / '_FULL_EPISODE_OVERLAY.json').read_text())
    for name, expected in files.items():
        assert sha(target / name) == expected
    provenance['files_sha256'].update(files)
    provenance['collection_kind'] = 'native_full_episode_collection'
    provenance['base_public_test_source_sha256'] = payload['base_source_manifest_sha256']
    new_bytes = (json.dumps(provenance, sort_keys=True) + '\n').encode()
    (target / '_PUBLIC_TEST_SOURCE_READY').write_bytes(new_bytes)
    record = dict(overlay_sha256=payload['overlay_sha256'], files_sha256=files,
                  source_manifest_sha256=hashlib.sha256(new_bytes).hexdigest())
    ready.write_text(json.dumps(record, indent=2) + '\n')
    return record['source_manifest_sha256']


def _run(payload):
    root = Path(payload['assignment_root'])
    root.mkdir(parents=True, exist_ok=True)
    log = (root / 'collection_driver.log').open('a', buffering=1)
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    code_target = Path(payload['code'])
    code_target.parent.mkdir(parents=True, exist_ok=True)
    with code_target.with_suffix('.staging.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_sha = stage_source(payload)
    code = Path(payload['code'])
    sys.path[:0] = [str(code / 'sco'), str(code / 'scripts')]
    scratch = Path('/var/tmp/behavior-full-episode') / payload['execution_name']
    scratch.mkdir(parents=True, exist_ok=False)
    assert scratch.stat().st_dev != root.stat().st_dev
    assert shutil.disk_usage(scratch).free >= 128 * 1024**3
    codec_env = dict(os.environ, PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='')
    if payload['pilot']:
        python = Path(payload['runtime']) / 'envs/b1k-pi05/bin/python'
        tag = subprocess.check_output([str(python), '-c',
            'import sys; print("cp"+str(sys.version_info.major)+str(sys.version_info.minor))'],
            env=codec_env, text=True).strip()
        wheels = list((code / 'vendor/full_episode_codec').glob('h5py-*-' + tag + '-' + tag + '-*.whl'))
        assert len(wheels) == 1, 'No unique frozen h5py wheel matches the actual codec Python ABI'
        dependencies = scratch / 'codec_dependencies'
        dependencies.mkdir()
        with zipfile.ZipFile(wheels[0]) as wheel:
            for name in wheel.namelist():
                path = Path(name)
                assert not path.is_absolute() and '..' not in path.parts
            wheel.extractall(dependencies)
        codec_env.update(PYTHONPATH=str(dependencies) + ':' + str(code / 'scripts'),
            LD_LIBRARY_PATH=str(python.parent.parent / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
        subprocess.run([str(python), '-c',
            'import h5py, av, pyarrow; from lerobot.datasets import LeRobotDataset; print("REAL_CODEC_DEPENDENCIES_READY", flush=True)'],
            env=codec_env, check=True, timeout=120)
    from pi05_eval_queue import EvaluationQueue
    from collection_checkpoint import request, upload_json
    import run_public_test_assignment as baseline
    # RGB histories live on scratch; AFS holds compact queue/JSON/log records.
    manifest = dict(kind='native_full_episode_collection_manifest', run_id=payload['run_id'],
        checkpoint=payload['checkpoint_identity'], source_manifest_sha256=source_sha,
        task_prompts=payload['task_prompts'], trajectory_root=str(scratch / 'trajectories'),
        object_storage=payload['object_storage'], official_benchmark=False,
        all_public_instances_training_seen=True, release_uploaded_scratch=not payload['pilot'])
    manifest_path = root / 'collection.private.json'
    manifest_path.write_text(json.dumps(manifest)); manifest_path.chmod(0o600)
    queue = EvaluationQueue(root / 'queue.sqlite3')
    if not (root / 'queue.sqlite3').exists():
        queue.initialize([(payload['task'], i) for i in payload['instance_ids']], max_attempts=1,
            metadata=dict(run_id=payload['run_id'], mode='public_test', num_rollouts=1,
                          require_videos=False, checkpoint_step=1688977,
                          source_manifest_sha256=source_sha))
    real_subprocess = baseline.subprocess

    class CollectionSubprocess:
        def __getattr__(self, name):
            return getattr(real_subprocess, name)

        def run(self, command, **kwargs):
            if command[:2] == ['bash', str(code / 'scripts/eval_b1k.sh')]:
                kwargs['env'] = dict(kwargs['env'], PI05_FULL_EPISODE_COLLECTION_MANIFEST=str(manifest_path))
            return real_subprocess.run(command, **kwargs)

    baseline.subprocess = CollectionSubprocess()
    ordinary = dict(payload, source_manifest_sha256=source_sha, pilot=False, write_video=False)
    baseline.run(ordinary)
    records = []
    raw_paths = []
    for instance in payload['instance_ids']:
        path = root / 'results/pi05' / payload['task'] / 'json' / f"{payload['task']}_{instance}_0.json"
        result = json.loads(path.read_text())
        receipt = result['full_episode_object_checkpoint']
        key = f"{payload['task']}:{instance}:0"
        storage = manifest['object_storage'][key]
        body, _ = request(storage['receipt']['get'])
        assert json.loads(body) == receipt
        assert receipt['run_id'] == payload['run_id'] and receipt['source_manifest_sha256'] == source_sha
        assert receipt['task'] == payload['task'] and receipt['instance_id'] == instance
        assert receipt['steps'] == result['steps'] and receipt['success'] == result['success']
        assert receipt['result'] == {k: v for k, v in result.items()
                                    if k not in ('full_episode_trajectory', 'full_episode_object_checkpoint')}
        reference = result['full_episode_trajectory']
        raw_paths.append(reference['path'])
        assert receipt['trajectory_sha256'] == reference['sha256']
        assert receipt['content_sha256'] == reference['content_sha256']
        assert receipt['subtask_sha256'] == reference['subtask_sha256']
        assert receipt['archive']['object_key'] == storage['archive']['key']
        _, head = request(storage['archive']['head'], method='HEAD')
        head = {key.lower(): value for key, value in head.items()}
        assert int(head['content-length']) == receipt['archive']['bytes']
        assert head['etag'] == receipt['archive']['etag']
        records.append(receipt)
    codec = None
    if payload['pilot']:
        probe_output = scratch / 'format_probe'
        probe_python = Path(payload['runtime']) / 'envs/b1k-pi05/bin/python'
        subprocess.run([str(probe_python), str(code / 'scripts/verify_full_episode_codec.py'),
            '--trajectory', raw_paths[0], '--output', str(probe_output)],
            env=codec_env, check=True, timeout=1200)
        codec = json.loads((probe_output / 'format_probe_report.json').read_text())
        assert codec['real_lerobot_reader'] is True
        assert codec['native_frame_action_alignment_verified'] is True
        assert codec['source_trajectory_sha256'] == records[0]['trajectory_sha256']
        assert codec['source_content_sha256'] == records[0]['content_sha256']
    completion = dict(kind='native_full_episode_collection_complete', complete=True,
        official_benchmark=False, run_id=payload['run_id'], checkpoint=payload['checkpoint_identity'],
        source_manifest_sha256=source_sha, episodes=len(records),
        successful_episodes=sum(r['success'] for r in records), frames=sum(r['steps'] for r in records),
        object_checkpoints=records, completed_at=time.time(), real_codec_reader_verified=bool(codec),
        codec_verification=codec, ready_for_training=False)
    upload_json(completion, payload['completion_object'])
    (root / '_FULL_EPISODE_SUCCESS').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps({k: completion[k] for k in ('complete', 'episodes', 'successful_episodes', 'frames')}), flush=True)


def run(payload):
    try:
        return _run(payload)
    except Exception:
        # Preserve a small private diagnostic even after the worker exits.
        text = re.sub(r'https?://[^\s\"\']+', '[URL REDACTED]', traceback.format_exc()[-16000:])
        failure = dict(kind='native_full_episode_collection_failure', run_id=payload['run_id'],
            plan_fingerprint=payload['fingerprint'], execution_name=payload['execution_name'],
            failed_at=time.time(), traceback=text, official_benchmark=False)
        body = json.dumps(failure, sort_keys=True).encode()
        try:
            target = payload['failure_object']
            request = urllib.request.Request(target['put'], data=body, method='PUT')
            urllib.request.urlopen(request, timeout=30).close()
            with urllib.request.urlopen(target['get'], timeout=30) as response:
                assert response.read() == body
        except Exception:
            pass
        raise
