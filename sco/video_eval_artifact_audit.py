"""Complete artifact checks for an already finished episode; never run a model."""


def audit_saved_episode(payload, completion_source):
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import subprocess
    import sys
    import tarfile
    import time
    import urllib.request

    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    code, runtime = Path(payload['code']), Path(payload['runtime'])
    root = Path(payload['assignment_root'])
    sys.path.insert(0, str(code / 'sco'))
    sys.path.insert(0, str(code / 'scripts'))
    from video_eval_worker import (PROGRESS_HOOK, VIDEO_CHECK, contained_file, put_file, sha256,
                                   validate_result, verify_inputs, verify_remote_archive, write_json)
    from pi05_eval_queue import EvaluationQueue
    assert payload['pilot'] and payload['task'] == 'turning_on_radio' and payload['instance_id'] == 301
    assert payload['old_gpu_state'] == 'FAILED' and payload['old_writer_stopped'] is True
    verify_inputs(payload)
    old = json.loads((root / 'receipt.json').read_bytes())
    assert old['state'] == 'failed' and old['error_type'] == 'FileNotFoundError'
    assert old['error'].endswith("native_progress.json'")
    assert old['run_id'] == payload['run_id'] and old['source_manifest_sha256'] == payload['source_manifest_sha256']
    backup = root / 'receipt.gpu_observer_failure.json'
    if not backup.exists():
        shutil.copyfile(root / 'receipt.json', backup)
    else:
        assert sha256(backup) == sha256(root / 'receipt.json')
    assert not (root / 'bundle').exists(), 'Inspect an existing audit; never overwrite it blindly'
    canonical = root / 'results/pi05' / payload['task'] / 'json' / (payload['task'] + '_301_0.json')
    original = json.loads(canonical.read_bytes())
    assert original.get('terminated') or original.get('truncated')
    # This is explicitly derived terminal evidence, not a fabricated live hook.
    progress = dict(run_id=payload['run_id'], steps=original['steps'], terminal=True,
        source='completed original JSON and independently decoded paired video',
        live_native_hook_applied=False, audit_only=True)
    write_json(root / 'native_progress.json', progress)
    env = dict(os.environ, **json.loads((root / 'execution.json').read_bytes())['env'])
    queue = EvaluationQueue(root / 'queue.sqlite3')
    hook_dir = root / 'audit_hook_source'
    hook_dir.mkdir(exist_ok=False)
    (hook_dir / 'sitecustomize.py').write_text(PROGRESS_HOOK)
    assert sha256(hook_dir / 'sitecustomize.py') == json.loads((root / 'execution.json').read_bytes())['native_progress_hook_sha256']
    hostname = os.uname().nodename
    match = re.fullmatch(r'pt-([a-z0-9]+)-worker-0', hostname)
    assert match, 'Need exact CPU audit worker identity'
    receipt = {k: v for k, v in old.items() if k not in ('error', 'error_type', 'failed_at', 'state', 'verified')}
    receipt.update(artifact_observer_recovery=True, original_model_execution_complete=True,
        old_gpu_uid=payload['old_gpu_uid'], old_gpu_state='FAILED', old_writer_stopped=True,
        failure_limited_to_post_rollout_observer=True, no_model_or_simulator_rerun=True,
        audit_job_name=payload['audit_job_name'], audit_job_uid=match.group(1))
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs/artifact_only_audit.py').write_text(payload['audit_source'])
    write_json(root / 'logs/artifact_only_audit_identity.json', dict(old_gpu_uid=payload['old_gpu_uid'],
        audit_job_name=payload['audit_job_name'], audit_job_uid=match.group(1), no_rollout_rerun=True,
        original_failure_receipt_sha256=sha256(backup)))
    # Reuse only the frozen worker's post-rollout verification/packaging code.
    # It requires one succeeded queue attempt, identical original JSON/video,
    # full video decoding and complete uploaded-object SHA256 readback.
    namespace = dict(locals())
    exec(compile(completion_source, 'frozen_post_rollout_artifact_checks', 'exec'), namespace, namespace)
