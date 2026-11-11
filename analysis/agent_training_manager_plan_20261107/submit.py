#!/usr/bin/env python3
"""Submit this campaign with its isolated, frozen source overlay."""
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import boto3  # Validate the transfer dependency before reserving a submission slot.

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'cluster_probe_20261104'
sys.path.insert(0, str(LEGACY))
spec = importlib.util.spec_from_file_location('cluster_submit', LEGACY / 'submit_rlinf.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def sources():
    entries = {}
    for folder, prefix in [(ROOT / 'runtime/RLinf/rlinf', 'RLinf/rlinf'),
                           (ROOT / 'configs', 'configs')]:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                entries[prefix + '/' + str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def verify_frozen():
    frozen = json.loads((ROOT / 'frozen_inputs.json').read_text())
    if sources() != frozen['overlay_files']:
        raise RuntimeError('Campaign source/config differs from the submitted and frozen version')
    for name, expected in frozen['support_files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen campaign input changed: {name}')
    return frozen


def frozen_overlay():
    verify_frozen()
    memory = io.BytesIO()
    names = []
    with tarfile.open(fileobj=memory, mode='w:gz') as archive:
        for folder, prefix in [(ROOT / 'runtime/RLinf/rlinf', 'RLinf/rlinf'),
                               (ROOT / 'configs', 'configs')]:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    name = prefix + '/' + str(path.relative_to(folder))
                    archive.add(path, arcname=name, recursive=False)
                    names.append(name)
    return memory.getvalue(), names


def enforce_policy(tag):
    from dispatch import current_jobs, required_storage_gib, storage_ready

    active = current_jobs()
    own = [row for row in active if row['name'].startswith('b1k-rl-pc-')]
    if sum(row['gpus'] for row in own) + 4 > 8 or sum(row['gpus'] for row in active) + 4 > 32:
        raise RuntimeError('Concurrent GPU budget is already allocated')
    if sum(row['state'] != 'RUNNING' for row in active) >= 2:
        raise RuntimeError('Account queue limit reached')
    if '--quota' not in sys.argv or sys.argv[sys.argv.index('--quota') + 1] != 'spot':
        raise RuntimeError('User policy from 2026-11-08 requires Spot resources for every experiment')
    reviews = json.loads((ROOT / 'reviewed_continuations.json').read_text())
    review = next((job for job in reviews if job['tag'] == tag), {})
    if not storage_ready(required_gib=required_storage_gib(review, active)):
        raise RuntimeError('Insufficient verified AFS headroom')
    now = dt.datetime.now()
    ledger = json.loads(submit.LEDGER.read_text()) if submit.LEDGER.exists() else []
    times = [dt.datetime.fromisoformat(row['time']) for row in ledger]
    if times and (now - max(times)).total_seconds() < 900:
        raise RuntimeError('Minimum 15-minute submission interval has not elapsed')
    if sum((now - when).total_seconds() < 3600 for when in times) >= 3:
        raise RuntimeError('Hourly submission limit reached')
    if sum(when.date() == now.date() for when in times) >= 50:
        raise RuntimeError('Daily submission limit reached')
    ledger.append({'time': now.isoformat(timespec='seconds'), 'tag': tag})
    temporary = submit.LEDGER.with_suffix('.tmp')
    temporary.write_text(json.dumps(ledger, indent=1))
    os.replace(temporary, submit.LEDGER)


if __name__ == '__main__':
    for key in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        os.environ.pop(key, None)
    tag = sys.argv[sys.argv.index('--tag') + 1]
    planned = {job['tag']: job for job in json.loads((ROOT / 'jobs.json').read_text())}
    continuation_file = ROOT / 'reviewed_continuations.json'
    if continuation_file.exists():
        planned.update({job['tag']: job for job in json.loads(continuation_file.read_text())})
    if tag not in planned:
        raise SystemExit('Run is outside the authorized primary campaign')
    if (LEGACY / 'runs' / f'{tag}.json').exists():
        raise SystemExit('A submission receipt already exists for this exact run')
    verify_frozen()
    if planned[tag]['arm'] == 'agent' and '--relay' not in sys.argv:
        sys.argv.extend(['--relay', 'train'])
    if planned[tag].get('resume_from') and '--resume-from' not in sys.argv:
        sys.argv.extend(['--resume-from', planned[tag]['resume_from']])
    if '--dry-run' not in sys.argv:
        intent = ROOT / f'intent_{tag}.json'
        with intent.open('x') as handle:
            json.dump({'tag': tag, 'created_at': dt.datetime.now().isoformat(),
                       'stage_sha256': hashlib.sha256((ROOT / 'stages' / f'{tag}.json').read_bytes()).hexdigest()}, handle)
    submit.overlay_bytes = frozen_overlay
    submit.enforce_submission_policy = enforce_policy
    submit.main()
