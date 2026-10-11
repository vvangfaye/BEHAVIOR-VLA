"""Prepare a fresh staging attempt from verified files of a stopped failed one.

This only validates local/AFS artifacts and writes a private payload. It performs
no downloads, no staging run, and no cloud operations. The caller must first
verify the exact prior cloud job is terminal and supply that status receipt.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import shutil

import prepare_codex_original_data as staging


def load(path):
    return json.loads(Path(path).read_text())


def identity(row):
    return {key: row[key] for key in ('relative_path', 'bytes', 'etag', 'sha256') if key in row}


def checked_file(root, row):
    relative = staging.relative(row['path'])
    path = root / relative
    if (path.resolve() != root / relative or not path.is_file()
            or type(row.get('bytes')) is not int or row['bytes'] < 1
            or not re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256')))
            or path.stat().st_size != row['bytes'] or staging.sha(path) != row['sha256']):
        raise ValueError('Previously receipted file failed size/hash/path verification')
    return path


def prepare_resume(failed_root, payload, *, previous_job, reuse_root, payload_path):
    staging.validate_payload(payload)
    if (previous_job.get('state') not in ('FAILED', 'SUSPENDED', 'SUCCEEDED')
            or not previous_job.get('name') or not previous_job.get('uid')):
        raise ValueError('Exact prior cloud job needs independently verified terminal status')
    root = Path(failed_root).resolve(strict=True)
    state, old_payload = load(root / 'status.json'), load(root / 'payload.json')
    if ((root / '_RUNNING').exists() or not (root / '_FAILED').is_file()
            or (root / '_ORIGINAL_DATA_READY.json').exists()
            or state.get('kind') != staging.KIND or state.get('state') != 'failed'
            or state.get('failure_phase') != 'selected_files'
            or Path(state.get('root', '')).resolve() != root):
        raise ValueError('Resume is limited to a stopped failed selected-file transfer')
    if (state.get('source') != staging.SOURCE
            or state.get('selected_episodes') != payload['episodes']
            or old_payload.get('episodes') != payload['episodes']
            or old_payload.get('demonstrations') != payload['demonstrations']):
        raise ValueError('Resume changed original task/episode/success-proof scope')
    for name in ('segments', 'retained_episodes'):
        key = name + '_sha256'
        if (not old_payload.get(key) or payload.get(key) != old_payload[key]
                or staging.sha(payload[name + '_path']) != old_payload[key]):
            raise ValueError('Original labels or retained-episode list changed')
    selected = state.get('selected_metadata', [])
    if not selected or any(not 0 <= row['task_instance_id'] <= 300 for row in selected):
        raise ValueError('Original transfer has no verified train-only selection')
    required = state.get('required_files', [])
    if not required or len(set(required)) != len(required):
        raise ValueError('Missing unique frozen required-file list')
    for name in required:
        if staging.relative(name).parts[0] not in ('data', 'videos'):
            raise ValueError('Only original data/video files may be resumed')
    old_signed = {row['relative_path']: identity(row) for row in old_payload.get('signed_files', [])}
    new_signed = {row['relative_path']: identity(row) for row in payload.get('signed_files', [])}
    if set(old_signed) != set(required) or new_signed != old_signed:
        raise ValueError('Refreshed signed URLs must preserve every frozen object identity')
    metadata = state.get('metadata_files', [])
    if not metadata or any(not row['path'].startswith('meta/') for row in metadata):
        raise ValueError('Original metadata receipt is missing')
    names = {row['path'] for row in metadata}
    actual = {p.relative_to(root).as_posix() for p in (root / 'meta').rglob('*') if p.is_file()}
    if len(names) != len(metadata) or actual != names:
        raise ValueError('Original metadata file inventory changed')
    for row in metadata:
        checked_file(root, row)
    transfers = state.get('transfers', [])
    names = [row['path'] for row in transfers]
    if not transfers or len(set(names)) != len(names) or not set(names) <= set(required):
        raise ValueError('No unique verified partial transfer inventory')
    verified = []
    for row in transfers:
        source = checked_file(root, row)
        expected = old_signed[row['path']]
        if (row['bytes'] != expected['bytes'] or row.get('etag') != expected.get('etag')
                or expected.get('sha256', row['sha256']) != row['sha256']):
            raise ValueError('Transferred object differs from frozen remote identity')
        verified.append((source, row))
    reuse, private, output = Path(reuse_root).resolve(), Path(payload_path).resolve(), Path(payload['output_root']).resolve()
    if reuse.exists() or private.exists() or output.exists():
        raise ValueError('Reuse directory, payload and next output must all be fresh')
    for a, b in ((root, reuse), (root, output), (reuse, output)):
        if a == b or a in b.parents or b in a.parents:
            raise ValueError('Old evidence, reuse files and next output must be separate')
    if private == output or output in private.parents or root in private.parents or reuse in private.parents:
        raise ValueError('Private payload must be outside dataset directories')
    reuse.mkdir(parents=True, mode=0o700)
    copied = 0
    for source, row in verified:
        destination = reuse / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination, follow_symlinks=False)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, destination)
            copied += row['bytes']
        if staging.sha(destination) != row['sha256']:
            raise ValueError('Reusable file changed during preparation')
    result = dict(kind='codex_original_resume_v1',
        previous_job={key: previous_job[key] for key in ('name', 'uid', 'state')},
        failed_root=str(root), previous_status_sha256=staging.sha(root / 'status.json'),
        output_root=str(output), reuse_root=str(reuse), reused_files=len(verified),
        reused_bytes=sum(row['bytes'] for _, row in verified),
        remaining_files=len(required)-len(verified),
        remaining_download_bytes=sum(old_signed[name]['bytes'] for name in set(required)-set(names)),
        next_output_expected_data_bytes=sum(row['bytes'] for row in old_signed.values()),
        local_copy_fallback_bytes=copied, ignored_partial_files=True,
        source_files=[dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256']) for _, row in verified])
    prepared = {**payload, 'metadata_source': str(root), 'local_source_roots': [str(reuse)]}
    private.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(prepared, stream, indent=2); stream.write('\n')
    with (reuse / 'resume_receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('failed-root', 'payload', 'previous-job', 'reuse-root', 'output-payload'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare_resume(args.failed_root, load(args.payload), previous_job=load(args.previous_job),
                            reuse_root=args.reuse_root, payload_path=args.output_payload)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
