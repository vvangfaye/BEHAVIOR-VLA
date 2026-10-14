"""Bounded CPU-only preparation of the four explicitly authorized data sources.

Private download URLs are read from the user's cache, never written to receipts.
This module does not load a policy, construct an optimizer or submit jobs.
"""
from __future__ import annotations

import argparse
import configparser
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request

GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def relocate_index(index, dataset, root_id):
    """Make label references portable while retaining immutable content hashes."""
    from openpi.training.recovery_index import canonical_digest, validate_index
    index = copy.deepcopy(index)
    index['dataset_root'] = '/prepared/' + root_id
    for entry in index['provenance']['source_files']:
        path = Path(entry['path'])
        if path.is_absolute():
            entry['path'] = path.relative_to(dataset.resolve()).as_posix()
    if 'labels_path' in index['provenance']:
        index['provenance']['labels_path'] = Path(index['provenance']['labels_path']).relative_to(dataset.resolve()).as_posix()
    index['fingerprint'] = canonical_digest({k: v for k, v in index.items() if k != 'fingerprint'})
    validate_index(index)
    return index


def safe_extract(archive, destination, *, max_bytes=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 1000000:
            raise ValueError('Too many archive members')
        if max_bytes is not None and sum(m.size for m in members if m.isfile()) > max_bytes:
            raise ValueError('Expanded archive exceeds bounded scratch budget')
        names = set()
        for member in members:
            relative = Path(member.name)
            if (relative.is_absolute() or '..' in relative.parts or member.name in names
                    or not (member.isfile() or member.isdir())
                    or not (destination / relative).resolve().is_relative_to(destination)):
                raise ValueError('Unsafe or duplicate archive member')
            names.add(member.name)
        source.extractall(destination, members=members, filter='data')


class RangeResponseError(ValueError):
    """An object response failed an exact range/identity check."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def error_detail(error):
    def redact(text):
        return re.sub(r'https?://[^\s\"\'<>]+', '<URL>', text)
    return {'error_type': type(error).__name__, 'error_message': redact(str(error))[:2000],
            'traceback': redact(''.join(traceback.format_exception(type(error), error, error.__traceback__)))[-10000:],
            'details': getattr(error, 'details', {}), 'time': time.time(), 'training_executed': False}


def download(row, path, opener, *, part_bytes=1024**2, workers=8, progress=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = path.with_suffix(path.suffix + '.download_events.jsonl')
    lock = threading.Lock()
    def record(value):
        value = dict(value, time=time.time(), source_key=row.get('key'), expected_sha256=row.get('sha256'))
        with lock, events.open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            stream.flush()
    partial = path.with_suffix(path.suffix + '.partial')
    record({'event': 'download_started', 'resumed_from_bytes': partial.stat().st_size if partial.exists() else 0,
            'source_bytes': row['bytes']})
    try:
        _download(row, path, opener, part_bytes=part_bytes, workers=workers, progress=progress, record=record)
    except BaseException as error:
        detail = error_detail(error)
        detail.update(source_key=row.get('key'), expected_sha256=row.get('sha256'), source_bytes=row['bytes'],
                      retained_partial_bytes=partial.stat().st_size if partial.exists() else 0)
        save(path.with_suffix(path.suffix + '.download_failure.json'), detail)
        record({'event': 'download_failed', **detail})
        raise
    record({'event': 'download_complete', 'full_sha256_verified': True, 'bytes': path.stat().st_size})


def _download(row, path, opener, *, part_bytes=1024**2, workers=8, progress=None, record=None):
    path = Path(path)
    if path.exists():
        if path.stat().st_size == row['bytes'] and sha(path) == row['sha256']:
            return
        raise ValueError('Existing source file identity differs')
    part = path.with_suffix(path.suffix + '.partial')
    if part_bytes <= 0 or not 1 <= workers <= 8:
        raise ValueError('Invalid bounded transfer configuration')
    offset = part.stat().st_size if part.exists() else 0
    if offset > row['bytes']:
        raise ValueError('Partial source exceeds frozen size')
    def fetch(bounds):
        start, end = bounds
        request = urllib.request.Request(row['signed_get_url'], headers={
            'If-Match': row['etag'], 'Range': f'bytes={start}-{end}'})
        for attempt in range(4):
            try:
                with opener.open(request, timeout=60) as response:
                    if (response.status != 206 or response.headers.get('Content-Range')
                            != f'bytes {start}-{end}/{row["bytes"]}'):
                        raise RangeResponseError('Source did not honor exact bounded range',
                            status=response.status, content_range=response.headers.get('Content-Range'),
                            expected_content_range=f'bytes {start}-{end}/{row["bytes"]}')
                    if response.headers.get('ETag') != row['etag']:
                        raise RangeResponseError('Source ETag changed',
                            expected_etag=row['etag'], actual_etag=response.headers.get('ETag'))
                    content = response.read(end - start + 2)
                    if len(content) != end - start + 1:
                        raise RangeResponseError('Source range length differs',
                            expected_bytes=end-start+1, actual_bytes=len(content))
                    return content
            except urllib.error.HTTPError as error:
                if record:
                    record({'event': 'range_failed', 'start': start, 'end': end, 'attempt': attempt+1,
                            **error_detail(error)})
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except (RangeResponseError, http.client.IncompleteRead, TimeoutError,
                    ConnectionError, urllib.error.URLError, OSError) as error:
                if record:
                    record({'event': 'range_failed', 'start': start, 'end': end, 'attempt': attempt+1,
                            **error_detail(error)})
                if attempt == 3:
                    raise
            # Exact identity checks remain mandatory on every retry.
            time.sleep(min(2**attempt, 4))
    # Keep a bounded request window full even when one range is slow. Completed
    # out-of-order ranges are spooled to disk, never accumulated in RAM. Append
    # only a contiguous prefix so both it and the range spool survive a restart.
    spool=path.with_suffix(path.suffix+'.ranges');spool.mkdir(exist_ok=True)
    bounds=iter((start,min(start+part_bytes,row['bytes'])-1)
                for start in range(offset,row['bytes'],part_bytes))
    exhausted=False; pending={}; last_progress=0
    with ThreadPoolExecutor(max_workers=workers) as pool, part.open('ab') as stream:
        while pending or not exhausted:
            while len(pending)<workers and not exhausted:
                try: start,end=next(bounds)
                except StopIteration: exhausted=True;break
                cached=spool/str(start)
                if cached.exists() and cached.stat().st_size==end-start+1:continue
                pending[pool.submit(fetch,(start,end))]=(start,end)
            if pending:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    start,end=pending.pop(future); content=future.result()
                    temporary=spool/(str(start)+'.tmp');temporary.write_bytes(content)
                    temporary.replace(spool/str(start))
            while offset<row['bytes']:
                cached=spool/str(offset);size=min(part_bytes,row['bytes']-offset)
                if not cached.exists() or cached.stat().st_size!=size:break
                content=cached.read_bytes();stream.write(content);stream.flush()
                offset+=len(content);cached.unlink()
                if progress is not None and time.monotonic()-last_progress>=1:
                    progress(offset,row['bytes']);last_progress=time.monotonic()
    if part.stat().st_size != row['bytes'] or sha(part) != row['sha256']:
        raise ValueError('Source archive failed full SHA256 verification')
    part.replace(path)
    if not any(spool.iterdir()):spool.rmdir()


def verify_dataset(root, repo_id, *, permit_empty_labels=False):
    """Read every action/state row and real RGB at each episode's boundaries."""
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets import LeRobotDataset
    from subtask_trajectory import CAMERAS, array
    info = json.loads((root / 'meta/info.json').read_text())
    assert info['fps'] == 30
    assert {k for k, v in info['features'].items() if v['dtype'] == 'video'} == set(CAMERAS)
    metadata = [r for p in sorted((root / 'meta/episodes').glob('chunk-*/*.parquet'))
                for r in pq.read_table(p).to_pylist()]
    labels = pq.read_table(root / 'meta/subtask_segments.parquet').to_pylist()
    grouped = {}
    for r in labels:
        grouped.setdefault(r['episode_index'], []).append(r)
    reader = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False, video_backend='pyav',
                            delta_timestamps={'action': [i / 30 for i in range(32)]})
    offset = decoded = 0
    for episode in sorted(metadata, key=lambda r: r.get('dataset_from_index', r['episode_index'])):
        ep, n = episode['episode_index'], episode['length']
        segments = sorted(grouped[ep], key=lambda r: r['start'])
        cursor = 0
        for segment in segments:
            assert segment['start'] == cursor and segment['end'] > cursor
            assert permit_empty_labels or (segment['subtask'] and segment['subtask'].strip())
            cursor = segment['end']
        assert cursor == n
        data = reader.hf_dataset.select_columns(['action', 'observation.state', 'episode_index', 'frame_index'])[offset:offset+n]
        actions, states = array(data['action']), array(data['observation.state'])
        assert actions.shape == (n, 23) and states.ndim == 2 and states.shape[0] == n and states.shape[1] >= 57
        assert np.isfinite(actions).all() and np.isfinite(states).all()
        assert array(data['frame_index']).reshape(-1).tolist() == list(range(n))
        assert array(data['episode_index']).reshape(-1).tolist() == [ep] * n
        for frame in sorted({0, n // 2, n - 1}):
            row = reader[offset + frame]
            assert np.array_equal(array(row['action']), actions[np.minimum(frame + np.arange(32), n - 1)])
            assert np.array_equal(array(row['observation.state']), states[frame])
            for key in CAMERAS:
                pixels = array(row[key])
                assert pixels.shape[0] == 3 and np.isfinite(pixels).all()
                assert pixels.min() >= 0 and pixels.max() <= 1
            decoded += 1
        offset += n
    assert offset == len(reader)
    return {'episodes': len(metadata), 'frames': offset, 'decoded_samples': decoded,
            'all_action_state_rows_checked': True, 'real_lerobot_reader': True,
            'video_backend': 'pyav', 'training_executed': False}


class Preparation:
    def __init__(self, cache, public, *, lock_name='controller.lock'):
        self.cache, self.public = Path(cache), Path(public)
        self.inputs = json.loads((self.cache / 'private_sources.json').read_text())
        self.auth = self.inputs['authorization']
        assert self.auth['training_allowed'] is False and self.auth['cpu_only'] is True
        self.public.mkdir(parents=True, exist_ok=True)
        if not re.fullmatch(r'(controller|accept-[a-f0-9]{20})\.lock', lock_name):
            raise ValueError('Invalid preparation lock scope')
        self.lock = (self.public / lock_name).open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Respect the source host's network policy; never retry a domain denial
        # through another proxy. Native macOS trust includes the managed CA.
        if sys.platform == 'darwin':
            import truststore
            context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        else:
            context = ssl.create_default_context()
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context))
        import oss2
        config = configparser.ConfigParser(interpolation=None)
        config.read(Path.home() / '.ossutilconfig')
        credentials = config['Credentials']
        network = json.loads((self.cache / 'connectivity.json').read_text())
        self.bucket = oss2.Bucket(oss2.Auth(credentials['accesskeyid'], credentials['accesskeysecret']),
                                 network['endpoint'], network['bucket'], connect_timeout=60)
        self.prefix = network['prefix']
        response = self.bucket.get_object(self.prefix + '_DATA_PREP_CLAIM.json')
        try:
            assert json.loads(response.read())['source_fingerprint'] == self.inputs['fingerprint']
        finally:
            response.resp.response.close()

    def upload(self, path, key):
        import oss2
        digest, size = sha(path), path.stat().st_size
        try:
            head = self.bucket.head_object(key)
            if head.content_length != size or head.headers.get('x-oss-meta-sha256') != digest:
                raise ValueError('Destination exists with a different content identity')
        except oss2.exceptions.NoSuchKey:
            oss2.resumable_upload(self.bucket, key, str(path),
                                 headers={'x-oss-meta-sha256': digest, 'x-oss-forbid-overwrite': 'true'},
                                 store=oss2.ResumableStore(root=str(self.cache / 'upload_checkpoints')),
                                 multipart_threshold=32*1024**2, part_size=16*1024**2, num_threads=4)
        head = self.bucket.head_object(key)
        response = self.bucket.get_object(key, headers={'If-Match': head.etag})
        checked, n = hashlib.sha256(), 0
        try:
            while block := response.read(8*1024**2):
                checked.update(block); n += len(block)
        finally:
            response.resp.response.close()
        assert n == size and checked.hexdigest() == digest
        return {'bucket': self.bucket.bucket_name, 'key': key, 'bytes': size,
                'sha256': digest, 'etag': head.etag, 'full_destination_readback': True}

    def process(self, row):
        identity = hashlib.sha256(row['key'].encode()).hexdigest()[:20]
        receipt_path = self.public / 'receipts' / (identity + '.json')
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous.get('accepted') and previous.get('index_target') and previous['source_sha256'] == row['sha256']:
                return previous
        if shutil.disk_usage(self.cache).free < self.auth['minimum_free_disk_bytes'] + row['bytes'] * 2:
            raise ValueError('Insufficient bounded scratch space')
        work = self.cache / 'work' / identity
        work.mkdir(parents=True, exist_ok=True)
        archive = work / 'source.tar'
        partial = archive.with_suffix('.tar.partial')
        self._transfer_progress = {'downloaded_bytes': partial.stat().st_size if partial.exists() else 0,
            'resumed_from_bytes': partial.stat().st_size if partial.exists() else 0, 'source_bytes': row['bytes']}
        self.status('downloading', row)
        if (self.public / 'source_transfer_blocker.json').exists() and not archive.exists():
            raise RuntimeError('Source network policy requires a successful direct preflight before transfer')
        last_status = 0
        def progress(done, total):
            nonlocal last_status
            self._transfer_progress['downloaded_bytes'] = done
            if time.monotonic() - last_status >= 10:
                self.status('downloading', row)
                last_status = time.monotonic()
        download(row, archive, self.opener, progress=progress)
        extracted = work / 'extracted'
        if extracted.exists():
            shutil.rmtree(extracted)
        used=sum(p.stat().st_size for p in (self.cache/'work').rglob('*') if p.is_file())
        budget=min(self.auth['maximum_scratch_bytes']-used,
                   shutil.disk_usage(self.cache).free-self.auth['minimum_free_disk_bytes'])
        safe_extract(archive, extracted, max_bytes=budget)
        self.status('validating', row)
        root_id = row['kind'] + '_' + identity
        if row['kind'] == 'subtask':
            reports = list(extracted.rglob('export_report.json'))
            assert len(reports) == 1
            dataset = reports[0].parent
            export = json.loads(reports[0].read_text())
            acceptance = verify_dataset(dataset, export['repo_id'])
            assert acceptance['episodes'] == row['episodes'] and acceptance['frames'] == row['frames']
            from openpi.training.recovery_index import prepare_index
            provenance = json.loads((dataset / 'meta/collection_provenance.json').read_text())
            ids = [r['episode_index'] for r in provenance if r['source_kind'] == 'collected']
            try:
                index = prepare_index(dataset, dataset / 'meta/subtask_segments.parquet', ids,
                                      repo_id=export['repo_id'], root_id=root_id, source_kind='collected',
                                      held_out_instances=self.inputs['held_out_task_instances'])
            except ValueError as error:
                if str(error) != 'No legal labelled anchors remain after source and holdout filtering':
                    raise
                # Preserve complete short episodes, but do not invent 32-frame targets.
                index = None
            provenance_rows = provenance
            supervision = 'human_fixed_subtask_action_and_language'
            self.status('uploading', row)
            target = self.upload(archive, self.prefix + 'subtask/archives/' + row['sha256'] + '.tar')
        else:
            raws = [p for p in extracted.rglob('*') if p.suffix in ('.hdf5', '.h5')]
            assert len(raws) == 1 and sha(raws[0]) == row['raw_sha256']
            import full_episode_trajectory as native
            import export_full_episode_collection as exporter
            raw = raws[0]
            raw.chmod(0o444)
            inspected = native.inspect_full_episode_trajectory(raw, require_success=True)
            version = (raw.stat().st_ino, raw.stat().st_size, raw.stat().st_mtime_ns)
            original_inspector = exporter.inspect_full_episode_trajectory
            def frozen_inspector(path, **kwargs):
                path = Path(path)
                if path.resolve() != raw.resolve():
                    return original_inspector(path, **kwargs)
                assert version == (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                return copy.deepcopy(inspected)
            exporter.inspect_full_episode_trajectory = frozen_inspector
            try:
                dataset = work / 'export'
                if dataset.exists():
                    shutil.rmtree(dataset)
                from lerobot.configs.video import VideoEncoderConfig
                exporter.export_full_episode_collection([raw], dataset, root_id,
                    held_out_instances=self.inputs['held_out_task_instances'], writer_options={
                        'streaming_encoding': True, 'encoder_threads': 4,
                        'camera_encoder': VideoEncoderConfig(vcodec='h264', crf=18, g=15, preset='fast')})
                acceptance = exporter.verify_export(dataset)
                assert acceptance['real_lerobot_reader'] and acceptance['frames'] == row['frames']
                try:
                    index = exporter.prepare_training_index(dataset, root_id=root_id,
                        held_out_instances=self.inputs['held_out_task_instances'])
                except ValueError as error:
                    if str(error) != 'No nonempty weak-label interval has a complete 32-step action window':
                        raise
                    index = None
                save(dataset / 'native_training_index.json', index)
            finally:
                exporter.inspect_full_episode_trajectory = original_inspector
            packed = work / 'training.tar'
            with tarfile.open(packed, 'w') as tar:
                tar.add(dataset, arcname='dataset')
            self.status('uploading', row)
            target = self.upload(packed, self.prefix + 'full_episode/exports/' + row['sha256'] + '.tar')
            provenance_rows = json.loads((dataset / 'meta/collection_provenance.json').read_text())
            supervision = 'action_only; policy_subtask_text_is_weak_not_human'
        if index is not None:
            index = relocate_index(index, dataset, root_id)
        index_path = self.public / 'indexes' / (root_id + '.json')
        portable = {'root_id': root_id, 'index': index, 'episodes': provenance_rows,
                    'data_target': target, 'supervision_policy': supervision,
                    'raw_source': {k: row[k] for k in ('key', 'sha256', 'bytes')},
                    'action_horizon': 32, 'training_executed': False}
        if row['kind'] == 'full_episode':
            portable['action_only_episodes'] = [
                {'episode_index': p['episode_index'], 'length': p['length'],
                 'task_name': p['task_name'], 'task_instance_id': p['task_instance_id'],
                 'content_sha256': p['content_sha256'], 'start': 0, 'end': p['length'],
                 'valid_anchors': max(0, p['length'] - 31),
                 'language_loss_enabled': False} for p in provenance_rows]
        save(index_path, portable)
        index_target = self.upload(index_path, self.prefix + 'indexes/' + root_id + '.json')
        receipt = {'accepted': True, 'kind': row['kind'], 'source_key': row['key'],
                   'source_sha256': row['sha256'], 'root_id': root_id,
                   'target': target, 'index_target': index_target, 'acceptance': acceptance,
                   'training_eligible_episodes': (sum(p['length'] >= 32 for p in provenance_rows)
                       if row['kind'] == 'full_episode' else len(index['episodes']) if index else 0),
                   'supervision_policy': supervision,
                   'training_executed': False, 'verified_at': time.time()}
        save(receipt_path, receipt)
        shutil.rmtree(work)
        return receipt

    def status(self, phase, row=None):
        receipts = [json.loads(p.read_text()) for p in (self.public / 'receipts').glob('*.json')]
        value = {'pid': os.getpid(), 'phase': phase, 'updated_at': time.time(),
                 'source_count': len(self.inputs['source_archives']),
                 'accepted_archives': len([r for r in receipts if r.get('accepted')]),
                 'accepted_episodes': sum(r['acceptance']['episodes'] for r in receipts if r.get('accepted')),
                 'training_executed': False}
        if row:
            value.update(current_id=row['id'], current_kind=row['kind'])
        if phase == 'downloading':
            value.update(getattr(self, '_transfer_progress', {}))
        save(self.public / 'status.json', value)
        print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--kind', choices=['subtask', 'full_episode'])
    args = parser.parse_args()
    prep = Preparation(args.cache, args.output)
    rows = [r for r in prep.inputs['source_archives'] if args.kind is None or r['kind'] == args.kind]
    if args.limit:
        rows = rows[:args.limit]
    try:
        for row in rows:
            prep.process(row)
        prep.status('selected_archives_complete')
    except Exception as error:
        message = re.sub(r'https?://[^\s\"\']+', '[redacted URL]', str(error))
        save(args.output / 'failure.json', {'type': type(error).__name__, 'message': message[:2000],
                                           'time': time.time(), 'training_executed': False})
        prep.status('failed')
        raise SystemExit(type(error).__name__) from None


if __name__ == '__main__':
    main()
