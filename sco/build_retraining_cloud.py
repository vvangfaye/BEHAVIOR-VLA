"""Build verified training exports next to AOSS, without loading any policy.

Only presigned, exact input/output objects are accessible to this worker. Raw
archives stay in the source bucket; only verified training exports are relayed.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.request

from prepare_retraining_data import Preparation, save, sha


class SignedPreparation(Preparation):
    def __init__(self, cache, public):
        self.cache, self.public = Path(cache), Path(public)
        self.inputs = json.loads((self.cache / 'private_sources.json').read_text())
        self.storage = json.loads((self.cache / 'private_storage.json').read_text())
        self.auth = self.inputs['authorization']
        assert self.auth['training_allowed'] is False and self.auth['cpu_only'] is True
        self.prefix = self.storage['prefix']
        self.public.mkdir(parents=True, exist_ok=True)
        self.lock = (self.public / 'controller.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self.current_row = None

    def publish_status(self, value):
        value = {**value, 'training_executed': False, 'updated_at': time.time()}
        save(self.public / 'cloud_status.json', value)
        data = (json.dumps(value, indent=2) + '\n').encode()
        with self.opener.open(urllib.request.Request(self.storage['status_put'],
                data=data, method='PUT'), timeout=60):
            pass
        print(json.dumps(value), flush=True)

    def upload(self, path, key):
        digest, size = sha(path), path.stat().st_size
        if key.startswith(self.prefix + 'subtask/archives/'):
            # The source subtask TAR is already a training export. Validation
            # hashes its complete bytes; reference that same immutable object.
            row = self.current_row
            assert row['kind'] == 'subtask' and digest == row['sha256'] and size == row['bytes']
            return {'bucket': '260010165', 'key': row['key'], 'bytes': size,
                'sha256': digest, 'etag': row['etag'].strip('"'),
                'full_destination_readback': True, 'source_object_reused': True}
        urls = self.storage['outputs'][key]
        if size > 5 * 1024**3:
            raise ValueError('Prepared output exceeds the bounded single-object upload limit')
        existing = False
        try:
            response = self.opener.open(urls['get'], timeout=60)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        else:
            with response:
                checked, count = hashlib.sha256(), 0
                while block := response.read(8 * 1024**2):
                    checked.update(block); count += len(block)
                assert count == size and checked.hexdigest() == digest, 'Existing prepared object differs'
                etag = response.headers['ETag']
                existing = True
        if not existing:
            with path.open('rb') as stream:
                request = urllib.request.Request(urls['put'], data=stream, method='PUT',
                    headers={'Content-Length': str(size), 'If-None-Match': '*'})
                with self.opener.open(request, timeout=600):
                    pass
            with self.opener.open(urls['get'], timeout=60) as response:
                checked, count = hashlib.sha256(), 0
                while block := response.read(8 * 1024**2):
                    checked.update(block); count += len(block)
                assert count == size and checked.hexdigest() == digest
                etag = response.headers['ETag']
        return {'bucket': '260010165', 'key': key, 'bytes': size, 'sha256': digest,
            'etag': etag.strip('"'), 'full_destination_readback': True}

    def status(self, phase, row=None):
        # Keep the legacy local status format and a fresh cloud-visible report.
        super().status(phase, row)
        self.publish_status(json.loads((self.public / 'status.json').read_text()))


def run(cache, public):
    prep = SignedPreparation(cache, public)
    rows = prep.inputs['source_archives']
    assert len(rows) == 463 and len({r['key'] for r in rows}) == 463
    existing = prep.storage['already_transferred']
    receipts = []
    # The first full-episode export is the real-data pilot. Remaining production
    # only starts after its source hash, all action/state rows and RGB readback pass.
    pilot = next(r for r in rows if r['kind'] == 'full_episode')
    order = [pilot] + [r for r in rows if r is not pilot]
    try:
        for row in order:
            if (Path(public) / 'STOP').exists():
                raise RuntimeError('Cloud data preparation STOP is present')
            if row['key'] in existing and row is not pilot:
                old = existing[row['key']]
                assert old['accepted'] and old['source_sha256'] == row['sha256']
                receipts.append(old)
                continue
            prep.current_row = row
            receipt = prep.process(row)
            receipt.update(execution_location='source_cloud_cpu', training_executed=False)
            identity = hashlib.sha256(row['key'].encode()).hexdigest()[:20]
            path = Path(public) / 'receipts' / (identity + '.json')
            save(path, receipt)
            prep.upload(path, prep.prefix + 'receipts/' + identity + '.json')
            receipts.append(receipt)
            if row is pilot:
                assert receipt['acceptance']['real_lerobot_reader']
                prep.publish_status({'phase': 'pilot_accepted', 'source_id': row['id'],
                    'source_bytes': row['bytes'], 'training_bytes': receipt['target']['bytes'],
                    'source_sha256': row['sha256'], 'acceptance': receipt['acceptance']})
        assert len(receipts) == len(rows)
        final = {'kind': 'source_cloud_prepared_training_exports',
            'source_fingerprint': prep.inputs['fingerprint'], 'sources': receipts,
            'expected_sources': len(rows), 'all_sources_accounted_for': True,
            'cloud_exports_accepted': True, 'destination_transfer_pending': True,
            'data_ready': False, 'training_allowed': False, 'training_executed': False}
        path = Path(public) / 'cloud_prepared_manifest.json'
        save(path, final)
        target = prep.upload(path, prep.prefix + 'cloud_prepared_manifest.json')
        prep.publish_status({'phase': 'all_cloud_exports_verified', 'manifest': target,
            'source_count': len(rows), 'episodes': sum(r['acceptance']['episodes'] for r in receipts),
            'training_bytes': sum(r['target']['bytes'] for r in receipts)})
    except Exception as error:
        prep.publish_status({'phase': 'needs_diagnostic', 'error_type': type(error).__name__,
            'source_id': prep.current_row['id'] if prep.current_row else None})
        raise SystemExit(1) from None


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1]), Path(sys.argv[2]))
