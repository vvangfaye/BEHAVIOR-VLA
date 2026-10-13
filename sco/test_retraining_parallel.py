import copy
import base64
import io
import shlex
import socket
import urllib.error
import zlib

import pytest

from build_retraining_parallel import balanced_parts, checked_receipts, check_resume_assignment, record_failure
from relay_cloud_prepared import merge_parallel
import bootstrap_retraining_cloud as bootstrap
from recover_retraining_parallel import startup_payload


def row(i):
    return {'key': str(i), 'sha256': str(i) * 64, 'kind': 'full_episode',
            'bytes': (i + 1) * 1000000, 'frames': (i + 1) * 100, 'episodes': 1}


def receipt(r):
    return {'source_key': r['key'], 'source_sha256': r['sha256'], 'accepted': True,
            'target': {'full_destination_readback': True},
            'index_target': {'full_destination_readback': True},
            'acceptance': {'real_lerobot_reader': True, 'frames': r['frames'], 'episodes': 1}}


def test_partition_is_complete_disjoint_deterministic_and_balanced():
    rows = [row(i) for i in range(32)]
    parts = balanced_parts(rows, 4)
    assert parts == balanced_parts(list(reversed(rows)), 4)
    assert len({r['key'] for part in parts for r in part}) == len(rows)
    assert sum(map(len, parts)) == len(rows)
    loads = [sum(r['frames'] for r in part) for part in parts]
    assert max(loads) - min(loads) <= max(r['frames'] for r in rows)


def test_partition_rejects_duplicate_input():
    with pytest.raises(ValueError, match='partition'):
        balanced_parts([row(1), row(1)], 4)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'sha', 'readback', 'frames'])
def test_incomplete_or_changed_receipt_cannot_be_accepted(damage):
    r = row(1); proof = receipt(r); proofs = [proof]
    if damage == 'missing': proofs = []
    if damage == 'duplicate': proofs.append(copy.deepcopy(proof))
    if damage == 'sha': proof['source_sha256'] = 'bad'
    if damage == 'readback': proof['target']['full_destination_readback'] = False
    if damage == 'frames': proof['acceptance']['frames'] += 1
    with pytest.raises(ValueError): checked_receipts([r], proofs)


def test_merge_preserves_baseline_and_requires_exact_frozen_coverage():
    rows = [row(i) for i in range(5)]
    frozen = {'fingerprint': 'frozen', 'source_archives': rows}
    shards = [{'name': str(i), 'source_keys': [str(i)]} for i in range(1, 5)]
    done = {str(i): {'source_fingerprint': 'frozen', 'all_assigned_sources_accounted_for': True,
                    'training_allowed': False, 'training_executed': False,
                    'sources': [receipt(rows[i])]} for i in range(1, 5)}
    result = merge_parallel(frozen, [receipt(rows[0])], done, shards)
    assert result['all_sources_accounted_for'] and len(result['sources']) == 5
    assert result['training_allowed'] is False
    shards[0]['source_keys'] = ['0']
    with pytest.raises(AssertionError): merge_parallel(frozen, [receipt(rows[0])], done, shards)


def test_merge_rejects_shard_source_hash_drift():
    r = row(1)
    done = {'s': {'source_fingerprint': 'changed', 'all_assigned_sources_accounted_for': True,
                  'training_allowed': False, 'training_executed': False, 'sources': [receipt(r)]}}
    with pytest.raises(AssertionError):
        merge_parallel({'fingerprint': 'original', 'source_archives': [r]}, [], done,
                       [{'name': 's', 'source_keys': ['1']}])


def test_bootstrap_status_retries_transient_dns_only(monkeypatch):
    waits = []
    monkeypatch.setattr(bootstrap.time, 'sleep', waits.append)
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            if self.calls < 3:
                raise urllib.error.URLError(socket.gaierror(-3, 'temporary DNS'))
            return io.BytesIO(b'ok')
    opener = Opener()
    assert bootstrap.open_with_dns_retry(opener, 'test', 60).read() == b'ok'
    assert opener.calls == 3 and waits == [1, 2]
    class Denied:
        def open(self, request, timeout):
            raise urllib.error.HTTPError('test', 403, 'forbidden', {}, None)
    with pytest.raises(urllib.error.HTTPError):
        bootstrap.open_with_dns_retry(Denied(), 'test', 60)
    assert waits == [1, 2]


def test_recovery_extracts_only_literal_payload_without_running_startup():
    payload = {'root': '/frozen', 'assets': [{'name': 'test', 'get': 'signed-placeholder'}]}
    source = 'raise RuntimeError("must not execute")\nrun(' + repr(payload) + ')\n'
    encoded = base64.b64encode(zlib.compress(source.encode())).decode()
    command = 'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(encoded) + ')))'
    assert startup_payload('timeout 86400 python3 -c ' + shlex.quote(command)) == payload


def test_resume_assignment_rejects_changed_frozen_identity(tmp_path):
    import json
    path = tmp_path / 'private_sources.json'
    original = {'fingerprint': 'frozen', 'source_archives': [{**row(1), 'etag': 'same'}]}
    path.write_text(json.dumps(original))
    renewed = copy.deepcopy(original); renewed['source_archives'][0]['signed_get_url'] = 'renewed'
    check_resume_assignment(path, renewed)
    renewed['source_archives'][0]['sha256'] = 'changed'
    with pytest.raises(ValueError, match='Resume assignment'):
        check_resume_assignment(path, renewed)


def test_local_failure_survives_status_publication_failure(tmp_path):
    import json
    class Prep:
        public = tmp_path
        def publish_status(self, value):
            raise ConnectionError('cannot publish https://invalid/?secret=no')
    try:
        raise ValueError('bad range https://invalid/?secret=no')
    except ValueError as error:
        record_failure(Prep(), error, source_id='exact-source')
    detail = json.loads((tmp_path / 'failure.json').read_text())
    assert detail['error_type'] == 'ValueError' and detail['source_id'] == 'exact-source'
    assert 'secret=no' not in json.dumps(detail)
    assert (tmp_path / 'failure_publication.json').exists()
