"""Fault injection for transport-only failure prevention, without episodes."""
import hashlib
import io
import json
from pathlib import Path
import ssl
import sys
import tarfile
import urllib.error

import pytest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / 'scripts')]
import full_episode_transport as transport
from full_episode_transport_revision import patched_files


class Response(io.BytesIO):
    headers = {'ETag': 'same-part'}


def test_repeat_identical_put_after_timeout_and_rewind(monkeypatch, capsys):
    attempts, delays = [], []
    def put(request, **kwargs):
        attempts.append(request.data.read())
        if len(attempts) < 3:
            raise urllib.error.URLError(TimeoutError('https://secret.invalid/signed'))
        return Response(b'ok')
    monkeypatch.setattr(transport.urllib.request, 'urlopen', put)
    monkeypatch.setattr(transport.time, 'sleep', delays.append)
    body = io.BytesIO(b'exact bytes')
    assert transport.request('https://secret.invalid/signed', method='PUT', body=body)[0] == b'ok'
    assert attempts == [b'exact bytes'] * 3 and len(delays) == 2
    assert 'secret.invalid' not in capsys.readouterr().out


@pytest.mark.parametrize('error', [
    urllib.error.HTTPError('secret-url', 403, 'secret', {}, None),
    urllib.error.URLError(ssl.SSLCertVerificationError(1, 'secret-url')),
])
def test_permanent_failures_not_retried_or_leaked(monkeypatch, error):
    calls = []
    def fail(*a, **k):
        calls.append(1); raise error
    monkeypatch.setattr(transport.urllib.request, 'urlopen', fail)
    monkeypatch.setattr(transport.time, 'sleep', lambda *_: pytest.fail('no retry'))
    with pytest.raises(transport.ObjectTransportError) as caught:
        transport.request('https://secret.invalid/signed', method='PUT', body=b'x')
    assert len(calls) == 1 and 'secret-url' not in str(caught.value)


def test_post_is_never_blindly_repeated(monkeypatch):
    calls = []
    def fail(*a, **k):
        calls.append(1); raise urllib.error.URLError(TimeoutError())
    monkeypatch.setattr(transport.urllib.request, 'urlopen', fail)
    with pytest.raises(transport.ObjectTransportError):
        transport.request('https://secret.invalid', method='POST', body=b'xml')
    assert len(calls) == 1


def test_transient_exhaustion_is_bounded(monkeypatch):
    calls = []
    def fail(*a, **k):
        calls.append(1); raise urllib.error.URLError(TimeoutError())
    monkeypatch.setattr(transport.urllib.request, 'urlopen', fail)
    monkeypatch.setattr(transport.time, 'sleep', lambda *_: None)
    with pytest.raises(transport.ObjectTransportError):
        transport.request('https://secret.invalid', method='PUT', body=b'x')
    assert len(calls) == 6


@pytest.mark.parametrize('corrupt', [False, True])
def test_lost_completion_response_reconciles_digest_without_second_post(corrupt):
    root = Path(__file__).resolve().parents[1] / 'eval_results/sco_full_episode_collection_20261012_v2'
    files = patched_files(root, Path(transport.__file__).read_bytes())
    module = {}; exec(files['scripts/full_episode_multipart.py'], module)
    calls = []
    data = b'one part'
    expected = hashlib.md5(hashlib.md5(data).digest()).hexdigest() + '-1'
    def request(url, *, method='GET', **kwargs):
        calls.append(method)
        if method == 'PUT':
            return b'', {'etag': 'part'}
        if method == 'POST':
            raise transport.ObjectTransportError('Lost response')
        return b'', {'content-length': str(len(data)), 'etag': 'wrong' if corrupt else expected}
    sink = module['MultipartArchive'](dict(parts=['part'], complete='complete', head='head', key='archive'), request)
    sink.write(data)
    if corrupt:
        with pytest.raises(ValueError, match='digest'):
            sink.finish()
    else:
        assert sink.finish()['sha256'] == hashlib.sha256(data).hexdigest()
    assert calls == ['PUT', 'POST', 'HEAD']


def test_revision_preserves_frozen_plan_and_rejects_unverified_deployment(tmp_path):
    from full_episode_transport_revision import freeze, validate, digest
    repo = Path(__file__).resolve().parents[1]
    baseline = json.dumps(dict(files_sha256={'policy.py': digest(b'unchanged policy')})).encode()
    (tmp_path / 'plan.json').write_text(json.dumps(dict(fingerprint='f' * 64)))
    (tmp_path / 'pilot_accepted.json').write_text(json.dumps(dict(source_manifest_sha256=digest(baseline))))
    with tarfile.open(tmp_path / 'overlay.tar.gz', 'w:gz') as archive:
        files = {'policy.py': b'unchanged policy'}
        files.update({name: (repo / name).read_bytes() for name in
                      ('scripts/collection_checkpoint.py', 'scripts/full_episode_multipart.py')})
        files['_FULL_EPISODE_OVERLAY.json'] = b'{}'
        for name, body in files.items():
            item = tarfile.TarInfo(name); item.size = len(body)
            archive.addfile(item, io.BytesIO(body))
    before = {name: digest((tmp_path / name).read_bytes()) for name in ('plan.json', 'overlay.tar.gz')}
    revision = freeze(tmp_path, Path(transport.__file__).read_bytes(), baseline)
    assert validate(tmp_path, require_accepted=False) == revision
    with pytest.raises(FileNotFoundError):
        validate(tmp_path)
    acceptance = dict(revision_sha256=revision['revision_sha256'], network_roundtrip_verified=True,
                      deployed_source_sha256=revision['source_manifest_sha256'])
    (tmp_path / 'transport-v1/accepted.json').write_text(json.dumps(acceptance))
    assert validate(tmp_path) == revision
    assert before == {name: digest((tmp_path / name).read_bytes()) for name in before}
    acceptance['deployed_source_sha256'] = 'a' * 64
    (tmp_path / 'transport-v1/accepted.json').write_text(json.dumps(acceptance))
    with pytest.raises(AssertionError):
        validate(tmp_path)
