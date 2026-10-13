import hashlib
import io
import json
import tarfile
import urllib.error

import pytest

from prepare_retraining_data import download, safe_extract


@pytest.mark.parametrize('kind', ['parent', 'absolute', 'symlink', 'duplicate'])
def test_archive_rejects_path_escape_and_links(tmp_path, kind):
    archive = tmp_path / 'bad.tar'
    with tarfile.open(archive, 'w') as tar:
        item = tarfile.TarInfo('../outside' if kind == 'parent' else '/outside' if kind == 'absolute' else 'item')
        if kind == 'symlink':
            item.type = tarfile.SYMTYPE
            item.linkname = '../outside'
        tar.addfile(item)
        if kind == 'duplicate':
            tar.addfile(item)
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / 'extract')
    assert not (tmp_path / 'outside').exists()


def test_resume_preserves_bytes_and_checks_full_hash(tmp_path):
    content = b'previous-and-new-bytes'
    target = tmp_path / 'archive.tar'
    target.with_suffix('.tar.partial').write_bytes(content[:8])
    class Response(io.BytesIO):
        status = 206
    class Opener:
        def open(self, request, timeout):
            assert request.headers['If-match'] == 'fixed'
            start, end = map(int, request.headers['Range'][6:].split('-'))
            response = Response(content[start:end+1])
            response.headers = {'ETag': 'fixed', 'Content-Range': f'bytes {start}-{end}/{len(content)}'}
            return response
    row = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
           'etag': 'fixed', 'signed_get_url': 'https://example.invalid/object'}
    download(row, target, Opener(), part_bytes=3, workers=2)
    assert target.read_bytes() == content
    row['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        download(row, target, Opener())


def test_existing_matching_source_does_not_repeat_download(tmp_path):
    target = tmp_path / 'archive.tar'
    target.write_bytes(b'complete')
    download({'bytes': 8, 'sha256': hashlib.sha256(b'complete').hexdigest()}, target, None)


def test_archive_expansion_respects_scratch_budget(tmp_path):
    source=tmp_path/'large.tar'
    with tarfile.open(source,'w') as tar:
        member=tarfile.TarInfo('file');member.size=100
        tar.addfile(member,io.BytesIO(b'a'*100))
    with pytest.raises(ValueError,match='scratch budget'):
        safe_extract(source,tmp_path/'expanded',max_bytes=99)


def test_permission_failure_is_not_retried(tmp_path):
    calls=[]
    class Opener:
        def open(self,request,timeout):
            calls.append(request)
            raise urllib.error.HTTPError('https://example.invalid',403,'Forbidden',{},None)
    row={'bytes':10,'etag':'fixed','signed_get_url':'https://example.invalid'}
    with pytest.raises(urllib.error.HTTPError):download(row,tmp_path/'archive',Opener(),workers=1)
    assert len(calls)==1


@pytest.mark.parametrize('damage', ['range', 'etag', 'length', 'incomplete', 'status'])
def test_transient_range_validation_retries_then_verifies(tmp_path, monkeypatch, damage):
    import http.client
    import prepare_retraining_data as module
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    content = b'abcdefgh'
    class Response(io.BytesIO):
        status = 206
        def read(self, *args):
            if damage == 'incomplete' and opener.calls == 1:
                raise http.client.IncompleteRead(b'a', 7)
            return super().read(*args)
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            response = Response(content[:-1] if damage == 'length' and self.calls == 1 else content)
            response.headers = {'ETag': 'fixed', 'Content-Range': 'bytes 0-7/8'}
            if self.calls == 1:
                if damage == 'range': response.headers['Content-Range'] = 'bytes 1-7/8'
                if damage == 'etag': response.headers['ETag'] = 'changed'
                if damage == 'status': response.status = 200
            return response
    opener = Opener()
    target = tmp_path / 'archive.tar'
    row = {'bytes': 8, 'sha256': hashlib.sha256(content).hexdigest(), 'etag': 'fixed',
           'signed_get_url': 'https://example.invalid/object?secret=never-log'}
    download(row, target, opener, part_bytes=8, workers=1)
    assert opener.calls == 2 and target.read_bytes() == content
    events = [json.loads(line) for line in (tmp_path / 'archive.tar.download_events.jsonl').read_text().splitlines()]
    failure = next(e for e in events if e['event'] == 'range_failed')
    assert failure['start'] == 0 and failure['end'] == 7 and failure['attempt'] == 1
    assert 'secret=never-log' not in json.dumps(events)
    assert events[-1]['full_sha256_verified']


def test_persistent_range_failure_is_bounded_preserves_resume_and_logs(tmp_path, monkeypatch):
    import prepare_retraining_data as module
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    target = tmp_path / 'archive.tar'
    partial = tmp_path / 'archive.tar.partial'
    partial.write_bytes(b'abcd')
    class Response(io.BytesIO):
        status = 200
        headers = {'Content-Range': 'wrong'}
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            assert request.headers['Range'] == 'bytes=4-7'
            return Response(b'efgh')
    opener = Opener()
    row = {'bytes': 8, 'sha256': hashlib.sha256(b'abcdefgh').hexdigest(), 'etag': 'fixed',
           'signed_get_url': 'https://example.invalid/object?secret=no'}
    with pytest.raises(ValueError, match='exact bounded range'):
        download(row, target, opener, part_bytes=4, workers=1)
    assert opener.calls == 4 and partial.read_bytes() == b'abcd' and not target.exists()
    detail = json.loads((tmp_path / 'archive.tar.download_failure.json').read_text())
    assert detail['retained_partial_bytes'] == 4 and detail['details']['status'] == 200
    assert 'traceback' in detail and 'secret=no' not in json.dumps(detail)


def test_resume_reuses_valid_range_spool_but_still_checks_full_sha(tmp_path):
    target = tmp_path / 'archive.tar'
    (tmp_path / 'archive.tar.partial').write_bytes(b'abcd')
    spool = tmp_path / 'archive.tar.ranges'; spool.mkdir(); (spool / '4').write_bytes(b'efgh')
    row = {'bytes': 8, 'sha256': '0' * 64, 'etag': 'fixed', 'signed_get_url': 'https://invalid'}
    with pytest.raises(ValueError, match='full SHA256'):
        download(row, target, None, part_bytes=4, workers=1)
    assert not target.exists() and (tmp_path / 'archive.tar.partial').read_bytes() == b'abcdefgh'
