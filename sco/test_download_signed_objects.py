"""Check that recovery reuses only complete files with verified contents."""
import hashlib
import io

import pytest

from sco import download_signed_objects as downloader


class Response(io.BytesIO):
    status = 206
    headers = {'Content-Range': 'bytes 0-2/3'}


def payload(tmp_path, expected):
    return {'root': str(tmp_path), 'report_put': 'https://example.invalid/report',
            'items': [{'relative_path': 'file', 'bytes': 3, 'etag': 'fixture',
                       'expected_sha256': hashlib.sha256(expected).hexdigest(),
                       'url': 'https://example.invalid/file'}]}


def test_completed_file_is_verified_without_downloading(tmp_path, monkeypatch):
    (tmp_path / 'file').write_bytes(b'abc')

    def open_request(request, **kwargs):
        assert request.get_method() == 'PUT', 'Verified files must not be downloaded again'
        return io.BytesIO()

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_request)
    downloader.run(payload(tmp_path, b'abc'))
    assert (tmp_path / '_DOWNLOAD_COMPLETE').is_file()
    assert b'"reused_verified_file": true' in (tmp_path / 'download_status.json').read_bytes()


@pytest.mark.parametrize('existing', [b'bad', b'ab'])
def test_corrupt_or_incomplete_file_is_replaced(tmp_path, monkeypatch, existing):
    (tmp_path / 'file').write_bytes(existing)
    gets = []

    def open_request(request, **kwargs):
        if request.get_method() == 'PUT':
            return io.BytesIO()
        gets.append(request)
        return Response(b'abc')

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_request)
    downloader.run(payload(tmp_path, b'abc'))
    assert len(gets) == 1
    assert (tmp_path / 'file').read_bytes() == b'abc'


def test_wrong_source_hash_does_not_publish_completion(tmp_path, monkeypatch):
    def open_request(request, **kwargs):
        return io.BytesIO() if request.get_method() == 'PUT' else Response(b'bad')

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_request)
    with pytest.raises(AssertionError):
        downloader.run(payload(tmp_path, b'abc'))
    assert not (tmp_path / '_DOWNLOAD_COMPLETE').exists()
