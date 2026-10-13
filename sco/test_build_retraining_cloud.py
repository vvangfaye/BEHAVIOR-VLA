import hashlib
import io
import urllib.error

import pytest

from build_retraining_cloud import SignedPreparation


class Reply(io.BytesIO):
    headers = {'ETag': '"fixed"'}


def prepared(opener):
    result = SignedPreparation.__new__(SignedPreparation)
    result.prefix = 'prepared/'
    result.storage = {'outputs': {'prepared/data.tar': {'get': 'https://unit/get', 'put': 'https://unit/put'}}}
    result.opener = opener
    return result


def test_existing_output_mismatch_never_overwrites(tmp_path):
    path = tmp_path / 'data'; path.write_bytes(b'correct')
    class Opener:
        def open(self, url, timeout):
            assert url == 'https://unit/get'
            return Reply(b'wrong')
    with pytest.raises(AssertionError, match='differs'):
        prepared(Opener()).upload(path, 'prepared/data.tar')


def test_new_output_is_exclusive_and_fully_read_back(tmp_path):
    path = tmp_path / 'data'; path.write_bytes(b'correct')
    class Opener:
        state = 0
        def open(self, request, timeout):
            if self.state == 0:
                self.state += 1
                raise urllib.error.HTTPError(request, 404, 'missing', {}, None)
            if self.state == 1:
                assert request.get_method() == 'PUT'
                assert request.headers['If-none-match'] == '*'
                assert request.data.read() == b'correct'
                self.state += 1
                return Reply()
            return Reply(b'correct')
    result = prepared(Opener()).upload(path, 'prepared/data.tar')
    assert result['full_destination_readback']
    assert result['sha256'] == hashlib.sha256(b'correct').hexdigest()


def test_reused_subtask_checks_frozen_content(tmp_path):
    path = tmp_path / 'data'; path.write_bytes(b'correct')
    prep = prepared(None)
    prep.current_row = {'kind': 'subtask', 'key': 'source/frozen.tar', 'etag': '"source-etag"',
        'bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}
    result = prep.upload(path, 'prepared/subtask/archives/hash.tar')
    assert result['key'] == 'source/frozen.tar' and result['source_object_reused']
    path.write_bytes(b'changed')
    with pytest.raises(AssertionError):
        prep.upload(path, 'prepared/subtask/archives/hash.tar')
