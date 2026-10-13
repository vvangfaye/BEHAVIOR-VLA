import hashlib
import io
import pytest
from direct_oss_worker import readback

class Response(io.BytesIO):
    headers={'ETag':'"actual"'}

class Opener:
    def __init__(self,data):self.data=data
    def open(self,*args,**kwargs):return Response(self.data)


def test_readback_requires_complete_destination_hash(tmp_path):
    data=b'exact object'
    target={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'key':'exact'}
    result=readback(Opener(data),{'target':target,'get':'unused'},tmp_path/'result')
    assert result['etag']=='actual' and result['full_destination_readback']
    assert (tmp_path/'result').read_bytes()==data

@pytest.mark.parametrize('data',[b'truncated',b'exact obJect',b'exact object plus extra'])
def test_changed_or_incomplete_destination_not_accepted(tmp_path,data):
    target={'bytes':12,'sha256':hashlib.sha256(b'exact object').hexdigest()}
    with pytest.raises((ValueError,AssertionError)):
        readback(Opener(data),{'target':target,'get':'unused'},tmp_path/'result')
