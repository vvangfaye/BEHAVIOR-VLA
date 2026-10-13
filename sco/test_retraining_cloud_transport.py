import hashlib
import io

from launch_retraining_cloud import frozen_tar, range_hash


def test_range_hash_keeps_order_and_requires_exact_source_identity():
    content = b'a' * 1024**2 + b'b' * 1024**2 + b'last'
    class Store:
        def get_object(self, *, Bucket, Key, Range, IfMatch):
            assert Bucket == '260010165' and Key == 'frozen' and IfMatch == 'same-etag'
            start, end = map(int, Range.removeprefix('bytes=').split('-'))
            return {'Body': io.BytesIO(content[start:end+1]),
                    'ContentRange': f'bytes {start}-{end}/{len(content)}'}
    assert range_hash(Store(), 'frozen', len(content), 'same-etag') == hashlib.sha256(content).hexdigest()


def test_frozen_tool_archive_is_reused_only_for_identical_contents(tmp_path):
    source = tmp_path / 'source'; source.write_text('first')
    target = tmp_path / 'tools.tar.gz'
    frozen_tar(target, [('module.py', source)])
    original = target.read_bytes()
    frozen_tar(target, [('module.py', source)])
    assert target.read_bytes() == original
    source.write_text('second')
    frozen_tar(target, [('module.py', source)])
    assert target.read_bytes() != original
