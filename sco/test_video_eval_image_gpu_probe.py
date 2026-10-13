"""Ensure image validation cannot accidentally exercise mounted AFS code."""
import pytest

from video_eval_image_gpu_probe import IMAGE_PATHS, assert_unshadowed_image_paths


ROOT = '21 0 0:1 / / rw,relatime - overlay overlay rw\n'


def mounted(path):
    return '22 21 0:2 / ' + path + ' rw,relatime - nfs shared rw\n'


def test_separate_output_mount_preserves_image():
    assert assert_unshadowed_image_paths(ROOT + mounted('/outputs'))


@pytest.mark.parametrize('path', ['/mnt', '/mnt/afs', *IMAGE_PATHS,
                                  IMAGE_PATHS[0] + '/openpi', IMAGE_PATHS[1] + '/envs'])
def test_reject_mounts_over_or_inside_image_sources(path):
    with pytest.raises(ValueError, match='shadows'):
        assert_unshadowed_image_paths(ROOT + mounted(path))


def test_decode_escaped_mount_path():
    with pytest.raises(ValueError, match='shadows'):
        assert_unshadowed_image_paths(ROOT + mounted(r'/mnt/\141fs'))


@pytest.mark.parametrize('text', ['', 'unparseable', mounted('/outputs')])
def test_incomplete_mount_provenance_is_rejected(text):
    with pytest.raises(ValueError):
        assert_unshadowed_image_paths(text)
