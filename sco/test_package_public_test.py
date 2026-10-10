import json
from pathlib import Path
import zipfile

import pytest

from finalize_public_test import run
from package_public_test import package
from test_finalize_public_test import complete_fixture


def test_complete_archive_preserves_every_original_byte(complete_fixture):
    """Synthetic integrity fixture; this does not certify any model evaluation."""
    run(complete_fixture)
    root = Path(complete_fixture['destination'])
    archive = root.parent / 'submission.zip'
    receipt = package(root, archive)
    assert receipt['public_test_rollouts'] == 2000 and receipt['roundtrip_verified']
    assert receipt['q_score'] == 0.625 and receipt['task_sr'] == 0.5
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) == 2003
        for assignment in complete_fixture['assignments']:
            for original in (Path(assignment['root']) / 'results').glob('*.json'):
                assert bundle.read(root.name + '/json/' + original.name) == original.read_bytes()
    with pytest.raises(AssertionError, match='Preserve'):
        package(root, archive)


def test_changed_or_missing_episode_cannot_be_packaged(complete_fixture):
    run(complete_fixture)
    root = Path(complete_fixture['destination'])
    archive = root.parent / 'submission.zip'
    path = next((root / 'json').glob('*.json'))
    original = path.read_bytes()
    result = json.loads(original)
    result['q_score']['final'] = 0.0
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError, match='bytes changed'):
        package(root, archive)
    assert not archive.exists()
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        package(root, archive)
    assert not archive.exists()
