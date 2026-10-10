"""Export a fully finalized public test directory without rewriting original results."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import zipfile


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(destination, archive):
    destination, archive = Path(destination).resolve(), Path(archive).resolve()
    assert destination.is_dir() and not destination.is_symlink()
    assert destination not in archive.parents, 'Archive must be outside the result directory'
    assert not archive.exists(), 'Preserve existing archives'
    track, split, team, affiliation, date = destination.name.split('.')
    assert track == 'standard' and split == 'public' and all((team, affiliation, date))
    report = json.loads((destination / 'VERIFICATION.json').read_bytes())
    success = json.loads((destination / '_SUCCESS').read_bytes())
    assert report['state'] == 'complete' and report['checkpoint_step'] == success['checkpoint_step'] == 1688977
    assert report['public_test_rollouts'] == success['rollouts'] == 2000 and report['tasks'] == 100
    rows = report['files']
    tasks = {row['task'] for row in rows}
    expected = {(task, instance, 0) for task in tasks for instance in range(301, 321)}
    keys = [(row['task'], row['instance_id'], row['rollout_id']) for row in rows]
    assert len(tasks) == 100 and len(keys) == 2000 and set(keys) == expected
    assert len(set(keys)) == len(keys), 'Duplicate episode in finalized manifest'
    contents = {}
    digests = {}
    q_scores = []
    for row, key in zip(rows, keys):
        relative = Path('json') / f'{key[0]}_{key[1]}_{key[2]}.json'
        path = destination / relative
        assert path.resolve().parent == destination / 'json' and not path.is_symlink()
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['sha256'], 'Finalized result bytes changed'
        value = json.loads(raw)
        assert (value['task'], value['instance_id'], value['rollout_id']) == key
        q = value['q_score']['final']
        assert not isinstance(q, bool) and isinstance(q, (int, float)) and math.isfinite(q) and 0 <= q <= 1
        q_scores.append(q)
        contents[relative.as_posix()] = path
        digests[relative.as_posix()] = row["sha256"]
    scores_relative = 'scores/' + destination.name + '.json'
    for relative in ('VERIFICATION.json', '_SUCCESS', scores_relative):
        assert not (destination / relative).is_symlink()
        contents[relative] = destination / relative
        digests[relative] = file_digest(contents[relative])
    scores = json.loads(contents[scores_relative].read_bytes())
    assert scores == report['scores']
    mean_q = sum(q_scores) / 2000
    task_sr = sum(q == 1 for q in q_scores) / 2000
    for metric, value in (('q_score', mean_q), ('task_sr', task_sr)):
        assert abs(scores['overall_scores'][metric] - value) < 1e-12
        assert abs(success[metric] - value) < 1e-12
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob('*') if path.is_file()}
    assert actual == set(contents), 'Unexpected or missing finalized files'
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a prior package, even if another writer races.
    with archive.open('xb') as output:
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for relative, path in sorted(contents.items()):
                with path.open('rb') as source, bundle.open(destination.name + '/' + relative, 'w', force_zip64=True) as member:
                    shutil.copyfileobj(source, member, length=1024 * 1024)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {destination.name + '/' + name for name in contents}
        for relative, path in contents.items():
            with path.open('rb') as source, bundle.open(destination.name + '/' + relative) as member:
                digest = hashlib.sha256()
                while True:
                    chunk = source.read(1024 * 1024)
                    assert member.read(len(chunk) if chunk else 1) == chunk, 'Archive roundtrip changed bytes'
                    if not chunk:
                        break
                    digest.update(chunk)
                assert digest.hexdigest() == digests[relative], 'Finalized source changed during packaging'
    return dict(state='packaged', checkpoint_step=1688977, public_test_rollouts=2000,
        destination=str(destination), archive=str(archive), bytes=archive.stat().st_size,
        sha256=file_digest(archive), roundtrip_verified=True,
        source_manifest_sha256=report['source_manifest_sha256'],
        official_scorer_sha256=report['official_scorer_sha256'], q_score=mean_q, task_sr=task_sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.destination, args.archive), indent=2))
