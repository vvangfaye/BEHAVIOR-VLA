"""Separate, audited transport-only deployment over an unchanged frozen plan."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

ALLOWED = {'scripts/collection_checkpoint.py', 'scripts/full_episode_multipart.py',
           'scripts/full_episode_transport.py'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def archive_files(path):
    files = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = Path(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            files[member.name] = archive.extractfile(member).read()
    return files


def patched_files(root, transport):
    files = archive_files(Path(root) / 'overlay.tar.gz')
    files.pop('_FULL_EPISODE_OVERLAY.json')
    checkpoint = files['scripts/collection_checkpoint.py'].decode()
    node = next(x for x in ast.parse(checkpoint).body if isinstance(x, ast.FunctionDef) and x.name == 'request')
    lines = checkpoint.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ['from full_episode_transport import request\n']
    files['scripts/collection_checkpoint.py'] = ''.join(lines).encode()
    multipart = files['scripts/full_episode_multipart.py'].decode()
    assert multipart.count('part_bytes=128*1024**2') == 1
    multipart = multipart.replace('part_bytes=128*1024**2', 'part_bytes=8*1024**2')
    old = "        response,_=self.request(self.urls['complete'],method='POST',body=body,headers={'Content-Type':'application/xml'})\n        tree=ET.fromstring(response)\n        if tree.tag.rsplit('}',1)[-1]=='Error':raise ValueError('Object store rejected multipart completion')\n        _,head=self.request(self.urls['head'],method='HEAD')"
    new = """        from full_episode_transport import ObjectTransportError
        import time
        # Send completion once. A lost response is reconciled against exact
        # length and multipart digest; never start a second upload/episode.
        try:
            response,_=self.request(self.urls['complete'],method='POST',body=body,headers={'Content-Type':'application/xml'})
            tree=ET.fromstring(response)
            if tree.tag.rsplit('}',1)[-1]=='Error':raise ObjectTransportError('Multipart completion returned an error')
        except (ObjectTransportError, FileNotFoundError):
            pass
        for attempt in range(6):
            try:
                _,head=self.request(self.urls['head'],method='HEAD')
                break
            except FileNotFoundError:
                if attempt == 5:raise ObjectTransportError('Multipart completion remains unknown; preserve original upload') from None
                time.sleep(min(2**attempt,16))"""
    assert multipart.count(old) == 1
    files['scripts/full_episode_multipart.py'] = multipart.replace(old, new).encode()
    files['scripts/full_episode_transport.py'] = transport
    for name in ALLOWED:
        compile(files[name], name, 'exec')
    return files


def freeze(root, transport, baseline_manifest):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text())
    gate = json.loads((root / 'pilot_accepted.json').read_text())
    assert digest(baseline_manifest) == gate['source_manifest_sha256']
    target = root / 'transport-v1'
    target.mkdir(exist_ok=False)
    (target / 'baseline_source_manifest.json').write_bytes(baseline_manifest)
    files = patched_files(root, transport)
    hashes = {name: digest(body) for name, body in files.items()}
    with tarfile.open(target / 'overlay.tar.gz', 'w:gz') as archive:
        for name, body in sorted(files.items()):
            item = tarfile.TarInfo(name); item.size = len(body)
            archive.addfile(item, io.BytesIO(body))
        body = json.dumps(hashes, sort_keys=True).encode()
        item = tarfile.TarInfo('_FULL_EPISODE_OVERLAY.json'); item.size = len(body)
        archive.addfile(item, io.BytesIO(body))
    source = json.loads(baseline_manifest)
    source['files_sha256'].update(hashes)
    source_sha = digest((json.dumps(source, sort_keys=True) + '\n').encode())
    revision = dict(kind='full_episode_transport_only_revision_v1', plan_fingerprint=plan['fingerprint'],
        baseline_source_sha256=digest(baseline_manifest), source_manifest_sha256=source_sha,
        overlay_sha256=digest((target / 'overlay.tar.gz').read_bytes()), overlay_files_sha256=hashes,
        transport_sha256=digest(transport), part_bytes=8 * 1024**2, maximum_parts=8192,
        maximum_archive_bytes=64 * 1024**3, authorized_by='user_20261012_fix_failure_prevention',
        future_initial_submissions_only=True, accepted=False)
    revision['revision_sha256'] = digest(json.dumps(revision, sort_keys=True).encode())
    (target / 'revision.json').write_text(json.dumps(revision, indent=2) + '\n')
    return revision


def validate(root, *, require_accepted=True):
    root = Path(root); target = root / 'transport-v1'
    revision = json.loads((target / 'revision.json').read_text())
    assert revision['kind'] == 'full_episode_transport_only_revision_v1'
    # Acceptance is separate from the immutable revision identity.
    identity = {k: v for k, v in revision.items() if k != 'revision_sha256'}
    assert digest(json.dumps(identity, sort_keys=True).encode()) == revision['revision_sha256']
    if require_accepted:
        acceptance = json.loads((target / 'accepted.json').read_text())
        assert acceptance['revision_sha256'] == revision['revision_sha256']
        assert acceptance['network_roundtrip_verified'] is True
        assert acceptance['deployed_source_sha256'] == revision['source_manifest_sha256']
    plan = json.loads((root / 'plan.json').read_text())
    assert revision['plan_fingerprint'] == plan['fingerprint']
    assert revision['future_initial_submissions_only'] is True
    old = archive_files(root / 'overlay.tar.gz'); old.pop('_FULL_EPISODE_OVERLAY.json')
    files = archive_files(target / 'overlay.tar.gz'); meta = json.loads(files.pop('_FULL_EPISODE_OVERLAY.json'))
    assert digest((target / 'overlay.tar.gz').read_bytes()) == revision['overlay_sha256']
    assert meta == revision['overlay_files_sha256'] == {k: digest(v) for k, v in files.items()}
    assert {k for k in files if files.get(k) != old.get(k)} == ALLOWED
    assert set(old).issubset(files)
    assert files == patched_files(root, files['scripts/full_episode_transport.py'])
    assert digest(files['scripts/full_episode_transport.py']) == revision['transport_sha256']
    baseline = (target / 'baseline_source_manifest.json').read_bytes()
    assert digest(baseline) == revision['baseline_source_sha256']
    assert json.loads((root / 'pilot_accepted.json').read_text())['source_manifest_sha256'] == digest(baseline)
    source = json.loads(baseline); source['files_sha256'].update(meta)
    assert digest((json.dumps(source, sort_keys=True) + '\n').encode()) == revision['source_manifest_sha256']
    assert revision['part_bytes'] * revision['maximum_parts'] == revision['maximum_archive_bytes'] == 64 * 1024**3
    return revision
