"""Freeze code locally and render one finite, independent cluster installer.

No uploads, cloud submissions, model calls, authentication, or dataset transfers.
The caller uploads the tar and supplies its private download URL separately.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = '/mnt/afs/260010165/behavior_1k/code_snapshots/public-test-1688977-aoss-20261010'
PILOT = '/mnt/afs/260010165/behavior_1k/codex_pilot_20261011'
SOURCE_SHA = '7f2a16738961dfb1ca5b95d280ee61450f996757c5514b8c144e86dce3c982d4'
FILES = (
    'scripts/collect_b1k_codex.py', 'scripts/codex_collection_teacher.py',
    'scripts/codex_teacher_bridge.py', 'scripts/codex_teacher_mailbox.py',
    'scripts/codex_collection_actions.py', 'scripts/codex_collection_trajectory.py',
    'scripts/codex_collection_demonstrations.py', 'scripts/export_codex_collection.py',
    'scripts/prepare_codex_training_data.py', 'scripts/finetune_codex_collection.py',
    'scripts/run_codex_collection.sh', 'sco/run_codex_collection.py',
    'sco/codex_cluster_proxy.py', 'scripts/evaluate_codex_pilot.py',
    'sco/run_codex_pilot_evaluation.py', 'sco/prepare_codex_original_data.py',
    'sco/prepare_codex_reader_resume.py',
    'scripts/subtask_trajectory.py', 'scripts/subtask_sample_panel.py',
    'scripts/stage_subtask_replay.py', 'scripts/strip_depth_from_info.py',
    'scripts/export_subtask_collection.py', 'scripts/subtask_improvement.py',
    'scripts/finetune_subtask_collection.py', 'scripts/eval_b1k_queue.py',
    'scripts/pi05_eval_queue.py', 'scripts/train_pi05_subtask_b1k.sh',
    'scripts/train_pi05_b1k.sh',
    'BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml',
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze_bundle(destination, *, version='v3', root=ROOT):
    if not re.fullmatch(r'v[1-9][0-9]*', version):
        raise ValueError('Version must be a fresh vN label')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Bundle output must be new')
    blobs = {name: (Path(root) / name).read_bytes() for name in FILES}
    manifest = dict(kind='codex_collection_code_bundle_v1', version=version,
        base_code=BASE, base_source_manifest_sha256=SOURCE_SHA,
        code=PILOT + '/code-' + version, original_data=PILOT + '/original-v2',
        source_files_sha256={name: digest(raw) for name, raw in blobs.items()},
        files={name: dict(bytes=len(raw), sha256=digest(raw)) for name, raw in blobs.items()})
    blobs['manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode() + b'\n'
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for name, raw in sorted(blobs.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode, member.mtime = len(raw), 0o600, 0
                tar.addfile(member, io.BytesIO(raw))
    raw = stream.getvalue()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(raw)
    result = {**manifest, 'bundle': str(destination.resolve()), 'bundle_bytes': len(raw),
              'bundle_sha256': digest(raw)}
    with destination.with_suffix(destination.suffix + '.manifest.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return result


INSTALLER = r'''import hashlib,io,json,os,shutil,signal,tarfile,time,urllib.request
from pathlib import Path
settings=SETTINGS
signal.alarm(600)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def publish(value):
 if os.environ.get('CODEX_PILOT_REPORT_PUT'):
  request=urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'],data=json.dumps(value).encode(),method='PUT')
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=30):pass
base=Path(settings['base_code']);code=Path(settings['code'])
assert not code.exists(),'New independent code directory is required'
assert sha(base/'_PUBLIC_TEST_SOURCE_READY')==settings['base_source_manifest_sha256'],'Base source changed'
publish(dict(state='downloading_code_bundle',code=str(code)))
try:
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(settings['url'],timeout=30) as response:
  raw=response.read(20*1024*1024+1)
 assert len(raw)<=20*1024*1024 and hashlib.sha256(raw).hexdigest()==settings['bundle_sha256'],'Bundle mismatch'
 blobs={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
  total=0
  for item in tar.getmembers():
   relative=Path(item.name)
   assert item.isfile() and not relative.is_absolute() and '..' not in relative.parts and item.name not in blobs
   total+=item.size;assert total<=30*1024*1024,'Expanded bundle exceeds limit'
   blobs[item.name]=tar.extractfile(item).read()
 manifest=json.loads(blobs.pop('manifest.json'))
 assert manifest['kind']=='codex_collection_code_bundle_v1'
 for key in ('version','base_code','base_source_manifest_sha256','code','original_data'):
  assert manifest[key]==settings[key],'Bundle target differs'
 assert set(blobs)==set(manifest['files'])
 for name,data in blobs.items():
  expected=manifest['files'][name]
  assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
 shutil.copytree(base,code,symlinks=True,copy_function=lambda source,dest:os.symlink(str(Path(source).resolve()),dest))
 for name,data in blobs.items():
  dest=code/name
  assert all(not parent.is_symlink() for parent in dest.parents if parent!=code and code in parent.parents),'Linked overlay directory is unsafe'
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.is_symlink():dest.unlink()
  assert not dest.exists(),'Unexpected existing overlay file'
  with dest.open('xb') as stream:stream.write(data)
  dest.chmod(0o700 if name.endswith('.sh') else 0o600)
  assert sha(dest)==manifest['source_files_sha256'][name]
 ready={**manifest,'state':'code_verified','bundle_sha256':settings['bundle_sha256'],
        'reused_original_data':True,'installed_at':time.time()}
 with (code/'_CODEX_PILOT_CODE_READY').open('x') as stream:json.dump(ready,stream,indent=2)
 publish(ready)
 print(json.dumps(dict(state='code_verified',code=str(code),files=len(blobs))),flush=True)
except BaseException as error:
 publish(dict(state='failed',code=str(code),error_type=type(error).__name__))
 raise
finally:signal.alarm(0)
'''


def installer_source(manifest, bundle_url):
    if not bundle_url.startswith('https://') or not re.fullmatch(r'[a-f0-9]{64}', manifest['bundle_sha256']):
        raise ValueError('Installer requires an HTTPS bundle URL and exact SHA256')
    settings = {key: manifest[key] for key in ('version', 'base_code', 'base_source_manifest_sha256',
        'code', 'original_data', 'bundle_sha256')}
    settings['url'] = bundle_url
    source = INSTALLER.replace('settings=SETTINGS', 'settings=' + repr(settings))
    compile(source, '<collection-code-installer>', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--output', type=Path, required=True)
    freeze.add_argument('--version', default='v3')
    installer = commands.add_parser('installer')
    installer.add_argument('--manifest', type=Path, required=True)
    installer.add_argument('--bundle-url-file', type=Path, required=True)
    installer.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze_bundle(args.output, version=args.version)
        print(json.dumps({key: result[key] for key in ('bundle', 'bundle_sha256', 'bundle_bytes', 'code', 'original_data')}))
    else:
        source = installer_source(json.loads(args.manifest.read_text()), args.bundle_url_file.read_text().strip())
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:stream.write(source)
        print(json.dumps(dict(installer=str(args.output.resolve()), sha256=digest(source.encode()))))


if __name__ == '__main__':
    main()
