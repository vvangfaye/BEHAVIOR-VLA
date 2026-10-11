"""Freeze and upload the isolated pilot code/metadata bundle, then save a CPU driver.

Uses existing local AOSS authentication. Signed source URLs live only in a private
bundle; the reviewable manifest and stdout contain paths, sizes and hashes only.
"""
from __future__ import annotations
import hashlib
import argparse
import io
import json
import os
from pathlib import Path
import tarfile

from codex_pilot_cloud import storage, BUCKET, PREFIX, AFS

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'eval_results/codex_pilot_20261011'
FILES = [
    'scripts/collect_b1k_codex.py', 'scripts/codex_collection_teacher.py',
    'scripts/codex_teacher_bridge.py', 'scripts/codex_teacher_mailbox.py',
    'scripts/codex_collection_actions.py', 'scripts/codex_collection_trajectory.py',
    'scripts/codex_collection_demonstrations.py', 'scripts/export_codex_collection.py',
    'scripts/prepare_codex_training_data.py', 'scripts/finetune_codex_collection.py',
    'scripts/run_codex_collection.sh', 'sco/run_codex_collection.py',
    'sco/prepare_codex_original_data.py', 'scripts/subtask_trajectory.py',
    'scripts/subtask_sample_panel.py', 'scripts/stage_subtask_replay.py',
    'scripts/strip_depth_from_info.py', 'scripts/export_subtask_collection.py',
    'scripts/subtask_improvement.py', 'scripts/finetune_subtask_collection.py',
    'scripts/eval_b1k_queue.py', 'scripts/pi05_eval_queue.py',
    'BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    version = args.version
    assert version.startswith('v') and version[1:].isdigit()
    receipt = EVIDENCE / ('staging_bundle_'+version+'.json')
    assert not receipt.exists(), 'Inspect existing bundle before choosing another version'
    selection = json.loads((ROOT.parent/'analysis/codex_teacher_pilot_20261011/demonstration_selection.json').read_text())
    signed = json.loads(Path('/private/tmp/codex-pilot-original-signed-files.json').read_text())
    blobs = {'code/'+name:(ROOT/name).read_bytes() for name in FILES}
    meta = EVIDENCE/'original_metadata'
    blobs.update({'metadata/'+p.relative_to(meta).as_posix():p.read_bytes() for p in sorted((meta/'meta').rglob('*')) if p.is_file()})
    blobs['labels/segments.parquet'] = Path(selection['segments']['path']).read_bytes()
    blobs['labels/retained.txt'] = Path(selection['retained_episode_list']['path']).read_bytes()
    blobs['checkpoint_upload_plan.json'] = (ROOT/'eval_results/sco_test2000_1688977_20261010/aoss_checkpoint_upload_plan.json').read_bytes()
    blobs['private_signed_files.json'] = json.dumps(signed).encode()
    manifest = {name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in blobs.items()}
    blobs['manifest.json'] = json.dumps(manifest).encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name,raw in blobs.items():
            entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mode=0o600
            tar.addfile(entry,io.BytesIO(raw))
    raw=stream.getvalue();sha=hashlib.sha256(raw).hexdigest()
    key=PREFIX+'bundles/'+version+'-'+sha+'.tar.gz'
    record=dict(state='upload_intent',key=key,bytes=len(raw),sha256=sha,
                source_files_sha256={name:manifest['code/'+name]['sha256'] for name in FILES},
                original_download_bytes=sum(r['bytes'] for r in signed),version=version)
    with receipt.open('x') as f:json.dump(record,f,indent=2)
    s=storage();s.put_object(Bucket=BUCKET,Key=key,Body=raw)
    assert hashlib.sha256(s.get_object(Bucket=BUCKET,Key=key)['Body'].read()).hexdigest()==sha
    url=s.generate_presigned_url('get_object',Params=dict(Bucket=BUCKET,Key=key),ExpiresIn=86400)
    bundle=AFS+'/bundle-'+version
    code=AFS+'/code-'+version
    output=AFS+'/original-'+version
    payload=dict(kind='codex_original_pilot_staging_v1',output_root=output,code_root=code,
        episodes=[r['episode_index'] for r in selection['original_episodes']],
        metadata_source=bundle+'/metadata',segments_path=bundle+'/labels/segments.parquet',
        retained_episodes_path=bundle+'/labels/retained.txt',
        segments_sha256=selection['segments']['sha256'],
        retained_episodes_sha256=selection['retained_episode_list']['sha256'],
        demonstrations=[dict(episode_index=r['episode_index'],raw_path=r['raw_path']) for r in selection['original_episodes']],
        max_seconds=3600,max_download_bytes=8*1024**3)
    # This wrapper is persisted privately because it carries one expiring bundle URL.
    driver = '''import hashlib,io,json,os,shutil,subprocess,tarfile,time,urllib.request
from pathlib import Path
bundle=Path(BUNDLE);code=Path(CODE);output=Path(OUTPUT)
base=Path('/mnt/afs/260010165/behavior_1k/code_snapshots/public-test-1688977-aoss-20261010')
def publish(value):
 data=json.dumps(value).encode()
 with urllib.request.urlopen(urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'],data=data,method='PUT'),timeout=30):pass
publish(dict(state='preparing_code'))
raw=urllib.request.urlopen(URL,timeout=30).read()
assert hashlib.sha256(raw).hexdigest()==SHA
bundle.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for entry in tar.getmembers():
  p=Path(entry.name);assert not p.is_absolute() and '..' not in p.parts and entry.isfile()
  target=bundle/p;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(tar.extractfile(entry).read())
  target.chmod(0o600)
manifest=json.loads((bundle/'manifest.json').read_text())
for name,item in manifest.items():
 b=(bundle/name).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
shutil.copytree(base,code,symlinks=True,copy_function=lambda source,dest:os.symlink(str(Path(source).resolve()),dest))
for p in (bundle/'code').rglob('*'):
 if p.is_file():
  dest=code/p.relative_to(bundle/'code');dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.is_symlink():dest.unlink()
  elif dest.exists():raise RuntimeError('Unexpected nonlinked overlay destination')
  shutil.copyfile(p,dest)
payload=PAYLOAD
payload['signed_files']=json.loads((bundle/'private_signed_files.json').read_text())
private=bundle/'private_staging_payload.json';private.write_text(json.dumps(payload));private.chmod(0o600)
env=dict(os.environ,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
 PYTHONPATH=str(code/'scripts')+':'+str(code/'openpi/src')+':'+str(code/'openpi/packages/openpi-client/src'))
python='/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-eval/bin/python'
with (bundle/'staging.log').open('x') as log:
 p=subprocess.Popen([python,str(code/'sco/prepare_codex_original_data.py'),'--payload',str(private)],env=env,stdout=log,stderr=subprocess.STDOUT)
 while True:
  try:rc=p.wait(timeout=20)
  except subprocess.TimeoutExpired:rc=None
  status=json.loads((output/'status.json').read_text()) if (output/'status.json').is_file() else dict(state='starting_stager')
  status.update(worker_exit_code=rc,code=str(code),bundle=str(bundle),output_root=str(output))
  publish(status)
  if rc is not None:break
assert rc==0,'Original data staging failed; inspect isolated staging.log'
assert (output/'_ORIGINAL_DATA_READY.json').is_file()
print(json.dumps(dict(state='original_demonstrations_verified',code=str(code),output_root=str(output))),flush=True)
'''
    for tag,value in [('BUNDLE',bundle),('CODE',code),('OUTPUT',output),('URL',url),('SHA',sha),('PAYLOAD',payload)]:
        # Token replacement avoids accidental rewriting of substituted values.
        import re
        driver=re.sub(r'\b'+tag+r'\b',lambda _:repr(value),driver)
    compile(driver,'private_staging_driver','exec')
    path=EVIDENCE/('private_staging_driver_'+version+'.py')
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:f.write(driver)
    record.update(state='uploaded_verified',bundle=bundle,code=code,original=output,driver=str(path))
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['state','bytes','original_download_bytes','code','original','driver']}))


if __name__=='__main__':main()
