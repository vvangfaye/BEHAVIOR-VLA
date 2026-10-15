"""Publish frozen public code to existing AOSS and prepare a tiny AIHub entry."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]
sys.path.insert(0,str(WORKSPACE/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def main():
    folder=HERE/'small_submit';folder.mkdir(exist_ok=True)
    archive=HERE/'frozen_cpu_runtime.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted((HERE/'bundle').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                tar.add(path,arcname=path.relative_to(HERE/'bundle'),recursive=False)
    body=archive.read_bytes();digest=hashlib.sha256(body).hexdigest()
    key='behavior1k/retraining_20261015_round3_v2/two_source_preflight/inputs/'+digest+'/runtime.tar.gz'
    client=aoss_client()
    try:client.put_object(Bucket='260010165',Key=key,Body=body,IfNoneMatch='*',Metadata={'sha256':digest})
    except client.exceptions.ClientError as e:
        if e.response['ResponseMetadata']['HTTPStatusCode'] not in (409,412):raise
    assert client.head_object(Bucket='260010165',Key=key)['ContentLength']==len(body)
    url=client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=129600)
    config=dict(get=url,bytes=len(body),sha256=digest,output=json.loads((HERE/'bundle/probe_job.json').read_text())['output'])
    (folder/'bootstrap_input.json').write_text(json.dumps(config));(folder/'bootstrap_input.json').chmod(0o600)
    code='''import hashlib,json,os,subprocess,sys,tarfile,time,urllib.request
from pathlib import Path
cfg=json.loads(Path(__file__).with_name('bootstrap_input.json').read_text())
root=Path('/var/tmp/wangfaye/two-source-cpu-bootstrap')/str(os.getpid());root.mkdir(parents=True,exist_ok=False)
output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
try:
 archive=root/'runtime.tar.gz'
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 digest=hashlib.sha256();size=0
 with opener.open(cfg['get'],timeout=120) as response,archive.open('wb') as stream:
  while True:
   block=response.read(1024**2)
   if not block:break
   size+=len(block);assert size<=cfg['bytes'];stream.write(block);digest.update(block)
 assert size==cfg['bytes'] and digest.hexdigest()==cfg['sha256']
 bundle=root/'bundle';bundle.mkdir()
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  tar.extractall(bundle)
 print('Frozen CPU runtime downloaded and SHA verified; no GPU or optimizer',flush=True)
 result=subprocess.run([sys.executable,str(bundle/'four_source_cpu_probe.py')],cwd=bundle,timeout=10800)
 raise SystemExit(result.returncode)
except Exception as error:
 (output/'bootstrap_transport_failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,observed_at=time.time(),training=False)))
 raise
'''
    (folder/'bootstrap.py').write_text(code)
    shutil.copyfile(HERE/'bundle/probe_cluster.json',folder/'probe_cluster.json')
    (HERE/'small_submit_artifact.json').write_text(json.dumps(dict(key=key,bytes=len(body),sha256=digest,small_files_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),no_new_cloud_job_yet=True),indent=2)+'\n')
    print(json.dumps(dict(frozen_code_bytes=len(body),bootstrap_package_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),training=False)))


if __name__=='__main__':main()
