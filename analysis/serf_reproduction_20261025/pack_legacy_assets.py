"""Wait for bounded range downloads, verify selected files, and stage one asset bundle."""
import hashlib,io,json,shutil,sys,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

deadline=time.monotonic()+1800
checks=[(HERE/'legacy_assets/behavior-1k-assets',1687),(HERE/'legacy_assets/omnigibson-robot-assets',100),(HERE/'legacy_system_defs',996)]
while True:
    ready=all((root/'_VERIFIED_FILES.json').exists() for root,n in checks)
    if ready:break
    if time.monotonic()>deadline:raise TimeoutError('Legacy extraction did not finish within 30 minutes')
    time.sleep(5)
for root,count in checks:
    d=json.loads((root/'_VERIFIED_FILES.json').read_text());assert len(d['files'])==count
    for item in d['files']:
        path=root/item['path'];assert path.stat().st_size==item['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
shutil.copytree(HERE/'legacy_system_defs/systems',HERE/'legacy_assets/behavior-1k-assets/systems',dirs_exist_ok=True)
archive=HERE/'legacy_task22_assets.tar.gz';manifest={}
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for path in sorted((HERE/'legacy_assets').rglob('*')):
        if not path.is_file() or path.name.startswith('_'):continue
        rel='data/'+str(path.relative_to(HERE/'legacy_assets'))
        data=path.read_bytes();sha=hashlib.sha256(data).hexdigest();manifest[rel]={'bytes':len(data),'sha256':sha}
        item=tarfile.TarInfo(rel);item.size=len(data);item.mode=0o644;item.mtime=0
        tar.addfile(item,io.BytesIO(data))
    data=json.dumps(manifest,indent=2).encode();item=tarfile.TarInfo('legacy_manifest.json');item.size=len(data);tar.addfile(item,io.BytesIO(data))
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
key='behavior1k/serf-reproduction-20261025/legacy-assets/task22-'+digest+'.tar.gz'
print('verified files',len(manifest),'archive bytes',archive.stat().st_size,flush=True)
c=aoss_client()
try:
    c.upload_file(str(archive),'260010165',key,ExtraArgs={'Metadata':{'sha256':digest}})
    head=c.head_object(Bucket='260010165',Key=key);assert head['ContentLength']==archive.stat().st_size
    result={'key':key,'sha256':digest,'bytes':archive.stat().st_size,'files':len(manifest),'uncompressed_bytes':sum(x['bytes'] for x in manifest.values())}
    (HERE/'legacy_asset_upload.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:c.close()
