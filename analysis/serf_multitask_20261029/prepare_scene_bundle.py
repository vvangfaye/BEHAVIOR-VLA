"""Stage only additional scene assets and the fixed new task-state files."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import zlib

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'serf_reproduction_20261025'
sys.path.insert(0,str(OLD))
from legacy_zip import fetch_range as raw_fetch
import extract_legacy_assets as extractor
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def fetch_range(url,start,end):
    for attempt in range(3):
        try:return raw_fetch(url,start,end)
        except Exception:
            if attempt==2:raise
            time.sleep(1+attempt)


extractor.fetch_range=fetch_range

def main():
    deadline=time.monotonic()+3600
    while True:
        files=list((HERE/'assets/instances/scenes').rglob('*_0_0_template.json'))
        ready=(HERE/'small_upload.json').exists() and json.loads((HERE/'small_upload.json').read_text()).get('complete')
        if len(files)==2:break
        if time.monotonic()>deadline:raise TimeoutError('Task-state and map preparation did not finish')
        time.sleep(5)
    index=json.loads((OLD/'legacy_asset_zip_index.json').read_text())
    have={x['name'] for x in json.loads((OLD/'legacy_asset_selection.json').read_text())['entries']}
    models=set();scenes=set()
    for f in files:
        d=json.loads(f.read_text());scenes.add(d['init_info']['args']['scene_model'])
        for obj in d['objects_info']['init_info'].values():
            a=obj['args']
            if a.get('category') and a.get('model'):models.add((a['category'],a['model']))
    needed=[x for x in index['entries'] if not x['directory'] and ((x['name'].startswith('objects/') and tuple(x['name'].split('/')[1:3]) in models) or (x['name'].startswith('scenes/') and x['name'].split('/')[1] in scenes) or x['name'].startswith('metadata/') or x['name']=='VERSION')]
    found={tuple(x['name'].split('/')[1:3]) for x in needed if x['name'].startswith('objects/')};assert models<=found,models-found
    missing=[x for x in needed if x['name'] not in have]
    selection={k:v for k,v in index.items() if k!='entries'}
    selection.update(entries=missing,models=sorted(models),scenes=sorted(scenes),selected_files=len(needed),reused_files=len(needed)-len(missing))
    (HERE/'extra_scene_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('New scene assets:',len(missing),'compressed bytes:',sum(x['compressed_size'] for x in missing),flush=True)
    target=HERE/'assets/extra_scene_assets'
    pending=[]
    for entry in missing:
        path=target/entry['name']
        valid=False
        if path.is_file() and path.stat().st_size==entry['size']:
            valid=(zlib.crc32(path.read_bytes())&0xffffffff)==entry['crc']
        if not valid:pending.append(entry)
    remainder=dict(selection,entries=pending)
    records=extractor.extract(remainder,index['entries'],target)
    for entry in missing:
        path=target/entry['name'];assert path.stat().st_size==entry['size'] and (zlib.crc32(path.read_bytes())&0xffffffff)==entry['crc']

    while not ((HERE/'small_upload.json').exists() and json.loads((HERE/'small_upload.json').read_text()).get('complete')):
        if time.monotonic()>deadline:raise TimeoutError('Instance data still incomplete')
        time.sleep(5)
    manifest={};archive=HERE/'extra_data.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
        for base,prefix in [(target,'behavior-1k-assets'),(HERE/'assets/instances','2025-challenge-task-instances')]:
            for f in sorted(base.rglob('*')):
                if not f.is_file() or f.name.startswith('_') or f.name.endswith('.part'):continue
                content=f.read_bytes();rel=prefix+'/'+str(f.relative_to(base))
                entry=tarfile.TarInfo(rel);entry.size=len(content);entry.mode=0o644;entry.mtime=0
                tf.addfile(entry,io.BytesIO(content));manifest[rel]={'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    h=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    sha=h.hexdigest();key='behavior1k/serf-multitask-20261029/extra-data-'+sha+'.tar.gz'
    c=aoss_client()
    try:
        c.upload_file(str(archive),'260010165',key,ExtraArgs={'Metadata':{'sha256':sha}})
        head=c.head_object(Bucket='260010165',Key=key);assert head['ContentLength']==archive.stat().st_size
    finally:c.close()
    result={'complete':True,'key':key,'sha256':sha,'size':archive.stat().st_size,'files':manifest,'models':sorted(models),'scenes':sorted(scenes)}
    (HERE/'data_bundle_upload.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Data bundle complete',len(manifest),'files',archive.stat().st_size,'bytes',flush=True)


if __name__=='__main__':main()
