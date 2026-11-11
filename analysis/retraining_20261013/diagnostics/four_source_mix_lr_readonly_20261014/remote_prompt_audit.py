"""Inspect only tar headers and small training metadata, without video download."""
import hashlib
import json
import time
from nebula.submit_oss import OSSClient

c=OSSClient()

def read_range(key,start,count,etag):
    body=c.bucket.get_object(key,byte_range=(start,start+count-1),headers={'If-Match':etag})
    try:raw=body.read()
    finally:body.resp.response.close()
    if len(raw)!=count:raise ValueError('Archive metadata read truncated')
    return raw

results=[]
for target in TARGETS:
    key=target['target']['key'];head=c.bucket.head_object(key)
    if head.content_length != target['target']['bytes']:raise ValueError('Archive size changed')
    offset=0;metadata={};header_count=0;etag=head.etag;longname=None
    while offset+512<=head.content_length and header_count<160:
        hdr=read_range(key,offset,512,etag);header_count+=1
        if not hdr.strip(b'\0'):break
        name=hdr[:100].split(b'\0',1)[0].decode();prefix=hdr[345:500].split(b'\0',1)[0].decode()
        if prefix:name=prefix+'/'+name
        size=int(hdr[124:136].strip(b'\0 ') or b'0',8);kind=hdr[156:157]
        if longname:name=longname;longname=None
        if kind==b'L':
            if size>65536:raise ValueError('Oversized tar longname')
            longname=read_range(key,offset+512,size,etag).strip(b'\0').decode()
        if name.endswith('/meta/tasks.jsonl') or name.endswith('/meta/info.json'):
            if size>1024**2:raise ValueError('Training metadata unexpectedly large')
            raw=read_range(key,offset+512,size,etag)
            if name.endswith('/meta/tasks.jsonl'):
                digest=hashlib.sha256(raw).hexdigest()
                if target.get('meta_tasks_sha256') and digest!=target['meta_tasks_sha256']:
                    raise ValueError('Frozen task prompt metadata SHA differs')
                metadata['tasks']=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
                metadata['tasks_sha256']=digest
            else:
                info=json.loads(raw);metadata['info']={'fps':info.get('fps'),'total_episodes':info.get('total_episodes'),
                    'total_frames':info.get('total_frames'),'state_shape':info.get('features',{}).get('observation.state',{}).get('shape')}
        offset+=512+((size+511)//512)*512
        if 'tasks' in metadata and 'info' in metadata:break
    if 'tasks' not in metadata:raise ValueError('Task metadata not found within bounded tar scan')
    results.append({'task':target['task'],'kind':target['kind'],'example_subtask':target['example_subtask'],
                    'headers_read':header_count,'metadata':metadata})
print(json.dumps({'observed_at':time.time(),'only_metadata_read':True,'training_modified':False,'results':results},ensure_ascii=False))
