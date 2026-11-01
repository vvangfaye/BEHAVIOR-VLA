"""Bounded archive of inactive experiment files; release after independent verification."""
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import ssl
import stat
import tarfile
import time
import urllib.parse

BASE=Path('/mnt/afs/260010165/behavior_1k')
PROTECTED_NAMES={'checkpoint','checkpoints','params','resume','videos','demonstrations','demonstration','expert','experts','2026-challenge-rawdata','datasets','dataset','data','demos','expert_data','rollout_inputs','.aws','.ssh','.git','secrets','credentials'}
PROTECTED_SUFFIXES={'.pt','.pth','.ckpt','.safetensors','.hdf5','.h5','.parquet','.arrow','.tfrecord','.pem','.key','.npy','.npz','.pkl'}

def raw(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n').encode()
def digest(value):return hashlib.sha256(value).hexdigest()
def identity(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode)
 return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'blocks':s.st_blocks,'mode':stat.S_IMODE(s.st_mode)}
def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def connection(url,method):
 u=urllib.parse.urlsplit(url)
 assert u.scheme=='https' and u.hostname in ['aoss.cn-sh-01b.sensecoreapi-oss.cn','aoss-internal.cn-sh-01b.sensecoreapi-oss.cn']
 c=http.client.HTTPSConnection(u.hostname,timeout=120,context=ssl.create_default_context())
 path=u.path+('?' +u.query if u.query else '')
 return c,path

def put(url,body=None,path=None):
 c,target=connection(url,'PUT');n=path.stat().st_size if path else len(body)
 try:
  c.putrequest('PUT',target);c.putheader('Content-Length',str(n));c.putheader('Cache-Control','no-cache');c.endheaders()
  if path:
   with path.open('rb') as f:
    for b in iter(lambda:f.read(8*1024**2),b''):c.send(b)
  else:c.send(body)
  r=c.getresponse();assert r.status==200, ('PUT',r.status);r.read()
 finally:c.close()

def get_bytes(url,limit=64*1024**2,missing_ok=False):
 c,target=connection(url,'GET')
 try:
  c.request('GET',target,headers={'Cache-Control':'no-cache'});r=c.getresponse()
  if missing_ok and r.status==404:return None
  assert r.status==200, ('GET',r.status)
  assert not r.getheader('Via') and 'HIT' not in (r.getheader('X-Cache') or '').upper()
  n=int(r.getheader('Content-Length'));assert n<=limit
  b=r.read(limit+1);assert len(b)==n
  return b
 finally:c.close()

class HashReader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.n+=len(b);return b


def eligible(p,root,protected):
 rel=p.relative_to(root)
 if any(x.lower() in PROTECTED_NAMES for x in rel.parts):return False
 if any(x.lower().startswith('checkpoint-') for x in rel.parts):return False
 if p.name.lower() in {'.env','model.bin','pytorch_model.bin','weights.bin','optimizer.bin','credentials.json'}:return False
 if p.suffix.lower() in PROTECTED_SUFFIXES and not (p.name=='rollout.pt' or p.name.startswith('replay_')):return False
 if any(p==r or p.is_relative_to(r) for r in protected):return False
 return True


def inventory(root,protected):
 assert root.is_relative_to(BASE) and root.resolve()==root
 selected=[];kept=[]
 for folder,dirs,names in os.walk(root,followlinks=False):
  parent=Path(folder)
  dirs[:]=[d for d in dirs if not (parent/d).is_symlink() and (parent/d).name.lower() not in PROTECTED_NAMES and not any((parent/d)==r or (parent/d).is_relative_to(r) for r in protected)]
  for name in names:
   p=parent/name
   if p.is_symlink() or not p.is_file() or not eligible(p,root,protected):
    kept.append(str(p.relative_to(root)));continue
   s=identity(p)
   if s['nlink']!=1 or s['size']>3*1024**3:
    kept.append(str(p.relative_to(root)));continue
   assert p.resolve()==p
   selected.append({'relative_path':str(p.relative_to(root)),**s})
 return selected,kept


def groups(entries,limit=3*1024**3):
 result=[];current=[];size=0
 for item in sorted(entries,key=lambda x:x['relative_path']):
  if current and size+item['size']>limit:result.append(current);current=[];size=0
  current.append(item);size+=item['size']
 if current:result.append(current)
 return result


def pack(root,entries,destination,meta):
 manifest={'schema':1,'root':str(root),'files':[],'metadata':meta,'restore':'Verify COMPLETE and archive SHA256, then extract regular files into a new directory and verify each file against manifest.'}
 with tarfile.open(destination,'w:gz',compresslevel=1) as tar:
  for item in entries:
   assert item['relative_path']!='__B1K_ARCHIVE_MANIFEST__.json'
   p=root/item['relative_path'];assert identity(p)=={k:v for k,v in item.items() if k!='relative_path'}
   info=tarfile.TarInfo(item['relative_path']);info.size=item['size'];info.mode=item['mode'];info.mtime=item['mtime_ns']//10**9
   with p.open('rb') as f:
    reader=HashReader(f);tar.addfile(info,reader)
   assert reader.n==item['size'] and identity(p)=={k:v for k,v in item.items() if k!='relative_path'}
   manifest['files'].append({**item,'sha256':reader.h.hexdigest()})
  body=raw(manifest);info=tarfile.TarInfo('__B1K_ARCHIVE_MANIFEST__.json');info.size=len(body);tar.addfile(info,io.BytesIO(body))
 assert destination.stat().st_size<5*1024**3
 return manifest


def verify_archive(url,expected_sha,expected_bytes,manifest):
 c,target=connection(url,'GET')
 try:
  c.request('GET',target,headers={'Cache-Control':'no-cache'});response=c.getresponse()
  assert response.status==200 and int(response.getheader('Content-Length'))==expected_bytes
  assert not response.getheader('Via') and 'HIT' not in (response.getheader('X-Cache') or '').upper()
  assert response.getheader('Content-Encoding') in (None,'identity')
  reader=HashReader(response);observed={};embedded=None
  with tarfile.open(fileobj=reader,mode='r|gz') as tar:
   for member in tar:
    assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
    with tar.extractfile(member) as f:
     if member.name=='__B1K_ARCHIVE_MANIFEST__.json':embedded=json.loads(f.read())
     else:
      h=hashlib.sha256();n=0
      for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);n+=len(block)
      assert member.name not in observed
      observed[member.name]={'size':n,'sha256':h.hexdigest()}
  while reader.read(8*1024**2):pass
  assert reader.n==expected_bytes and reader.h.hexdigest()==expected_sha
  assert embedded==manifest
  assert observed=={x['relative_path']:{'size':x['size'],'sha256':x['sha256']} for x in manifest['files']}
  return {'full_origin_readback':True,'archive_sha256':expected_sha,'archive_bytes':expected_bytes,'files':len(observed),'at':time.time()}
 finally:c.close()


def run(config):
 root=Path(config['scratch']);root.mkdir(parents=True,exist_ok=False)
 protected=[Path(p) for p in config['protected']]
 started=time.time();state={'stage':'inventory','roots':len(config['roots']),'released_bytes':0,'released_units':[], 'kept_roots':[]}
 def publish(**fields):
  state.update(fields,at=time.time());put(config['status_put'],raw(state));print(json.dumps(state),flush=True)
 publish()
 for root_index,source in enumerate(config['roots']):
  if time.time()-started>config['deadline_seconds']:break
  source=Path(source)
  if not source.is_dir():continue
  if get_bytes(config['stop_get'],missing_ok=True):break
  entries,kept=inventory(source,protected)
  parts=groups(entries)
  publish(stage='root_inventoried',root=str(source),selected_bytes=sum(e['size'] for e in entries),selected_files=len(entries),kept_files=len(kept),parts=len(parts))
  for part_index,part in enumerate(parts):
   if time.time()-started>config['deadline_seconds'] or get_bytes(config['stop_get'],missing_ok=True):break
   unit=f'{root_index:03d}-{part_index:03d}'
   endpoint=config['units'][unit]
   archive=root/(unit+'.tar.gz')
   assert shutil.disk_usage(root).free>sum(e['size'] for e in part)+512*1024**2
   publish(stage='packing',unit=unit)
   manifest=pack(source,part,archive,{'unit':unit,'kept_file_count':len(kept),'protected_policy':config['protected_policy']})
   archive_sha=sha_file(archive);archive_bytes=archive.stat().st_size
   publish(stage='uploading',unit=unit,archive_bytes=archive_bytes)
   put(endpoint['archive_put'],path=archive)
   publish(stage='verifying',unit=unit)
   verification=verify_archive(endpoint['archive_get'],archive_sha,archive_bytes,manifest)
   manifest_body=raw(manifest);report_body=raw(verification)
   put(endpoint['manifest_put'],manifest_body);assert get_bytes(endpoint['manifest_get'])==manifest_body
   put(endpoint['verification_put'],report_body);assert get_bytes(endpoint['verification_get'])==report_body
   complete={'unit':unit,'manifest_sha256':digest(manifest_body),'verification_sha256':digest(report_body),'archive_sha256':archive_sha,'archive_bytes':archive_bytes,'verified':True,'key':endpoint['archive_key']}
   complete_body=raw(complete);put(endpoint['complete_put'],complete_body);assert get_bytes(endpoint['complete_get'])==complete_body
   publish(stage='awaiting_recovery_index',unit=unit,complete=complete)
   ack=None;until=min(started+config['deadline_seconds'],time.time()+900)
   while time.time()<until:
    body=get_bytes(endpoint['release_get'],missing_ok=True)
    if body:ack=json.loads(body);break
    time.sleep(3)
   assert ack and ack['manifest_sha256']==complete['manifest_sha256'] and ack['external_index_verified']
   # Validate the entire source set before releasing its first file.
   for item in manifest['files']:
    p=source/item['relative_path'];assert identity(p)=={k:v for k,v in item.items() if k not in ['relative_path','sha256']}
    assert sha_file(p)==item['sha256'] and eligible(p,source,protected)
   released=0
   for item in manifest['files']:
    p=source/item['relative_path'];assert identity(p)=={k:v for k,v in item.items() if k not in ['relative_path','sha256']}
    p.unlink();released+=item['blocks']*512
   result={'unit':unit,'root':str(source),'files':len(part),'released_allocated_bytes':released,'manifest_sha256':complete['manifest_sha256'],'external_recovery_index':ack['external_index_path'],'at':time.time()}
   body=raw(result);put(endpoint['released_put'],body);assert get_bytes(endpoint['released_get'])==body
   state['released_bytes']+=released;state['released_units'].append(result)
   try:
    pointer=source/('ARCHIVED_TO_AOSS_'+unit+'.json');pointer.write_bytes(raw({**complete,'recovery_index':ack['external_index_path']}))
   except OSError as e:
    if e.errno not in (28,122):raise
   archive.unlink()
   publish(stage='unit_released',unit=unit)
 publish(stage='complete',elapsed=time.time()-started)
