"""Remote archive/release executor. Configuration and short-lived URLs stay in memory."""
import hashlib,http.client,json,os,pathlib,ssl,stat,time,urllib.parse


def encode(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def identity(path):
 s=path.lstat()
 if not stat.S_ISREG(s.st_mode):raise RuntimeError('Source is not a regular file')
 return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink,blocks=s.st_blocks,mode=stat.S_IMODE(s.st_mode))

class TransientReadError(OSError):pass

def ranged_get(url,expected,fallback=None,progress=None):
 import concurrent.futures,collections
 size=expected['size'];chunk_size=8*1024*1024;workers=4
 def read_range_once(bounds,target):
  start,end=bounds;u=urllib.parse.urlsplit(target)
  if u.scheme!='https' or u.hostname not in ('aoss.cn-sh-01b.sensecoreapi-oss.cn','aoss-internal.cn-sh-01b.sensecoreapi-oss.cn'):raise RuntimeError('Unapproved HTTPS origin')
  conn=http.client.HTTPSConnection(u.hostname,timeout=120,context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt"))
  try:
   conn.request('GET',u.path+('?' +u.query if u.query else ''),headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'})
   response=conn.getresponse()
   if response.status in (408,429,500,502,503,504):raise TransientReadError('Range HTTP '+str(response.status))
   if response.status!=206:raise RuntimeError('Range HTTP '+str(response.status))
   if response.getheader('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError('Range response mismatch')
   if response.getheader('Via') or 'HIT' in (response.getheader('X-Cache') or '').upper() or response.getheader('Content-Encoding') not in (None,'identity'):raise RuntimeError('Invalid range origin response')
   if int(response.getheader('Content-Length','-1'))!=end-start+1:raise RuntimeError('Range length mismatch')
   raw=response.read(end-start+2)
   if len(raw)!=end-start+1:raise RuntimeError('Truncated range')
   return raw
  finally:conn.close()
 def read_range(bounds):
  for attempt in range(4):
   try:return read_range_once(bounds,fallback if fallback and attempt==3 else url)
   except (OSError,http.client.HTTPException):
    if attempt==3:raise
    time.sleep(min(2**attempt,4))
 ranges=iter((start,min(size-1,start+chunk_size-1)) for start in range(0,size,chunk_size));pending=collections.deque();digest=hashlib.sha256();count=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  for _ in range(workers):
   item=next(ranges,None)
   if item is not None:pending.append(pool.submit(read_range,item))
  while pending:
   raw=pending.popleft().result();digest.update(raw);count+=len(raw)
   if progress and (count%(128*1024*1024)<len(raw) or count==size):progress(count,size)
   item=next(ranges,None)
   if item is not None:pending.append(pool.submit(read_range,item))
 value={'size':count,'sha256':digest.hexdigest()}
 if value!=expected:raise RuntimeError('Full range readback hash mismatch')
 return value

def _request_once(url,method,payload=None,path=None,expected=None,tag=None,fallback=None,progress=None):
 if method=='GET' and expected is not None and expected['size']>=16*1024*1024:return ranged_get(url,expected,fallback,progress)
 u=urllib.parse.urlsplit(url)
 if u.scheme!='https' or u.hostname not in ('aoss.cn-sh-01b.sensecoreapi-oss.cn','aoss-internal.cn-sh-01b.sensecoreapi-oss.cn'):raise RuntimeError('Unapproved HTTPS origin')
 conn=http.client.HTTPSConnection(u.hostname,timeout=120,context=ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt"))
 size=path.stat().st_size if path else len(payload) if payload is not None else None
 digest=hashlib.sha256();count=0
 try:
  conn.putrequest(method,u.path+('?' +u.query if u.query else ''))
  conn.putheader('Cache-Control','no-cache')
  if size is not None:conn.putheader('Content-Length',str(size))
  conn.endheaders()
  if path:
   with path.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):
     conn.send(block);digest.update(block);count+=len(block)
     if tag and count%(256*1024*1024)<len(block):print('STORAGE_PROGRESS '+json.dumps({'unit':tag,'stage':'uploading','file':path.name,'transferred_bytes':count,'file_bytes':size}),flush=True)
  elif payload is not None:conn.send(payload);digest.update(payload);count=len(payload)
  response=conn.getresponse()
  if method=='GET' and response.status in (408,429,500,502,503,504):raise TransientReadError('GET HTTP '+str(response.status))
  if response.status!=200:raise RuntimeError('Object '+method+' status '+str(response.status))
  if response.getheader('Via') or 'HIT' in (response.getheader('X-Cache') or '').upper():raise RuntimeError('Cached response forbidden')
  if response.getheader('Content-Encoding') not in (None,'identity'):raise RuntimeError('Encoded response forbidden')
  if method=='GET':
   digest=hashlib.sha256();count=0
   while True:
    block=response.read(8*1024*1024)
    if not block:break
    digest.update(block);count+=len(block)
   if response.getheader('Content-Length') is None or int(response.getheader('Content-Length'))!=count:raise RuntimeError('Readback length mismatch')
  else:response.read()
  value={'size':count,'sha256':digest.hexdigest()}
  if expected is not None and value!=expected:raise RuntimeError('Readback content mismatch')
  return value
 finally:conn.close()
def request(url,method,**kwargs):
 for attempt in range(4 if method=='GET' else 1):
  try:return _request_once(url,method,**kwargs)
  except (OSError,http.client.HTTPException):
   if attempt==3 or method!='GET':raise
   time.sleep(min(2**attempt,4))
def save(path,value):
 data=encode(value);tmp=path.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(path);return data

def run(config):
 started=time.time();out=pathlib.Path(config['evidence_root']);out.mkdir(parents=True,exist_ok=True)
 summaries=[]
 for unit in config['units']:
  if time.time()-started>config['max_seconds']:raise RuntimeError('Storage operation time budget exceeded')
  root=pathlib.Path(unit['source']);assert root.resolve()==root and root.is_relative_to('/mnt/afs/260010165') and str(root) in config['allowed_sources']
  local=out/unit['id'];local.mkdir(exist_ok=True)
  expected_files={item['path'] for item in unit['files']}
  actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
  if actual!=expected_files:raise RuntimeError('Source file set changed')
  for item in unit['files']:
   path=root/item['path'];assert path.resolve().is_relative_to(root)
   current=identity(path)
   keys=('size','mtime_ns','nlink') if config.get('allow_mount_rebind') else ('size','mtime_ns','inode','device','nlink')
   if any(current[k]!=item[k] for k in keys):raise RuntimeError('Source identity changed')
   if config.get('allow_mount_rebind') and config['action']=='archive':item.update(current)
  if config['action']=='archive':
   if (local/'COMPLETE.json').exists():raise RuntimeError('Completed unit requires explicit reuse; will not overwrite')
   manifest={'version':1,'archive_id':config['archive_id'],'source':str(root),'bucket':'260010165','prefix':unit['prefix'],'created_at':time.time(),'files':[],'dependencies':config['dependencies']}
   for index,item in enumerate(unit['files']):
    path=root/item['path'];before=identity(path)
    uploaded=({'size':before['size'],'sha256':sha_file(path)} if item.get('existing_object_key') else request(item['put_url'],'PUT',path=path,tag=unit['id']))
    if uploaded['size']!=item['size'] or (item.get('expected_sha256') and uploaded['sha256']!=item['expected_sha256']):raise RuntimeError('Original content hash mismatch')
    if identity(path)!=before:raise RuntimeError('Source changed during upload')
    def progress(count,total):save(local/'TRANSFER.json',{'file':item['path'],'stage':'readback','bytes':count,'total':total,'time':time.time()})
    save(local/'TRANSFER.json',{'file':item['path'],'stage':'readback','bytes':0,'total':uploaded['size'],'time':time.time()})
    request(item['get_url'],'GET',expected=uploaded,fallback=item.get('get_url_public'),progress=progress)
    manifest['files'].append({**{k:v for k,v in item.items() if '_url' not in k},**uploaded,'mode':before['mode']})
    save(local/'MANIFEST.partial.json',manifest)
    print('STORAGE_PROGRESS '+json.dumps({'unit':unit['id'],'verified_files':index+1,'total_files':len(unit['files']),'verified_bytes':sum(f['size'] for f in manifest['files'])}),flush=True)
   data=save(local/'MANIFEST.json',manifest);mh=hashlib.sha256(data).hexdigest()
   request(unit['manifest_put'],'PUT',payload=data);request(unit['manifest_get'],'GET',expected={'size':len(data),'sha256':mh})
   receipt={'archive_id':config['archive_id'],'unit':unit['id'],'manifest_sha256':mh,'files':len(manifest['files']),'bytes':sum(f['size'] for f in manifest['files']),'full_https_origin_readback':True,'verified_at':time.time()}
   rb=save(local/'VERIFIED.json',receipt);rh=hashlib.sha256(rb).hexdigest();request(unit['receipt_put'],'PUT',payload=rb);request(unit['receipt_get'],'GET',expected={'size':len(rb),'sha256':rh})
   complete={'archive_id':config['archive_id'],'manifest_sha256':mh,'receipt_sha256':rh,'completed_at':time.time()};cb=encode(complete)
   request(unit['complete_put'],'PUT',payload=cb);request(unit['complete_get'],'GET',expected={'size':len(cb),'sha256':hashlib.sha256(cb).hexdigest()});save(local/'COMPLETE.json',complete)
   summary={**receipt,'source':str(root),'prefix':unit['prefix'],'state':'complete_source_retained'};summaries.append(summary)
  elif config['action']=='release':
   complete=json.loads((local/'COMPLETE.json').read_text());manifest=json.loads((local/'MANIFEST.json').read_text());receipt=(local/'VERIFIED.json').read_bytes();raw=(local/'MANIFEST.json').read_bytes();cb=(local/'COMPLETE.json').read_bytes()
   assert complete['archive_id']==config['archive_id'] and hashlib.sha256(raw).hexdigest()==complete['manifest_sha256'] and hashlib.sha256(receipt).hexdigest()==complete['receipt_sha256']
   assert {f['path'] for f in manifest['files']}==expected_files
   request(unit['complete_get'],'GET',expected={'size':len(cb),'sha256':hashlib.sha256(cb).hexdigest()})
   request(unit['manifest_get'],'GET',expected={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
   release_identities={}
   for item in manifest['files']:
    path=root/item['path'];before=identity(path)
    keys=('size','mtime_ns','nlink') if config.get('allow_mount_rebind') else ('size','mtime_ns','inode','device','nlink')
    if any(before[k]!=item[k] for k in keys) or sha_file(path)!=item['sha256'] or identity(path)!=before:raise RuntimeError('Source content changed before release')
    release_identities[item['path']]=before
   pointer={'state':'archived','bucket':'260010165','prefix':unit['prefix'],'archive_id':config['archive_id'],'manifest_sha256':complete['manifest_sha256'],'restore':'Download MANIFEST.json and COMPLETE.json; retrieve each files/<relative path> into a new directory and verify every SHA256. Keep file modes from manifest.'}
   save(root/'AFS_ARCHIVED.json',pointer)
   released=[]
   for item in manifest['files']:
    path=root/item['path'];before=identity(path)
    if before!=release_identities[item['path']]:raise RuntimeError('Source identity changed during release')
    path.unlink();released.append({'path':item['path'],'blocks':item['blocks'],'nlink':item['nlink']});save(local/'RELEASE.partial.json',released)
   result={'archive_id':config['archive_id'],'source':str(root),'released_files':len(released),'unlinked_allocated_bytes':sum(f['blocks']*512 for f in released if f['nlink']==1),'released_at':time.time(),'retained_pointer':str(root/'AFS_ARCHIVED.json')}
   data=save(local/'RELEASED.json',result);request(unit['release_put'],'PUT',payload=data);request(unit['release_get'],'GET',expected={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()});summaries.append(result)
   print('STORAGE_PROGRESS '+json.dumps(result),flush=True)
  else:raise RuntimeError('Unknown action')
 result={'action':config['action'],'archive_id':config['archive_id'],'units':summaries,'elapsed_seconds':time.time()-started};save(out/(config['action']+'_'+'_'.join(u['id'] for u in config['units'])+'_summary.json'),result)
 return result


def run_all(config):
 if config['action']!='archive' or len(config['units'])<2:return run(config)
 import concurrent.futures
 def one(unit):
  try:
   local=pathlib.Path(config['evidence_root'])/unit['id']
   if (local/'COMPLETE.json').exists():
    complete=(local/'COMPLETE.json').read_bytes();manifest=(local/'MANIFEST.json').read_bytes();assert hashlib.sha256(manifest).hexdigest()==json.loads(complete)['manifest_sha256']
    request(unit['complete_get'],'GET',expected={'size':len(complete),'sha256':hashlib.sha256(complete).hexdigest()})
    return {'units':[{'unit':unit['id'],'source':unit['source'],'prefix':unit['prefix'],'state':'complete_source_retained'}]}
   return run({**config,'units':[unit]})
  except Exception as error:
   value={'unit':unit['id'],'error_type':type(error).__name__,'error':str(error) if '://' not in str(error) else type(error).__name__,'time':time.time()}
   local=pathlib.Path(config['evidence_root'])/unit['id'];local.mkdir(parents=True,exist_ok=True);raw=save(local/'ERROR.json',value)
   if unit.get('error_put'):
    try:request(unit['error_put'],'PUT',payload=raw)
    except Exception:pass
   return {'units':[],'failure':value}
 with concurrent.futures.ThreadPoolExecutor(max_workers=config.get('archive_workers',4)) as pool:
  results=list(pool.map(one,config['units']))
 result={'action':config['action'],'archive_id':config['archive_id'],'units':[u for r in results for u in r['units']]}
 failures=[r['failure'] for r in results if 'failure' in r]
 if failures:result.update(error_type='PartialArchiveFailure',error='One or more units failed',failed_units=failures)
 return result
