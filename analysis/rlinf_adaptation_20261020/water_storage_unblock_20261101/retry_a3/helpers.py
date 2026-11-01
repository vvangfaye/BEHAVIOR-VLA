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
 size=expected['size'];chunk_size=128*1024*1024;workers=8
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
 # Stream the entire origin response; avoid range fan-out during reclamation.
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
    if progress and count%(64*1024*1024)<len(block):progress(count,int(response.getheader('Content-Length') or 0))
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
