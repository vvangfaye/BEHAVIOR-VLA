"""Precisely remove inactive, regenerable simulation caches; retain all research artifacts."""
import concurrent.futures,gzip,hashlib,json,os,pathlib,stat,subprocess,time
from archive_helpers import request,encode
ALLOWED={'rlinf_multiround_20261023','rlinf_adapter_20261023','rlinf_conservative_20261024'}
CACHE={'cache','warp','appdata'}

def identify(path):
 s=path.lstat()
 return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'dev':s.st_dev,'ino':s.st_ino,'nlink':s.st_nlink,'blocks':s.st_blocks,'regular':stat.S_ISREG(s.st_mode)}

def gather(roots,cutoff_ns):
 selected=[];directories=[];skipped=[]
 for root in map(pathlib.Path,roots):
  assert root.name in ALLOWED and root.parent.name=='behavior_1k' and root.resolve()==root
  for folder,subdirs,files in os.walk(root,followlinks=False):
   folder=pathlib.Path(folder);rel=folder.relative_to(root)
   subdirs[:]=[n for n in subdirs if not (folder/n).is_symlink() and n not in ['checkpoints','checkpoint']]
   if not any(x in CACHE for x in rel.parts):continue
   directories.append(str(folder))
   for name in files:
    p=folder/name;s=identify(p)
    if not s['regular'] or s['nlink']!=1 or s['mtime_ns']>=cutoff_ns or p.resolve()!=p:
     skipped.append({'path':str(p),'reason':'linked/special/recent'});continue
    selected.append({'path':str(p),**s})
 return selected,directories,skipped

def measure(roots):
 totals={};errors=[]
 for root in roots:
  total=0;seen=set()
  def onerror(error):errors.append(str(error))
  for folder,subdirs,files in os.walk(root,followlinks=False,onerror=onerror):
   for name in files:
    try:
     value=pathlib.Path(folder,name).lstat();key=(value.st_dev,value.st_ino)
     if key not in seen:total+=value.st_blocks*512;seen.add(key)
    except OSError as error:errors.append(str(error))
  totals[root]=total
 return totals,errors

def run(config):
 started=time.time();roots=config['roots']
 files,directories,skipped=gather(roots,config['cutoff_ns'])
 total=sum(x['blocks']*512 for x in files)
 assert 0<total<=config['max_allocated_bytes'] and len(files)<=250000
 manifest={'source_roots':roots,'files':files,'directories':directories,'skipped':skipped,
           'reason':'Inactive per-episode NVIDIA/Isaac/Warp caches, reconstructible from retained code and runtime.',
           'not_a_research_data_backup':True,'content_backup_required':False,
           'retain':['all checkpoints and optimizer files','raw rollouts','videos','logs','initial states','experiment configs'],
           'cutoff_ns':config['cutoff_ns'],'active_dependency_audit_sha256':config['active_dependency_audit_sha256'],
           'generated_at':time.time(),'allocated_bytes':total}
 raw=gzip.compress(encode(manifest),compresslevel=1);signature={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 request(config['manifest_put'],'PUT',payload=raw)
 request(config['manifest_get'],'GET',expected=signature)
 before,before_errors=measure(roots)
 # Whole manifest identity preflight completes before the first unlink.
 for f in files:
  assert identify(pathlib.Path(f['path']))=={k:v for k,v in f.items() if k!='path'}, 'Cache changed since inventory'
 count=0;released=0;errors=[]
 def one(f):
  p=pathlib.Path(f['path'])
  try:
   if identify(p)!={k:v for k,v in f.items() if k!='path'}:return (0,0,{'path':str(p),'error':'changed before unlink'})
   p.unlink();return (1,f['blocks']*512,None)
  except OSError as e:return (0,0,{'path':str(p),'error':str(e)})
 def publish(stage):
  value={'state':stage,'manifest_key':config['manifest_key'],'manifest_sha256':signature['sha256'],
         'metadata_manifest_verified':True,'content_archival':False,'reconstruction':'Re-run the retained simulator environment; shader caches are rebuilt.',
         'selected_files':len(files),'selected_allocated_bytes':total,'removed_files':count,'removed_allocated_bytes':released,
         'errors':errors[:30],'error_count':len(errors),'elapsed_seconds':time.time()-started}
  request(config['status_put'],'PUT',payload=encode(value));return value
 publish('manifest_verified')
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  # Bounded windows prevent queuing the entire filesystem at once.
  for offset in range(0,len(files),2000):
   if time.time()-started>config['deadline_seconds']:break
   for n,b,e in pool.map(one,files[offset:offset+2000]):
    count+=n;released+=b
    if e:errors.append(e)
   publish('removing_regenerable_cache')
 for folder in sorted(directories,key=lambda x:len(pathlib.Path(x).parts),reverse=True):
  try:pathlib.Path(folder).rmdir()
  except OSError:pass
 after,after_errors=measure(roots)
 result=publish('complete' if count==len(files) else 'partial')
 result.update(directory_before=before,directory_after=after,directory_reduction_bytes=sum(before.values())-sum(after.values()),measurement_errors=before_errors+after_errors)
 probe=pathlib.Path(roots[0]).parent/('.storage_write_probe_'+str(time.time_ns()))
 try:
  with probe.open('xb') as f:f.write(b'1'*4096);f.flush();os.fsync(f.fileno())
  probe.unlink();result['write_probe_succeeded']=True
 except OSError as e:result['write_probe_succeeded']=False;result['write_error']=str(e)
 request(config['status_put'],'PUT',payload=encode(result))
 return result
