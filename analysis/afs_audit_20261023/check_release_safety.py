"""Behavioral checks for fail-closed cleanup; no cloud access."""
import importlib.util,pathlib,tempfile,hashlib,json
spec=importlib.util.spec_from_file_location('worker',pathlib.Path(__file__).with_name('storage_worker.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

def case(mode):
 with tempfile.TemporaryDirectory(dir='/tmp') as temp:
  base=pathlib.Path(temp);source=base/'source';source.mkdir();f=source/'weights';f.write_bytes(b'preserve this checkpoint')
  # The production allowlist remains unchanged; test only the Path membership guard.
  real_relative=w.pathlib.Path.is_relative_to
  w.pathlib.Path.is_relative_to=lambda self,other: real_relative(self,other) or str(other)=='/mnt/afs/260010165'
  item={'path':'weights',**w.identity(f),'put_url':'weights','get_url':'weights'}
  unit={'id':'u','source':str(source),'prefix':'test','files':[item]}
  for label in ['manifest','receipt','complete','release']:
   unit[label+'_put']=label;unit[label+'_get']=label
  config={'evidence_root':str(base/'evidence'),'archive_id':'safety-check','max_seconds':60,'units':[unit],'dependencies':{},'action':'archive','allowed_sources':[str(source)]}
  objects={}
  def request(url,method,payload=None,path=None,expected=None,tag=None,**kwargs):
   if method=='PUT':objects[url]=path.read_bytes() if path else payload
   if method=='GET' and mode=='corrupt':raise RuntimeError('Injected failed readback')
   raw=objects[url];value={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   if expected is not None:assert value==expected
   return value
  w.request=request
  try:
   try:w.run(config)
   except RuntimeError:
    assert mode=='corrupt' and f.exists();return
   if mode=='mutate':f.write_bytes(b'changed after verified backup')
   config['action']='release'
   try:w.run(config)
   except RuntimeError:
    assert mode=='mutate' and f.exists();return
   assert mode=='success' and not f.exists() and (source/'AFS_ARCHIVED.json').is_file()
  finally:w.pathlib.Path.is_relative_to=real_relative
for mode in ['corrupt','mutate','success']:case(mode)
print('PASS: failed readback and changed source preserve checkpoint; verified stable source releases with recovery pointer')
