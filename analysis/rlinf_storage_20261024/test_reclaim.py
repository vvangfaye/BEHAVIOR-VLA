import hashlib,json,pathlib,tempfile
import reclaim_worker as worker

def exercise(corrupt=False):
 with tempfile.TemporaryDirectory() as temp:
  root=pathlib.Path(temp)/'cache';root.mkdir();p=root/'test.bin';p.write_bytes(b'archive-preserve-release-test'*8)
  original=p.read_bytes();meta=worker.identity(p);store={};gets=0
  def fake(url,method,payload=None,expected=None,**kwargs):
   nonlocal gets
   key=url.split(':',1)[-1]
   if method=='PUT':store[key]=payload;data=payload
   else:
    data=store[key];gets+=1
    if corrupt and key.startswith('part'):data=b'broken'
   v={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   if expected is not None and expected!=v:raise ValueError('checksum mismatch')
   return v
  worker.request=fake
  parts=[]
  size=32
  for i,off in enumerate(range(0,len(original),size)):
   key='part'+str(i);parts.append({'key':key,'put':'put:'+key,'get':'get:'+key,'bytes':min(size,len(original)-off)})
  item={'path':str(p),**meta,'parts':parts,'reason':'test derived cache','manifest_key':'manifest'}
  for label in ['manifest','verified','complete']:
   item[label+'_put']='put:'+label;item[label+'_get']='get:'+label
  result=worker.run({'roots':[str(root)],'archive_id':'test','deadline_seconds':60,'status_put':'put:status','part_bytes':size,'files':[item]})
  if corrupt:
   assert p.read_bytes()==original and result['released_files']==0
  else:
   assert not p.exists() and result['released_files']==1
   manifest=json.loads(store['manifest']);recovered=b''.join(store[x['key']] for x in manifest['parts'])
   assert recovered==original and hashlib.sha256(recovered).hexdigest()==manifest['sha256']
  return {'corrupt':corrupt,'passed':True,'verified_gets':gets}
print(json.dumps([exercise(),exercise(True)]))
