import pathlib,tempfile,hashlib,json,time
import delete_regenerable_cache as w

def test(corrupt):
 with tempfile.TemporaryDirectory() as tmp:
  root=pathlib.Path(tmp)/'behavior_1k'/'rlinf_conservative_20261024';cache=root/'pilot/results/i1/appdata';cache.mkdir(parents=True)
  p=cache/'shader.bin';p.write_bytes(b'x'*65536);data=root/'pilot/results/rollout.pt';data.write_bytes(b'important')
  store={}
  def request(url,method,payload=None,expected=None,**kw):
   key=url.split(':')[-1]
   if method=='PUT':store[key]=payload;raw=payload
   else:raw=store[key] if not corrupt else b'corrupt'
   found={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   if expected is not None:assert found==expected
   return found
  w.request=request
  cfg={'roots':[str(root)],'cutoff_ns':time.time_ns()+1,'max_allocated_bytes':1024**2,'active_dependency_audit_sha256':'test','manifest_put':'put:m','manifest_get':'get:m','manifest_key':'m','status_put':'put:s','deadline_seconds':60}
  try:r=w.run(cfg)
  except AssertionError:
   assert corrupt and p.exists() and data.read_bytes()==b'important';return
  assert not corrupt and not p.exists() and data.read_bytes()==b'important' and r['write_probe_succeeded']
for corrupt in [False,True]:test(corrupt)
print(json.dumps({'passed':True,'research_artifacts_retained':True,'corrupt_manifest_prevents_delete':True}))
