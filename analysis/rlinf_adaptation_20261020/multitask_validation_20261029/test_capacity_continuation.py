import unittest,time,json,pathlib,tempfile,hashlib
import io,types,sys,importlib.util
from unittest.mock import patch
from capacity_admission import evidence,acceptable,GIB
ROOT=pathlib.Path(__file__).resolve().parent

class CapacityTests(unittest.TestCase):
 def test_scaled_suite_limit_preserves_task_shards_and_rejects_larger_ceiling(self):
  from recovery_policy import global_gpu_limit
  with tempfile.TemporaryDirectory() as d:
   parent=pathlib.Path(d)/'multitask_validation_20261029';parent.mkdir();leaf=parent/'trash';leaf.mkdir()
   (parent/'campaign.json').write_text(json.dumps({'max_live_gpus':16}))
   self.assertEqual(global_gpu_limit(leaf,{'max_live_gpus':8}),16)
   shards=[list(range(rank,32,8)) for rank in range(8)]
   self.assertEqual(sorted(x for shard in shards for x in shard),list(range(32)))
   self.assertEqual(len(set(x for shard in shards for x in shard)),32)
   (parent/'campaign.json').write_text(json.dumps({'max_live_gpus':17}))
   with self.assertRaises(AssertionError):global_gpu_limit(leaf,{'max_live_gpus':8})
 def test_parallel_reconciliation_preserves_worker_identity(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);(root/'jobs').mkdir()
   for i in range(3):
    p=root/'jobs'/str(i);p.mkdir();(p/'submission.json').write_text(json.dumps({'job':str(i),'uid':str(i),'aoss_prefix':str(i)+'/'}))
   fake=types.SimpleNamespace(HERE=root,describe=lambda n:{'uid':n,'state':'RUNNING','ownership':{'user_name':'260010165'}},submit=None,write=None,aoss_client=None)
   spec=importlib.util.spec_from_file_location('reconcile_under_test',ROOT/'trash/control_support.py');module=importlib.util.module_from_spec(spec)
   with patch.dict(sys.modules,{'deploy':fake,'eval_utils':types.SimpleNamespace(paired_report=None)}):spec.loader.exec_module(module)
   client=types.SimpleNamespace(get_object=lambda **kw:{'ContentLength':2,'Body':io.BytesIO(b'{}')},exceptions=types.SimpleNamespace(ClientError=RuntimeError))
   rows=module.inspect(client);self.assertEqual([r['uid'] for r in rows],['0','1','2'])
   module.describe=lambda n:{'uid':'wrong','state':'RUNNING','ownership':{'user_name':'260010165'}}
   with self.assertRaises(AssertionError):module.inspect(client)
   self.assertTrue(module.retryable({'phase':'sample1','progress':{'error_type':'TimeoutError','error':'Simulator initialization/step timed out: ready'}}))
   self.assertFalse(module.retryable({'phase':'train1','progress':{'error_type':'TimeoutError','error':'Simulator initialization/step timed out: ready'}}))
   self.assertFalse(module.retryable({'phase':'sample1','progress':{'error_type':'TimeoutError','error':'Unknown timeout'}}))


 def test_both_trainers_cannot_share_one_reservation(self):
  q={'observed_at':time.time(),'method':'test','rows':[{'quota_bytes':200*GIB,'used_bytes':100*GIB}]}
  for slug,selected in [('trash',True),('water',False)]:
   p={'experiment':slug,'runtime_storage':{'max_persistent_bytes':48*GIB}}
   proof=evidence(p,q,selected,True)
   self.assertEqual(acceptable(proof,p),selected)
 def test_low_stale_or_wrong_task_proof_rejected(self):
  p={'experiment':'trash','runtime_storage':{'max_persistent_bytes':48*GIB}}
  q={'observed_at':time.time(),'method':'test','rows':[{'quota_bytes':200*GIB,'used_bytes':150*GIB}]}
  self.assertFalse(acceptable(evidence(p,q,True,False),p))
  q['rows'][0]['used_bytes']=100*GIB
  proof=evidence(p,q,True,False);self.assertTrue(acceptable(proof,p))
  proof['observed_at']-=7*3600;self.assertFalse(acceptable(proof,p))
 def test_rollout_and_optimizer_are_unchanged(self):
  for slug in ['trash','water']:
   p=json.loads((ROOT/slug/'plan.json').read_text())
   self.assertEqual(len(p['train_instances']),32)
   self.assertEqual(p['optimizer']['updates_per_round'],[17])
   self.assertEqual(hashlib.sha256((ROOT/slug/'training.py').read_bytes()).hexdigest(),
                    'ff35bf3f660bf16695b53c0d57129d4b022d62d156c2a3993cba478588165b1b')

class ArchiveTests(unittest.TestCase):
 def exercise(self,corrupt=False,worker="archive_worker.py"):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);source=root/'old/optimizer.pt';source.parent.mkdir();source.write_bytes(b'original model bytes')
   s=source.stat();objects={}
   helpers='''import hashlib, pathlib, json, stat
'''+(ROOT.parent.parent/'rlinf_storage_20261024/archive_helpers.py').read_text() if False else None
   import importlib.util
   spec=importlib.util.spec_from_file_location('old_archive_helpers',ROOT.parent.parent/'rlinf_storage_20261024/archive_helpers.py')
   helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
   def request(url,method,payload=None,path=None,expected=None,**kwargs):
    if method=='PUT':objects[url]=path.read_bytes() if path else payload
    data=objects[url];v={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    if method=='GET' and url=='blob' and corrupt:raise RuntimeError('readback mismatch')
    if expected is not None and v!=expected:raise RuntimeError('mismatch')
    return v
   ns={'__name__':'test_archive'}
   script=(ROOT/'capacity_repair_20261030'/worker).read_text().replace('/mnt/afs/260010165/behavior_1k',str(source.parent))
   exec(script,ns)
   ns.update({n:getattr(helper,n) for n in ['identity','encode','sha_file','save']});ns['request']=request
   row=dict(path=str(source),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,nlink=1,key='archive-key',put='blob',get='blob',manifest_put='manifest',manifest_get='manifest',complete_put='complete',complete_get='complete')
   cfg={'helpers_source':'','output':str(root/'audit'),'status_put':'status','files':[row],'allowed_roots':[str(source.parent)],'deadline_seconds':60}
   if corrupt:
    with self.assertRaises(RuntimeError):ns['run'](cfg)
    self.assertTrue(source.exists());self.assertNotIn('complete',objects)
   else:
    ns['run'](cfg);self.assertFalse(source.exists());self.assertTrue(source.with_name('optimizer.pt.aoss.json').exists())
    self.assertIn('complete',objects);self.assertEqual(objects['blob'],b'original model bytes')
 def test_verified_archive_then_release(self):self.exercise()
 def test_failed_readback_preserves_source(self):self.exercise(corrupt=True)
 def test_parallel_archive_verified_before_release(self):self.exercise(worker="archive_worker_parallel.py")
 def test_parallel_archive_failure_keeps_source(self):self.exercise(corrupt=True,worker="archive_worker_parallel.py")
 def test_streaming_archive_verified_before_release(self):self.exercise(worker="archive_worker_streaming.py")
 def test_streaming_archive_failure_keeps_source(self):self.exercise(corrupt=True,worker="archive_worker_streaming.py")

if __name__=='__main__':unittest.main()
