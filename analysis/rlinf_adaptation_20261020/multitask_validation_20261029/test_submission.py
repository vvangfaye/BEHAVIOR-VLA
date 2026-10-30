"""Exercise real bundle construction with cloud calls mocked; allocate no jobs."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import time
from unittest.mock import patch
from capacity_admission import evidence,GIB

ROOT=Path(__file__).resolve().parent


class FakeClient:
    def __init__(self):self.objects={}
    def put_object(self,**kw):self.objects[kw['Key']]=kw['Body']
    def get_object(self,**kw):return {'Body':io.BytesIO(self.objects[kw['Key']])}
    def generate_presigned_url(self,*args,**kw):return 'https://invalid.local/test-only'
    def close(self):pass


class SubmissionTests(unittest.TestCase):
    def test_training_rejects_storage_before_creating_an_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'trash'
            shutil.copytree(ROOT/'trash',folder,ignore=shutil.ignore_patterns('__pycache__','jobs','retrieved'))
            spec=importlib.util.spec_from_file_location('test_capacity_deploy',ROOT/'trash/deploy.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.HERE=folder
            (folder/'training_storage_clearance.json').write_text(json.dumps({'passed':False}))
            with patch.object(module,'describe',side_effect=AssertionError('must not contact cloud')):
                with self.assertRaises(AssertionError):module.submit('sample1',0)
            self.assertFalse((folder/'jobs').exists())

    def test_both_task_bundles_preserve_original_checkpoint_and_exact_predicates(self):
        for slug,phase in [('trash','baseline'),('water','baseline'),('trash','sample1')]:
            with self.subTest(task=slug,phase=phase),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory)/slug
                shutil.copytree(ROOT/slug,folder,ignore=shutil.ignore_patterns('__pycache__','jobs','retrieved'))
                spec=importlib.util.spec_from_file_location('test_deploy_'+slug,ROOT/slug/'deploy.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                module.HERE=folder
                if phase!='baseline':
                    plan=json.loads((folder/'plan.json').read_text())
                    quota={'observed_at':time.time(),'method':'test only',
                           'rows':[{'quota_bytes':4096*GIB,'used_bytes':1000*GIB}]}
                    (folder/'training_storage_clearance.json').write_text(json.dumps(evidence(plan,quota,True,True)))
                client=FakeClient();commands=[]
                def execute(args,**kwargs):
                    commands.append(args)
                    return subprocess.CompletedProcess(args,0,'','')
                def describe(name):
                    template={'ownership':{'user_name':'260010165'},
                              'resource_pool':{'name':'computing-cluster-5090-01g'},
                              'roles':[{'resource_spec':[{'name':'n12lp.nn.i10a.1'}],'image_path':'test-image'}]}
                    if name=='b1k-repeat1688-1019-batch-00':return template
                    return {**template,'uid':'test-uid','state':'QUEUEING',
                            'scheduling':{'quota_type':'RESERVED'},
                            'roles':[{'startup_script':next(x.split('=',1)[1] for x in commands[-1] if x.startswith('--command='))}]}
                with patch.object(module,'describe',describe),patch.object(module,'aoss_client',lambda:client),\
                     patch.object(module,'query_standard_capacity',lambda *a:{'used_gpus':0,'standard_limit':4}),\
                     patch.object(module.subprocess,'run',execute):
                    receipt=module.submit(phase,0)
                self.assertIn(slug,receipt['job']);self.assertEqual(receipt['quota'],'reserved')
                blob=next(iter(client.objects.values()))
                with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
                    plan=json.load(archive.extractfile('run/plan.json'))
                    observer=archive.extractfile('run/goals.py').read().decode()
                    self.assertIn('run/capacity_admission.py',archive.getnames())
                    self.assertIn('class TaskObserver',observer)
                    self.assertEqual(len(plan['goal_contract']['predicates']),3)
                    self.assertIn('original1688977',plan['original'])
                    self.assertEqual(plan['optimizer']['actor_lr'],1e-7)
                    self.assertEqual(plan['rounds'],1)
                    self.assertFalse(any(n.endswith(('.pt','.safetensors')) for n in archive.getnames()))


if __name__=='__main__':unittest.main(verbosity=2)
