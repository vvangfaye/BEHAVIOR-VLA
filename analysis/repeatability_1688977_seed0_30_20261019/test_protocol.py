"""Local validation of the repeat schedule and read-only evaluator instrumentation."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
import zlib
import numpy as np

ROOT=Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


class ProtocolTests(unittest.TestCase):
    def test_thirty_unique_repetitions_in_ten_gpu_groups(self):
        plan=json.loads((ROOT/'plan.json').read_text())
        assembly=module('repeat_assembly_test',ROOT/'assemble.py')
        files={p.name:p.read_text() for p in (ROOT/'frozen').glob('*.py')}
        files['batch_runner.py']='def run(config):\n global captured\n captured=config\n return 0\n'
        urls={g['id']:{'report':'http://test.invalid/group'} for g in plan['groups']}
        for i in range(30):urls[f'baseline-{i:02d}']={k:'http://test.invalid/'+k for k in ['report','heartbeat','diagnostics']}
        outputs=assembly.assemble(dict(plan=plan,files=files,urls=urls,bundle_hash='test'))
        identities=[]
        for key,source in outputs.items():
            ns={}
            try:exec(compile(source,key,'exec'),ns)
            except SystemExit as exc:self.assertEqual(exc.code,0)
            self.assertEqual(len(ns['captured']['members']),3)
            for member in ns['captured']['members']:
                identities.append(member['id'])
                decoded=zlib.decompress(base64.b64decode(member['source'])).decode()
                compile(decoded,member['id'],'exec')
        self.assertEqual(sorted(identities),[f'baseline-{i:02d}' for i in range(30)])
        self.assertEqual(len(set(identities)),30)
        for group in plan['groups']:
            self.assertEqual({m['repeat'] for m in group['members']},{0,1,2})
            self.assertEqual(len({tuple(plan['pairs'][m['index']]) for m in group['members']}),1)

    def test_snapshot_is_read_only_and_every_action_is_recorded(self):
        class Object:
            name='robot';n_joints=2
            def get_position_orientation(self):return np.array([0.,1.,2.]),np.array([0.,0.,0.,1.])
            def get_joint_positions(self):return np.array([.1,.2])
            def get_joint_velocities(self):return np.zeros(2)
        class Policy:
            def reset(self):pass
        class Original:
            def __init__(self):
                self.env=types.SimpleNamespace(scene=types.SimpleNamespace(objects=[Object()]))
                self.obs={'rgb':np.zeros((2,2,3),dtype=np.uint8),'state':np.zeros(2)}
                self.policy=Policy();self.native_resets=0;self.native_steps=0
            def reset(self,*a,**k):self.native_resets+=1
            def step(self,*a,**k):
                self.native_steps+=1;self.robot_action=np.array([self.native_steps,.25])
                return self.native_steps==2,False
        native=types.ModuleType('omnigibson.eval.eval');native.Evaluator=Original;native.main=lambda:None
        evaluation=types.ModuleType('omnigibson.eval');evaluation.eval=native
        og=types.ModuleType('omnigibson');og.eval=evaluation
        seed=module('repeat_seed_test',ROOT/'frozen/seed_protocol.py')
        saved={k:sys.modules.get(k) for k in ['omnigibson','omnigibson.eval','omnigibson.eval.eval','seed_protocol']}
        sys.modules.update({'omnigibson':og,'omnigibson.eval':evaluation,'omnigibson.eval.eval':native,'seed_protocol':seed})
        old_env={k:os.environ.get(k) for k in ['REPEATABILITY_OUTPUT','PAIRED_SEED_PROOF']}
        try:
            with tempfile.TemporaryDirectory() as temp:
                os.environ['REPEATABILITY_OUTPUT']=temp;os.environ['PAIRED_SEED_PROOF']=temp+'/seed.json'
                result=runpy.run_path(str(ROOT/'frozen/evaluator_entry.py'))
                obj=result['Evaluator']();obj.reset()
                state=json.loads(Path(temp+'/initial_state.json').read_text())
                self.assertEqual(obj.native_resets,1)
                self.assertEqual(state['objects']['robot']['joint_positions'],[.1,.2])
                obj.policy.acknowledged=True
                self.assertEqual(obj.step(),(False,False));self.assertEqual(obj.step(),(True,False))
                self.assertEqual(obj.native_steps,2)
                trace=json.loads(Path(temp+'/action_trace.json').read_text())
                self.assertEqual((trace['steps'],trace['action_dim']),(2,2))
                np.testing.assert_array_equal(np.fromfile(temp+'/actions.f32',dtype='<f4'),[1,.25,2,.25])
                self.assertEqual(json.loads(Path(temp+'/seed.json').read_text()),
                    dict(protocol=seed.PROTOCOL,policy_seed=0,server_seed_acknowledged=True,environment_seed=0))
        finally:
            for k,v in saved.items():
                if v is None:sys.modules.pop(k,None)
                else:sys.modules[k]=v
            for k,v in old_env.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v


if __name__=='__main__':unittest.main()
