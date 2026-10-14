"""Scope, checkpoint-commit, seed wiring and bounded queue regression checks."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
spec=importlib.util.spec_from_file_location('paired_controller',ROOT/'controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
from seed_protocol import SeededClient,SeededPolicy,FIELD,ACK,PROTOCOL


class CampaignTests(unittest.TestCase):
    def test_frozen_scope_and_no_training_control(self):
        plan=controller.load(ROOT/'plan.json')
        self.assertEqual(plan['pairs'],controller.load(ROOT/'pairs.json')['pairs'])
        self.assertEqual(len(plan['pairs']),20)
        self.assertEqual(len({p[0] for p in plan['pairs']}),20)
        self.assertEqual(plan['episodes'],40)
        self.assertEqual(plan['maximum_attempts_per_episode'],1)
        self.assertFalse(plan['modify_or_stop_training'])
        self.assertEqual(plan['training_iid'],234354)
        protocol=controller.load(ROOT/'protocol.json')
        self.assertEqual(protocol['environment_seed'],protocol['policy_seed'])
        self.assertTrue(protocol['native_task_step_limits'])
        self.assertEqual(plan['protocol_sha256'],controller.sha_bytes((ROOT/'protocol.json').read_bytes()))

    def test_unknown_and_starting_consume_slots_other_campaigns_absent(self):
        states=['UNKNOWN','PENDING','STARTING','RUNNING','SUCCEEDED','FAILED']
        rows={str(i):dict(intent={'gpu':1},observation={'state':s}) for i,s in enumerate(states)}
        rows['cpu']={'intent':{'gpu':0},'observation':{'state':'PENDING'}}
        self.assertEqual(controller.pending_and_active(rows),(3,4))

    def test_not_yet_uploaded_report_counts_as_zero_accepted(self):
        rows={'baseline-00':{'report':None}}
        self.assertIs(controller.accepted(rows,'baseline-00'),False)
        self.assertEqual(sum(controller.accepted(rows,id) for id in rows),0)

    def test_pending_gpu_spec_with_zero_allocated_replicas_still_requests_one(self):
        intent={'name':'own','uid':'uid','gpu':1,'image':'image'}
        job={'name':'own','uid':'uid','ownership':{'user_name':'260010165'},
             'workspace_name':'share-space','scheduling':{'quota_type':'SPOT'},
             'resource_pool':{'name':'computing-cluster-5090-01g'},'state':'PENDING',
             'roles':[{'image_path':'image','total_replicas':1,'resource_spec':[{'replicas':0,'limits':{'nvidia.com/gpu':'1'}}]}]}
        self.assertEqual(controller.verify_job(job,intent)['requested_gpus'],1)
        job['ownership']['user_name']='someone_else'
        with self.assertRaises(ValueError):controller.verify_job(job,intent)

    def test_baseline_pilot_can_start_without_candidate_checkpoint(self):
        plan=controller.load(ROOT/'plan.json');rows={};created=[]
        def submit(client,plan,id,mode,**extra):
            created.append(id);rows[id]={'intent':{'gpu':1},'observation':{'state':'PENDING'},'report':None}
        with patch.object(controller,'reconcile',side_effect=lambda *a:dict(rows)),patch.object(controller,'submit',side_effect=submit):
            controller.fill_baseline(None,plan,{})
        self.assertEqual(created,['baseline-00'])
        self.assertNotIn('export',rows)

    def test_baseline_fill_requires_accepted_pilot_and_stops_at_five_pending(self):
        plan=controller.load(ROOT/'plan.json')
        rows={'baseline-00':{'intent':{'gpu':1},'observation':{'state':'SUCCEEDED'},'report':{'state':'accepted'}}};created=[]
        def submit(client,plan,id,mode,**extra):
            created.append(id);rows[id]={'intent':{'gpu':1},'observation':{'state':'PENDING'},'report':None}
        with patch.object(controller,'reconcile',side_effect=lambda *a:dict(rows)),patch.object(controller,'submit',side_effect=submit):
            result=controller.fill_baseline(None,plan,dict(rows))
        self.assertEqual(len(created),5)
        self.assertEqual(controller.pending_and_active(result),(5,5))

    def test_actual_policy_rng_is_seeded_cache_reset_and_ack_required(self):
        calls=[];raw=SimpleNamespace(_is_pytorch_model=False,_rng='old')
        base=SimpleNamespace(policy=raw,reset=lambda:calls.append('reset'),act=lambda obs:raw._rng)
        policy=SeededPolicy(base,key_factory=lambda seed:('key',seed))
        with self.assertRaises(ValueError):policy.act({})
        self.assertEqual(policy.act({FIELD:42}),('key',42))
        self.assertEqual(policy.act({}),('key',42));self.assertEqual(calls,['reset'])
        for ack in (None,True,41,'42'):
            socket=SimpleNamespace(recv=lambda:json.dumps({ACK:ack}))
            net=SimpleNamespace(_ws=socket,get_server_metadata=lambda:{'full_episode_seed_protocol':PROTOCOL})
            base=SimpleNamespace(policy=net,reset=lambda:None,forward=lambda obs:net._ws.recv())
            with self.assertRaises(RuntimeError):SeededClient(base,{'policy_seed':42},unpack=json.loads).forward({})

    def test_evaluator_entry_keeps_native_loop_and_captures_terminal_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Native:
                def __init__(self):
                    socket=SimpleNamespace(recv=lambda:json.dumps({ACK:42}))
                    net=SimpleNamespace(_ws=socket,get_server_metadata=lambda:{'full_episode_seed_protocol':PROTOCOL})
                    self.policy=SimpleNamespace(policy=net,reset=lambda:None,forward=lambda obs:net._ws.recv())
                def reset(self,*a,**k):self.policy.reset()
                def step(self):self.policy.forward({});return False,True
            native=ModuleType('omnigibson.eval.eval');native.Evaluator=Native
            def main():
                self.assertEqual(native.DEFAULT_EVAL_SEED,42)
                e=native.Evaluator();e.reset(prepare_for_task_instance=True);e.reset()
                self.assertEqual(e.step(),(False,True))
            native.main=main
            package=ModuleType('omnigibson.eval');package.eval=native
            network=ModuleType('omnigibson.eval.utils.network_utils');network.unpackb=json.loads
            proof=Path(tmp)/'proof.json'
            with patch.dict(sys.modules,{'omnigibson':ModuleType('omnigibson'),'omnigibson.eval':package,
                    'omnigibson.eval.utils.network_utils':network}),patch.dict('os.environ',{'PAIRED_SEED_PROOF':str(proof)}):
                runpy.run_path(str(ROOT/'frozen/evaluator_entry.py'))
            self.assertEqual(json.loads(proof.read_text()),dict(protocol=PROTOCOL,policy_seed=42,server_seed_acknowledged=True,environment_seed=42))

    def probe(self,commit,step=5000):
        run=controller.load(ROOT/'plan.json')['run_id'];root='wangfaye/result/behavior-challenge/four_source_finetune/'+run
        ckpt=root+'/checkpoints/pi05_b1k/'+run+'/5000'
        data={'_CHECKPOINT_METADATA':{'commit_timestamp_nsecs':commit},
              'assets/formal_run_contract.json':{'run_id':run,'max_authorized_updates':20000},
              'assets/subtask_mixture.json':{'step':step,'consumed_samples':step*192},
              'params/_METADATA':{'tree':'valid'}}
        values={ckpt+'/'+k:json.dumps(v).encode() for k,v in data.items()}
        class Client:
            def __init__(self):self.bucket=self
            def head(self,k):return len(values[k]) if k in values else None
            def read(self,k):return values[k]
            def inventory(self,k):return {key:len(value) for key,value in values.items()}
            def head_object(self,k):return SimpleNamespace(content_length=len(values[k]),etag='etag')
            def sign_url(self,*a):return 'https://test.invalid/signed'
        module=ModuleType('nebula.submit_oss');module.OSSClient=Client
        with patch.dict(sys.modules,{'nebula':ModuleType('nebula'),'nebula.submit_oss':module}),contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(ROOT/'frozen/remote_probe.py'))
        return json.loads(output.getvalue())

    def test_checkpoint_requires_committed_exact_completed5000(self):
        for bad in (None,0,-1,'100'):
            with self.assertRaises(ValueError):self.probe(bad)
        with self.assertRaises(ValueError):self.probe(100,4999)
        snap=self.probe(100)
        self.assertEqual(snap['state'],'committed_checkpoint5000')
        self.assertTrue(all('train_state' not in row['relative_path'] for row in snap['files']))

    def test_bootstrap_source_is_hash_pinned_and_compiles(self):
        plan=controller.load(ROOT/'plan.json')
        source=controller.bootstrap(plan,dict(id='prepare',mode='prepare',plan=plan),'https://test.invalid/private')
        compile(source,'test_bootstrap','exec')
        self.assertNotIn('OSS_ACCESS_KEY',source)
        self.assertNotIn('AI_HUB_TOKEN',source)
        self.assertIn('exist_ok=False',source)


if __name__=='__main__':unittest.main()
