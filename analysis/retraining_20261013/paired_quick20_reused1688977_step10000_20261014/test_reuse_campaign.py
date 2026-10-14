import copy
import contextlib
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
spec=importlib.util.spec_from_file_location('reuse_controller',ROOT/'controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
import scorer
from seed_protocol import ACK,PROTOCOL,SeededPolicy,SeededClient,FIELD


class Tests(unittest.TestCase):
    def test_three_way_reuses_real5000_and_preserves_seed_provenance(self):
        plan=controller.load(ROOT/'plan.json');base=controller.load(ROOT/'baseline_accepted.json')
        previous=controller.load(ROOT/'step5000_candidate_accepted.json');candidate=copy.deepcopy(previous)
        candidate['checkpoint_manifest_sha256']='2'*64
        result=scorer.three_way(base,previous,candidate,plan['pairs'])
        self.assertEqual([r['checkpoint'] for r in result['summaries']],[1688977,5000,10000])
        self.assertEqual([r['successes'] for r in result['summaries']],[0,1,1])
        self.assertEqual(result['comparisons']['5000_vs1688977']['candidate_completed_updates'],5000)
        self.assertEqual(result['comparisons']['10000_vs5000']['mean_q_delta'],0)
        with self.assertRaises(ValueError):scorer.three_way(base,previous,previous,plan['pairs'])
        candidate['protocol_sha256']='3'*64
        with self.assertRaises(ValueError):scorer.three_way(base,previous,candidate,plan['pairs'])
        candidate=copy.deepcopy(previous);candidate['checkpoint_manifest_sha256']='2'*64
        previous=copy.deepcopy(previous);previous['records'][0]['policy_seed_acknowledged']=False
        with self.assertRaises(ValueError):scorer.three_way(base,previous,candidate,plan['pairs'])

    def test_checkpoint5000_cannot_be_exported_as10000(self):
        import worker
        plan=controller.load(ROOT/'plan.json')
        snapshot=dict(state='committed_checkpoint10000',formal_contract=plan['formal_contract'],
                      consumed_step=5000,metadata={'commit_timestamp_nsecs':1})
        with self.assertRaises(ValueError):worker.export({'plan':plan,'snapshot':snapshot})

    def test_parallel_admission_does_not_wait_for_pilot_and_keeps_pending_limit(self):
        for limit in (5,20):
            with self.subTest(limit=limit):
                plan=controller.load(ROOT/'plan.json');plan['max_pending']=limit
                rows={'export':{'intent':{'gpu':0},'observation':{'state':'SUCCEEDED'},'report':{'state':'accepted'}},
                      'candidate-00':{'intent':{'gpu':1},'observation':{'state':'STARTING'},'report':None}}
                created=[]
                def submit(client, plan, id, mode, **extra):
                    self.assertNotIn(id, rows)
                    created.append(id)
                    rows[id]={'intent':{'gpu':1},'observation':{'state':'CREATING'},'report':None}
                with patch.object(controller,'reconcile',return_value=rows),patch.object(controller,'save'),patch.object(controller,'submit',side_effect=submit):
                    self.assertFalse(controller.tick(None,plan))
                self.assertEqual(created,[f'candidate-{i:02d}' for i in range(1,limit)])
                self.assertEqual(controller.pending_and_active(rows)[0],limit)
                self.assertIsNone(rows['candidate-00']['report'])

    def test_large_consumed_asset_bounded_read_and_integrity(self):
        module = ModuleType('nebula.submit_oss')
        module.OSSClient = lambda: SimpleNamespace(head=lambda key: None)
        with patch.dict(sys.modules, {'nebula.submit_oss': module}), contextlib.redirect_stdout(io.StringIO()):
            namespace = runpy.run_path(str(ROOT/'frozen/remote_probe.py'))
        value = dict(schema_version=1, kind='b1k_consumed_mixture_contract', step=5000,
                     consumed_samples=960000, index='x' * (5 * 1024**2))
        canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        value['fingerprint'] = hashlib.sha256(canonical).hexdigest()
        raw = json.dumps(value).encode()
        def client(data, size=None):
            stream = io.BytesIO(data)
            body = SimpleNamespace(read=stream.read, resp=SimpleNamespace(response=stream))
            return SimpleNamespace(head=lambda key: len(data) if size is None else size,
                                   bucket=SimpleNamespace(get_object=lambda key: body))
        read = namespace['read_consumed_mixture']
        got, digest = read(client(raw), 'asset')
        self.assertEqual(got['step'], 5000)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError): read(client(raw), 'asset', max_bytes=4*1024**2)
        with self.assertRaises(ValueError): read(client(raw[:-1], len(raw)), 'asset')
        value['step'] = 4999
        with self.assertRaises(ValueError): read(client(json.dumps(value).encode()), 'asset')
        select = namespace['inference_objects']
        self.assertEqual(select({'ck/params/': 0, 'ck/params/_METADATA': 5,
                                 'ck/assets/norm.json': 4, 'ck/train_state/data': 8}, 'ck'),
                         {'ck/params/_METADATA': 5, 'ck/assets/norm.json': 4})
        with self.assertRaises(ValueError): select({'ck/params/data': 0}, 'ck')

    def test_scope_new20_reused20_with_frozen_public_instances(self):
        p=controller.load(ROOT/'plan.json');s=controller.load(ROOT/'selection_frozen.json')
        self.assertEqual(p['pairs'],s['pairs']);self.assertEqual(p['episodes'],20)
        self.assertEqual(p['baseline_reused_episodes'],20);self.assertEqual(p['new_baseline_rollouts'],0)
        self.assertEqual(len({t for t,_ in p['pairs']}),20)
        self.assertTrue(all(i==301 for _,i in p['pairs']))
        self.assertTrue(p['training_seen']);self.assertFalse(p['modify_or_stop_training'])

    def test_historical_baseline_has_complete_raw_results_and_no_fabricated_ack(self):
        p=controller.load(ROOT/'plan.json');b=controller.load(ROOT/'baseline_accepted.json')
        rows=scorer.validate(b,p['pairs'],'baseline')
        self.assertEqual(len(rows),20)
        self.assertTrue(all(r['policy_seed_acknowledged'] is False for r in rows.values()))
        self.assertTrue(all(r['policy_seed_source_verified'] for r in rows.values()))

    def test_baseline_ack_cannot_be_fabricated_and_candidate_requires_ack(self):
        p=controller.load(ROOT/'plan.json');b=controller.load(ROOT/'baseline_accepted.json')
        b=copy.deepcopy(b);b['records'][0]['policy_seed_acknowledged']=True
        with self.assertRaises(ValueError):scorer.validate(b,p['pairs'],'baseline')
        b['records'][0]['policy_seed_acknowledged']=False
        with self.assertRaises(ValueError):scorer.validate(b,p['pairs'],'candidate')

    def test_success_paired_comparison_rejects_missing_and_changed_protocol(self):
        p=controller.load(ROOT/'plan.json');b=controller.load(ROOT/'baseline_accepted.json');c=copy.deepcopy(b)
        c['checkpoint_manifest_sha256']='1'*64
        for row in c['records']:row['policy_seed_acknowledged']=True
        result=scorer.compare(b,c,p['pairs'])
        self.assertEqual(result['baseline_successes'],result['candidate_successes'])
        self.assertEqual(result['improved_pairs'],[]);self.assertTrue(result['training_seen'])
        c['records'].pop()
        with self.assertRaises(ValueError):scorer.compare(b,c,p['pairs'])
        c=copy.deepcopy(b);c['protocol_sha256']='2'*64
        with self.assertRaises(ValueError):scorer.compare(b,c,p['pairs'])

    def test_controller_refuses_any_new_baseline_rollout(self):
        p=controller.load(ROOT/'plan.json')
        with self.assertRaises(ValueError):controller.submit(None,p,'baseline-00','evaluate',role='baseline',pair=p['pairs'][0])

    def test_unknown_and_starting_requests_hold_pending_slot(self):
        rows={str(i):{'intent':{'gpu':1},'observation':{'state':s},'report':None} for i,s in enumerate(['UNKNOWN','PENDING','STARTING','RUNNING','SUCCEEDED'])}
        self.assertEqual(controller.pending_and_active(rows),(3,4))
        self.assertIs(controller.accepted(rows,'0'),False)

    def test_candidate_evaluator_applies_zero_seed_and_keeps_native_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Native:
                def __init__(self):
                    socket=SimpleNamespace(recv=lambda:json.dumps({ACK:0}))
                    net=SimpleNamespace(_ws=socket,get_server_metadata=lambda:{'full_episode_seed_protocol':PROTOCOL})
                    self.policy=SimpleNamespace(policy=net,reset=lambda:None,forward=lambda obs:net._ws.recv())
                def reset(self,*a,**k):self.policy.reset()
                def step(self):self.policy.forward({});return False,True
            native=ModuleType('omnigibson.eval.eval');native.Evaluator=Native
            def main():
                self.assertEqual(native.DEFAULT_EVAL_SEED,0)
                e=native.Evaluator();e.reset(prepare_for_task_instance=True);e.reset();self.assertEqual(e.step(),(False,True))
            native.main=main;package=ModuleType('omnigibson.eval');package.eval=native
            network=ModuleType('omnigibson.eval.utils.network_utils');network.unpackb=json.loads
            proof=Path(tmp)/'proof.json'
            with patch.dict(sys.modules,{'omnigibson':ModuleType('omnigibson'),'omnigibson.eval':package,'omnigibson.eval.utils.network_utils':network}),patch.dict('os.environ',{'PAIRED_SEED_PROOF':str(proof)}):
                runpy.run_path(str(ROOT/'frozen/evaluator_entry.py'))
            self.assertEqual(json.loads(proof.read_text()),dict(protocol=PROTOCOL,policy_seed=0,server_seed_acknowledged=True,environment_seed=0))

    def test_actual_policy_rng_zero_and_wrong_seed_ack_rejected(self):
        raw=SimpleNamespace(_is_pytorch_model=False,_rng='old');base=SimpleNamespace(policy=raw,reset=lambda:None,act=lambda obs:raw._rng)
        seeded=SeededPolicy(base,key_factory=lambda seed:('key',seed));self.assertEqual(seeded.act({FIELD:0}),('key',0))
        sock=SimpleNamespace(recv=lambda:json.dumps({ACK:42}));net=SimpleNamespace(_ws=sock,get_server_metadata=lambda:{'full_episode_seed_protocol':PROTOCOL})
        cli=SimpleNamespace(policy=net,reset=lambda:None,forward=lambda obs:net._ws.recv())
        with self.assertRaises(RuntimeError):SeededClient(cli,{'policy_seed':0},unpack=json.loads).forward({})

    def test_bootstrap_pinned_and_no_optimizer_export(self):
        p=controller.load(ROOT/'plan.json');s=controller.bootstrap(p,{'id':'export','mode':'export','plan':p,'worker_bundle_sha256':'0'*64},'https://test.invalid/private')
        compile(s,'bootstrap','exec');self.assertNotIn('OSS_ACCESS_KEY',s)
        self.assertIn("'train_state' in relative.parts",(ROOT/'frozen/worker.py').read_text())


if __name__=='__main__':unittest.main()
