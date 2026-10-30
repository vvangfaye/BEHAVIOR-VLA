import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import time
from unittest.mock import patch

from goals import TaskObserver,potential,shaping_bonus
from storage_guard import admit_storage

ROOT=Path(__file__).resolve().parent


class GoalTests(unittest.TestCase):
    def test_progress_can_be_lost_and_terminal_discounted_shaping_telescopes(self):
        config={'gamma':.99,'beta':1.}
        def state(a,b,c):return {'goals':dict(a=a,b=b,c=c)}
        path=[state(False,False,True),state(True,False,True),state(False,False,False),state(True,True,True)]
        value=sum(.99**i*shaping_bonus(a,b,i==2,config) for i,(a,b) in enumerate(zip(path,path[1:])))
        self.assertAlmostEqual(value,-1.)
        self.assertLess(shaping_bonus(path[1],path[2],False,config),0)

    def test_negative_open_predicate_and_exact_binary_relation(self):
        Rel=type('OnTop',(),{});Open=type('Open',(),{})
        table=SimpleNamespace(name='table')
        bottle=SimpleNamespace(states={Rel:SimpleNamespace(get_value=lambda other:other is table)})
        fridge=SimpleNamespace(states={Open:SimpleNamespace(get_value=lambda:False)})
        scope={'bottle':bottle,'table':table,'fridge':fridge}
        contract={'goals':['placed','closed'],'predicates':[
            {'name':'placed','predicate':'ontop','arguments':['bottle','table']},
            {'name':'closed','predicate':'open','arguments':['fridge'],'negated':True}]}
        obs=TaskObserver(SimpleNamespace(env=SimpleNamespace(task=SimpleNamespace(object_scope=scope))),
                         contract,{'ontop':Rel,'open':Open}).observe()
        self.assertTrue(obs['all_goals']);self.assertEqual(potential(obs),2.)

    def test_missing_truth_never_becomes_a_proxy_label(self):
        with self.assertRaises(ValueError):potential({'goals':{'a':1}})
        with self.assertRaises(ValueError):potential({'goals':{}})

    def test_preregistered_split_budget_and_original_start(self):
        for task in ['trash','water']:
            plan=json.loads((ROOT/task/'plan.json').read_text())
            self.assertEqual(plan['train_instances'],list(range(1,17))*2)
            self.assertEqual(plan['optimizer']['updates_per_round'],[17])
            self.assertEqual(plan['rounds'],1)
            self.assertEqual(plan['evaluation']['heldout_cases'],[[i,0] for i in range(301,321)])
            self.assertEqual(len(plan['evaluation']['cases']),24)
            self.assertEqual(len({tuple(x) for x in plan['evaluation']['cases']}),24)
            self.assertEqual(plan['action_interface']['gripper_open_tolerance'],0.)
            self.assertEqual(plan['optimizer']['actor_lr'],1e-7)
            self.assertEqual(plan['expected_original_trainable_sha256'],
                             'cc2b68e2950f05827e4d50d8cbe391c04ee6514cdd9e774fcbc203be992df51c')
            self.assertLessEqual(plan['max_live_gpus'],8)

    def test_storage_guard_rejects_insufficient_account_headroom(self):
        with tempfile.TemporaryDirectory() as folder:
            plan={'output':folder,'runtime_storage':{'max_persistent_bytes':200},
                  'storage_admission':{'account_root':'/mnt/afs/260010165','quota_bytes':1000,'scan_timeout_seconds':1}}
            with patch('storage_guard.subprocess.run',return_value=SimpleNamespace(stdout='900\t/mnt/afs/260010165\n')):
                with self.assertRaises(RuntimeError):admit_storage(plan)
            self.assertFalse(json.loads((Path(folder)/'storage_admission.json').read_text())['passed'])

    def test_storage_guard_reserves_the_whole_task_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            plan={'output':folder,'runtime_storage':{'max_persistent_bytes':200},
                  'storage_admission':{'account_root':'/mnt/afs/260010165','quota_bytes':1000,'scan_timeout_seconds':1}}
            with patch('storage_guard.subprocess.run',return_value=SimpleNamespace(stdout='700\t/mnt/afs/260010165\n')):
                self.assertTrue(admit_storage(plan)['passed'])

    def test_gpu_guard_consumes_fresh_cpu_proof_without_directory_scan(self):
        with tempfile.TemporaryDirectory() as folder:
            plan={'experiment':'rlinf_multitask_test','output':folder,
                  'runtime_storage':{'max_persistent_bytes':200},
                  'storage_admission':{'account_root':'/mnt/afs/260010165'}}
            proof={'complete':True,'passed':True,'experiment':plan['experiment'],
                   'account_root':'/mnt/afs/260010165','observed_at':time.time(),
                   'new_task_reserve_bytes':200}
            (Path(folder)/'storage_inventory.json').write_text(json.dumps(proof))
            with patch('storage_guard.subprocess.run',side_effect=AssertionError('GPU must not scan')):
                self.assertTrue(admit_storage(plan)['passed'])

    def test_real_task_without_cpu_proof_waits_instead_of_scanning(self):
        with tempfile.TemporaryDirectory() as folder:
            plan={'experiment':'rlinf_multitask_test','output':folder,
                  'storage_admission':{'account_root':'/mnt/afs/260010165'}}
            with patch('storage_guard.subprocess.run',side_effect=AssertionError('GPU must not scan')):
                with self.assertRaises(RuntimeError):admit_storage(plan)

    def test_baseline_probe_does_not_claim_training_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            plan={'experiment':'rlinf_multitask_test','output':folder,
                  'storage_admission':{'account_root':'/mnt/afs/260010165'}}
            with patch('storage_guard.subprocess.run',side_effect=AssertionError('Must not scan')):
                proof=admit_storage(plan,baseline_probe=True)
            self.assertTrue(proof['passed'])
            self.assertFalse(proof['full_training_clearance'])
            self.assertEqual(proof['shared_baseline_limit_bytes'],8*1024**3)


if __name__=='__main__':unittest.main(verbosity=2)
