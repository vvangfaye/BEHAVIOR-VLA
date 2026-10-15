import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
import scorer
spec=importlib.util.spec_from_file_location('new_controller',ROOT/'controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class FourCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.plan=c.load(ROOT/'plan.json')
        self.base=c.load(ROOT/'baseline_accepted.json')
        self.mid=c.load(ROOT/'step5000_candidate_accepted.json')
        self.last=c.load(ROOT/'step10000_candidate_accepted.json')
        self.new=copy.deepcopy(self.last)
        self.new.update(completed_updates=20000,checkpoint_manifest_sha256='unit-test-distinct-20000')

    def test_four_way_uses_all_real_prior_results_without_new_baselines(self):
        result=scorer.four_way(self.base,self.mid,self.last,self.new,self.plan['pairs'])
        self.assertEqual([r['checkpoint'] for r in result['summaries']],[1688977,5000,10000,20000])
        self.assertEqual([r['successes'] for r in result['summaries']],[0,1,1,1])
        self.assertEqual(len(result['per_instance']),20)
        self.assertEqual(result['comparisons']['20000_vs10000']['mean_q_delta'],0)
        self.assertEqual(result['comparisons']['20000_vs5000']['baseline_checkpoint'],5000)
        self.assertFalse(result['historical_baseline_acknowledged'])
        self.assertTrue(result['trained_checkpoints_actual_seed_acknowledged'])

    def test_four_way_rejects_mislabeled_or_duplicate_checkpoints(self):
        for update in ({'completed_updates':10000}, {'checkpoint_manifest_sha256':self.last['checkpoint_manifest_sha256']}):
            wrong=copy.deepcopy(self.new);wrong.update(update)
            with self.assertRaises(ValueError):scorer.four_way(self.base,self.mid,self.last,wrong,self.plan['pairs'])

    def test_four_way_rejects_mixed_protocol_or_unacknowledged_seed(self):
        wrong=copy.deepcopy(self.new);wrong['protocol_sha256']='different-native-settings'
        with self.assertRaises(ValueError):scorer.four_way(self.base,self.mid,self.last,wrong,self.plan['pairs'])
        wrong=copy.deepcopy(self.new);wrong['records'][0]['policy_seed_acknowledged']=False
        with self.assertRaises(ValueError):scorer.four_way(self.base,self.mid,self.last,wrong,self.plan['pairs'])

    def test_evaluation_bootstrap_has_independent_supervision(self):
        payload=dict(id='candidate-04',mode='evaluate',plan=self.plan,role='candidate',
                     pair=self.plan['pairs'][4],job_name='unit',worker_bundle_sha256='unit',
                     heartbeat_put_url='https://test.invalid/heartbeat',diagnostics_put_url='https://test.invalid/errors')
        source=c.bootstrap(self.plan,payload,'https://test.invalid/report')
        compile(source,'supervised-bootstrap','exec')
        self.assertIn("'checkpoint_step': 20000",source)
        self.assertIn('putting_up_Christmas_decorations_inside_301_0',source)
        self.assertIn('memory.events',source)
        self.assertNotIn('access_key_secret',source)

    def test_native_worker_and_seed_protocol_unchanged_except_checkpoint(self):
        old=ROOT.parent/'paired_quick20_reused1688977_step10000_20261014'
        self.assertEqual((ROOT/'frozen/worker.py').read_text(),(old/'frozen/worker.py').read_text().replace('10000','20000'))
        for name in ['seed_protocol.py','server_entry.py','evaluator_entry.py','native_helpers.py']:
            self.assertEqual((ROOT/'frozen'/name).read_bytes(),(old/'frozen'/name).read_bytes())
        self.assertEqual((ROOT/'protocol.json').read_bytes(),(old/'protocol.json').read_bytes())
        self.assertEqual(self.plan['pairs'],c.load(old/'plan.json')['pairs'])


if __name__=='__main__':unittest.main()
