import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('recovery',ROOT/'controller.py')
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.plan=r.c.load(ROOT.parent/'plan.json')
        self.report=copy.deepcopy(r.c.load(ROOT.parent/'reports/candidate-00.json'))
        self.report.update(id=r.ID)
        self.report['record'].update(task=self.plan['pairs'][4][0],instance_id=301)
        self.intent={'worker_bundle_sha256': self.report['worker_bundle_sha256']}

    def test_exact_frozen_episode_and_checkpoint_accepted(self):
        self.assertTrue(r.verify_recovery_report(self.report,self.plan,self.intent))

    def test_wrong_checkpoint_rejected(self):
        self.report['checkpoint_manifest_sha256']='wrong-step'
        with self.assertRaises(ValueError):
            r.verify_recovery_report(self.report,self.plan,self.intent)

    def test_wrong_seed_or_unacknowledged_rejected(self):
        for change in ({'policy_seed':1},{'environment_seed':1},{'policy_seed_acknowledged':False}):
            report=copy.deepcopy(self.report);report['record'].update(change)
            with self.assertRaises(ValueError):
                r.verify_recovery_report(report,self.plan,self.intent)

    def test_failed_report_never_accepted(self):
        self.report.update(state='failed')
        self.assertFalse(r.verify_recovery_report(self.report,self.plan,self.intent))

    def test_existing_unknown_intent_only_described_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            intent={'creation_attempted':True,'name':r.NAME,'uid':None}
            (root/'intent.json').write_text(json.dumps(intent))
            observation={'uid':'existing-exact-uid','state':'QUEUEING','observed_at':1}
            with patch.object(r,'ROOT',root),patch.object(r.c,'observe',return_value=observation) as observe,patch.object(r.subprocess,'run') as run:
                result=r.submit_once(None,self.plan)
                self.assertEqual(result['uid'],'existing-exact-uid')
                observe.assert_called_once()
                run.assert_not_called()

    def test_original_19_and_frozen_identity_pinned(self):
        identity=r.immutable_identity()
        self.assertEqual(sum(k.startswith('reports/candidate-') for k in identity),19)
        self.assertIn('frozen/worker.py',identity)
        self.assertIn('baseline_accepted.json',identity)
        self.assertIn('step5000_candidate_accepted.json',identity)


if __name__=='__main__':
    unittest.main()
