import unittest
from compare_partial_fixed20 import calculate


class PartialTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(pairs=[['a',301],['b',301]],protocol_sha256='p',
                       expected_checkpoint_manifest_sha256='checkpoint',candidate_step=1000)
        self.row=dict(task='a',instance_id=301,artifact_accepted=True,policy_seed=0,environment_seed=0,
                      policy_seed_acknowledged=True,steps=12,terminated=True,truncated=False,
                      model_success=True,q_score=1.0)
        self.report=dict(id='candidate-00',state='accepted',protocol_sha256='p',
                         worker_bundle_sha256='bundle',checkpoint_manifest_sha256='checkpoint',record=self.row)

    def test_empty_is_unknown_not_zero_accuracy(self):
        result=calculate(self.plan,{},'bundle',{})
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['unobserved_or_unaccepted_episodes'],2)
        self.assertFalse(result['complete'])

    def test_only_matched_completed_denominator(self):
        old={('a',301):dict(self.row,model_success=False,q_score=.25),
             ('b',301):dict(self.row,task='b',model_success=True,q_score=1.)}
        result=calculate(self.plan,{'candidate-00':self.report},'bundle',{'old':old})
        comparison=result['comparisons']['old']
        self.assertEqual(comparison['matched_completed_episodes'],1)
        self.assertEqual(comparison['mean_q_delta'],.75)
        self.assertEqual(result['full_panel_success_rate_bounds'],[.5,1.])
        self.assertFalse(result['complete'])

    def test_wrong_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            calculate(self.plan,{'candidate-00':dict(self.report,checkpoint_manifest_sha256='wrong')},'bundle',{})

    def test_failed_job_not_failed_model(self):
        report=dict(self.report,state='failed')
        result=calculate(self.plan,{'candidate-00':report},'bundle',{})
        self.assertIsNone(result['candidate'])
        self.assertEqual(result['infrastructure_failed_reports'],['candidate-00'])


if __name__=='__main__':unittest.main()
