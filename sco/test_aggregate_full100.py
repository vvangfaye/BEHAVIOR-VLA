import copy
import unittest

from aggregate_full100 import verify_full_report


class FullReportTests(unittest.TestCase):
    def setUp(self):
        self.success = {'expected_episode_jobs': 400, 'expected_segment_rollouts': 7223,
                        'expected_annotated_segments': 8112, 'panel_fingerprint': 'frozen'}
        self.summary = {**self.success, 'complete': True, 'valid_segment_rollouts': 7223,
                        'accounted_annotations': 8112, 'per_task': {str(i): {} for i in range(100)},
                        'overall': {'unsupported': 889},
                        'state_fidelity': {'contract': 'recorded_native_fields_v1', 'verified': True},
                        'video_completeness': {'required': 15, 'completed': 15}}

    def test_partial_panel_cannot_be_exported_as_full100(self):
        verify_full_report(self.summary, self.success)
        for key, value in [('complete', False), ('expected_episode_jobs', 4),
                           ('expected_segment_rollouts', 6), ('valid_segment_rollouts', 7222),
                           ('accounted_annotations', 8111), ('panel_fingerprint', 'other'),
                           ('per_task', {'only_one': {}}), ('overall', {'unsupported': 888})]:
            changed = copy.deepcopy(self.summary)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_full_report(changed, self.success)

    def test_missing_native_proof_or_rollout_video_blocks_export(self):
        for key, value in [('state_fidelity', {'contract': 'synthetic', 'verified': True}),
                           ('state_fidelity', {'contract': 'recorded_native_fields_v1', 'verified': False}),
                           ('video_completeness', {'required': 15, 'completed': 14})]:
            changed = copy.deepcopy(self.summary)
            changed[key] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_full_report(changed, self.success)


if __name__ == '__main__':
    unittest.main()
