import copy
import unittest

import compare_screens as mod


def fixture():
    result = dict(kind='pi05_native_cfm_and_euler_endpoint_screen', status='screen_complete',
        normalization_asset_id='asset', normalization_sha256='norm', use_quantile_norm=True,
        action_conditioning='task_and_subtask', anchor_count=3, draws_per_anchor=8,
        effective_model_config={'subtask_loss_weight': 1}, max_token_len=512,
        action_horizon=32, model_action_dim=32, parameter_dtype_counts={'bfloat16': 1},
        jax_version='test', numpy_version='test', backend='cpu', protocol={'seed':0},
        raw_cache={'manifest_sha256':'cache'}, samples=[])
    for anchor in range(3):
        for draw in range(8):
            array = {'shape':[32,23], 'values':[[0.0]*23 for _ in range(32)]}
            row = dict(episode_index=anchor, frame_index=10, draw_index=draw,
                sample_id=str(anchor), task_index=anchor, subtask='move',
                raw_sample_sha256='raw'+str(anchor), policy_input_sha256='input',
                transformed_training_sample_sha256='sample',
                target_leakage_control={'observation_identical':True}, training_target={'sha256':'target'},
                rng={'noise_sha256':str(draw)},
                cfm={'metrics':{'action_loss':{'shape':[], 'values':float(draw)},
                                'subtask_loss':{'shape':[], 'values':1.0}}},
                endpoint={'actions':copy.deepcopy(array), 'ground_truth':array,
                          'valid_mask':[True]*16+[False]*16})
            result['samples'].append(row)
    result['samples'][0]['same_model_repeat']={'all_passed':True}
    return result


class ComparisonTests(unittest.TestCase):
    def test_energy_known_values_and_symmetry(self):
        # U statistic differs from the biased plug-in score (which is 0.5 here).
        self.assertEqual(mod.energy([[0.0],[2.0]], [0.0])['score'], 0.0)
        self.assertEqual(mod.energy([[3.0]]*8, [0.0])['score'], 3.0)
        draws = [[0,1], [2,3], [4,5]]
        self.assertAlmostEqual(mod.energy(draws,[2,1])['score'], mod.energy(draws[::-1],[2,1])['score'])

    def test_identical_and_constant_coordinate_shift(self):
        base = fixture()
        candidate = copy.deepcopy(base)
        for row in candidate['samples']:
            for actions in row['endpoint']['actions']['values']:
                actions[7] = 2.0
        result = mod.compare(base, candidate)
        for anchor in result['anchors']:
            scores = anchor['endpoint']['valid_prefix16']['coordinates']
            self.assertEqual(scores[7]['energy_delta'], 2.0)
            self.assertTrue(all(score['energy_delta']==0 for score in scores if score['coordinate']!=7))
            self.assertEqual(anchor['native_metrics']['action_loss']['candidate_minus_baseline_mean'],0)

    def test_noise_or_backend_mismatch_rejected(self):
        base = fixture()
        for path in ('noise','backend'):
            candidate = copy.deepcopy(base)
            if path == 'noise':
                candidate['samples'][1]['rng']['noise_sha256'] = 'other'
            else:
                candidate['backend'] = 'gpu'
            with self.assertRaises(ValueError):
                mod.compare(base,candidate)

    def test_duplicate_or_incomplete_input_rejected(self):
        base=fixture()
        for rows in (base['samples'][:-1], base['samples'][:-1]+[base['samples'][0]]):
            candidate=copy.deepcopy(base)
            candidate['samples']=rows
            with self.assertRaises(ValueError):
                mod.compare(base,candidate)


if __name__ == '__main__':
    unittest.main()
