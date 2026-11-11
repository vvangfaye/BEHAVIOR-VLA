import copy
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'run'))
sys.path.insert(0, str(HERE.parent/'upstream'))
import batched
from replay_contract import HardwareReplayValidator, audit_replay, configure_sampler_precision
from source_contract import original_training_source


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.3]))

    def default_forward(self, inputs, **kwargs):
        output = inputs['x'] @ self.weight
        return {'logprobs': output[:, None, None].expand(-1, 2, 23)/46,
                'values': output*.7}


class BatchEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        self.rows = []
        for i in range(7):
            self.rows.append({'forward_inputs': {'x': torch.randn(1, 2)},
                              'prev_logprobs': torch.full((1,2,23), (i-3)*.003),
                              'prev_values': torch.tensor([.1*i])})
        self.advantages = torch.tensor([-1.,2.,.5,-.3,1.5,-2.,.4])
        self.returns = torch.arange(7)*.13
        self.config = {'clip_ratio': .2}

    def test_uneven_microbatches_preserve_gradients_and_adam_updates(self):
        a, b = Toy(), Toy()
        oa, ob = [torch.optim.AdamW(m.parameters(), lr=.01) for m in (a,b)]
        for indices in [[0,1,2,3,4,5,6], [2,5,0,4,6], [1,3,6]]:
            oa.zero_grad(); ob.zero_grad()
            ma = batched.backward_group(a,self.rows,self.advantages,self.returns,indices,self.config,1)
            mb = batched.backward_group(b,self.rows,self.advantages,self.returns,indices,self.config,4)
            torch.testing.assert_close(a.weight.grad,b.weight.grad,rtol=1e-5,atol=1e-7)
            self.assertEqual([x['chunk'] for x in ma],[x['chunk'] for x in mb])
            torch.testing.assert_close(torch.tensor([x['loss'] for x in ma]),torch.tensor([x['loss'] for x in mb]))
            oa.step(); ob.step()
            torch.testing.assert_close(a.weight,b.weight,rtol=1e-5,atol=1e-7)

    def test_kl_keeps_all_rows_including_short_tail(self):
        model=Toy()
        expected=batched.batch_kl(model,self.rows,1)
        actual=batched.batch_kl(model,self.rows,4)
        self.assertAlmostEqual(expected['mean'],actual['mean'],places=6)
        self.assertAlmostEqual(expected['max_abs_log_ratio'],actual['max_abs_log_ratio'],places=6)

    def test_mismatched_replay_schema_is_rejected(self):
        rows=copy.deepcopy(self.rows[:2]);rows[1]['forward_inputs']['extra']=torch.ones(1)
        with self.assertRaises(AssertionError):
            batched.stack_inputs(rows,'cpu')

    def test_sampler_precision_restores_convolution_setting(self):
        old_matmul=torch.get_float32_matmul_precision()
        old_cudnn=torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32=True
            settings=configure_sampler_precision()
            self.assertEqual(settings['float32_matmul_precision'],'highest')
            self.assertFalse(settings['matmul_allow_tf32'])
            self.assertFalse(settings['cudnn_allow_tf32'])
        finally:
            torch.set_float32_matmul_precision(old_matmul)
            torch.backends.cudnn.allow_tf32=old_cudnn

    def test_replay_audit_rejects_mismatch_without_relabeling(self):
        model=Toy()
        rows=copy.deepcopy(self.rows)
        for row in rows:
            row['prev_logprobs']=model.default_forward(row['forward_inputs'])['logprobs'].detach().clone()
        self.assertTrue(audit_replay(model,rows,list(range(len(rows))))['passed'])
        rows[2]['prev_logprobs'] += .005
        before=rows[2]['prev_logprobs'].clone()
        report=audit_replay(model,rows,list(range(len(rows))))
        self.assertFalse(report['passed'])
        self.assertEqual(report['failed_rows'][0]['row'],2)
        torch.testing.assert_close(rows[2]['prev_logprobs'],before,rtol=0,atol=0)

    def test_hardware_contract_checks_joint_ratio_not_only_coordinates(self):
        validator=HardwareReplayValidator(clip_ratio=.2,kl_target=.01)
        # Every coordinate passes the old atol, but 368 dimensions accumulate.
        with self.assertRaises(AssertionError):
            validator(torch.full((1,16,23),.00004),torch.zeros(1,16,23))

    def test_hardware_contract_keeps_original_denominator(self):
        validator=HardwareReplayValidator(clip_ratio=.2,kl_target=.01)
        old=torch.zeros(1,16,23)
        actual=old.clone();actual[0,0,0]=.0005
        validator(actual,old)
        report=validator.finalize()
        self.assertEqual(report['legacy_elementwise_mismatches'],1)
        self.assertLess(report['max_probability_ratio_error'],.001)
        self.assertEqual(float(old.abs().sum()),0.)

    def test_hardware_contract_does_not_hide_cancelled_large_errors(self):
        validator=HardwareReplayValidator(clip_ratio=.2,kl_target=.01)
        actual=torch.tensor([[.02,-.02]])
        with self.assertRaises(AssertionError):
            validator(actual,torch.zeros_like(actual))

    def test_hardware_contract_rejects_nonfinite_inputs(self):
        validator=HardwareReplayValidator(clip_ratio=.2,kl_target=.01)
        with self.assertRaises(AssertionError):
            validator(torch.tensor([[float('nan')]]),torch.zeros(1,1))

    def test_native_training_equations_are_unchanged(self):
        import hashlib
        import json
        normalized=original_training_source((HERE/'run/training.py').read_text())
        expected=json.loads((HERE/'validation.json').read_text())['frozen_sources']['run/training.py']
        self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(),expected)


if __name__=='__main__':
    unittest.main()
