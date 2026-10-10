"""Selection and provenance regressions without requiring simulator imports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from accepted_subtask_collection import candidates, select_starts, freeze, task_shards
from subtask_improvement import load_plan, digest


class AcceptedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.panel = dict(fingerprint='panel', samples=[dict(sample_id='s'+str(i), task='task',
            subtask='pick up', task_instance_id=i, task_index=3, raw_episode_id=i,
            episode_index=i, start=0) for i in range(3)])
        a = dict(index=0, audited_samples=4, expected_samples=4,
            original_scope_and_sample_ids_matched=True, all_required_video_sha_and_first_frames_passed=True,
            independent_all_raw_native_fields_matched=True, corrective_audit_uid='audit', producing_run_root='/baseline')
        row = dict(task='task', subtask='pick up', assignment_index=0, metric_eligible=4,
                   successes=1, completion_rate=.25)
        self.metrics = dict(panel_fingerprint='panel', assignments=[a], subtasks=[row])
        self.acceptance = dict(kind='independently_accepted_task_and_subtask_success_rates',
                               panel_fingerprint='panel', accepted_task_indices=[0])
        self.identity = dict(checkpoint_step=1375000, checkpoint_source='/checkpoint',
                             checkpoint_manifest_sha256='c'*64)
        self.proofs = {s['sample_id']:dict(raw_sha256='a'*64, selected_state_sha256='b'*64,
            baseline_result_sha256='d'*64, baseline_metric_eligible=True) for s in self.panel['samples']}

    def plan(self):
        groups, _ = candidates(self.metrics, self.acceptance, self.panel)
        starts = select_starts(self.panel, groups, [('task', 0)], starts_per_group=2)
        return freeze(self.panel, groups, starts, self.proofs, self.identity, [('task', 0)],
                      accepted_input_sha256={'metrics':'f'*64})

    def load(self, plan):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            path.write_text(json.dumps(plan))
            return load_plan(path, self.panel)

    def test_pending_task_is_not_selected(self):
        self.acceptance['accepted_task_indices'] = []
        self.assertEqual(candidates(self.metrics, self.acceptance, self.panel), ([], []))

    def test_history_is_deferred_until_separate_gate(self):
        self.metrics['assignments'][0]['independent_all_raw_native_fields_matched'] = False
        groups, deferred = candidates(self.metrics, self.acceptance, self.panel)
        self.assertFalse(groups)
        self.assertEqual(deferred[0]['reason'], 'physical_history_collection_requires_separate_gpu_gate')

    def test_rate_denominator_mismatch_is_rejected(self):
        self.metrics['subtasks'][0]['completion_rate'] = .5
        with self.assertRaises(ValueError): candidates(self.metrics, self.acceptance, self.panel)

    def test_older_acceptance_without_producing_receipt_is_deferred(self):
        self.metrics['assignments'][0].pop('producing_run_root')
        groups, deferred = candidates(self.metrics, self.acceptance, self.panel)
        self.assertFalse(groups)
        self.assertEqual(deferred[0]['reason'], 'producing_receipt_not_embedded_in_accepted_metrics')

    def test_holdout_and_finite_seed_schedule(self):
        p = self.plan()
        self.load(p)
        self.assertEqual(p['expected_trials'], 6)
        self.assertEqual(len({t['policy_seed'] for t in p['trials']}), 6)
        self.assertNotIn('s0', {t['sample_id'] for t in p['trials']})
        self.assertTrue(p['selected_groups'][0]['underpowered'])
        self.assertFalse(p['full100_baseline_complete'])

    def test_modified_seeds_are_rejected_even_with_recomputed_fingerprint(self):
        p = self.plan(); p['trials'][0]['policy_seed'] += 1
        p['fingerprint'] = digest({k:v for k,v in p.items() if k != 'fingerprint'})
        with self.assertRaises(ValueError): self.load(p)

    def test_provenance_cannot_be_removed(self):
        p = self.plan(); p['baseline_result_sha256'].clear()
        p['fingerprint'] = digest({k:v for k,v in p.items() if k != 'fingerprint'})
        with self.assertRaises(ValueError): self.load(p)

    def test_already_satisfied_starts_are_excluded(self):
        groups, _ = candidates(self.metrics, self.acceptance, self.panel)
        self.proofs['s1']['baseline_metric_eligible'] = False
        p = freeze(self.panel, groups, self.panel['samples'][1:], self.proofs, self.identity, [('task', 0)],
                   accepted_input_sha256={'metrics':'f'*64})
        self.load(p)
        self.assertEqual(p['expected_trials'], 3)
        self.assertEqual(p['baseline_setup_excluded_sample_ids'], ['s1'])

    def test_checkpoint_must_match_measured_baseline(self):
        groups, _ = candidates(self.metrics, self.acceptance, self.panel)
        self.identity['checkpoint_step'] = 1688977
        with self.assertRaises(ValueError):
            freeze(self.panel, groups, self.panel['samples'][1:], self.proofs, self.identity, [],
                   accepted_input_sha256={'metrics':'f'*64})

    def test_task_shards_preserve_every_seed_exactly_once(self):
        plan = self.plan()
        samples = {s['sample_id']:s for s in self.panel['samples']}
        shards = task_shards(plan, samples)
        self.assertEqual(shards[0]['expected_trials'], len(plan['trials']))
        self.assertEqual(shards[0]['trial_ids'], [t['trial_id'] for t in plan['trials']])


if __name__ == '__main__': unittest.main()
