import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full100_shards as sco


class FullPanelAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = sco.load_panel(sco.REPO /
            'eval_results/subtask100_native_baseline_1375000_20261007_091214/panel.json')

    def document(self, count):
        d = {'panel_fingerprint': self.panel['fingerprint'], 'node_count': count,
             'assignments': sco.task_assignments(self.panel, count)}
        d['fingerprint'] = sco.runner.dynamic._fingerprint(d)
        return d

    def test_all_400_instances_and_7223_starts_are_disjoint_at_each_scale(self):
        for count in [1, 2, 4, 8, 16, 100]:
            with self.subTest(nodes=count):
                d = self.document(count)
                assignments = sco.verify_assignments(d, self.panel)
                keys = [tuple(k) for a in assignments for k in a['episode_keys']]
                ids = [s for a in assignments for s in a['sample_ids']]
                tasks = [t for a in assignments for t in a['tasks']]
                self.assertEqual(len(keys), 400)
                self.assertEqual(len(set(keys)), 400)
                self.assertEqual(set(ids), {s['sample_id'] for s in self.panel['samples']})
                self.assertEqual(len(ids), 7223)
                self.assertEqual(len(tasks), 100)
                self.assertEqual(len(set(tasks)), 100)
                self.assertTrue(all(a['episode_keys'] for a in assignments))

    def test_resigned_incomplete_or_overlapping_assignments_are_rejected(self):
        original = self.document(4)
        for field in ['tasks', 'episode_keys', 'sample_ids']:
            for mutation in ['drop', 'duplicate']:
                d = copy.deepcopy(original)
                if mutation == 'drop':
                    d['assignments'][0][field].pop()
                else:
                    d['assignments'][1][field].append(d['assignments'][0][field][0])
                d.pop('fingerprint')
                d['fingerprint'] = sco.runner.dynamic._fingerprint(d)
                with self.assertRaisesRegex(ValueError, 'missing, overlapping or changed'):
                    sco.verify_assignments(d, self.panel)

    def test_wrong_parent_panel_and_impossible_node_counts_are_rejected(self):
        d = self.document(4)
        d['panel_fingerprint'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'identity'):
            sco.verify_assignments(d, self.panel)
        for count in [0, -1, 101, True]:
            with self.assertRaises(ValueError):
                sco.task_assignments(self.panel, count)

    def test_gate_requires_actual_matching_model_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            ready = {'step': 1375000, 'manifest_sha256': 'a' * 64}
            (checkpoint / '_READY').write_text(json.dumps(ready))
            proof = {'state': 'complete', 'checkpoint_step': ready['step'],
                     'checkpoint_manifest_sha256': ready['manifest_sha256'],
                     'summary': {'complete': True, 'eligible_rollouts': 1,
                                 'panel_fingerprint': self.panel['fingerprint']}}
            path = root / 'proof.json'
            path.write_text(json.dumps(proof))
            sco.gate(path, checkpoint, self.panel)
            for field, value in [('eligible_rollouts', 0), ('panel_fingerprint', 'b' * 64),
                                 ('complete', False)]:
                changed = copy.deepcopy(proof)
                changed['summary'][field] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    sco.gate(path, checkpoint, self.panel)
            proof['checkpoint_manifest_sha256'] = 'c' * 64
            path.write_text(json.dumps(proof))
            with self.assertRaises(ValueError):
                sco.gate(path, checkpoint, self.panel)


class SupervisorCompletionTests(unittest.TestCase):
    def test_node_completion_uses_assignment_handler_without_global_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'queue_db':str(root / 'queue.sqlite3'), 'runtime_root':str(root / 'runtime'),
                        'gpus':[0], 'base_port':8810, 'tasks':['alpha','beta'],
                        'max_restarts_per_slot':8, 'run_id':'test'}
            queue = Mock()
            queue.snapshot.return_value = {'jobs':[{'task':'alpha'}]}
            queue.counts.return_value = {'counts':{'exhausted':0,'succeeded':1,'pending':0},'total':1}
            callback = Mock()
            with patch.object(sco.runner, 'finalize_results') as global_finalizer:
                sco.runner.supervise(manifest, queue, tasks=('alpha',), completion_handler=callback)
                callback.assert_called_once_with(manifest, queue)
                global_finalizer.assert_not_called()
            self.assertFalse((root / '_SUCCESS').exists())
            self.assertFalse((root / '_RUNNING').exists())
            with self.assertRaisesRegex(ValueError, 'assignment'):
                sco.runner.supervise(manifest, queue, tasks=('beta',), completion_handler=callback)


if __name__ == '__main__':
    unittest.main()
