"""Cover full rate scope, both source protocols, and frozen holdout/seed rules."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parents[1]/'scripts')]
from test_accepted_subtask_collection import AcceptedCollectionTests
from full_subtask_collection import targets, choose_starts, freeze, HISTORY, NATIVE
from subtask_improvement import load_plan, digest


class FullCollectionTests(unittest.TestCase):
    def setUp(self):
        f = AcceptedCollectionTests(); f.setUp()
        self.fixture = f
        self.group = targets(f.metrics, f.acceptance)[0]
        for p in f.proofs.values():
            p.update(state_protocol=HISTORY, prefix_actions_sha256='e'*64,
                     raw_sample_evidence=dict(selected_state_size=400, transition_count_before_frame=0))

    def plan(self):
        f = self.fixture
        return freeze(f.panel, self.group, f.panel['samples'][1:], f.proofs, f.identity, [('task',0)],
            accepted_input_sha256={'metrics':'f'*64}, evidence_manifest_sha256='9'*64, repeats=5)

    def load(self, plan):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'; path.write_text(json.dumps(plan))
            return load_plan(path, self.fixture.panel)

    def test_zero_success_and_one_eligible_are_included(self):
        f = self.fixture
        f.metrics['subtasks'][0].update(metric_eligible=1, successes=0, completion_rate=0)
        self.assertEqual(len(targets(f.metrics, f.acceptance)),1)
        f.metrics['subtasks'][0].update(metric_eligible=5, successes=4, completion_rate=.8)
        self.assertEqual(targets(f.metrics, f.acceptance),[])

    def test_history_keeps_its_distinct_contract(self):
        p = self.plan(); self.load(p)
        self.assertFalse(p['require_recorded_native_state'])
        self.assertEqual(p['expected_trials'],10)
        self.assertEqual(set(p['source_protocol_by_sample'].values()),{HISTORY})

    def test_native_group_still_requires_native_proof(self):
        for p in self.fixture.proofs.values(): p['state_protocol']=NATIVE
        p=self.plan(); self.load(p)
        self.assertTrue(p['require_recorded_native_state'])
        self.assertEqual(p['prefix_actions_sha256'],{})

    def test_no_heldout_or_already_true_substitution(self):
        f=self.fixture; f.proofs['s1']['baseline_metric_eligible']=False
        chosen=choose_starts(f.panel,self.group,f.proofs,[('task',0)],maximum=3)
        self.assertEqual([s['sample_id'] for s in chosen],['s2'])

    def test_changed_prefix_or_protocol_cannot_be_hidden_by_rehash(self):
        p=self.plan();p['source_protocol_by_sample']['s1']='unknown'
        p['fingerprint']=digest({k:v for k,v in p.items() if k!='fingerprint'})
        with self.assertRaises(ValueError):self.load(p)


if __name__=='__main__':unittest.main()


def test_history_collection_checks_independent_raw_dimensions():
    import pytest
    from test_history_artifact_accounting import fixture
    from full_subtask_collection import validate_result
    result,sample,fingerprint,raw,prefix=fixture()
    result['production_difficulty_baseline']=False
    sid=sample['sample_id']
    relative=f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
    plan=dict(source_protocol_by_sample={sid:HISTORY},prefix_actions_sha256={sid:prefix},
        raw_source_sha256={relative:raw['sha256']},selected_state_sha256={sid:raw['selected_state_sha256']},
        history_raw_sample_evidence={sid:{k:raw[k] for k in ('selected_state_size','transition_count_before_frame')}})
    validate_result(result,sample,fingerprint,plan)
    result['restoration']['topology_transition_steps']=[]
    with pytest.raises(ValueError,match='transition count'):validate_result(result,sample,fingerprint,plan)
