import copy
import json
from pathlib import Path
import pytest
from trained_full_episode_collection import metric_summary, validate_model_plan, MODEL
from full_episode_scope import validate_final1000
ROOT=Path(__file__).resolve().parents[1]/'eval_results/sco_full_episode_collection_20261015_quick20_step20000_round4/round-4'

def row(i,success,q):
 return dict(task='task',instance_id=i,success=success,q_score_final=q,steps=20,terminated=True,truncated=False)

def test_partial_scores_exclude_unpublished_and_separate_success():
 r=metric_summary([row(301,True,1.),row(302,False,.6)],['task','other'])
 assert r['completed_episodes']==2 and r['missing_episodes']==998
 assert r['success_rate']==.5 and r['mean_q_score']==.8
 assert r['final_competition_score'] is False
 assert r['per_task']['other']['mean_q_score'] is None
 assert r['bc_eligible_episodes']==1

@pytest.mark.parametrize('q',[float('nan'),float('inf'),-.1,1.1,None,True])
def test_invalid_competition_scores_rejected(q):
 with pytest.raises((AssertionError,TypeError)): metric_summary([row(301,False,q)],['task'])

def test_duplicate_results_cannot_inflate_scores():
 with pytest.raises(AssertionError):metric_summary([row(301,True,1.),row(301,True,1.)],['task'])

def test_truncated_success_is_not_bc():
 v=row(301,True,1.);v.update(terminated=False,truncated=True)
 assert metric_summary([v],['task'])['bc_eligible_episodes']==0


def test_model_scope_exact_files_and_source_are_bound():
 p=json.loads((ROOT/'plan.json').read_text());validate_final1000(p)
 assert p['checkpoint']==MODEL and p['sampling_round']==4
 assert p['checkpoint_identity']['checkpoint_step']==20000
 for key in ['checkpoint','sampling_round']:
  v=copy.deepcopy(p);v[key]='wrong'
  with pytest.raises(AssertionError):validate_model_plan(v)


def test_frozen_worker_checks_new_weights_and_limits_only_threads():
 s=ROOT/'overlay_sources'
 baseline=(s/'sco/run_public_test_assignment.py').read_text()
 assert "verify_checkpoint(checkpoint, payload['trained_model_export'])" in baseline
 assert "OMP_NUM_THREADS='8'" in baseline and "OPENBLAS_NUM_THREADS='1'" in baseline
 assert "PI05_CKPT=str(checkpoint)" in baseline and "EXECUTE_HORIZON='16'" in baseline
 queue=(s/'scripts/eval_b1k_queue.py').read_text()
 assert 'torch.set_num_threads' in queue and 'torch.set_num_interop_threads' in queue
 checkpoint=(s/'scripts/full_episode_checkpoint.py').read_text()
 assert checkpoint.index("upload_json(receipt, storage['receipt'])")<checkpoint.index("upload_json(metrics,")
 assert "q = result['q_score']['final']" in checkpoint
