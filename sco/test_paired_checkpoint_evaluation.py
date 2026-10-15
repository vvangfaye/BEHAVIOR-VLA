import copy
import json
from pathlib import Path
import pytest
from paired_checkpoint_evaluation import validate_plan,aggregate,score_row,evaluation_gate_allowed
from restore_full_episode_collection import read_plan

ROOT=Path(__file__).resolve().parents[1]/'eval_results/sco_paired_eval962_step10000_20261015/round-5'

def plan():return json.loads((ROOT/'plan.json').read_text())

def test_frozen_exact_paired_subset_and_checkpoint():
 p=read_plan(ROOT);assert len(p['baseline_scores'])==962
 assert sum(len(s['instance_ids']) for s in p['shards'])==962
 assert len(p['shards'])==197 and evaluation_gate_allowed(ROOT,p)
 old=json.loads((ROOT.parents[1]/'sco_full_episode_collection_20261014_final1000_round3/round-3/plan.json').read_text())
 assert all(old['policy_seeds'][k]==v for k,v in p['policy_seeds'].items())
 assert p['checkpoint_identity']['checkpoint_step']==10000

@pytest.mark.parametrize('field,value',[('checkpoint','wrong'),('sampling_round',4),('expected_episodes',1000),('record_rgb_trajectories',True),('cpu_thread_policy','limited')])
def test_reject_scope_model_or_runtime_drift(field,value):
 p=plan();p[field]=value
 with pytest.raises(AssertionError):validate_plan(p)

def test_reject_previously_missing_instance():
 p=plan();p['shards'][1]['instance_ids'][0]=302
 with pytest.raises(AssertionError):validate_plan(p)

def test_no_zero_imputation_and_no_success_equals_q1_assumption():
 p=plan();old=p['baseline_scores'][0];row=dict(task=old['task'],instance_id=old['instance_id'],success=False,q_score_final=1.)
 s=aggregate(p,[row]);assert s['completed_episodes']==1 and s['missing_episodes']==961
 assert s['candidate']['success_rate']==0 and s['candidate']['competition_sr']==1
 assert s['baseline_same_instances']['episodes']==1 and not s['final_comparison']
 assert s['baseline_all_962']['successes']==82
 with pytest.raises(AssertionError):aggregate(p,[row,row])

def test_worker_preserves_native_policy_and_skips_video_archive_uploads():
 root=ROOT/'overlay_sources'
 baseline=(root/'sco/run_public_test_assignment.py').read_text()
 assert "verify_checkpoint(checkpoint,payload['trained_model_export'])" in baseline
 assert 'OMP_NUM_THREADS' not in baseline and 'PI05_EVAL_TORCH_THREADS' not in baseline
 assert "PI05_CKPT=str(checkpoint)" in baseline
 queue=(root/'scripts/eval_b1k_queue.py').read_text()
 assert "if not manifest.get('paired_evaluation_only')" in queue
 assert "result['repeat_sampling'] = evaluator.policy.proof()" in queue
 assert 'upload_json(row,manifest' in queue
 worker=(root/'sco/run_full_episode_collection.py').read_text()
 assert "write_video=False" in worker and 'verify_full_episode_codec.py' not in worker
 assert 'checkpoint_step=10000' in worker
 for f in root.rglob('*.py'):compile(f.read_bytes(),str(f),'exec')
