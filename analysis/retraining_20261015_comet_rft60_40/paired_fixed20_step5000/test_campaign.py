from pathlib import Path
import copy,importlib.util,json,sys
import pytest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
import scorer
spec=importlib.util.spec_from_file_location('controller',ROOT/'controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def data():
 p=c.load(ROOT/'plan.json');history=[c.load(ROOT/f) for f in ['baseline_accepted.json','step5000_candidate_accepted.json','step10000_candidate_accepted.json','step20000_candidate_accepted.json']]
 new=copy.deepcopy(history[1]);new['checkpoint_manifest_sha256']='unit-new-two-source-5000';new['completed_updates']=5000
 return p,history,new

def test_distinct_same_step_models_and_real_history():
 p,h,n=data();r=scorer.five_way(*h,n,p['pairs'])
 assert [x['model'] for x in r['summaries']]==['1688977','previous_5000','previous_10000','previous_20000','rft60_40_5000']
 assert [x['successes'] for x in r['summaries']]==[0,1,1,1,1]
 assert len(r['comparisons'])==4 and len(r['per_instance'])==20
 assert r['comparisons']['rft60_40_5000_vs_previous_5000']['mean_q_delta']==0
 assert not r['training_continues_to20000'] and not r['heldout_generalization_claim']

def test_reject_mixed_protocol_or_reused_checkpoint():
 p,h,n=data()
 for key,value in [('protocol_sha256','bad'),('checkpoint_manifest_sha256',h[1]['checkpoint_manifest_sha256']),('completed_updates',10000)]:
  x=copy.deepcopy(n);x[key]=value
  with pytest.raises(ValueError):scorer.five_way(*h,x,p['pairs'])

def test_no_new_historical_rollouts_or_automatic_retries():
 p,h,n=data()
 assert p['episodes']==20 and p['new_candidate_rollouts']==20
 assert p['new_baseline_rollouts']==p['new_previous_rollouts']==p['new_additional_rollouts']==p['new_extra_rollouts']==0
 assert p['maximum_attempts_per_episode']==1 and not p['modify_or_stop_training']
 assert p['training_iid']==238414 and p['candidate_step']==5000


def test_exact_protocol_and_scope_preserved():
 p,h,n=data();old=ROOT.parents[1]/'retraining_20261013/paired_quick20_reused1688977_step20000_20261015'
 # Resolve old path explicitly from workspace instead of construction-sensitive relative chains.
 old=ROOT.parents[2]/'analysis/retraining_20261013/paired_quick20_reused1688977_step20000_20261015'
 assert (ROOT/'protocol.json').read_bytes()==(old/'protocol.json').read_bytes()
 assert (ROOT/'frozen/worker.py').read_text()==(old/'frozen/worker.py').read_text().replace('20000','5000')
 for name in ['server_entry.py','evaluator_entry.py','seed_protocol.py','native_helpers.py']:
  assert (ROOT/'frozen'/name).read_bytes()==(old/'frozen'/name).read_bytes()

def test_bootstrap_is_new_scope_and_5000_only():
 p,h,n=data();payload=dict(id='candidate-04',mode='evaluate',plan=p,role='candidate',pair=p['pairs'][4],job_name='unit',worker_bundle_sha256='unit',heartbeat_put_url='https://example.invalid/heartbeat',diagnostics_put_url='https://example.invalid/diag')
 text=c.bootstrap(p,payload,'https://example.invalid/report');compile(text,'bootstrap','exec')
 assert "'checkpoint_step': 5000" in text and p['afs_root'] in text
 assert '/paired_quick20_reused1688977_step20000_20261015/' not in text
 assert 'access_key_secret' not in text
