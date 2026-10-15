import importlib.util,json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gates',P/'launch_gated_training.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def proof():
 return dict(passed=True,real_two_source_batches=2,model_loaded=False,optimizer_initialized=False,mixture_fingerprint='f',batches=[dict(source_counts={'0':115,'3':77},full_episode_text_target_tokens=0,supervised_text_target_tokens=100,actions_shape=[192,32,32],images={'a':[192,224,224,3],'b':[192,224,224,3],'c':[192,224,224,3]}) for _ in range(2)])
def test_real_cpu_gate():g.cpu_validate(proof(),'f')
def test_reject_full_text_loss():
 p=proof();p['batches'][0]['full_episode_text_target_tokens']=1
 with pytest.raises(AssertionError):g.cpu_validate(p,'f')
def test_reject_old_four_source_proof():
 p=proof();p['batches'][0]['source_counts']={'0':100,'1':15,'3':77}
 with pytest.raises(AssertionError):g.cpu_validate(p,'f')
def test_wrong_mixture_rejected():
 with pytest.raises(AssertionError):g.cpu_validate(proof(),'wrong')
def test_fixed_budget_and_clock():
 b=P/'formal_stage/bundle';j=g.read(b/'nebula/worker_job.json');a=g.read(b/'inputs/formal_authorization.json')
 m=g.module('fw',b/'nebula/train_four_source.py');m.validate(j,a,{'completed_updates':2,'checkpoint_restore':True,'gpus':16})
 assert j['stage_end_step']==j['max_steps']==a['max_completed_updates']==5000
 assert g.read(b/'inputs/compiled/learning_rate_contract.json')['schedule']=={'warmup_steps':200,'peak_lr':1e-6,'decay_steps':20000,'decay_lr':2e-7}
 assert g.digest(b/'inputs/compiled/learning_rate_contract.json')==g.read(b/'inputs/formal_run_contract.json')['learning_rate_contract_sha256']
