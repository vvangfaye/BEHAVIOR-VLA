"""Freeze two independent task replications without altering the modem study."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
SOURCE=BASE/'modem_parallel_20261029'
PROJECT=BASE.parents[1]

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)

plan0=json.loads((SOURCE/'plan.json').read_text())
inventory=json.loads((HERE/'inventory.json').read_text())
stats={x['task_name']:x for x in json.loads((HERE/'task_stats.json').read_text())}
tree=ast.parse((PROJECT/'BEHAVIOR-VLAs/openpi/src/openpi/configs/tasks/b1k.py').read_text())
prompts=ast.literal_eval(next(x.value for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASKS' for t in x.targets)))
definitions=PROJECT/'analysis/organizer_rehearsal_1688977_20261016/frozen_source/BEHAVIOR-1K/bddl3/bddl/activity_definitions'
if not definitions.is_dir():definitions=PROJECT/'BEHAVIOR-VLAs/BEHAVIOR-1K/bddl3/bddl/activity_definitions'

task_specs=[
    ('trash','picking_up_trash',1,96,48,10800,86400,290100,[
        {'name':f'can_{i}_inside_bin','predicate':'inside','arguments':[f'can__of__soda.n.01_{i}','ashcan.n.01_1']} for i in range(1,4)]),
    ('water','bringing_water',17,160,64,21600,172800,290170,[
        {'name':f'bottle_{i}_on_table','predicate':'ontop','arguments':[f'bottle.n.01_{i}','coffee_table.n.01_1']} for i in range(1,3)]+
        [{'name':'fridge_closed','predicate':'open','arguments':['electric_refrigerator.n.01_1'],'negated':True}]),
]
records=[]
for slug,task,tid,budget,storage,phase_seconds,train_seconds,seed,predicates in task_specs:
    folder=HERE/slug;folder.mkdir(exist_ok=True)
    assert not (folder/'jobs').exists(),'Do not regenerate a submitted campaign'
    for name in ['common.py','training.py','experiment.py','simulator.py','helpers.py','driver.py','bootstrap.py','eval_utils.py','resource_watch.py']:
        shutil.copyfile(SOURCE/name,folder/name)
    shutil.copytree(SOURCE/'b1k_rlinf',folder/'b1k_rlinf',dirs_exist_ok=True)
    shutil.copyfile(HERE/'goals.py',folder/'goals.py')
    shutil.copyfile(HERE/'storage_guard.py',folder/'storage_guard.py')
    shutil.copyfile(HERE/'recovery_policy.py',folder/'recovery_policy.py')
    s=(folder/'simulator.py').read_text()
    s=replace_once(s,'from goals import ModemObserver','from goals import TaskObserver')
    s=replace_once(s,"observer = ModemObserver(evaluator, spec['goal_contract'])","observer = TaskObserver(evaluator, spec['goal_contract'])")
    s=replace_once(s,"goal_names = ('toggled_on','nextto_tv','shared_tv_cabinet')","goal_names = tuple(spec['goal_contract']['goals'])")
    (folder/'simulator.py').write_text(s)
    s=(folder/'driver.py').read_text()
    s=replace_once(s,"index = 2 if phase=='final' else 1","index = plan['rounds'] if phase=='final' else 1")
    (folder/'driver.py').write_text(s)
    s=(folder/'bootstrap.py').read_text()
    s=replace_once(s,"assert task == 'installing_a_modem'","assert task == plan['task'] and task in ('picking_up_trash','bringing_water')")
    s=replace_once(s,"plan['source_identity']['modem_bddl_sha256']","plan['source_identity']['task_bddl_sha256']")
    s=replace_once(s,'    plan = json.loads((root / "run/plan.json").read_text())',
        '    plan = json.loads((root / "run/plan.json").read_text())\n'
        '    if config["task_phase"]=="baseline" and config["rank"]==0:\n'
        '        import sys\n'
        '        sys.path.insert(0,str(root/"run"))\n'
        '        from storage_guard import admit_storage\n'
        '        admit_storage(plan)')
    (folder/'bootstrap.py').write_text(s)

    p=copy.deepcopy(plan0)
    for key in ['baseline_sources','baseline_reuse_audit','resume_authorization','first_case_dependency','previous_generation_directory','first_case_gate','startup_repair','attempt_caps','prior_terminal_gpu_seconds_upper_bound']:
        p.pop(key,None)
    p.update(task=task,task_prompt=prompts[task],experiment=f'rlinf_multitask_{slug}_20261029',
             task_id=tid,rounds=1,phase_order=['baseline','sample1','train1','final'],
             job_prefix=f'b1k-mtv-1029-{slug}',aoss_prefix=f'behavior1k/rlinf-multitask-20261029/{slug}/',
             worker_prefix='m1_',generation='multitask_pilot_v1',parallel_workers=8,
             max_live_gpus=8,max_total_gpu_hours=budget,max_wall_hours=7*24,
             remote_root=f'/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029/{slug}',
             output=f'/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029/{slug}/results')
    p['source_identity']['task_bddl_sha256']=sha(definitions/task/'problem0.bddl')
    p['source_identity'].pop('modem_bddl_sha256',None)
    p['goal_contract']={'goals':[x['name'] for x in predicates],'predicates':predicates,
                        'bddl_sha256':p['source_identity']['task_bddl_sha256'],'read_only':True}
    p['eval_steps']=p['train_steps_per_round']=int(stats[task]['length']*1.5)
    p['train_instances']=list(range(1,17))*2
    p['train_seeds']=[[seed+i for i in range(32)]]
    p['eval_instances']=list(range(301,321));p['eval_seeds']=[0]
    training=[[i,0] for i in range(1,5)];heldout=[[i,0] for i in range(301,321)]
    p['evaluation']={'cases':training+heldout,'training_cases':training,'heldout_cases':heldout,
                      'checkpoints':['original1688977','after_17_updates'],
                      'primary':'20 distinct held-out instances, each seed 0'}
    p['evaluation_runtime_order']=heldout+training
    p['optimizer']['updates_per_round']=[17]
    p['critic_warmup']['source']='Only this task first 32 online trajectories; no transferred critic'
    p['protocol'].update(total_training_episodes=32,baseline_episodes=24,final_episodes=24,max_new_episodes=80,new_baseline_episodes=24)
    p['phase_limits_seconds']={'baseline':phase_seconds,'sample1':phase_seconds,'train1':train_seconds,'final':phase_seconds}
    p['runtime_storage']['max_persistent_bytes']=storage*1024**3
    p['runtime_storage']['shared_budget_roots']=[p['output']]
    p['storage_admission']={'account_root':'/mnt/afs/260010165','quota_bytes':2000000000000,
                            'scan_timeout_seconds':900,'gate':'baseline rank 0 must pass before parallel expansion'}
    p['comparisons']={'primary':'heldout mean score and success rate, paired initial states',
                       'no_checkpoint_selection':True,'report_negative_results':True,
                       'historical_radio_and_modem':'separate reference panels; their settings differ'}
    p['authorization']={'request':'再用多个任务说明下目前的RL是有效的','date':'2026-10-29',
                         'purpose':'Test, rather than assume, multi-task improvement','automatic_private_code_upload':True,
                         'global_new_campaign_concurrency':8,'other_projects_unchanged':True}
    p['metadata_verified']=True
    write(folder/'plan.json',p)
    write(folder/'preflight_result.json',{'passed':True,'type':'read-only inventory and frozen source validation',
          'task_id':tid,'train_instances':inventory['tasks'][task]['train'],
          'heldout_instances':inventory['tasks'][task]['test'],'human_stats':stats[task],
          'original_runtime_and_checkpoint':'same verified assets as modem; verified again in every job',
          'not_a_closed_loop_smoke':True})
    shutil.copyfile(HERE/'task_controller.py',folder/'controller.py')
    shutil.copyfile(HERE/'deploy_template.py',folder/'deploy.py')
    shutil.copyfile(SOURCE/'controller.py',folder/'control_support.py')
    write(folder/'auto_recovery_policy.json',{'eligible_phases':['baseline','sample1','train1','final'],
         'max_attempts_per_shard':8,'within_original_gpu_hour_budget':True,'preserve_all_completed_cases':True})
    source_names=['common.py','training.py','experiment.py','simulator.py','helpers.py','driver.py','bootstrap.py','eval_utils.py','resource_watch.py','goals.py','plan.json','controller.py','deploy.py','control_support.py','storage_guard.py','recovery_policy.py','auto_recovery_policy.json']
    source_names += [str(x.relative_to(folder)) for x in sorted((folder/'b1k_rlinf').glob('*.py'))]
    write(folder/'execution_manifest.json',{'source_sha256':{name:sha(folder/name) for name in source_names},
          'created_at':datetime.now(timezone.utc).isoformat(),'base_algorithm_source':str(SOURCE),
          'unchanged_ppo_training_sha256':sha(SOURCE/'training.py')})
    records.append({'slug':slug,'task':task,'task_id':tid,'directory':str(folder),
                    'max_steps':p['eval_steps'],'new_episode_budget':80,'gpu_hours_limit':budget,
                    'storage_gib_limit':storage,'training_trajectories':32,'optimizer_updates':17,
                    'heldout_instances':20,'training_sanity_instances':4})
write(HERE/'campaign.json',{'created_at':datetime.now(timezone.utc).isoformat(),
      'task_selection':'Declared before new baseline outcomes; two distinct task families with existing exact predicates and available public instances',
      'preregistered_tasks':records,'max_live_gpus':8,'max_gpu_hours_new_campaigns':256,
      'order':'Complete the resumed modem campaign, then trash, then water',
      'evidence_standard':'Paired held-out success/score per task; no selection of successful tasks; one training run per task is preliminary evidence only'})
print(json.dumps(records,ensure_ascii=False,indent=2))
