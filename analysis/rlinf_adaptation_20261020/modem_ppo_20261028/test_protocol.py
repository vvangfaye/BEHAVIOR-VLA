import ast
from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from goals import ModemObserver, potential, shaping_bonus, whole_task_prompt

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE/'plan.json').read_text())


def signals(a=False,b=False,c=False):
    return dict(toggled_on=a,nextto_tv=b,shared_tv_cabinet=c)


class RewardTests(unittest.TestCase):
    def test_potential_telescope_independent_of_order_and_repeated_visits(self):
        config = PLAN['shaping']; gamma = config['gamma']
        paths = [[signals(),signals(True),signals(True,True),signals(True,True,True)],
                 [signals(),signals(False,False,True),signals(),signals(False,False,True),signals(True,True,True)],
                 [signals(),signals(),signals()]]
        for path in paths:
            shaped = sum(gamma**i * shaping_bonus(a,b,i==len(path)-2,config)
                         for i,(a,b) in enumerate(zip(path,path[1:])))
            self.assertAlmostEqual(shaped,-potential(path[0])*config['beta'])

    def test_initially_satisfied_goal_does_not_create_extra_return(self):
        path = [signals(True),signals(),signals(True),signals(True)]
        gamma = PLAN['shaping']['gamma']
        total = sum(gamma**i * shaping_bonus(a,b,i==2,PLAN['shaping'])
                    for i,(a,b) in enumerate(zip(path,path[1:])))
        self.assertAlmostEqual(total,-1.0)

    def test_lost_goal_has_negative_bonus(self):
        self.assertLess(shaping_bonus(signals(True),signals(),False,PLAN['shaping']),0)

    def test_missing_or_non_boolean_truth_fails_closed(self):
        with self.assertRaises(KeyError): potential({'toggled_on':False})
        with self.assertRaises(ValueError): potential(dict(toggled_on=0,nextto_tv=False,shared_tv_cabinet=False))

    def test_shared_cabinet_requires_same_cabinet_as_tv(self):
        classes = tuple(type(name,(),{}) for name in ('NextTo','OnTop','ToggledOn'))
        NextTo,OnTop,ToggledOn = classes
        first = SimpleNamespace(name='c1');second=SimpleNamespace(name='c2')
        def state(fn):return SimpleNamespace(get_value=fn)
        modem=SimpleNamespace(name='modem',states={NextTo:state(lambda other:True),
                  ToggledOn:state(lambda:True),OnTop:state(lambda c:c is second)})
        tv=SimpleNamespace(name='tv',states={OnTop:state(lambda c:c is first)})
        contract=PLAN['goal_contract']; scope={contract['target_bddl_id']:modem,contract['tv_bddl_id']:tv,
                      contract['cabinet_bddl_ids'][0]:first,contract['cabinet_bddl_ids'][1]:second}
        evaluator=SimpleNamespace(env=SimpleNamespace(task=SimpleNamespace(object_scope=scope)))
        observer=ModemObserver(evaluator,contract,classes)
        self.assertFalse(observer.observe()['shared_tv_cabinet'])
        modem.states[OnTop]=state(lambda c:c is first)
        self.assertTrue(observer.observe()['all_goals'])


class ProtocolTests(unittest.TestCase):
    def test_supervisor_preserves_sigkill_exit_and_memory_evidence(self):
        import resource_watch,sys
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            receipt=resource_watch.run([sys.executable,'-c','import os,signal;os.kill(os.getpid(),signal.SIGKILL)'],
                                       root,dict(__import__('os').environ))
            self.assertEqual(receipt['returncode'],-9)
            self.assertEqual(receipt['signal'],'SIGKILL')
            self.assertTrue((root/'resource_latest.json').is_file())
            self.assertEqual(json.loads((root/'driver_exit.json').read_text())['returncode'],-9)

    def test_whole_task_uses_registered_text_without_obs_prompt(self):
        prompt=PLAN['task_prompt']
        self.assertEqual(whole_task_prompt({'task_id':torch.tensor([77])},77,prompt),prompt)
        self.assertEqual(whole_task_prompt({'task_id':np.array([77]),'prompt':prompt},77,prompt),prompt)

    def test_whole_task_rejects_other_task_and_prompt_override(self):
        with self.assertRaises(ValueError):whole_task_prompt({'task_id':torch.tensor([0])},77,PLAN['task_prompt'])
        with self.assertRaises(ValueError):whole_task_prompt({'task_id':torch.tensor([77]),'prompt':'radio'},77,PLAN['task_prompt'])

    def test_disjoint_instances_seeds_and_two_on_policy_rounds(self):
        self.assertTrue(Path(PLAN['dependency_status_file']).is_absolute())
        self.assertTrue(Path(PLAN['dependency_campaign_directory']).is_absolute())
        self.assertEqual(set(PLAN['train_instances']),set(range(1,17)))
        self.assertEqual(len(PLAN['train_instances']),32)
        self.assertEqual(len(PLAN['evaluation']['cases']),46)
        self.assertFalse(set(i for i,s in PLAN['evaluation']['heldout_cases']) & set(PLAN['train_instances']))
        train_seeds=[s for values in PLAN['train_seeds'] for s in values]
        self.assertEqual(len(train_seeds),len(set(train_seeds)))
        self.assertFalse(set(train_seeds)&{s for i,s in PLAN['evaluation']['cases']})
        self.assertEqual(PLAN['phase_order'],['baseline','sample1','train1','sample2','train2','final'])

    def test_actual_bootstrap_keeps_workflow_after_diagnostic_mutation(self):
        tree=ast.parse((HERE/'bootstrap.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                  and isinstance(n.func.value,ast.Name) and n.func.value.id=='env' and n.func.attr=='update')
        code=compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)],type_ignores=[])),
                     '<real bootstrap env>','exec')
        for phase in PLAN['phase_order']:
            config={'rank':0,'attempt':1,'task_phase':phase,'phase':'graphics_setup','job_name':'test',
                    'status_put':'private','result_put':'private','artifact_puts':{}}
            env={}; runtime=Path('/runtime');base=Path('/base');root=Path('/root')
            exec(code,dict(env=env,plan=PLAN,config=config,runtime=runtime,base=base,root=root,
                           library_path='/library',vk='/vk',egl='/egl',json=json))
            self.assertEqual(env['B1K_PHASE'],phase)

    def test_update_partition_is_one_pass_and_compute_budget_is_bounded(self):
        from training import partition_updates
        order=list(range(4599));groups=partition_updates(order,17)
        self.assertEqual([i for group in groups for i in group],order)
        self.assertEqual(len(groups),17)
        nominal=sum(seconds*(1 if phase.startswith('train') else 2)
                    for phase,seconds in PLAN['phase_limits_seconds'].items())/3600
        self.assertLessEqual(nominal,PLAN['max_total_gpu_hours'])
        self.assertEqual(PLAN['max_live_gpus'],2)

    def test_scheduler_waits_for_both_baseline_shards(self):
        from controller import next_pending
        one=[dict(phase='baseline',rank=0,attempt=1,state='SUCCEEDED')]
        self.assertEqual(next_pending(PLAN,one)[0],'baseline')
        both=one+[dict(phase='baseline',rank=1,attempt=1,state='SUCCEEDED')]
        self.assertEqual(next_pending(PLAN,both)[0],'sample1')
        both+=[dict(phase='sample1',rank=0,attempt=1,state='RUNNING'),
               dict(phase='sample1',rank=1,attempt=1,state='SUCCEEDED')]
        self.assertEqual(next_pending(PLAN,both)[0],'sample1')

    def test_timeout_is_saved_as_finite_horizon_terminal(self):
        import experiment
        class Model:
            def predict_action_batch(self,obs,mode):
                return torch.zeros(1,16,23),dict(forward_inputs={'x':torch.zeros(1)},
                                               prev_values=torch.ones(1),prev_logprobs=torch.zeros(1,16,23))
        class Sim:
            def __init__(self):
                self.current={'physical_goals':signals(),'status':{'steps':0},'observation':{}}
            def step(self,actions,text):
                self.current=dict(physical_goals=signals(True),status={'steps':16,'score':0.,'success':False},
                                  rewards=[0.]*16,done=True,terminal=False,truncated=True)
                return self.current
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            plan=copy.deepcopy(PLAN);plan['output']=directory
            output=Path(directory)/'training/round_01/episode_01';output.mkdir(parents=True)
            def start(*args):
                self.assertTrue(args[0]['collecting_training'])
                return Sim()
            with patch.object(experiment,'enforce_storage_budget'),patch.object(experiment,'start_simulator',start),\
                 patch.object(experiment,'env_observation',lambda x:x),patch.object(experiment,'label_text',lambda *x:'subtask'):
                result=experiment.collect(Model(),plan,1,0,lambda **kw:None)
            row=result['transitions'][-1]
            self.assertTrue(row['terminal']);self.assertFalse(row['environment_terminal'])
            self.assertEqual(float(result['bootstrap']),0.)
            self.assertEqual(row['shaping_reward'],0.)  # terminal potential is zero
            self.assertEqual(result['seed'],PLAN['train_seeds'][0][0])

    def test_final_report_ignores_partial_case_from_earlier_worker_snapshot(self):
        import controller
        plan=copy.deepcopy(PLAN)
        plan['evaluation'].update(cases=[[1,0],[301,0]],training_cases=[[1,0]],heldout_cases=[[301,0]])
        initial={'physical_state':{'object':[[0.,0.,0.],[0.,0.,0.,1.]]},'proprio':[0.],
                 'physics_dt':.01,'rendering_dt':.02,'observation_dt':.02}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);complete=root/'retrieved/complete';partial=root/'retrieved/partial'
            for phase in ('baseline','final'):
                for instance in (1,301):
                    case=complete/phase/f'i{instance}_s0';case.mkdir(parents=True)
                    items={'result.json':{'instance':instance,'seed':0,'score':1.,'success':True},
                           'initial.json':initial,'spec.json':{},'actions.json':[],
                           'goal_summary.json':{'final':signals(True,True,True),'ever_achieved':signals(True,True,True)}}
                    for name,data in items.items():(case/name).write_text(json.dumps(data))
                    old=partial/phase/f'i{instance}_s0';old.mkdir(parents=True)
                    (old/'goal_summary.json').write_text(json.dumps({'wrong':'unfinished case'}))
            for index in (1,2):
                checkpoint=complete/'checkpoints'/f'round_{index:02d}';checkpoint.mkdir(parents=True)
                (checkpoint/'ready.json').write_text(json.dumps({'optimizer_steps':17,'cumulative_optimizer_steps':index*17}))
            with patch.object(controller,'HERE',root):
                report=controller.finalize(plan)
            self.assertTrue(report['heldout_instances']['comparison_valid'])
            self.assertEqual(report['heldout_instances']['completed_pairs'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
