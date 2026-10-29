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
    def test_small_pilot_is_fixed_four_rollouts_two_updates_four_pairs(self):
        self.assertEqual(PLAN['train_instances'],[1,2,3,4])
        self.assertEqual(PLAN['optimizer']['updates_per_round'],[2])
        self.assertEqual(PLAN['rounds'],1)
        self.assertEqual(PLAN['evaluation']['cases'],[[1,0],[2,0],[301,0],[302,0]])
        self.assertEqual(PLAN['max_live_gpus'],8)

    def test_parallel_sample_and_eval_shards_do_not_duplicate(self):
        for source in (list(range(4)),PLAN['evaluation']['cases']):
            parts=[source[rank::4] for rank in range(4)]
            self.assertEqual(sum(len(x) for x in parts),len(source))
            self.assertEqual(len({str(v) for part in parts for v in part}),len(source))

    def test_whole_task_registry_not_empty_observation_prompt(self):
        self.assertEqual(whole_task_prompt({'task_id':torch.tensor([77])},77,PLAN['task_prompt']),PLAN['task_prompt'])

    def test_one_training_pass_exactly_two_updates(self):
        from training import partition_updates
        order=list(range(700));groups=partition_updates(order,2)
        self.assertEqual(len(groups),2)
        self.assertEqual([v for g in groups for v in g],order)

    def test_bootstrap_uses_immutable_phase(self):
        tree=ast.parse((HERE/'bootstrap.py').read_text())
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                  and isinstance(n.func.value,ast.Name) and n.func.value.id=='env' and n.func.attr=='update')
        code=compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)],type_ignores=[])),'<bootstrap>','exec')
        for phase in PLAN['phase_order']:
            config={'rank':0,'attempt':1,'task_phase':phase,'phase':'graphics_setup','job_name':'test',
                    'status_put':'private','result_put':'private','artifact_puts':{}}
            env={};runtime=Path('/runtime');base=Path('/base');root=Path('/root')
            exec(code,dict(env=env,plan=PLAN,config=config,runtime=runtime,base=base,root=root,
                           library_path='/library',vk='/vk',egl='/egl',json=json))
            self.assertEqual(env['B1K_PHASE'],phase)

if __name__=='__main__':unittest.main(verbosity=2)
