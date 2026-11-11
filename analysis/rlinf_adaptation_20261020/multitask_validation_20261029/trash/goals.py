"""Read the frozen task's exact physical predicates; no learned reward labels."""
import math


def whole_task_prompt(observation, expected_task_id, registry_prompt):
    if 'task_id' not in observation:
        raise ValueError('Missing observation task ID')
    value=observation['task_id']
    if hasattr(value,'reshape'):value=value.reshape(-1)[0]
    if hasattr(value,'item'):value=value.item()
    if int(value)!=int(expected_task_id):raise ValueError('Wrong task ID')
    if not isinstance(registry_prompt,str) or not registry_prompt.strip():raise ValueError('Missing task prompt')
    if observation.get('prompt') is not None and observation['prompt']!=registry_prompt:
        raise ValueError('Unexpected prompt override')
    return registry_prompt


def potential(signals):
    goals=signals['goals']
    if not goals or not all(type(value) is bool for value in goals.values()):
        raise ValueError('Expected nonempty Boolean physical goals')
    return float(sum(goals.values()))


def shaping_bonus(before,after,done,config):
    if before['goals'].keys()!=after['goals'].keys():raise ValueError('Goal identities changed')
    value=float(config['beta'])*(float(config['gamma'])*(0. if done else potential(after))-potential(before))
    if not math.isfinite(value):raise ValueError('Nonfinite shaping')
    return value


class TaskObserver:
    def __init__(self,evaluator,contract,state_types=None):
        if state_types is None:
            from omnigibson.object_states import Inside,OnTop,Open
            state_types={'inside':Inside,'ontop':OnTop,'open':Open}
        self.spec=contract
        self.items=[]
        scope=evaluator.env.task.object_scope
        def resolve(name):
            item=scope[name];obj=getattr(item,'wrapped_obj',item)
            if obj is None:raise ValueError('Unresolved object '+name)
            return obj
        for item in contract['predicates']:
            kind=item['predicate'];cls=state_types[kind]
            objects=[resolve(name) for name in item['arguments']]
            if cls not in objects[0].states:raise ValueError('Missing exact object state '+kind)
            if kind=='open' and len(objects)!=1:raise ValueError('Open requires one argument')
            if kind!='open' and len(objects)!=2:raise ValueError('Relation requires two arguments')
            self.items.append((item['name'],objects,cls,bool(item.get('negated',False))))
        assert [x[0] for x in self.items]==contract['goals']

    def observe(self):
        goals={}
        for name,objects,cls,negated in self.items:
            value=bool(objects[0].states[cls].get_value(*objects[1:]))
            goals[name]=not value if negated else value
        result={'goals':goals,**goals}
        result['potential']=potential(result)
        result['all_goals']=all(goals.values())
        return result

    def contract(self):
        return {**self.spec,'read_only':True,'source':'official OmniGibson physical object states'}
