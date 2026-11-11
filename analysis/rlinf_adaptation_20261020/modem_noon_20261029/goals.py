"""Read exact task predicates; add a telescoping chunk-level potential in training."""
import math


def whole_task_prompt(observation, expected_task_id, registry_prompt):
    """Match the original server: whole-task text comes from its task registry."""
    if 'task_id' not in observation:
        raise ValueError('Missing observation task ID')
    value = observation['task_id']
    if hasattr(value, 'reshape'):
        value = value.reshape(-1)[0]
    if hasattr(value, 'item'):
        value = value.item()
    if int(value) != int(expected_task_id):
        raise ValueError('Observation belongs to another task')
    if not isinstance(registry_prompt,str) or not registry_prompt.strip():
        raise ValueError('Invalid registered task instruction')
    override = observation.get('prompt')
    if override is not None and override != registry_prompt:
        raise ValueError('Unexpected prompt override in whole-task mode')
    return registry_prompt


def potential(signals):
    names = ('toggled_on', 'nextto_tv', 'shared_tv_cabinet')
    if not all(type(signals[name]) is bool for name in names):
        raise ValueError('Expected three actual Boolean task predicates')
    return float(sum(signals[name] for name in names))


def shaping_bonus(before, after, done, config):
    previous = potential(before)
    following = 0.0 if done else potential(after)
    value = float(config['beta']) * (float(config['gamma']) * following - previous)
    if not math.isfinite(value):
        raise ValueError('Nonfinite shaping reward')
    return value


class ModemObserver:
    def __init__(self, evaluator, contract, state_types=None):
        if state_types is None:
            from omnigibson.object_states import NextTo, OnTop, ToggledOn
        else:
            NextTo, OnTop, ToggledOn = state_types
        self.types = NextTo, OnTop, ToggledOn
        scope = evaluator.env.task.object_scope
        def resolve(name):
            item = scope[name]
            obj = getattr(item, 'wrapped_obj', item)
            if obj is None:
                raise ValueError('Unresolved task object: ' + name)
            return obj
        self.modem = resolve(contract['target_bddl_id'])
        self.tv = resolve(contract['tv_bddl_id'])
        self.cabinets = [resolve(name) for name in contract['cabinet_bddl_ids']]
        # Missing states must fail visibly; never substitute AABB proxies.
        for obj, classes in [(self.modem, (NextTo, OnTop, ToggledOn)), (self.tv, (OnTop,))]:
            if not all(cls in obj.states for cls in classes):
                raise ValueError('Missing official object state on ' + obj.name)

    def observe(self):
        NextTo, OnTop, ToggledOn = self.types
        tv = [bool(self.tv.states[OnTop].get_value(c)) for c in self.cabinets]
        modem = [bool(self.modem.states[OnTop].get_value(c)) for c in self.cabinets]
        result = {'toggled_on': bool(self.modem.states[ToggledOn].get_value()),
                  'nextto_tv': bool(self.modem.states[NextTo].get_value(self.tv)),
                  'shared_tv_cabinet': any(a and b for a, b in zip(tv, modem)),
                  'tv_on_cabinets': tv, 'modem_on_cabinets': modem}
        result['potential'] = potential(result)
        result['all_goals'] = result['potential'] == 3.0
        return result

    def contract(self):
        return {'modem': self.modem.name, 'tv': self.tv.name,
                'cabinets': [c.name for c in self.cabinets],
                'states': ['ToggledOn', 'NextTo', 'OnTop'],
                'shared_cabinet': 'exists c: OnTop(tv,c) and OnTop(modem,c)',
                'read_only': True}
