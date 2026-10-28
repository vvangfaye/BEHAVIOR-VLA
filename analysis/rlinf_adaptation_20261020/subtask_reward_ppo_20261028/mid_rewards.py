"""Read-only physical milestones. Scores never depend on generated subtask text."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Signals:
    base_distance_m: float
    finger_marker_gap_m: float
    toggled_on: bool


def validate(signals):
    for value in (signals.base_distance_m, signals.finger_marker_gap_m):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Non-finite or negative physical reward distance')


class Milestones:
    def __init__(self, config, initial):
        validate(initial)
        self.config = config
        self.initial = initial
        near, finger = self.predicates(initial)
        self.awarded = {'near': near, 'finger': finger}
        self.initially_satisfied = dict(self.awarded)
        self.dwell = {'near': 0, 'finger': 0}
        self.total = 0.
        self.steps = 0
        self.events = []

    def predicates(self, signals):
        near = signals.base_distance_m <= self.config['near_distance_m']
        finger = near and signals.finger_marker_gap_m <= self.config['finger_marker_gap_m']
        return near, finger

    def step(self, signals):
        validate(signals)
        self.steps += 1
        added = 0.
        events = []
        for name, satisfied in zip(('near','finger'), self.predicates(signals)):
            self.dwell[name] = self.dwell[name]+1 if satisfied else 0
            if not self.awarded[name] and self.dwell[name] >= self.config['stable_control_steps']:
                value = self.config['near_bonus' if name=='near' else 'finger_bonus']
                added += value
                self.awarded[name] = True
                event = {'step':self.steps,'milestone':name,'reward':value,
                         'base_distance_m':signals.base_distance_m,
                         'finger_marker_gap_m':signals.finger_marker_gap_m}
                self.events.append(event); events.append(event)
        self.total += added
        assert self.total <= self.config['max_bonus_per_episode'] + 1e-8
        return added, events

    def summary(self):
        return {'total_mid_reward':self.total,'steps':self.steps,
                'initially_satisfied':self.initially_satisfied,'events':self.events,
                'label_independent':True,'maximum':self.config['max_bonus_per_episode']}


def vector(value):
    result = [float(x) for x in value]
    if len(result)!=3 or not all(math.isfinite(x) for x in result):
        raise ValueError('Invalid physical position')
    return result


def box(entity):
    lower, upper = entity.aabb
    lower, upper = vector(lower), vector(upper)
    if any(a>b for a,b in zip(lower,upper)):
        raise ValueError('Invalid AABB')
    return lower,upper


def box_gap(left, right):
    a,b=left;c,d=right
    return math.sqrt(sum(max(x1-y2,y1-x2,0.)**2 for x1,x2,y1,y2 in zip(a,b,c,d)))


class RadioObserver:
    def __init__(self, evaluator, config):
        from omnigibson.object_states import ToggledOn
        if config['task']!='turning_on_radio':
            raise ValueError('This reward contract supports only turning_on_radio')
        scope=evaluator.env.task.object_scope
        self.target=scope[config['target_bddl_id']]
        self.robot=evaluator.robot
        self.toggle=self.target.states[ToggledOn]
        self.marker=self.toggle.marker
        if self.marker is None:
            raise ValueError('Radio toggle marker is not observable; do not invent a reward')
        self.fingers=[link for links in self.robot.finger_links.values() for link in links]
        if not self.fingers:
            raise ValueError('No observable robot finger links')
        self.config=config

    def observe(self):
        lower,upper=box(self.target)
        position=vector(self.robot.get_position_orientation()[0])
        distance=math.sqrt(sum(max(lower[i]-position[i],position[i]-upper[i],0.)**2 for i in (0,1)))
        marker_box=box(self.marker)
        gap=min(box_gap(box(finger),marker_box) for finger in self.fingers)
        result=Signals(distance,gap,bool(self.toggle.get_value()))
        validate(result)
        return result

    def contract(self):
        return {'target':self.target.name,'marker':self.marker.name,
                'finger_links':[x.name for x in self.fingers],
                'hand_measure':'3D AABB separation between finger links and actual toggle marker',
                'near_measure':'XY point-to-target-AABB separation',
                'read_only':True}
