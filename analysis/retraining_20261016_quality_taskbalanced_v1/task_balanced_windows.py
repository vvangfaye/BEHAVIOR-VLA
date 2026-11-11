"""Finite globally task-balanced windows with strict per-rollout exposure limits."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import math


class TaskBalancedWindowSampler:
    def __init__(self, plan, policy):
        expected = {'kind', 'task_count', 'within_task_full_ceiling', 'full_episode_draw_cap',
                    'window_stride', 'maximum_full_window_passes','finite_budget_updates'}
        if set(policy) != expected or policy['kind'] != 'task_balanced_capped_windows_v1':
            raise ValueError('Unknown quality sampling policy')
        if len(plan._phases) != 1 or policy['window_stride'] != plan.action_horizon:
            raise ValueError('Quality sampler requires one finite phase and nonoverlapping windows')
        if type(policy['finite_budget_updates']) is not int or plan.num_steps != policy['finite_budget_updates']:
            raise ValueError('Changing the finite budget would change already consumed sampling')
        for key in ('task_count','full_episode_draw_cap','maximum_full_window_passes'):
            if type(policy[key]) is not int or policy[key] < 1:
                raise ValueError('Positive integer sampling limits required')
        if not 0 < policy['within_task_full_ceiling'] <= 1:
            raise ValueError('Invalid within-task full ceiling')
        sources = {s['kind']:s for s in plan._phases[0]['sources']}
        self.expert = dict(sources['replay']['balanced_tasks'])
        self.tasks = tuple(sorted(self.expert))
        if len(self.tasks) != policy['task_count'] or plan.num_samples % len(self.tasks):
            raise ValueError('Task count or finite balanced budget differs')
        self.plan, self.policy = plan, dict(policy)
        self.per_task = plan.num_samples // len(self.tasks)
        self._orders = OrderedDict()
        self.slots, self.episode_draws, self.task_full_draws = {}, {}, {}
        horizon, stride = plan.action_horizon, policy['window_stride']
        for task, instances in sources['full_episode']['balanced_tasks']:
            if task not in self.expert:
                raise ValueError('Full task missing from original expert source')
            target = math.floor(self.per_task * policy['within_task_full_ceiling'])
            selected = []
            for _, episodes in instances:
                for identity, segments in episodes:
                    windows = [(s, frame) for s in segments
                        for frame in range(s['start'], s['end']-horizon+1, stride)]
                    quota = min(policy['full_episode_draw_cap'], target//len(instances)//len(episodes),
                                policy['maximum_full_window_passes'] * len(windows))
                    ordered = []
                    for repeat in range(policy['maximum_full_window_passes']):
                        ordered.extend(sorted(windows,key=lambda x:self.draw(
                            'window-order', task, identity, repeat, x[1])))
                    selected.extend(ordered[:quota])
                    self.episode_draws[identity] = quota
            selected.sort(key=lambda x:self.draw('within-task-full-order',task,x[0]['root_id'],x[1]))
            positions = sorted(range(self.per_task),key=lambda p:self.draw('full-slot',task,p))[:len(selected)]
            self.slots[task] = dict(zip(positions,selected))
            self.task_full_draws[task] = len(selected)
            if len(selected) > target:
                raise ValueError('Within-task quota exceeded')
        self.full_draws = sum(self.episode_draws.values())
        if not math.isclose(sources['full_episode']['weight'], self.full_draws/plan.num_samples, abs_tol=1e-12):
            raise ValueError('Declared source weight differs from actual finite full budget')

    def draw(self,*parts):
        raw = repr((self.plan.seed,*parts)).encode()
        return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),'big')

    def task_at(self,index):
        cycle,position = divmod(index,len(self.tasks))
        if cycle not in self._orders:
            self._orders[cycle] = tuple(sorted(self.tasks,key=lambda task:self.draw('task-order',cycle,task)))
            while len(self._orders)>64:self._orders.popitem(last=False)
        self._orders.move_to_end(cycle)
        return self._orders[cycle][position],cycle

    def reference(self,index,*,expert_only=False):
        task,ordinal = self.task_at(index)
        full = None if expert_only else self.slots.get(task,{}).get(ordinal)
        if full is not None:
            segment,frame = full
            kind = 'full_episode'
        else:
            instances = self.expert[task]
            _,episodes = instances[self.draw(index,'instance')%len(instances)]
            _,segments = episodes[self.draw(index,'episode')%len(episodes)]
            total = sum(s['end']-s['start']-self.plan.action_horizon+1 for s in segments)
            offset = self.draw(index,'legal_window')%total
            for segment in segments:
                n = segment['end']-segment['start']-self.plan.action_horizon+1
                if offset<n:break
                offset-=n
            frame = segment['start']+offset
            kind = 'replay'
        return {'root_id':segment['root_id'],'episode_index':segment['episode_index'],
            'frame':frame,'subtask':segment['subtask'],'source_kind':kind,'task_name':task,
            'task_instance_id':segment['task_instance_id'],'group':(task,segment['subtask']),
            'segment_start':segment['start'],'segment_end':segment['end'],'global_index':index}
