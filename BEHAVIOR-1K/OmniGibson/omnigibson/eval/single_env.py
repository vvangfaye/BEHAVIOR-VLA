"""Single-environment view of the vectorized 3.9.3 evaluation API.

The diagnostic queue and exact-state subtask runner operate on one environment.
Keep their observations and predicates scalar while the simulator remains vectorized.
"""

from functools import partial


def _first(value):
    if isinstance(value, (list, tuple)) or getattr(value, "ndim", 0) > 0:
        if len(value) != 1:
            raise ValueError("Single-environment evaluation requires exactly one result")
        return value[0]
    return value


class SingleTaskAdapter:
    def __init__(self, task):
        self._task = task

    @property
    def object_scope(self):
        return self._task.object_scopes[0]

    @property
    def success(self):
        return bool(_first(self._task.success))

    def _evaluate_predicate(self, predicate_name, *entities):
        return self._task._evaluate_predicate(0, predicate_name, *entities)

    def get_goal_option_satisfaction(self):
        return self._task.get_goal_option_satisfaction(0)

    def __getattr__(self, name):
        if name == "assign_object_scope_with_cache":
            return partial(getattr(self._task, name), env_idx=0)
        return getattr(self._task, name)


class SingleEnvironmentAdapter:
    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("Subtask diagnostics require num_envs=1; use BatchedEvaluator for batches")
        object.__setattr__(self, "_wrapped_env", env)
        base = env
        seen = set()
        while hasattr(base, "env") and id(base) not in seen:
            seen.add(id(base))
            nested = base.env
            if nested is base:
                break
            base = nested
        object.__setattr__(self, "_base_env", base)

    @property
    def env(self):
        # Exact-state restoration must retain this scalar view when unwrapping.
        return self

    @property
    def task(self):
        return SingleTaskAdapter(self._wrapped_env.task)

    @property
    def robot(self):
        return self.scene.robots[0]

    @property
    def robots(self):
        return self.scene.robots

    @property
    def success(self):
        return self.task.success

    def get_goal_option_satisfaction(self):
        return self.task.get_goal_option_satisfaction()

    def get_obs(self, *args, **kwargs):
        obs, info = self._wrapped_env.get_obs(*args, **kwargs)
        return _first(obs), _first(info)

    def reset(self, *args, **kwargs):
        result = self._wrapped_env.reset(*args, **kwargs)
        if result is None:
            return None
        obs, info = result
        return _first(obs), _first(info)

    def step(self, action, *args, **kwargs):
        if getattr(action, "ndim", 0) == 1:
            action = action.unsqueeze(0)
        obs, reward, terminated, truncated, info = self._wrapped_env.step(action, *args, **kwargs)
        return _first(obs), _first(reward), bool(_first(terminated)), bool(_first(truncated)), _first(info)

    def __getattr__(self, name):
        if name in ("_current_step", "_current_episode"):
            plural = {"_current_step": "_current_steps", "_current_episode": "_current_episodes"}[name]
            if hasattr(self._base_env, plural):
                return _first(getattr(self._base_env, plural))
        return getattr(self._wrapped_env, name)

    def __setattr__(self, name, value):
        if name in ("_current_step", "_current_episode"):
            plural = {"_current_step": "_current_steps", "_current_episode": "_current_episodes"}[name]
            target_name = plural if hasattr(self._base_env, plural) else name
            counter = getattr(self._base_env, target_name, None)
            if hasattr(counter, "fill_"):
                counter.fill_(value)
            elif hasattr(counter, "fill"):
                counter.fill(value)
            else:
                setattr(self._base_env, target_name, value)
        else:
            setattr(self._wrapped_env, name, value)
