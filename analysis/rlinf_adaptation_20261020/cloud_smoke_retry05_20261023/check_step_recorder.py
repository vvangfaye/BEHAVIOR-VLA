"""Reproduce the recursion and verify the fix using the real adapter classes."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from simulator import install_step_recorder


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "BEHAVIOR-VLAs/BEHAVIOR-1K/OmniGibson/omnigibson"
spec = importlib.util.spec_from_file_location("single_env_regression", SOURCE / "eval/single_env.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Adapter = module.SingleEnvironmentAdapter
# Load the actual forwarding wrapper without importing the simulator runtime.
tree = ast.parse((SOURCE / "utils/python_utils.py").read_text())
wrapper_ast = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Wrapper")
namespace = {}
exec(compile(ast.Module(body=[wrapper_ast], type_ignores=[]), "actual_wrapper.py", "exec"), namespace)
Wrapper = namespace["Wrapper"]


class VectorEnv:
    num_envs = 1

    def __init__(self):
        self.calls = 0

    def step(self, action, *, scale=1):
        assert action.shape == (1, 23)
        self.calls += 1
        return [{"step": self.calls}], torch.tensor([self.calls * scale], dtype=torch.float32), [False], [False], [{"step": self.calls}]


class ObservationWrapper(Wrapper):
    @property
    def env(self):
        return self.wrapped_obj

    def step(self, action, **kwargs):
        return self.wrapped_obj.step(action, **kwargs)


def stack():
    base = VectorEnv()
    wrapper = ObservationWrapper(base)
    return Adapter(wrapper), wrapper, base


def main():
    env, wrapper, base = stack()
    original_step = env.step

    def old_recorder(*args, **kwargs):
        return original_step(*args, **kwargs)

    env.step = old_recorder
    reproduced = False
    try:
        env.step(torch.zeros(23))
    except RecursionError:
        reproduced = True
    assert reproduced and base.calls == 0

    env, wrapper, base = stack()
    underlying = base.step.__func__
    last = install_step_recorder(env)
    for step in range(1, 65):
        obs, reward, terminated, truncated, info = env.step(torch.zeros(23), scale=2)
        assert base.calls == step
        assert reward == step * 2 and last["reward"] == step * 2
        assert last["info"] == info == {"step": step}
        assert obs == {"step": step} and not terminated and not truncated
        assert base.step.__func__ is underlying
        assert "step" not in vars(wrapper) and "step" not in vars(base)
    report = {
        "passed": True,
        "old_recursion_reproduced": reproduced,
        "fixed_steps": base.calls,
        "scalar_reward_and_info_correct": True,
        "inner_step_unchanged": True,
        "actual_adapter_sha256": hashlib.sha256((SOURCE / "eval/single_env.py").read_bytes()).hexdigest(),
        "fixed_simulator_sha256": hashlib.sha256((HERE / "simulator.py").read_bytes()).hexdigest(),
        "scope": "CPU regression on actual adapter/wrapper code; GPU closed-loop remains required.",
    }
    (HERE / "regression_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
