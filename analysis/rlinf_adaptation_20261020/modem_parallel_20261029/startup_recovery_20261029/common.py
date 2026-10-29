import json
import os
import socket
import signal
import subprocess
import tempfile
import shutil
import hashlib
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def env_observation(raw):
    return {
        "states": raw["state"][None],
        "main_images": raw["images"][0][None],
        "wrist_images": np.stack(raw["images"][1:])[None],
        "task_descriptions": [raw["prompt"]],
    }


def reuse_baseline_case(plan, case, output):
    source=plan.get('baseline_sources',{}).get(case)
    if source is None:
        return False
    assert source['cold_trainable_hash']==plan['source_identity']['cold_trainable_hash']
    bodies={name:(Path(source['directory'])/name).read_bytes() for name in source['files']}
    for name,body in bodies.items():
        assert hashlib.sha256(body).hexdigest()==source['files'][name], 'Reused baseline file changed: '+case+'/'+name
    spec=json.loads(bodies['spec.json'])
    assert spec['task']==plan['task'] and spec['max_steps']==plan['eval_steps']
    assert spec['mode']==('train' if spec['instance']<301 else 'public_test')
    assert not spec['shaping_enabled']
    assert case==f"i{spec['instance']}_s{spec['seed']}"
    result=json.loads(bodies['result.json'])
    assert (result['instance'],result['seed'])==(spec['instance'],spec['seed'])
    assert json.loads(bodies['actions.json']), 'Completed baseline has no executed actions'
    prompt=json.loads(bodies['prompt_contract.json'])
    assert prompt['prompt']==plan['task_prompt'] and prompt['source_sha256']==plan['task_prompt_source_sha256']
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,body in bodies.items():
        target=output/name
        if target.exists():
            assert target.read_bytes()==body,'Conflicting completed baseline: '+case+'/'+name
        else:
            temp=target.with_suffix(target.suffix+'.reuse_tmp');temp.write_bytes(body);temp.replace(target)
    write(output/'reuse_receipt.json',{'source':source['directory'],'files':source['files'],
                                     'original_checkpoint':'1688977','no_new_controls':True})
    return True


class SimulatorInitializationError(RuntimeError):
    pass


class ExecutionStopped(BaseException):
    """Cancellation must clean up the simulator without retrying initialization."""


class Simulator:
    def __init__(self, plan, output, instance, mode, max_steps):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        spec = {
            "task": plan["task"],
            "instance": instance,
            "mode": mode,
            "seed": plan["seed"],
            "max_steps": max_steps,
            "output": str(self.output),
            "robot_config": plan["code"]
            + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
        }
        spec["goal_contract"] = plan["goal_contract"]
        for name in ('task_prompt','task_prompt_source','task_prompt_source_sha256'):
            spec[name] = plan[name]
        spec["shaping_enabled"] = bool(plan.get("collecting_training", False))
        write(self.output / "spec.json", spec)
        parent, child = socket.socketpair()
        self.conn = Connection(parent.detach())
        childfd = child.fileno()
        self.log = (self.output / "simulator.log").open("x")
        env = dict(os.environ)
        code = Path(plan["code"])
        env["PYTHONPATH"] = ":".join(
            [
                str(Path(__file__).parent),
                str(code / "scripts"),
                str(code / "BEHAVIOR-1K/OmniGibson"),
                str(code / "BEHAVIOR-1K/bddl3"),
            ]
        )
        self.scratch = tempfile.TemporaryDirectory(prefix="b1k-sim-cache-", dir="/tmp")
        for key, name in [
            ("OMNIGIBSON_APPDATA_PATH", "appdata"),
            ("XDG_CACHE_HOME", "cache"),
            ("XDG_CONFIG_HOME", "config"),
            ("WARP_CACHE_PATH", "warp"),
        ]:
            directory = Path(self.scratch.name) / name
            directory.mkdir()
            env[key] = str(directory)
        self.process = subprocess.Popen(
            [
                plan["runtime"] + "/envs/b1k-eval/bin/python",
                str(Path(__file__).with_name("simulator.py")),
                "--fd",
                str(childfd),
                "--spec",
                str(self.output / "spec.json"),
            ],
            pass_fds=(childfd,),
            env=env,
            cwd=str(code),
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        child.close()
        try:
            self.current = self.receive(plan["initialization_timeout_seconds"])
        except BaseException as error:
            returncode = self.process.poll()
            # EOF can precede delivery of the child exit status by a few ms.
            # Preserve its natural signal before aborting any surviving process.
            if isinstance(error, EOFError) and returncode is None:
                try:
                    returncode = self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            write(self.output / "process_failure.json", {
                "error_type": type(error).__name__,
                "natural_returncode": returncode,
                "natural_signal": signal.Signals(-returncode).name
                    if returncode is not None and returncode < 0 else None,
                "alive_before_cleanup": returncode is None,
            })
            self.abort()
            if not isinstance(error, Exception):
                raise
            failure = self.output / 'python_startup_failure.json'
            if failure.is_file():
                details = json.loads(failure.read_text())
                if details.get('error_type') in ('AssertionError','ValueError','KeyError','AttributeError',
                                                 'TypeError','ImportError','ModuleNotFoundError'):
                    raise ValueError('Simulator task/code setup failed: '+str(details)[:800]) from error
            raise SimulatorInitializationError(
                f'{type(error).__name__}; simulator returncode={returncode}; logs={self.output}'
            ) from error
        write(self.output / "initial.json", self.current["initial"])

    def receive(self, timeout=180):
        if not self.conn.poll(timeout):
            progress_path = self.output / "initialization.json"
            progress = (
                progress_path.read_text()
                if progress_path.exists()
                else "no initialization progress"
            )
            raise TimeoutError(f"Simulator initialization/step timed out: {progress}")
        result = self.conn.recv()
        if "error_type" in result:
            raise RuntimeError(str(result))
        return result

    def step(self, actions, subtask):
        self.conn.send({"op": "step", "actions": actions, "subtask": subtask})
        self.current = self.receive()
        return self.current

    def close(self):
        if self.process.poll() is None:
            try:
                self.conn.send({"op": "close"})
                self.receive(30)
                self.process.wait(timeout=60)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.conn.close()
        self.log.close()
        self.cleanup_scratch()

    def abort(self):
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=15)
        self.conn.close()
        self.log.close()
        self.cleanup_scratch()

    def cleanup_scratch(self):
        source = Path(self.scratch.name) / 'appdata/local/logs'
        for path in source.rglob('*.log') if source.exists() else []:
            if path.is_file() and not path.is_symlink():
                target = self.output / 'native_logs' / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        self.scratch.cleanup()


def torch_model(plan):
    import torch
    from b1k_rlinf.factory import build_model
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model_path": plan["converted"],
            "pi05": True,
            "precision": "fp32",
            "add_value_head": True,
            "action_dim": 23,
            "num_action_chunks": 16,
            "num_steps": 10,
            "openpi": {
                "model_action_dim": 32,
                "action_horizon": 32,
                "max_token_len": 512,
                "train_expert_only": True,
                "noise_method": "flow_sde",
                "noise_level": 0.4,
                "ignore_last": True,
                "joint_logprob": False,
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
                "config_name": "pi05_behavior",
            },
            "b1k_joint": {
                "enabled": True,
                "match_jax_compute": True,
                "execution_gripper_open_tolerance": plan["action_interface"]["gripper_open_tolerance"],
                "norm_stats_path": plan["converted"] + "/norm_stats.json",
                "tokenizer_path": plan["tokenizer"],
                "state_layout": "checkpoint61",
            },
        }
    )
    torch.manual_seed(plan["seed"])
    model = build_model(cfg, torch.float32).cuda().eval()
    assert model.rl_cfg.add_value_head and model.value_head is not None
    return model


def label_text(model, result):
    replay = result["forward_inputs"]
    n = int(replay["subtask_lengths"][0])
    return model.joint_tokenizer.processor.decode(
        replay["subtask_tokens"][0, :n].tolist()
    )
