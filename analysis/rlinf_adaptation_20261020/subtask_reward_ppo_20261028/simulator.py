"""Original competition evaluator in its unchanged simulator environment."""

import argparse
import faulthandler
import hashlib
import json
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path


def install_step_recorder(env):
    """Record scalar results without forwarding assignment into the vector env."""
    original_step = env.step
    last = {}

    def recording_step(*args, **kwargs):
        value = original_step(*args, **kwargs)
        last["reward"] = float(value[1])
        last["info"] = value[4]
        return value

    # SingleEnvironmentAdapter.__setattr__ forwards ordinary assignment. The
    # captured bound step calls that forwarded target, which would call us again.
    object.__setattr__(env, "step", recording_step)
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    conn = Connection(args.fd)
    faulthandler.enable()
    # Avoid asynchronous frame walking while Isaac Sim worker threads mutate frames.
    # Keep fatal-signal diagnostics; initialization.json already reports the load stage.
    started = time.monotonic()

    def stage(name):
        target = Path(spec["output"]) / "initialization.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"stage": name, "elapsed_seconds": time.monotonic() - started})
        )
        temporary.replace(target)
        print("INITIALIZATION " + name, flush=True)

    stage("importing_simulator")
    import numpy as np
    import omnigibson as og
    import torch
    from helpers import _build_evaluator_cfg
    from omnigibson.eval import eval as base
    from omnigibson.metrics.task_metric import TaskMetric

    base.seed_everything(spec["seed"])
    og.macros.gm.HEADLESS = True

    class ActionQueue:
        current_subtask = None
        current_subtask_source = "generated"

        def reset(self):
            self.actions = []

        def forward(self, obs):
            return self.actions.pop(0)

    policy = ActionQueue()

    class Evaluator(base.Evaluator):
        def load_policy(self):
            return policy

    cfgargs = base.parse_args(
        [
            "--task-name",
            spec["task"],
            "--mode",
            spec["mode"],
            "--prompt-mode",
            "whole_task",
            "--num-rollouts",
            "1",
            "--max-steps",
            str(spec["max_steps"]),
            "--headless",
            "--env-wrapper",
            "omnigibson.eval.wrappers.RGBDFullResWrapper",
            "--robot-config",
            spec["robot_config"],
            "--output-dir",
            spec["output"],
        ]
    )
    stage("creating_evaluator")
    evaluator = Evaluator(_build_evaluator_cfg(base, cfgargs, spec["seed"]))
    stage("first_reset")
    evaluator.reset(prepare_for_task_instance=True)
    stage("loading_task_instance")
    evaluator.load_task_instance(spec["instance"])
    stage("resetting_task_instance")
    evaluator.reset()
    assert evaluator.robot.action_dim == 23
    evaluator.start_recording(str(Path(spec["output"]) / "rollout.mp4"))
    metric = next(m for m in evaluator.metrics if isinstance(m, TaskMetric))
    last = install_step_recorder(evaluator.env)
    observer = milestones = reward_trace = None
    if spec.get("mid_reward"):
        from mid_rewards import RadioObserver, Milestones
        from dataclasses import asdict
        observer = RadioObserver(evaluator, spec["mid_reward"])
        initial_signal = observer.observe()
        milestones = Milestones(spec["mid_reward"], initial_signal)
        Path(spec["output"], "reward_contract.json").write_text(json.dumps(
            {**observer.contract(), "initial": asdict(initial_signal), "config": spec["mid_reward"]}, indent=2))
        reward_trace = Path(spec["output"], "reward_trace.jsonl").open("x", buffering=1)


    def observation():
        obs = evaluator.obs
        cams = [
            obs[evaluator.robot_camera_names[name] + "::rgb"][..., :3].cpu().numpy()
            for name in ["head", "left_wrist", "right_wrist"]
        ]
        proprio = obs[evaluator.robot.name + "::proprio"].cpu().numpy()
        return {"state": proprio, "images": cams, "prompt": str(obs.get("prompt", "Turn on the radio receiver that's on the table in the living room."))}

    def status():
        metrics = metric.aggregate(evaluator.env)
        score = metrics.get("q_score", {}).get("final", 0.0)
        return {
            "steps": evaluator._rollout_steps,
            "score": float(score),
            "success": bool(evaluator.env.success),
        }

    physical = {}
    for obj in evaluator.env.scene.objects:
        p, q = obj.get_position_orientation()
        physical[obj.name] = [p.cpu().tolist(), q.cpu().tolist()]
    initial_json = json.dumps(physical, sort_keys=True)
    initial = {
        "physical_state": physical,
        "physical_sha256": hashlib.sha256(initial_json.encode()).hexdigest(),
        "proprio": observation()["state"].tolist(),
        "physics_dt": og.sim.get_physics_dt(),
        "rendering_dt": og.sim.get_rendering_dt(),
        "observation_dt": og.sim.get_sim_step_dt(),
    }
    stage("ready")
    faulthandler.cancel_dump_traceback_later()
    done = False
    conn.send({"observation": observation(), "initial": initial, "status": status()})
    try:
        while True:
            command = conn.recv()
            if command["op"] == "close":
                evaluator.stop_recording()
                conn.send(status())
                break
            assert command["op"] == "step" and not done
            actions = np.asarray(command["actions"], dtype=np.float32)
            assert actions.ndim == 2 and actions.shape[1] == 23 and len(actions) <= 16
            assert np.isfinite(actions).all()
            policy.actions = [torch.from_numpy(x.copy()) for x in actions]
            policy.current_subtask = command["subtask"]
            rewards = []
            mid_rewards = []
            terminal = False
            truncated = False
            for _ in range(len(actions)):
                terminal, truncated = evaluator.step()
                rewards.append(last["reward"])
                if milestones is not None:
                    signal = observer.observe()
                    bonus, events = milestones.step(signal)
                    mid_rewards.append(bonus)
                    reward_trace.write(json.dumps({"step": evaluator._rollout_steps, **asdict(signal),
                        "environment_reward": last["reward"], "mid_reward": bonus, "events": events})+"\n")
                    if events:
                        Path(spec["output"], "reward_summary.json").write_text(json.dumps(milestones.summary(), indent=2))
                else:
                    mid_rewards.append(0.)

                if (
                    terminal
                    or truncated
                    or evaluator._rollout_steps >= spec["max_steps"]
                ):
                    done = True
                    break
            conn.send(
                {
                    "observation": observation(),
                    "status": status(),
                    "rewards": rewards,
                    "mid_rewards": mid_rewards,
                    "done": done,
                    "terminal": bool(terminal),
                    "truncated": bool(truncated),
                }
            )
    except BaseException as error:
        details = {
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "traceback": traceback.format_exc(),
        }
        # Persist before IPC: the parent terminates the stage on receiving error.
        Path(spec["output"], "failure.json").write_text(json.dumps(details, indent=2))
        print(details["traceback"], flush=True)
        try:
            conn.send(details)
        except Exception:
            pass
        raise
    finally:
        if reward_trace is not None:
            reward_trace.close()
            Path(spec["output"], "reward_summary.json").write_text(json.dumps(milestones.summary(), indent=2))
        evaluator.stop_recording()
        conn.close()
        og.shutdown()


if __name__ == "__main__":
    main()
