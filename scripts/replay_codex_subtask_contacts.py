"""Replay the frozen countertop attempt; observe contacts without model calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class RecordedActions:
    current_subtask = "place on tupperware to countertop"
    current_subtask_source = "provided"

    def __init__(self, actions):
        self.actions, self.index = actions, 0

    def reset(self):
        self.index = 0

    def forward(self, obs):
        import torch
        if self.index >= len(self.actions):
            raise RuntimeError("Replay exhausted the recorded actions")
        result = torch.as_tensor(self.actions[self.index].copy(), dtype=torch.float32)
        self.index += 1
        return result


def observe(evaluator, step, phase, *, raycasts=False):
    from omnigibson import object_states as states
    from omnigibson.macros import gm
    from omnigibson.utils.usd_utils import RigidContactAPI
    from omnigibson.utils.sampling_utils import raytest
    import omnigibson as og
    target = evaluator.env.scene.object_registry("name", "tupperware_238")
    objects = list(evaluator.env.scene.objects)
    candidates = [o for o in objects if o.category == "countertop" and "kitchen_0" in (o.in_rooms or [])]
    pairs = sorted(RigidContactAPI.get_contact_pairs(target.scene.idx, {target}, None, current_only=True))
    paths = {link.prim_path: o.name for o in objects for link in o.links.values()}

    def owner(path):
        return paths.get(path) or next((o.name for o in objects if path.startswith(o.prim_path + "/")), None)

    def tensor_pair(cls, other):
        si = target.scene.idx
        i, j = cls.OBJ_IDXS[target.relative_prim_path], cls.OBJ_IDXS[other.relative_prim_path]
        cpu = jsonable(cls.VALUES_CPU[si, i, j])
        gpu = jsonable(cls.VALUES[si, i, j])
        return dict(cpu=cpu, gpu=gpu, self_index=i, reference_index=j)

    lower, upper = target.aabb
    row = dict(phase=phase, step=step, target=target.name,
        position=jsonable(target.get_position_orientation()[0]),
        orientation=jsonable(target.get_position_orientation()[1]), aabb=jsonable([lower, upper]),
        grasped_by=[arm for arm in evaluator.robot.arm_names
                    if getattr(evaluator.robot._ag_obj_in_hand[arm], "name", None) == target.name],
        contacts=[dict(query=a, other=b, query_object=owner(a), other_object=owner(b)) for a, b in pairs],
        references={})
    for other in candidates:
        touching = bool(target.states[states.Touching].get_value(other))
        adjacent = jsonable(target.states[states.Adjacency].get_value(other))
        on_top = bool(target.states[states.OnTop].get_value(other))
        row["references"][other.name] = dict(touching=touching, above=bool(adjacent[0]),
            below=bool(adjacent[1]), on_top=on_top,
            conjunction=touching and bool(adjacent[1]) and not bool(adjacent[0]),
            position=jsonable(other.get_position_orientation()[0]), aabb=jsonable(other.aabb),
            touching_tensor=tensor_pair(states.Touching, other),
            adjacency_tensor=tensor_pair(states.Adjacency, other),
            on_top_tensor=tensor_pair(states.OnTop, other))
    if raycasts:
        center = (lower + upper) / 2
        row["rays"] = {}
        for name, dz in (("down", -5.0), ("up", 5.0)):
            end = center.clone()
            end[2] += dz
            hit = raytest(center, end, ignore_bodies=list(target.link_prim_paths))
            row["rays"][name] = jsonable(hit)
            if hit.get("hit"):
                row["rays"][name]["object"] = owner(hit.get("rigidBody", ""))
        row["state_runtime"] = dict(enable_object_states=bool(gm.ENABLE_OBJECT_STATES),
            update_order=[s.__name__ for s in og.sim.object_state_types_requiring_update],
            on_top_touch_indices=states.OnTop._touch_idx.numpy().tolist(),
            on_top_adjacency_indices=states.OnTop._adj_idx.numpy().tolist(),
            aabb_tensor=jsonable(target.states[states.AABB].get_value()),
            contact_row_paths=[p for p in target.link_prim_paths
                               if p in RigidContactAPI._PATH_TO_ROW_IDX.get(target.scene.idx, {})],
            contact_col_paths=[p for p in target.link_prim_paths
                               if p in RigidContactAPI._PATH_TO_COL_IDX.get(target.scene.idx, {})])
    return row


def run(config):
    import h5py
    import numpy as np
    from PIL import Image
    import eval_b1k_subtask_samples as runner
    from codex_subtask_collection import load_plan, inspect_attempt, file_sha256
    from collect_b1k_codex_subtask import _restore, make_monitor
    from collect_b1k_codex import collection_evaluator, time_limit
    from subtask_trajectory import capture_observation, CAMERAS
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    stage = "input_validation"

    def publish(**values):
        values.update(stage=stage, updated_at=time.time(), diagnostic_only=True,
                      training_eligible=False, model_calls=0)
        (root / "status.json").write_text(json.dumps(values, allow_nan=False, indent=2) + "\n")
        print("REPLAY_STATUS=" + json.dumps(values, allow_nan=False), flush=True)

    try:
        plan = load_plan(config["plan_path"])
        if plan["fingerprint"] != config["plan_fingerprint"]:
            raise ValueError("Original frozen plan differs")
        original = Path(config["attempt_root"])
        info = inspect_attempt(original, require_success=False)
        if (info["result"]["steps"] != 960 or info["result"]["actual_teacher_control_steps"] != 0
                or info["result"]["plan_fingerprint"] != plan["fingerprint"]):
            raise ValueError("This diagnostic admits only the existing 960-step attempt")
        with h5py.File(original / "trajectory.hdf5", "r", locking=False) as f:
            actions = f["action"][:]
        if actions.shape != (960, 23) or actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError("Original executed actions differ")
        action_sha = hashlib.sha256(actions.tobytes()).hexdigest()
        publish(state="preparing", original_content_sha256=info["content_sha256"], actions_sha256=action_sha)
        sample = plan["sample"]
        base = runner._load_base_module()
        base_args = base.parse_args(["--task-name", sample["task"], "--mode", "train",
            "--prompt-mode", "whole_task", "--num-rollouts", "1", "--max-steps", "2048",
            "--robot-config", config["robot_config"], "--output-dir", str(root), "--write-video", "--headless"])
        base_args.external_subtask = sample["subtask"]
        base.gm.HEADLESS = True
        base.seed_everything(plan["policy_seed"])
        policy = RecordedActions(actions)

        class ReplayEvaluator(base.Evaluator):
            def load_policy(self):
                return policy

        proxy = SimpleNamespace(**vars(base))
        proxy.Evaluator = ReplayEvaluator
        cache = runner.load_episode_cache(runner.raw_hdf5_path(plan["raw_data_root"], sample),
                                          [sample], allow_unverified_grasp_state=True)
        relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
        if cache["sha256"] != plan["source_plan"]["raw_source_sha256"][relative]:
            raise ValueError("Original raw source differs")
        cfg = runner._build_evaluator_cfg(base, base_args, plan["policy_seed"])
        stage = "simulator_startup"
        publish(state="starting")

        def failure(error):
            publish(state="failed", error_type=type(error).__name__)
            (root / "failure.json").write_text(json.dumps(dict(stage=stage, error_type=type(error).__name__)) + "\n")

        with time_limit(3900), collection_evaluator(
                lambda: runner.create_sample_evaluator(proxy, cfg, cache, sample, 2048), failure) as evaluator:
            stage = "original_start_materialization"
            publish(state="running")
            materialized = runner.materialize_episode_start_states(evaluator, cache, [sample])
            restoration = _restore(evaluator, cache, sample, materialized[sample["sample_id"]], "recorded_native_fields_v1")
            if restoration["recorded_native_source"]["selected_state_sha256"] != plan["source_plan"]["selected_state_sha256"][sample["sample_id"]]:
                raise ValueError("Initial restored native fields differ")
            monitor = make_monitor(evaluator, sample, cache, restoration)
            (root / "restoration.json").write_text(json.dumps(restoration, indent=2) + "\n")
            rows, differences = [], []
            keyframes = {0, 64, 128, 160, 192, 224, 256, 384, 512, 768, 896, 960}
            evaluator.start_recording(str(root / "replay.mp4"), rate=30)
            stage = "recorded_action_replay"
            with h5py.File(original / "trajectory.hdf5", "r", locking=False) as f, (root / "contacts.jsonl").open("x", buffering=1) as stream:
                for step in range(961):
                    row = observe(evaluator, step, "attempt_replay", raycasts=step in keyframes or 128 <= step <= 256 and step % 8 == 0)
                    row["monitor"] = monitor.result()
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if step in keyframes:
                        frame = capture_observation(evaluator.obs)
                        for index, key in enumerate(CAMERAS):
                            Image.fromarray(frame[key]).save(root / f"replay-step-{step:04d}-camera-{index}.png")
                    if step == 960:
                        break
                    frame = capture_observation(evaluator.obs)
                    delta = frame["observation.state"] - f["observation.state"][step]
                    diff = dict(step=step, state_max_abs=float(np.abs(delta).max()), state_rmse=float(np.sqrt(np.mean(delta**2))))
                    if step in keyframes:
                        diff["rgb_mae"] = {k: float(np.abs(frame[k].astype(float)-f[k][step].astype(float)).mean()) for k in CAMERAS}
                    differences.append(diff)
                    evaluator.external_subtask = sample["subtask"]
                    terminated, truncated = evaluator.step()
                    if not np.array_equal(np.asarray(evaluator.robot_action).reshape(-1), actions[step]):
                        raise ValueError("An executed action differs from its original row")
                    monitor.refresh(step + 1)
                    if terminated or truncated:
                        raise RuntimeError("Replay unexpectedly reached an environment terminal")
                    if (step + 1) % 64 == 0:
                        publish(state="running", executed_steps=step + 1, latest=row)
            evaluator.stop_recording()
            (root / "replay_fidelity.json").write_text(json.dumps(differences, indent=2) + "\n")
            stage = "original_demo_end_control"
            publish(state="running", executed_steps=960)
            reference = copy.deepcopy(sample)
            reference.update(start=1829, end=1830, sample_id=sample["sample_id"] + "-diagnostic-demo-end")
            control_cache = runner.load_episode_cache(cache["path"], [reference], allow_unverified_grasp_state=True)
            if control_cache["sha256"] != cache["sha256"]:
                raise ValueError("Original demonstration source changed")
            controls = runner.materialize_episode_start_states(evaluator, control_cache, [reference])
            runner.restore_materialized_sample_state(evaluator, control_cache, controls[reference["sample_id"]])
            with h5py.File(cache["path"], "r", locking=False) as f:
                control_actions = np.asarray(f["data"][cache["details"]["demo_group"]]["action"][1829:1834], dtype=np.float32)
            if control_actions.shape != (5, 23):
                raise ValueError("Original demonstration action semantics differ")
            evaluator.policy = RecordedActions(control_actions)
            controls_rows = []
            with (root / "demo_end_contacts.jsonl").open("x", buffering=1) as stream:
                for step in range(6):
                    row = observe(evaluator, step, "original_demo_end", raycasts=True)
                    controls_rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    frame = capture_observation(evaluator.obs)
                    for index, key in enumerate(CAMERAS):
                        Image.fromarray(frame[key]).save(root / f"demo-end-step-{step:02d}-camera-{index}.png")
                    if step < 5:
                        evaluator.step()
            summary = dict(kind="countertop_contact_replay_v1", state="complete", diagnostic_only=True,
                training_eligible=False, model_calls=0, original_results_modified=False,
                plan_fingerprint=plan["fingerprint"], actions_sha256=action_sha,
                original_trajectory_sha256=file_sha256(original / "trajectory.hdf5"),
                executed_steps=960, original_demo_control_steps=5,
                original_result=info["result"], initial_replay=rows[0], final_replay=rows[-1],
                first_released_step=next((r["step"] for r in rows if not r["grasped_by"]), None),
                reference_summaries={name: {k: [r["step"] for r in rows if r["references"][name][k]]
                    for k in ("touching", "above", "below", "on_top")} for name in rows[0]["references"]},
                contacts_objects=sorted({c["other_object"] for r in rows for c in r["contacts"] if c["other_object"]}),
                fidelity_keyframes=[d for d in differences if "rgb_mae" in d], original_demo_end=controls_rows,
                byte_identical_actions=True, unrecorded_physics_caches_verified=False)
            (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            stage = "complete"
            publish(state="complete", executed_steps=960, summary_file=str(root / "summary.json"))
        return summary
    except BaseException as error:
        publish(state="failed", error_type=type(error).__name__)
        (root / "failure.json").write_text(json.dumps(dict(stage=stage, error_type=type(error).__name__)) + "\n")
        raise


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
