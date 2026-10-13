"""Calibrate a source predicate, then run a paired student-only diagnostic.

Original end actions are diagnostic controls only and never training data.
Neither calibration nor the baseline calls an LLM or fabricates a teacher.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from codex_subtask_collection import NATIVE, HISTORY, file_sha256, validate_restoration_proof


def control_sample(sample):
    control = copy.deepcopy(sample)
    control.update(sample_id=sample["sample_id"] + "-predicate-calibration",
                   start=sample["end"] - 1, end=sample["end"], segment_frames=1)
    return control


def qualification_reason(calibration, baseline):
    if calibration.get("success") is not True:
        return "source_demonstration_predicate_not_calibrated"
    if baseline.get("metric_eligible") is not True:
        return "baseline_setup_excluded"
    if baseline.get("success") is True:
        return "student_already_succeeds"
    if baseline.get("terminated") or baseline.get("truncated"):
        return "baseline_environment_terminal"
    if baseline.get("failure_reason") != "sample_timeout":
        return "baseline_failure_not_established"
    return None


class OriginalActions:
    def __init__(self, actions, label):
        self.actions, self.index = actions, 0
        self.current_subtask, self.current_subtask_source = label, "provided"

    def reset(self):
        self.index = 0

    def forward(self, obs):
        import torch
        if self.index >= len(self.actions):
            raise ValueError("Diagnostic source actions exhausted")
        action = torch.as_tensor(self.actions[self.index].copy())
        self.index += 1
        return action


def calibrate(evaluator, plan, cache, state, output):
    import h5py
    import numpy as np
    from collect_b1k_codex_subtask import _restore
    from recorded_native_state import source_role_bindings
    from recorded_fixture_state import validate_runtime_fixture_roles
    from subtask_completion import RoleStateObserver
    from subtask_trajectory import CAMERAS, capture_observation
    from PIL import Image

    sample = control_sample(plan["sample"])
    protocol = plan["source_plan"].get("source_protocol_by_sample", {}).get(plan["sample"]["sample_id"], NATIVE)
    restored = _restore(evaluator, cache, sample, state, protocol)
    source = restored.get("recorded_native_source") or restored.get("history_recovery_source")
    binding = validate_runtime_fixture_roles(source_role_bindings(cache, source, sample["predicate"]), evaluator)
    observer = RoleStateObserver(evaluator, sample["predicate"], resolved_roles=binding["resolved_roles"])
    with h5py.File(cache["path"], "r", locking=False) as file:
        actions = np.asarray(file["data"][cache["details"]["demo_group"]]["action"][sample["start"]:sample["start"] + 5], dtype=np.float32)
    if actions.shape != (5, 23) or not np.isfinite(actions).all():
        raise ValueError("Original demonstration control is incomplete")
    original = evaluator.policy
    rows = []
    from omnigibson import object_states
    def pair(cls, target, reference, cpu):
        values = cls.VALUES_CPU if cpu else cls.VALUES
        value = values[target.scene.idx, cls.OBJ_IDXS[target.relative_prim_path], cls.OBJ_IDXS[reference.relative_prim_path]]
        return value.tolist()
    try:
        evaluator.policy = OriginalActions(actions, sample["subtask"])
        for step in range(6):
            value, snapshot = observer.observe()
            tensor_rows = []
            for target in (observer.targets if sample["predicate"]["operation"] == "placed" else []):
                reference = observer.references[0]
                tensors = {cls.__name__: dict(cpu=pair(cls, target, reference, True),
                                             gpu=pair(cls, target, reference, False))
                           for cls in (object_states.Touching, object_states.Adjacency, object_states.OnTop)}
                tensor_rows.append(dict(target=target.name, reference=reference.name, values=tensors))
            rows.append(dict(step=step, predicate_satisfied=value, snapshot=snapshot, tensors=tensor_rows))
            if step in (0, 5):
                frame = capture_observation(evaluator.obs)
                for index, camera in enumerate(CAMERAS):
                    Image.fromarray(frame[camera]).save(output / f"source-end-{step}-camera-{index}.png")
            if step < 5:
                evaluator.external_subtask = sample["subtask"]
                # Continue the five real demo actions even if the whole task
                # has ended; this is a diagnostic control, never a rollout score.
                evaluator.step()
    finally:
        evaluator.policy = original
    agreement = all(t["values"][name]["cpu"] == t["values"][name]["gpu"]
                    for row in rows[1:] for t in row["tensors"] for name in t["values"])
    agreement = agreement and all(
        target["relation"] == tensor["values"]["OnTop"]["cpu"]
        for row in rows[1:] for target, tensor in zip(row["snapshot"]["targets"], row["tensors"]))
    result = dict(success=agreement and all(row["predicate_satisfied"] for row in rows[-3:]),
                  api_cpu_gpu_agree_after_physics=agreement if sample["predicate"]["operation"] == "placed" else None,
                  source_role_binding=binding, original_actions_start=sample["start"],
                  original_actions_count=5, restoration=restored, observations=rows,
                  training_eligible=False, model_calls=0)
    (output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def qualify(base, evaluator, plan, cache, states, output):
    import eval_b1k_subtask_samples as runner

    output = Path(output)
    from codex_jacobian_validation import RUNTIME_CONTRACT
    early_validation = plan.get('teacher_motion_validation') == RUNTIME_CONTRACT
    output.mkdir(parents=True, exist_ok=early_validation)
    if early_validation:
        from codex_jacobian_validation import verify
        verify(output/'jacobian_validation.json', plan)
        if any((output/name).exists() for name in ('calibration.json','qualification.json','baseline')):
            raise ValueError('Never overwrite completed or partial qualification')
    sample = plan["sample"]
    calibration = calibrate(evaluator, plan, cache, states[control_sample(sample)["sample_id"]], output)
    baseline = None
    if calibration["success"]:
        if plan.get('teacher_motion_validation') and not early_validation:
            from codex_jacobian_validation import run as validate_jacobian
            validate_jacobian(evaluator, plan, cache, states[sample['sample_id']],
                              output/'jacobian_validation.json')
        protocol = plan["source_plan"].get("source_protocol_by_sample", {}).get(sample["sample_id"], NATIVE)
        # run_sample restores the same snapshot; its additional validator checks
        # the source fields after restoration, without a second scene reset.
        def validator(actual):
            if protocol == HISTORY:
                from legacy_history_recovery import restored_context
                return restored_context(actual, cache, sample, states[sample["sample_id"]])
            from recorded_native_state import decode_start, validate_runtime
            return {"recorded_native_source": validate_runtime(decode_start(cache, sample), actual)}
        base.seed_everything(plan["policy_seed"])
        evaluator.policy.next_seed = plan["policy_seed"]
        baseline = runner.run_sample(base, evaluator, SimpleNamespace(write_video=True,
            video_fps=int(evaluator.env.env_config["rendering_frequency"]), sample_max_steps=plan["budgets"]["max_steps"]),
            sample, cache, states[sample["sample_id"]], output / "baseline", plan["panel_fingerprint"],
            restoration_validator=validator)
        validate_restoration_proof(baseline, plan)
    reason = qualification_reason(calibration, baseline or {})
    result = dict(kind="codex_subtask_qualification_v1", plan_fingerprint=plan["fingerprint"],
        qualified=reason is None, rejection_reason=reason, calibration=calibration, baseline=baseline,
        checkpoint_identity=plan["checkpoint_identity"], policy_seed=plan["policy_seed"],
        baseline_teacher_calls=0, calibration_is_training_data=False, training_eligible=False,
        official_benchmark=False)
    if plan.get('teacher_motion_validation') and calibration['success']:
        from codex_jacobian_validation import verify
        result['jacobian_validation'] = verify(output/'jacobian_validation.json', plan)
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_qualification(path, plan):
    path = Path(path)
    receipt = json.loads(path.read_text())
    if (receipt.get("kind") != "codex_subtask_qualification_v1"
            or receipt.get("plan_fingerprint") != plan["fingerprint"]
            or receipt.get("checkpoint_identity") != plan["checkpoint_identity"]
            or receipt.get("policy_seed") != plan["policy_seed"]
            or receipt.get("baseline_teacher_calls") != 0
            or receipt.get("qualified") is not True
            or qualification_reason(receipt.get("calibration", {}), receipt.get("baseline") or {}) is not None):
        raise ValueError("Paired unassisted failure qualification is missing or invalid")
    if plan.get('teacher_motion_validation'):
        from codex_jacobian_validation import verify
        if receipt.get('jacobian_validation') != verify(path.parent/'jacobian_validation.json', plan):
            raise ValueError('Paired Jacobian validation differs')
    return dict(path=str(path.absolute()), sha256=file_sha256(path))
