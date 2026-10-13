"""Qualification gates; fixtures never establish simulator success."""
import copy
import json

import pytest

from codex_subtask_qualification import control_sample, qualification_reason, verify_qualification


@pytest.mark.parametrize("change,reason", [
    ({"metric_eligible": False}, "baseline_setup_excluded"),
    ({"success": True}, "student_already_succeeds"),
    ({"terminated": True}, "baseline_environment_terminal"),
    ({"truncated": True}, "baseline_environment_terminal"),
    ({"failure_reason": "environment_terminated"}, "baseline_failure_not_established"),
])
def test_baseline_success_setup_and_early_termination_do_not_qualify(change, reason):
    baseline = dict(metric_eligible=True, success=False, terminated=False,
                    truncated=False, failure_reason="sample_timeout")
    assert qualification_reason({"success": True}, {**baseline, **change}) == reason
    assert qualification_reason({"success": True}, baseline) is None
    assert qualification_reason({"success": False}, baseline) == "source_demonstration_predicate_not_calibrated"


def test_calibration_never_changes_original_start():
    sample = {"sample_id": "original", "start": 100, "end": 200, "segment_frames": 100}
    original = copy.deepcopy(sample)
    control = control_sample(sample)
    assert sample == original
    assert control["start"] == 199 and control["sample_id"] != sample["sample_id"]


def test_rejected_or_different_checkpoint_cannot_be_reused(tmp_path):
    plan = dict(fingerprint="frozen", checkpoint_identity={"checkpoint_step": 1688977}, policy_seed=9)
    receipt = dict(kind="codex_subtask_qualification_v1", plan_fingerprint="frozen",
        checkpoint_identity=plan["checkpoint_identity"], policy_seed=9, qualified=True,
        baseline_teacher_calls=0, calibration={"success": True},
        baseline=dict(metric_eligible=True, success=False, terminated=False, truncated=False, failure_reason="sample_timeout"))
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(receipt))
    assert len(verify_qualification(path, plan)["sha256"]) == 64
    receipt["baseline"]["success"] = True
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_qualification(path, plan)
