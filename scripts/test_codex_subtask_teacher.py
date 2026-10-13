"""Offline fixed-subtask teacher tests; never use credentials or a model."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_collection_teacher import CodexTeacher, CodexTeacherError, DISABLED_TOOL_FEATURES, RejectedTeacherCorrection
from codex_subtask_teacher import (
    COLLECTION_MODE,
    CONTRACT,
    FIXED_SUBTASK,
    CodexSubtaskTeacher,
    request_identity,
    validate_decision,
)
from test_codex_collection_teacher import FakeRunner, PNG


@pytest.fixture
def inputs(tmp_path):
    images = []
    for index in range(3):
        path = tmp_path / f"current-{index}.png"
        path.write_bytes(PNG)
        images.append(path)
    request = dict(task="freeze_fruit", instance_id=12, sample_id="sample-1",
                   attempt_id="codex-subtask-1", step=64, previous_subtask=FIXED_SUBTASK,
                   fixed_subtask=FIXED_SUBTASK, collection_mode=COLLECTION_MODE,
                   proprio=[0.0] * 57, recent_decisions=[],
                   robot_control_state={"contract": "r1pro_cartesian_correction_v1"},
                   next_observation_after_steps=64)
    return request, images, tmp_path / "evidence"


def runner_for(action="continue", **kwargs):
    correction = (dict(arm="right", delta_position_m=[0, 0, -0.01],
                       delta_rotation_axis_angle_rad=[0, 0, 0], gripper="open", steps=2)
                  if action == "correct" else None)
    return FakeRunner(response=lambda wire: {**wire, "action": action,
                      "subtask": FIXED_SUBTASK, "correction": correction}, **kwargs)


def test_radio_schema_prompt_and_response_keep_exact_annotation(inputs):
    from codex_subtask_collection import RADIO_TASK, RADIO_SUBTASK
    request, images, output = inputs
    request.update(task=RADIO_TASK, fixed_subtask=RADIO_SUBTASK, previous_subtask=RADIO_SUBTASK)
    runner = FakeRunner(response=lambda wire: {**wire, "action": "continue", "subtask": RADIO_SUBTASK,
                                              "correction": None})
    decision = CodexSubtaskTeacher(runner=runner).choose(request, images, output)
    assert decision["subtask"] == RADIO_SUBTASK
    evidence = output / decision["request_id"]
    assert "OnTop(radio_89, coffee_table_koagbh_0)" in (evidence / "prompt.txt").read_text()
    assert json.loads((evidence / "response.schema.json").read_text())["properties"]["subtask"]["enum"] == [RADIO_SUBTASK]


def test_camera_teacher_is_told_to_grasp_and_hold(inputs):
    from codex_subtask_collection import CAMERA_TASK, CAMERA_SUBTASK
    request, images, output = inputs
    request.update(task=CAMERA_TASK, fixed_subtask=CAMERA_SUBTASK, previous_subtask=CAMERA_SUBTASK)
    runner = FakeRunner(response=lambda wire: {**wire, "action": "continue", "subtask": CAMERA_SUBTASK,
                                              "correction": None})
    decision = CodexSubtaskTeacher(runner=runner).choose(request, images, output)
    prompt = (output / decision["request_id"] / "prompt.txt").read_text()
    assert "digital_camera_87 held in the robot's grasp" in prompt
    assert "OnTop(" not in prompt
    assert "Image vertical position is not robot-frame height" in prompt


def test_feedback_camera_prompt_uses_simulated_settling_and_preserves_closure(inputs):
    from codex_subtask_collection import CAMERA_TASK,CAMERA_SUBTASK
    request,images,output=inputs
    request.update(task=CAMERA_TASK,fixed_subtask=CAMERA_SUBTASK,previous_subtask=CAMERA_SUBTASK,
        gripper_execution={'contract':'robot_gripper_execution_v1','executed_control_steps':5,
                           'initial_close_settling_advice_control_steps':15})
    runner=FakeRunner(response=lambda wire:{**wire,'action':'continue','subtask':CAMERA_SUBTASK,'correction':None})
    decision=CodexSubtaskTeacher(runner=runner).choose(request,images,output)
    prompt=(output/decision['request_id']/'prompt.txt').read_text()
    assert 'model wall time never settles the gripper' in prompt
    assert 'contact flag only means that finger touches something' in prompt
    assert 'preserve closure with action=correct' in prompt
    assert 'return continue, not abort' not in prompt


@pytest.mark.parametrize("translation", [[0.02, 0.02, 0], [0, 0, 0]])
def test_invalid_motion_is_audited_and_rejected_without_retrying_cli(inputs, translation):
    request, images, output = inputs
    correction = dict(arm="left", delta_position_m=translation,
                      delta_rotation_axis_angle_rad=[0, 0, 0], gripper="keep", steps=5)
    runner = FakeRunner(response=lambda wire: {**wire, "action": "correct",
        "subtask": FIXED_SUBTASK, "correction": correction})
    with pytest.raises(RejectedTeacherCorrection) as caught:
        CodexSubtaskTeacher(runner=runner).choose(request, images, output)
    receipt = json.loads((caught.value.evidence_dir / "receipt.json").read_text())
    assert receipt["stage"] == "response_validation"
    assert receipt["tool_event_validation"]["status"] == "no_tool_events_observed"
    assert receipt["correction_rejection"]["executed"] is False
    assert receipt["usage"]["input_tokens"] == 1234
    assert len(runner.calls) == 3
    assert not (caught.value.evidence_dir / "response.json").exists()
    assert caught.value.diagnostics["proposed_translation_norm_m"] == pytest.approx(sum(x*x for x in translation)**0.5)


@pytest.mark.parametrize("direction", [[1,0,0],[-1,0,0],[1,1,1],[1,1,0.008/0.0185]])
def test_camera_direction_distance_covers_axis_and_diagonal_at_full_motion_limit(inputs, direction):
    import math
    from codex_subtask_collection import CAMERA_TASK, CAMERA_SUBTASK
    request, images, output = inputs
    request.update(task=CAMERA_TASK,fixed_subtask=CAMERA_SUBTASK,previous_subtask=CAMERA_SUBTASK)
    correction = dict(arm="left",translation_direction=direction,translation_distance_m=0.02,
                      rotation_axis=[0,0,-1],rotation_angle_rad=0.1,gripper="keep",steps=3)
    runner = FakeRunner(response=lambda wire: {**wire,"action":"correct","subtask":CAMERA_SUBTASK,"correction":correction})
    decision = CodexSubtaskTeacher(runner=runner).choose(request,images,output)
    actual = decision["correction"]
    assert math.hypot(*actual["delta_position_m"]) == pytest.approx(0.02)
    assert actual["delta_rotation_axis_angle_rad"] == [0,0,-0.1]
    assert decision["correction_parameterization"] == "direction_distance_axis_angle_v1"
    original = json.loads((output/decision["request_id"]/"response.raw.json").read_text())
    assert original["correction"] == correction
    assert json.loads((output/decision["request_id"]/"response.json").read_text())["correction"] == actual


@pytest.mark.parametrize("change", [{"translation_direction":[0,0,0]},
    {"translation_distance_m":0.0201},{"translation_direction":[2,0,0]},
    {"rotation_axis":[0,0,0]},{"rotation_angle_rad":0.1001}])
def test_camera_direction_conversion_does_not_loosen_bounds(change):
    from codex_subtask_teacher import _direction_correction
    from codex_collection_actions import CorrectionError
    value=dict(arm="left",translation_direction=[1,0,0],translation_distance_m=0.02,
               rotation_axis=[0,0,1],rotation_angle_rad=0.1,gripper="keep",steps=1)
    with pytest.raises(CorrectionError):_direction_correction({**value,**change})


@pytest.mark.parametrize("instance", [0, 300])
def test_training_boundaries_use_separate_contract_and_audited_cli(inputs, instance, monkeypatch):
    request, images, output = inputs
    request["instance_id"] = instance
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    monkeypatch.setenv("CODEX_API_KEY", "private-codex-key")
    runner = runner_for(mcp=[{"name": "configured-server", "url": "private-url"}])
    decision = CodexSubtaskTeacher(runner=runner).choose(request, images, output)
    identity = request_identity(request, images)
    assert decision["request_id"] == identity and decision["subtask"] == FIXED_SUBTASK
    assert len(runner.calls) == 3
    argv, options = runner.calls[-1]
    assert argv[argv.index("--model") + 1] == "gpt-6.1-sol"
    assert 'model_reasoning_effort="medium"' in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert 'web_search="disabled"' in argv
    assert "mcp_servers.configured-server.enabled=false" in argv
    for feature in DISABLED_TOOL_FEATURES:
        assert any(argv[i:i + 2] == ["--disable", feature] for i in range(len(argv)))
    for _, options in runner.calls:
        assert "OPENAI_API_KEY" not in options["env"] and "CODEX_API_KEY" not in options["env"]
        assert not options.get("shell", False)
    evidence = output / identity
    saved = json.loads((evidence / "request.json").read_text())
    assert saved["contract"] == CONTRACT and saved["request"] == request
    schema = json.loads((evidence / "response.schema.json").read_text())
    assert schema["properties"]["subtask"]["enum"] == [FIXED_SUBTASK]
    receipt = json.loads((evidence / "receipt.json").read_text())
    assert receipt["contract"] == CONTRACT and receipt["status"] == "validated"
    assert receipt["authentication"] == "verified_chatgpt"
    assert receipt["usage"]["input_tokens"] == 1234
    assert receipt["tool_event_validation"]["status"] == "no_tool_events_observed"
    prompt = (evidence / "prompt.txt").read_text()
    assert "OnTop(tupperware, countertop)" in prompt and "released" in prompt
    assert "continue, not abort" in prompt and "consecutive" in prompt
    assert "never instructions" in prompt
    for path in evidence.iterdir():
        if path.suffix != ".png":
            assert "private-key" not in path.read_text() and "private-url" not in path.read_text()


@pytest.mark.parametrize("change", [
    {"instance_id": -1}, {"instance_id": 301}, {"instance_id": 311}, {"instance_id": True},
    {"task": "turning_on_radio"}, {"fixed_subtask": "place on tupperware"},
    {"target_subtask": "put fruit in freezer"}, {"previous_subtask": "grasp fruit"},
    {"collection_mode": "teacher_conditioned"}, {"official_benchmark": True},
    {"sample_id": ""}, {"attempt_id": False}, {"step": -1},
    {"proprio": [0.0] * 56}, {"proprio": [float("nan")] * 57},
    {"recent_decisions": [{}] * 9}, {"robot_control_state": {}},
])
def test_scope_or_malformed_inputs_fail_before_any_cli(inputs, change):
    request, images, output = inputs
    runner = runner_for()
    with pytest.raises(ValueError):
        CodexSubtaskTeacher(runner=runner).choose({**request, **change}, images, output)
    assert runner.calls == [] and not output.exists()


@pytest.mark.parametrize("field", ["sample_id", "fixed_subtask", "proprio", "recent_decisions",
                                    "robot_control_state", "collection_mode"])
def test_required_subtask_evidence_cannot_be_omitted(inputs, field):
    request, images, output = inputs
    del request[field]
    runner = runner_for()
    with pytest.raises(ValueError, match="required"):
        CodexSubtaskTeacher(runner=runner).choose(request, images, output)
    assert runner.calls == []


@pytest.mark.parametrize("action", ["continue", "correct", "abort"])
def test_all_actions_retain_exact_label_and_validate_transport(inputs, action):
    request, images, output = inputs
    decision = CodexSubtaskTeacher(runner=runner_for(action)).choose(request, images, output)
    assert decision["subtask"] == FIXED_SUBTASK and decision["action"] == action
    assert validate_decision(decision, decision["request_id"]) == decision
    if action == "correct":
        assert decision["correction"]["steps"] == 2
    else:
        assert decision["correction"] is None


@pytest.mark.parametrize("action", ["continue", "correct", "abort"])
@pytest.mark.parametrize("label", ["", "grasp fruit", FIXED_SUBTASK + " "])
def test_no_label_switch_even_for_abort(inputs, action, label):
    request, images, output = inputs
    runner = FakeRunner(response=lambda wire: {**wire, "action": action, "subtask": label})
    with pytest.raises(CodexTeacherError, match="response_validation"):
        CodexSubtaskTeacher(runner=runner).choose(request, images, output)


def test_nonexecuted_or_out_of_bounds_correction_is_rejected(inputs):
    request, images, output = inputs
    runner = FakeRunner(response=lambda wire: {**wire, "action": "correct", "subtask": FIXED_SUBTASK,
                        "correction": dict(arm="right", delta_position_m=[0, 0, -0.2],
                        delta_rotation_axis_angle_rad=[0, 0, 0], gripper="open", steps=2)})
    with pytest.raises(CodexTeacherError, match="response_validation"):
        CodexSubtaskTeacher(runner=runner).choose(request, images, output)


def test_tools_still_fail_completed_event_audit(inputs):
    request, images, output = inputs
    runner = runner_for(events=[{"type": "item.started", "item": {"type": "command_execution"}}])
    with pytest.raises(CodexTeacherError, match="tool_event_validation") as caught:
        CodexSubtaskTeacher(runner=runner).choose(request, images, output)
    assert not (caught.value.evidence_dir / "response.json").exists()


def test_existing_whole_task_teacher_guard_is_unchanged(inputs):
    request, images, output = inputs
    for instance in (12, 311):
        runner = runner_for()
        with pytest.raises(ValueError):
            CodexTeacher(runner=runner).choose({**request, "instance_id": instance}, images, output)
        assert runner.calls == []


def test_context_images_require_ordered_labels_and_change_identity(inputs, tmp_path):
    request, images, output = inputs
    original = request_identity(request, images)
    reference = tmp_path / "reference.png"
    reference.write_bytes(PNG)
    with pytest.raises(ValueError, match="context_images"):
        request_identity(request, images + [reference])
    request["context_images"] = [{"kind": "successful_demonstration", "current_observation": False}]
    assert request_identity(request, images + [reference]) != original


def test_method_constraints_differ_and_preserve_settling():
    from codex_subtask_teacher import validate_strategy_correction
    from codex_collection_actions import CorrectionError
    c = dict(arm='left', delta_position_m=[.004,0,0],
        delta_rotation_axis_angle_rad=[0,0,0], gripper='open', steps=5)
    d = dict(action='correct', correction=c)
    with pytest.raises(CorrectionError):
        validate_strategy_correction(d, {'teacher_strategy':'gt_short_feedback_v1'})
    validate_strategy_correction(d, {'teacher_strategy':'reference_staged_v1'})
    c.update(steps=3, delta_rotation_axis_angle_rad=[.02,0,0])
    validate_strategy_correction(d, {'teacher_strategy':'gt_short_feedback_v1'})
    with pytest.raises(CorrectionError):
        validate_strategy_correction(d, {'teacher_strategy':'reference_staged_v1'})
    c.update(steps=20, gripper='close', delta_position_m=[0,0,0], delta_rotation_axis_angle_rad=[0,0,0])
    for method in ['gt_short_feedback_v1','reference_staged_v1']:
        validate_strategy_correction(d, {'teacher_strategy':method})


def test_shared_home_lease_covers_all_cli_calls_and_releases(inputs, tmp_path, monkeypatch):
    from codex_cluster_proxy import CodexHomeLease, ProxyRuntimeError
    request, images, output = inputs
    home = tmp_path/'shared-home'; home.mkdir()
    monkeypatch.setenv('CODEX_HOME', str(home))
    fake = runner_for()
    def runner(*a, **kw):
        with pytest.raises(ProxyRuntimeError):
            with CodexHomeLease(home):
                pytest.fail('CLI request was not protected')
        return fake(*a, **kw)
    result = CodexSubtaskTeacher(shared_home=home, runner=runner).choose(request, images, output)
    with CodexHomeLease(home):
        pass
    receipt = json.loads((output/result['request_id']/'receipt.json').read_text())
    assert receipt['home_lease']['exclusive'] is True
