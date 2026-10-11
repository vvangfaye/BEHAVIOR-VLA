"""Already-installed code-v8 continuation with one cross-version budget ledger."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_codex_next_collection as shared
import build_codex_next_collection_v8 as builder
import build_codex_radio_recovery_v8 as recovery
import test_build_codex_radio_recovery_v8 as recovery_tests


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def audit_next(tmp_path, source, receipt, *, number=1, unknown=2):
    job = dict(name=f"owned-code8-{number}", uid=f"code8-uid-{number}", state="SUCCEEDED", observed_at=1000)
    config = shared.frozen_config(source)
    job_path, submission_path, audit_path = [tmp_path / f"{name}-{number}.json" for name in ("terminal", "submission", "audit")]
    dump(job_path, job);dump(submission_path, {**job, "source_sha256": receipt["sha256"]})
    audit = dict(kind="codex_collection_readonly_audit_v1", state="audit_complete", source_unchanged_during_audit=True,
                 model_calls=0, network_calls=0, model_calls_scope="audit_process_only", source_job=job,
                 collection_source_sha256=receipt["sha256"], run_root=config["collection"]["run_root"],
                 audit_root=shared.AFS + f"/audit-code8-fixture-{number}", training_eligible=False,
                 issues=["fixture_failure"], actions={}, trajectory={},
                 teachers=dict(requests=5+unknown, model_calls_observed=5, requests_with_unknown_model_call=unknown,
                               requests_failed_before_model=0))
    dump(audit_path, audit)
    return dict(prior_job_path=job_path, prior_submission_path=submission_path, prior_source_path=source,
                prior_audit_path=audit_path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    first = recovery_tests.inputs.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(builder, "CODE8_SHA", first["expected_bundle_sha256"])
    source = tmp_path / "pilot003.py";receipt = recovery.build(source, **first)
    return dict(**audit_next(tmp_path, source, receipt), task="putting_shoes_on_rack", instance_id=312,
                attempt_id="next-001", demo_episode=4424, ledger_path=first["ledger_path"], now=1000)


@pytest.mark.parametrize("task,instance,episode,max_steps", [("turning_on_radio", 311, 28, 4500),
    ("turning_on_radio", 320, 102, 4500), ("putting_shoes_on_rack", 311, 4424, 10000),
    ("putting_shoes_on_rack", 320, 4517, 10000)])
def test_reuses_installed_code8_across_tasks_and_preserves_old_budget(tmp_path, inputs, task, instance, episode, max_steps):
    source = tmp_path / "next.py"
    result = builder.build(source, **{**inputs, "task": task, "instance_id": instance, "demo_episode": episode})
    config = shared.frozen_config(source)
    assert builder.validate_frozen_config(config)["task"] == task
    assert config["collection"]["max_steps"] == max_steps
    interval = 320 if task == "putting_shoes_on_rack" else 256
    assert config["collection"]["teacher_interval"] == interval
    assert config["admission"]["all_continue_action_ceiling"] == 32 * interval
    assert config["admission"]["step_limit_may_exceed_teacher_budget"] is False
    assert config["admission"]["correction_can_reduce_action_ceiling"] is True
    assert config["collection"]["code"].endswith("/code-v8")
    assert "installer" not in config and "recovery_from" not in config
    assert "private-fixture" not in source.read_text()
    assert result["code_installation"] is False and result["actual_job_submitted"] is False
    assert result["recommended_cloud_max_seconds"] == 9600
    assert result["budget_admission"]["total_reserved_decisions"] == 128
    ledger = shared.load(inputs["ledger_path"])
    assert len(ledger["entries"]) == 4 and sum(row["reserved_decisions"] for row in ledger["entries"]) == 128
    assert ledger["entries"][1]["teacher_accounting"]["model_calls_observed"] == 17
    assert ledger["entries"][2]["teacher_accounting"]["requests_with_unknown_model_call"] == 2
    assert ledger["entries"][-1]["collection_code"].endswith("/code-v8")
    compile(source.read_text(), str(source), "exec")


def test_audited_generic_code8_successor_can_continue_without_pilot003_gate(tmp_path, inputs):
    source = tmp_path / "next1.py";result = builder.build(source, **inputs)
    next_inputs = {**inputs, **audit_next(tmp_path, source, result, number=2, unknown=1),
                   "task": "turning_on_radio", "instance_id": 313, "attempt_id": "next-002", "demo_episode": 68}
    receipt = builder.build(tmp_path / "next2.py", **next_inputs)
    assert receipt["budget_admission"]["total_reserved_decisions"] == 160
    assert receipt["budget_admission"]["sequence"] == 5
    ledger = shared.load(inputs["ledger_path"])
    assert ledger["entries"][-2]["teacher_accounting"]["requests_with_unknown_model_call"] == 1
    assert ledger["entries"][-2]["reserved_decisions"] == 32


@pytest.mark.parametrize("damage", ["active", "stale", "uid", "audit_failed", "source", "over32"])
def test_requires_actual_exact_terminal_audit_before_any_reservation(tmp_path, inputs, damage):
    before = inputs["ledger_path"].read_bytes()
    if damage in {"active", "stale", "uid"}:
        path = inputs["prior_job_path"];value = shared.load(path)
        value.update({"active": {"state": "RUNNING"}, "stale": {"observed_at": 0}, "uid": {"uid": "other"}}[damage])
    elif damage == "source":
        inputs["prior_source_path"].write_text("changed")
        path = None
    else:
        path = inputs["prior_audit_path"];value = shared.load(path)
        if damage == "audit_failed":value["state"] = "audit_failed"
        else:value["teachers"]["requests_with_unknown_model_call"] = 32
    if path:dump(path, value)
    with pytest.raises(ValueError):builder.build(tmp_path / "blocked.py", **inputs)
    assert inputs["ledger_path"].read_bytes() == before


@pytest.mark.parametrize("change", [dict(instance_id=301), dict(instance_id=321), dict(task="other"),
                                     dict(demo_episode=28), dict(attempt_id="../old")])
def test_rejects_unsupported_selection_before_budget_write(tmp_path, inputs, change):
    before = inputs["ledger_path"].read_bytes()
    with pytest.raises(ValueError):builder.build(tmp_path / "blocked.py", **{**inputs, **change})
    assert inputs["ledger_path"].read_bytes() == before


@pytest.mark.parametrize("damage", ["installer", "task", "demo", "bundle", "teacher", "code", "account", "budget", "run_root", "wrapper_root", "interval", "ceiling", "step_warning"])
def test_source_bound_validator_rejects_configuration_tampering(tmp_path, inputs, damage):
    source = tmp_path / "next.py";builder.build(source, **inputs);config = shared.frozen_config(source)
    if damage == "installer":config["installer"] = "not allowed"
    elif damage == "task":config["collection"]["task"] = "turning_on_radio"
    elif damage == "demo":config["demo_episode"] = 28
    elif damage == "bundle":config["manifest"]["bundle_sha256"] = "f" * 64
    elif damage == "teacher":config["manifest"]["source_files_sha256"]["scripts/codex_collection_teacher.py"] = "f" * 64
    elif damage == "code":config["collection"]["code"] = shared.AFS + "/code-v7"
    elif damage == "account":config["collection"]["codex_home"] = "/other-account"
    elif damage == "budget":config["admission"]["total_reserved_decisions"] = 32
    elif damage == "run_root":config["collection"]["run_root"] = recovery.RUN_ROOT
    elif damage == "interval":config["collection"]["teacher_interval"] = 256
    elif damage == "ceiling":config["admission"]["all_continue_action_ceiling"] = 8192
    elif damage == "step_warning":config["admission"]["step_limit_may_exceed_teacher_budget"] = True
    else:config["report_root"] = recovery.REPORT_ROOT
    with pytest.raises(ValueError):builder.validate_frozen_config(config)


def test_cannot_reset_ledger_after_code8_audit(tmp_path, inputs):
    inputs["ledger_path"].unlink()
    with pytest.raises(ValueError, match="initialized ledger is missing"):
        builder.build(tmp_path / "blocked.py", **inputs)


def test_one_pending_successor_blocks_another_even_with_same_valid_predecessor(tmp_path, inputs):
    builder.build(tmp_path / "first.py", **inputs)
    before = inputs["ledger_path"].read_bytes()
    with pytest.raises(ValueError, match="latest reserved"):
        builder.build(tmp_path / "second.py", **{**inputs, "attempt_id": "next-002"})
    assert inputs["ledger_path"].read_bytes() == before


def test_generic_code8_worker_uses_hash_gates_without_any_installer():
    assert "'code-v8'" in builder.WORKER
    assert "code_v8_installer" not in builder.WORKER and "config['installer']" not in builder.WORKER
    assert "'codex_bounded_collection_gpu_v8'" in builder.WORKER
    assert "Codex-home lease" in builder.WORKER


def test_radio_interval_is_not_relaxed_to_shoe_interval(tmp_path, inputs):
    source = tmp_path / "radio.py"
    builder.build(source, **{**inputs, "task": "turning_on_radio", "demo_episode": 28})
    config = shared.frozen_config(source)
    config["collection"]["teacher_interval"] = 320
    with pytest.raises(ValueError, match="finite limits"):
        builder.validate_frozen_config(config)


def test_recovery_predecessor_still_requires_256_before_budget_write(tmp_path, inputs):
    source = inputs["prior_source_path"];config = shared.frozen_config(source)
    config["collection"]["teacher_interval"] = 320
    source.write_text(recovery.WORKER + "\nBOOTSTRAP_REPORT = run(" + repr(config) + ")\n")
    submission = shared.load(inputs["prior_submission_path"])
    submission["source_sha256"] = shared.sha(source);dump(inputs["prior_submission_path"], submission)
    before = inputs["ledger_path"].read_bytes()
    with pytest.raises(ValueError, match="bounded collection experiment"):
        builder.build(tmp_path / "blocked.py", **inputs)
    assert inputs["ledger_path"].read_bytes() == before
