"""Offline bounded continuation admission, immutable reuse and serial ledger tests."""
import copy
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_codex_next_collection as builder
import run_codex_collection
import test_build_codex_first_collection_v5 as v5


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "LEDGER_PATH", tmp_path / "ledger.json")
    names = ["scripts/run_codex_collection.sh", "scripts/collect_b1k_codex.py", "scripts/codex_collection_teacher.py",
             "scripts/codex_teacher_mailbox.py", "scripts/codex_collection_actions.py", "scripts/codex_collection_trajectory.py",
             "scripts/codex_collection_demonstrations.py", "sco/run_codex_collection.py", "sco/codex_cluster_proxy.py",
             run_codex_collection.ROBOT_CONFIG] + [f"scripts/fixture-{i}.py" for i in range(19)]
    template = dict(code=builder.AFS + "/code-v7", base_code="/fixture/base",
                    runtime=builder.PYTHON.removesuffix("/envs/b1k-eval/bin/python"), checkpoint="/fixture/checkpoint",
                    run_root=builder.AFS + "/collection/radio-311-pilot002", task="turning_on_radio", instance_id=311,
                    attempt_id="pilot-002", execution_name="fixture", teacher_backend="codex_local",
                    codex_binary="/usr/local/bin/codex", codex_home=builder.AFS + "/private-codex-home-v1",
                    mihomo_binary="/usr/local/bin/mihomo", mihomo_config_dir="/fixture/mihomo-config",
                    demonstration_manifest=builder.AFS + "/original-reader-v4/demonstrations/28/manifest.json",
                    robot_config=builder.AFS + "/code-v7/" + run_codex_collection.ROBOT_CONFIG,
                    source_files_sha256={name: "d" * 64 for name in names}, max_steps=4500, max_decisions=32,
                    max_seconds=7200, wall_timeout_seconds=8400, teacher_interval=256)
    common = dict(base_code=template["base_code"], base_source_manifest_sha256="a" * 64,
                  source_files_sha256=template["source_files_sha256"])
    old_report = v5.failed_report()
    old_identity = dict(state=old_report["state"], training_eligible=False, task="turning_on_radio", instance_id=311,
                        attempt_id="pilot-001", executed_steps_observed=0, teacher_usage=old_report["teacher_usage"])
    frozen = dict(collection=template, report_root=builder.AFS + "/first-radio311-gpu-v5",
                  recovery_from=dict(job=dict(name="prior-zero", uid="zero", state="SUCCEEDED"),
                                     source_sha256="e" * 64, report_identity=old_identity))
    source_path = tmp_path / "previous.py"
    source_path.write_text("BOOTSTRAP_REPORT = run(" + repr(frozen) + ")\n")
    source_sha = builder.sha(source_path)
    job = dict(name=builder.PILOT2_NAME, uid=builder.PILOT2_UID, state="FAILED", observed_at=1000)
    audit = dict(kind="codex_pilot002_readonly_audit_v5", state="audit_complete",
                 source_unchanged_during_audit=True, model_calls=0, network_calls=0,
                 model_calls_scope="audit_process_only", collection_source_sha256=source_sha,
                 source_job={key: job[key] for key in ("name", "uid", "state")},
                 run_root=template["run_root"], audit_root=builder.AFS + "/audit-radio311-pilot002-v5",
                 training_eligible=False, issues=["failed_attempt"], actions={}, trajectory={},
                 teachers=dict(requests=3, model_calls_observed=1, requests_with_unknown_model_call=2,
                               requests_failed_before_model=0))
    values = dict(prior_job_path=job, prior_submission_path={**job, "source_sha256": source_sha},
                  prior_audit_path=audit, collection_path=template,
                  manifest_path={**common, "version": "v7", "code": builder.AFS + "/code-v7", "bundle_sha256": builder.CODE_SHA},
                  preparation_manifest_path={**common, "version": "v6", "code": builder.AFS + "/code-v6", "bundle_sha256": builder.OLD_BUNDLE_SHA256},
                  encryption_path=dict(selected_auth_file="auth_weizhi.json", ciphertext_sha256="c" * 64))
    kwargs = dict(prior_source_path=source_path, task="putting_shoes_on_rack", instance_id=311,
                  attempt_id="followup-001", demo_episode=4424, ledger_path=tmp_path / "ledger.json", now=1000)
    for key, value in values.items():
        path = tmp_path / (key + ".json"); dump(path, value); kwargs[key] = path
    return kwargs


def test_builds_only_fresh_finite_job_with_installed_code_and_preserved_unknown(tmp_path, inputs):
    output = tmp_path / "new.py"
    result = builder.build(output, **inputs)
    config = builder.frozen_config(output)
    assert result["actual_job_submitted"] is False and result["code_installation"] is False
    assert result["recommended_cloud_max_seconds"] == 9600 and "/envs/b1k-eval/" in result["python_binary"]
    assert config["collection"]["max_steps"] == 10000 and config["collection"]["max_decisions"] == 32
    assert config["admission"]["all_continue_action_ceiling"] == 8192
    assert config["admission"]["step_limit_may_exceed_teacher_budget"] is True
    assert "installer" not in config and "recovery_from" not in config
    assert "exec(compile(config['installer']" not in output.read_text()
    ledger = builder.load(inputs["ledger_path"])
    assert len(ledger["entries"]) == 3 and sum(r["reserved_decisions"] for r in ledger["entries"]) == 96
    assert ledger["entries"][0]["observed_model_calls"] == 0
    assert ledger["entries"][1]["teacher_accounting"]["requests_with_unknown_model_call"] == 2
    assert ledger["entries"][-1]["state"] == "reserved"
    for path in (output, output.with_suffix(".receipt.json"), inputs["ledger_path"]):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    compile(output.read_text(), str(output), "exec")


@pytest.mark.parametrize("task", ["turning_on_radio", "putting_shoes_on_rack"])
def test_legacy_code7_predecessor_cannot_adopt_code8_shoe_interval(tmp_path, inputs, task):
    source = inputs["prior_source_path"];config = builder.frozen_config(source)
    config["collection"].update(task=task, teacher_interval=320)
    source.write_text("BOOTSTRAP_REPORT = run(" + repr(config) + ")\n")
    submission = builder.load(inputs["prior_submission_path"])
    submission["source_sha256"] = builder.sha(source);dump(inputs["prior_submission_path"], submission)
    with pytest.raises(ValueError, match="bounded collection experiment"):
        builder.build(tmp_path / "blocked.py", **inputs)
    assert not inputs["ledger_path"].exists()


@pytest.mark.parametrize("damage", ["active", "stale", "uid", "source", "audit_failed", "audit_identity", "audit_mutated", "over32", "missing_usage"])
def test_requires_exact_fresh_terminal_and_actual_audit(tmp_path, inputs, damage):
    if damage in ("active", "stale", "uid"):
        path = inputs["prior_job_path"]; value = builder.load(path)
        value.update({"active": {"state": "RUNNING"}, "stale": {"observed_at": 0}, "uid": {"uid": "another"}}[damage])
    elif damage == "source":
        inputs["prior_source_path"].write_text("# altered submitted source")
        path = None
    else:
        path = inputs["prior_audit_path"]; value = builder.load(path)
        if damage == "audit_failed":value["state"] = "audit_failed"
        elif damage == "audit_identity":value["source_job"]["uid"] = "another"
        elif damage == "audit_mutated":value["source_unchanged_during_audit"] = False
        elif damage == "over32":value["teachers"]["requests_with_unknown_model_call"] = 32
        else:value["teachers"].pop("model_calls_observed")
    if path:dump(path, value)
    with pytest.raises(ValueError):builder.build(tmp_path / "new.py", **inputs)
    assert not inputs["ledger_path"].exists()


@pytest.mark.parametrize("changes", [dict(instance_id=310), dict(instance_id=321), dict(instance_id=True),
                                      dict(task="other"), dict(demo_episode=28), dict(attempt_id="../overwrite")])
def test_rejects_wrong_task_instance_demo_or_attempt(tmp_path, inputs, changes):
    with pytest.raises(ValueError):builder.build(tmp_path / "new.py", **{**inputs, **changes})


def test_reserved_latest_blocks_another_build_even_when_old_audit_is_valid(tmp_path, inputs):
    builder.build(tmp_path / "new.py", **inputs)
    before = inputs["ledger_path"].read_bytes()
    with pytest.raises(ValueError, match="latest reserved"):
        builder.build(tmp_path / "another.py", **{**inputs, "attempt_id": "followup-002"})
    assert inputs["ledger_path"].read_bytes() == before and not (tmp_path / "another.py").exists()


def test_a_second_ledger_cannot_reset_experiment_budget(tmp_path, inputs):
    with pytest.raises(ValueError, match="single canonical"):
        builder.build(tmp_path / "new.py", **{**inputs, "ledger_path": tmp_path / "another-ledger.json"})


def test_missing_previously_initialized_ledger_cannot_reseed(tmp_path, inputs):
    builder.build(tmp_path / "new.py", **inputs)
    inputs["ledger_path"].unlink()
    with pytest.raises(ValueError, match="initialized ledger is missing"):
        builder.build(tmp_path / "replay.py", **{**inputs, "attempt_id": "another"})
    assert not (tmp_path / "replay.py").exists()


def advance_previous(inputs, source, receipt, *, unknown=0):
    job = dict(name="owned-next-job", uid="owned-next-uid", state="SUCCEEDED", observed_at=1000)
    config = builder.frozen_config(source)
    dump(inputs["prior_job_path"], job)
    dump(inputs["prior_submission_path"], {**job, "source_sha256": receipt["sha256"]})
    audit = builder.load(inputs["prior_audit_path"])
    audit.update(kind="codex_collection_readonly_audit_v1", source_job=job,
                 collection_source_sha256=receipt["sha256"], run_root=config["collection"]["run_root"],
                 audit_root=builder.AFS + "/audit-next-fixture")
    audit["teachers"].update(requests=unknown, model_calls_observed=0, requests_with_unknown_model_call=unknown)
    dump(inputs["prior_audit_path"], audit)
    return {**inputs, "prior_source_path": source, "attempt_id": "followup-002"}


def test_exact_terminal_audit_advances_chain_without_refunding_unknown(tmp_path, inputs):
    source = tmp_path / "first-next.py"; receipt = builder.build(source, **inputs)
    next_inputs = advance_previous(inputs, source, receipt, unknown=32)
    result = builder.build(tmp_path / "second-next.py", **next_inputs)
    assert result["budget_admission"]["total_reserved_decisions"] == 128
    ledger = builder.load(inputs["ledger_path"])
    assert ledger["entries"][-2]["teacher_accounting"]["requests_with_unknown_model_call"] == 32
    assert ledger["entries"][-2]["reserved_decisions"] == 32


def test_valid_old_ledger_cannot_roll_back_committed_reservation(tmp_path, inputs):
    source = tmp_path / "first-next.py"; receipt = builder.build(source, **inputs)
    previous_bytes = inputs["ledger_path"].read_bytes()
    next_inputs = advance_previous(inputs, source, receipt)
    builder.build(tmp_path / "second-next.py", **next_inputs)
    inputs["ledger_path"].write_bytes(previous_bytes)
    with pytest.raises(ValueError, match="journal disagrees"):
        builder.build(tmp_path / "replay.py", **{**next_inputs, "attempt_id": "another"})
    assert not (tmp_path / "replay.py").exists()


def test_failure_writing_source_retains_budget_and_blocks_automatic_replay(tmp_path, inputs, monkeypatch):
    original = builder.write_private
    output = tmp_path / "new.py"
    def write(path, value):
        if path == output:raise OSError("fixture local write failed")
        original(path, value)
    monkeypatch.setattr(builder, "write_private", write)
    with pytest.raises(OSError):builder.build(output, **inputs)
    assert builder.load(inputs["ledger_path"])["entries"][-1]["state"] == "reserved"
    with pytest.raises(ValueError, match="latest reserved"):
        builder.build(tmp_path / "retry.py", **inputs)


def test_cannot_initialize_new_ledger_from_arbitrary_later_job(tmp_path, inputs):
    for key in ("prior_job_path", "prior_submission_path"):
        value = builder.load(inputs[key]); value["uid"] = "different"; dump(inputs[key], value)
    audit = builder.load(inputs["prior_audit_path"]);audit["source_job"]["uid"] = "different";dump(inputs["prior_audit_path"], audit)
    with pytest.raises(ValueError, match="exact audited pilot-002"):
        builder.build(tmp_path / "new.py", **inputs)


def test_full_task_and_global_reservations_remain_bounded(inputs):
    previous = builder.checked_previous(inputs["prior_job_path"], inputs["prior_submission_path"],
                                        inputs["prior_source_path"], inputs["prior_audit_path"], now=1000)
    ledger = builder.seed_ledger(previous)
    builder.settle(ledger, previous)
    for task, count in (("turning_on_radio", 8), ("putting_shoes_on_rack", 10)):
        for i in range(count):
            ledger["entries"].append(dict(task=task, instance_id=311+i, attempt_id=f"fixture-{task}-{i}",
                                          run_root=builder.AFS + f"/collection/{task}-{i}",
                                          reserved_decisions=32, state="audited", source_sha256="f" * 64))
    builder.validate_ledger(ledger)
    assert sum(r["reserved_decisions"] for r in ledger["entries"]) == 640
    more = copy.deepcopy(ledger["entries"][-1]); more.update(attempt_id="extra", run_root=builder.AFS + "/collection/extra")
    ledger["entries"].append(more)
    with pytest.raises(ValueError):builder.validate_ledger(ledger)


@pytest.mark.parametrize("damage", ["release_unknown", "tamper", "duplicate", "early_unresolved"])
def test_ledger_cannot_release_reservations_or_skip_unresolved(inputs, damage):
    previous = builder.checked_previous(inputs["prior_job_path"], inputs["prior_submission_path"],
                                        inputs["prior_source_path"], inputs["prior_audit_path"], now=1000)
    ledger = builder.seed_ledger(previous)
    if damage == "release_unknown":ledger["entries"][-1]["reserved_decisions"] = 1
    elif damage == "tamper":ledger["max_teacher_decisions_total"] = 641
    elif damage == "duplicate":ledger["entries"].append(copy.deepcopy(ledger["entries"][-1]))
    else:ledger["entries"][0]["state"] = "reserved"
    with pytest.raises(ValueError):builder.validate_ledger(ledger)


@pytest.fixture
def worker():
    namespace = {}
    exec(compile(builder.WORKER, "next_collection_worker", "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_runtime_reuses_code_and_selects_matching_shoe_demonstration(tmp_path, worker):
    config = v5.gate_inputs.__wrapped__(tmp_path, worker)
    config.update(demo_episode=4424)
    config["collection"]["task"] = "putting_shoes_on_rack"
    base = Path(config["pilot_root"])
    demo_path = base / "original-reader-v4/demonstrations/4424/manifest.json"
    demo = dict(metadata=dict(task="putting_shoes_on_rack", episode_index=4424), fingerprint="shoe-demo")
    dump(demo_path, demo)
    data_path = base / "original-reader-v4/_ORIGINAL_DATA_READY.json";data = builder.load(data_path)
    row = next(r for r in data["demonstrations"] if r["episode_index"] == 4424)
    row.update(manifest=str(demo_path), fingerprint=demo["fingerprint"]); dump(data_path, data)
    prep_path = base / "preparation-v14/report.json"; prep = builder.load(prep_path)
    prep["data_ready_sha256"] = worker.sha(data_path);dump(prep_path, prep)
    config["collection"]["demonstration_manifest"] = str(demo_path)
    assert worker.gates(config)["collection_code"].endswith("/code-v7")
    demo["metadata"]["task"] = "turning_on_radio"; dump(demo_path, demo)
    with pytest.raises(ValueError, match="reference identity"):
        worker.gates(config)


def test_runtime_rejects_changed_audit_and_deadline_is_not_swallowed(tmp_path, worker):
    job = dict(name="fixture", uid="abc", state="FAILED")
    path = tmp_path / "report.json"
    audit = dict(state="audit_complete", source_unchanged_during_audit=True,
                 collection_source_sha256="a"*64, run_root="/fixture/old", source_job=job)
    dump(path, audit)
    config = dict(predecessor=dict(job=job, source_sha256="a"*64, run_root="/fixture/old",
                                  audit=dict(path=str(path), content_sha256=builder.fingerprint(audit))),
                  terminal_states=sorted(builder.TERMINAL))
    # AOSS bootstrap compact serialization and AFS pretty serialization must
    # agree on content while retaining separate byte hashes for provenance.
    path.write_text(json.dumps(audit, indent=2) + "\n")
    worker.verify_predecessor(config)
    audit["source_job"]["uid"] = "changed";dump(path, audit)
    with pytest.raises(ValueError):worker.verify_predecessor(config)
    assert not issubclass(worker.WrapperDeadline, Exception)


def test_source_collision_never_changes_prior_input(tmp_path, inputs):
    path = inputs["prior_source_path"]; before = path.read_bytes()
    with pytest.raises(ValueError):builder.build(path, **inputs)
    assert path.read_bytes() == before and not inputs["ledger_path"].exists()


def test_ledger_symlink_cannot_overwrite_another_file(tmp_path, inputs):
    target = tmp_path / "unrelated.json"; target.write_text("preserve")
    inputs["ledger_path"].symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        builder.build(tmp_path / "new.py", **inputs)
    assert target.read_text() == "preserve"
