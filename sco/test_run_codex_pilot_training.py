import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import run_codex_pilot_training as worker


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return worker.file_ref(path)


@pytest.fixture
def prepared(tmp_path):
    code, runtime = tmp_path / "code", tmp_path / "runtime"
    code.mkdir(); runtime.mkdir()
    for name in ("b1k-eval", "b1k-pi05"):
        binary = runtime / "envs" / name / "bin/python"
        binary.parent.mkdir(parents=True)
        binary.symlink_to(sys.executable)
        (binary.parent / "activate").touch()
    (runtime / "_RUNTIME_READY").touch()
    (runtime / "openpi-cache").mkdir()
    plan = dict(schema_version=1, kind=worker.PLAN_KIND, training_run_id="pilot", conditioning="task_and_subtask",
        dataset_root=str(tmp_path / "mixed"), checkpoint=str(tmp_path / "base-checkpoint"),
        run_root=str(tmp_path / "training"), repo_id="b1k_codex_pilot",
        base_repo_id="b1k_full100_joint_subtask_v1_drop_episode_s1",
        devices=4, batch_size=4, steps=256, smoke_steps=2, training_action_horizon=32, seed=42,
        learning_rate=dict(peak=1e-5, floor=1e-6, warmup=64, decay=256),
        execution_horizon=16, optimizer_resume=False, video_backend="pyav",
        evaluation_agent_enabled=False, collection_checkpoint_id="pi05_b1k_full100_joint_1688977",
        collection_instances=list(range(311, 321)), evaluation_instances=list(range(301, 311)),
        tasks=["turning_on_radio"], init_checkpoint_identity={"checkpoint_step": 1688977},
        base_norm_sha256="a"*64, dataset_files={"meta/info.json": "b"*64}, dataset_fingerprint="c"*64,
        export_verification=dict(complete=True, real_lerobot_reader=True, real_lerobot_writer=True,
            label_coverage=True, action_horizon=32, reader_tolerance_s=0.005, export_fingerprint="c"*64))
    plan["fingerprint"] = worker.digest(plan)
    payload = dict(kind=worker.KIND, code=str(code), runtime=str(runtime),
        report_root=str(tmp_path / "worker-report"), destination=str(tmp_path / "inference"),
        training_plan=save(Path(plan["run_root"]) / "training_plan.json", plan),
        code_ready=save(code / "_CODEX_PILOT_CODE_READY", {"base_code": str(tmp_path / "base-code")}),
        driver_sha256=worker.sha(worker.__file__))
    return payload, plan


def repin(payload, plan):
    plan["fingerprint"] = worker.digest({k: v for k, v in plan.items() if k != "fingerprint"})
    payload["training_plan"] = save(Path(plan["run_root"]) / "training_plan.json", plan)


@pytest.mark.parametrize("key,value", [("devices", 1), ("batch_size", 8), ("steps", 128),
    ("smoke_steps", 1), ("devices", True), ("evaluation_agent_enabled", True),
    ("collection_instances", [301]), ("evaluation_instances", [311]),
    ("collection_checkpoint_id", "old-1375000"), ("optimizer_resume", True),
    ("tasks", ["other"]), ("repo_id", "base"), ("training_run_id", "../escape"),
    ("seed", 0), ("learning_rate", {"peak": 1})])
def test_changed_scope_rejected_before_any_output(prepared, key, value):
    payload, plan = prepared
    plan[key] = value; repin(payload, plan)
    with pytest.raises(ValueError): worker.run(payload)
    assert not Path(payload["report_root"]).exists()
    assert not Path(payload["destination"]).exists()


@pytest.mark.parametrize("key,value", [("complete", False), ("real_lerobot_reader", False),
    ("real_lerobot_writer", False), ("reader_tolerance_s", 0.0001), ("action_horizon", 16),
    ("label_coverage", False), ("export_fingerprint", "d"*64)])
def test_fake_or_different_reader_proof_rejected(prepared, key, value):
    payload, plan = prepared
    plan["export_verification"][key] = value; repin(payload, plan)
    with pytest.raises(ValueError, match="Mixed dataset"): worker.verify_plan(payload)


def test_plan_hash_and_fresh_isolated_roots_are_mandatory(prepared):
    payload, plan = prepared
    worker.verify_plan(payload)
    (Path(plan["run_root"]) / "smoke").mkdir()
    with pytest.raises(ValueError, match="only its new"): worker.verify_plan(payload)
    (Path(plan["run_root"]) / "smoke").rmdir()
    payload["report_root"] = str(Path(plan["dataset_root"]) / "reports")
    with pytest.raises(ValueError, match="separate"): worker.run(payload)
    assert not Path(payload["report_root"]).exists()
    payload["report_root"] = str(Path(plan["run_root"]).parent / "report")
    Path(payload["training_plan"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="Pinned artifact changed"): worker.verify_plan(payload)


@pytest.mark.parametrize("key,value", [("wall_timeout_seconds", 0), ("full_timeout_seconds", 13801),
    ("wall_timeout_seconds", True), ("teacher", "codex"), ("resume", True),
    ("code", "relative"), ("driver_sha256", "missing")])
def test_payload_bounds(prepared, key, value):
    payload, _ = prepared; payload[key] = value
    with pytest.raises(ValueError): worker.validate_payload(payload)


def test_environment_has_explicit_runtime_and_no_teacher_or_proxy(prepared):
    payload, _ = prepared
    inherited = dict(PATH="/bin", HOME="/worker-home", OPENAI_API_KEY="private", CODEX_HOME="/auth",
        HTTPS_PROXY="private", PYTHONPATH="/old-source", PYTHONHOME="/wrong", TRAIN_STEPS="1", PI05_ENV="/old")
    env = worker.environment(payload, inherited)
    assert env["HOME"] == inherited["HOME"]
    assert env["PI05_ENV"] == payload["runtime"] + "/envs/b1k-pi05"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert env["PYTHONPATH"].startswith(payload["code"] + "/scripts")
    assert not {"OPENAI_API_KEY", "CODEX_HOME", "HTTPS_PROXY", "PYTHONHOME", "TRAIN_STEPS"} & env.keys()


@pytest.mark.parametrize("count,busy,passes", [(4, "", True), (1, "", False), (8, "", False), (4, "1234\n", False)])
def test_actual_gpu_inventory_requires_exactly_four_idle_cards(monkeypatch, count, busy, passes):
    def query(command, **kwargs):
        assert kwargs["timeout"] == 20
        return SimpleNamespace(stdout=busy if "--query-compute-apps=pid" in command else
            "".join(f"{i}, GPU-{i}, card, 32768, 32000\n" for i in range(count)))
    monkeypatch.setattr(worker.subprocess, "run", query)
    if passes: assert len(worker.gpu_inventory({})) == 4
    else:
        with pytest.raises(ValueError): worker.gpu_inventory({})


def phase_output(plan, phase):
    root = Path(plan["run_root"]) / phase; root.mkdir()
    (root / "train.log").write_text("actual simulated phase log\n")
    proof = dict(restored_step=2 if phase == "smoke" else 256, global_devices=4,
        processes=1, conditioning_mode="joint", action_horizon=32)
    save(root / "checkpoint_restore_verified.json", proof)
    save(root / "phase_receipt.json", dict(complete=True, kind=worker.PLAN_KIND, phase=phase,
        training_plan_fingerprint=plan["fingerprint"], checkpoint_restore_evidence=proof,
        base_norm_sha256=plan["base_norm_sha256"], training_log_sha256=worker.sha(root / "train.log")))


def fake_phases(monkeypatch, prepared, failure=None, missing_proof=False):
    payload, plan = prepared; calls = []
    monkeypatch.setattr(worker, "verify_code", lambda _: {"state": "test-only"})
    monkeypatch.setattr(worker, "gpu_inventory", lambda _: [{"index": i} for i in range(4)])
    def execute(command, **kwargs):
        phase = Path(kwargs["log_path"]).stem; calls.append((phase, command, kwargs["env"]))
        Path(kwargs["log_path"]).write_text("safe simulated log\n")
        if phase == failure:
            Path(kwargs["log_path"]).write_text("HTTPS_PROXY=https://user:password@proxy/key?signature=secret OPENAI_API_KEY=private")
            raise RuntimeError("https://host/private?token=secret TOKEN=private")
        if phase in ("smoke", "full"):
            if not missing_proof: phase_output(plan, phase)
        elif phase == "stage":
            save(Path(plan["run_root"]) / "training_receipt.json", {"fixture": True})
        else: save(Path(payload["report_root"]) / (phase + ".json"), {"state": "verified"})
        kwargs["heartbeat"]()
    monkeypatch.setattr(worker, "bounded_process", execute)
    monkeypatch.setattr(worker, "verify_final", lambda p, _: worker.file_ref(Path(plan["run_root"]) / "training_receipt.json"))
    return calls


def test_ordered_worker_requires_smoke_full_and_final_staging(prepared, monkeypatch):
    payload, plan = prepared; before = Path(payload["training_plan"]["path"]).read_bytes()
    calls = fake_phases(monkeypatch, prepared)
    result = worker.run(payload)
    assert result["state"] == "complete" and result["ready_for_evaluation"] is True
    assert [item[0] for item in calls] == ["data_preflight", "jax_preflight", "smoke", "full", "stage"]
    for phase, command, env in calls:
        expected = "b1k-pi05" if phase == "jax_preflight" else "b1k-eval"
        assert command[0] == payload["runtime"] + "/envs/" + expected + "/bin/python"
        assert (env.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false") == (phase == "jax_preflight")
    assert Path(payload["training_plan"]["path"]).read_bytes() == before
    assert not (Path(payload["report_root"]) / "_RUNNING").exists()
    assert Path(payload["report_root"]).stat().st_mode & 0o777 == 0o700
    assert (Path(payload["report_root"]) / "status.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["data_preflight", "jax_preflight", "smoke", "full", "stage"])
def test_phase_failure_prevents_later_phases_and_redacts_failure(prepared, monkeypatch, failure):
    payload, _ = prepared; calls = fake_phases(monkeypatch, prepared, failure=failure)
    result = worker.run(payload)
    assert result["state"] == "failed" and result["ready_for_evaluation"] is False
    assert calls[-1][0] == failure
    if failure != "stage": assert "stage" not in [row[0] for row in calls]
    assert not Path(payload["destination"]).exists()
    failure_text = (Path(payload["report_root"]) / "failure.json").read_text()
    for secret in ("user:password", "signature=secret", "TOKEN=private", "OPENAI_API_KEY=private"):
        assert secret not in failure_text


def test_success_exit_without_smoke_proof_cannot_start_full(prepared, monkeypatch):
    calls = fake_phases(monkeypatch, prepared, missing_proof=True)
    result = worker.run(prepared[0])
    assert result["state"] == "failed"
    assert calls[-1][0] == "smoke"


def test_real_short_child_deadline_cleans_only_its_group(tmp_path, monkeypatch):
    recorded = []; native = os.killpg
    def kill_group(pid, signum):
        recorded.append((pid, signum)); return native(pid, signum)
    monkeypatch.setattr(worker.os, "killpg", kill_group)
    with pytest.raises(TimeoutError):
        worker.bounded_process([sys.executable, "-c", "import time; time.sleep(60)"], env=os.environ.copy(),
            cwd=tmp_path, log_path=tmp_path / "timeout.log", seconds=0.05,
            deadline=time.monotonic() + 60, heartbeat=lambda: None)
    assert recorded and len({pid for pid, _ in recorded}) == 1
    pid = recorded[0][0]
    assert pid != os.getpgrp() and recorded[0][1] == signal.SIGTERM
    with pytest.raises(ProcessLookupError): os.kill(pid, 0)


def test_global_deadline_cannot_be_swallowed_as_dependency_exception():
    assert not issubclass(worker.WorkerDeadline, Exception)


def test_final_validation_uses_native_eval_contract(prepared):
    payload, plan = prepared
    code = Path(payload["code"]) / "sco/run_codex_pilot_evaluation.py"
    code.parent.mkdir()
    actual = Path(__file__).with_name("run_codex_pilot_evaluation.py")
    code.write_bytes(actual.read_bytes())
    root = Path(payload["destination"]); root.mkdir()
    entries = {"params/_METADATA": "metadata", "params/shard": "trained-weights",
        "assets/subtask_conditioning.json": json.dumps({"action_conditioning": "task_and_subtask"}),
        "assets/b1k_codex_pilot/norm_stats.json": "normalization"}
    for name, content in entries.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    manifest = root / "SHA256SUMS.inference"
    manifest.write_text("".join(worker.sha(root / name) + "  " + name + "\n" for name in sorted(entries)))
    plan["base_norm_sha256"] = worker.sha(root / "assets/b1k_codex_pilot/norm_stats.json")
    save(root / "_READY", dict(init_checkpoint_step=1688977, restored_train_state_step=256,
        manifest_sha256=worker.sha(manifest)))
    receipt = dict(kind="codex_pilot_trained_checkpoint_v1", checkpoint_id="pi05_b1k_codex_pilot_256",
        parent_checkpoint_step=1688977, optimizer_steps=256, ready_for_evaluation=True,
        real_lerobot_readback_verified=True, base_normalization_preserved=True,
        base_norm_sha256=plan["base_norm_sha256"], checkpoint_manifest=worker.file_ref(manifest),
        training_plan=payload["training_plan"])
    save(Path(plan["run_root"]) / "training_receipt.json", receipt)
    assert worker.verify_final(payload, plan)["sha256"] == worker.sha(Path(plan["run_root"]) / "training_receipt.json")
    (root / "params/shard").write_text("changed")
    with pytest.raises(ValueError): worker.verify_final(payload, plan)
