"""Safety tests for requested-GPU caps; no cloud resources or credentials."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import bounded_collection_admission as admission


def job(name, state="RUNNING", uid=None, gpu=1):
    return dict(name=name, uid=uid or "uid-" + name, state=state, ownership=dict(user_name="owner"),
                scheduling=dict(quota_type="SPOT"),
                roles=[dict(total_replicas=1, resource_spec=[dict(replicas=0, limits={"nvidia.com/gpu": str(gpu)})])])


def make_gate(tmp_path, jobs, limit=64, *, describe=None):
    specs = [dict(directory=tmp_path, prefix=f"job-{i}") for i in range(100)]
    helper = SimpleNamespace(describe=describe or (lambda cli, env, name: job(name)))
    policy = dict(max_active_gpus=limit, owner="owner", workspace="share-space")
    return admission.GPUAdmission(policy, tmp_path, specs, helper, "unused", {}, query=lambda: jobs)


def test_inherited_over_limit_and_exact_limit_never_admit(tmp_path):
    jobs = [job(f"job-{i}-g0") for i in range(80)]
    gate = make_gate(tmp_path, jobs)
    gate.begin_pass()
    with pytest.raises(admission.GPUAllowanceFull):
        gate.reserve("job-90-g0")
    assert gate.last["occupied_gpus"] == 80 and not gate.reservations
    del jobs[64:]
    gate.begin_pass()
    with pytest.raises(admission.GPUAllowanceFull):
        gate.reserve("job-90-g0")


def test_running_queued_starting_stopping_all_consume_slots(tmp_path):
    jobs = [job(f"job-{i}-g0", state) for i, state in enumerate(("RUNNING", "STARTING", "QUEUEING", "SUSPENDING"))]
    gate = make_gate(tmp_path, jobs, limit=4)
    gate.begin_pass()
    with pytest.raises(admission.GPUAllowanceFull):
        gate.reserve("job-90-g0")
    assert gate.last["occupied_gpus"] == 4
    assert gate.last["active_job_states"]["SUSPENDING"] == 1


def test_one_vacancy_one_reservation_persists_and_blocks_second(tmp_path):
    jobs = [job(f"job-{i}-g0") for i in range(63)]
    def not_visible(cli, env, name):
        raise subprocess.CalledProcessError(65, ["redacted"])
    gate = make_gate(tmp_path, jobs, describe=not_visible)
    gate.begin_pass()
    gate.reserve("job-90-g0")
    gate.before_copy(["sco", "acp", "jobs", "copy", "--name=job-90-g0"])
    with pytest.raises(admission.GPUAllowanceFull):
        gate.reserve("job-91-g0")
    restarted = make_gate(tmp_path, jobs, describe=not_visible)
    restarted.begin_pass()
    assert restarted.last["occupied_gpus"] == 64
    with pytest.raises(admission.GPUAllowanceFull):
        restarted.reserve("job-91-g0")


def test_visible_reserved_job_not_counted_twice_and_terminal_releases(tmp_path):
    jobs = []
    gate = make_gate(tmp_path, jobs, describe=lambda *args: job(args[-1], "SUCCEEDED"))
    gate.reserve("job-90-g0")
    jobs.append(job("job-90-g0", "STARTING"))
    assert gate.snapshot()["occupied_gpus"] == 1
    jobs.clear()
    assert gate.snapshot()["occupied_gpus"] == 0
    assert not gate.reservations


def test_unknown_intent_keeps_slot_without_retrying_creation(tmp_path):
    (tmp_path / "spot_090_g0.json").write_text(json.dumps(dict(name="job-90-g0", gpu_count=1, quota_type="spot")))
    def not_visible(*args):
        raise subprocess.CalledProcessError(65, ["redacted"])
    gate = make_gate(tmp_path, [], limit=1, describe=not_visible)
    gate.begin_pass()
    assert gate.last["occupied_gpus"] == 1
    with pytest.raises(admission.GPUAllowanceFull):
        gate.reserve("job-91-g0")


def test_platform_read_failure_blocks_the_whole_pass(tmp_path):
    gate = make_gate(tmp_path, [])
    gate.query = lambda: (_ for _ in ()).throw(subprocess.CalledProcessError(65, ["redacted"]))
    with pytest.raises(subprocess.CalledProcessError):
        gate.begin_pass()
    with pytest.raises(admission.GPUAllowanceFull):
        gate.reserve("job-90-g0")
    assert not gate.reservations


def test_actual_copy_guard_rechecks_increased_usage_and_duplicate_name(tmp_path):
    jobs = [job(f"job-{i}-g0") for i in range(63)]
    gate = make_gate(tmp_path, jobs)
    gate.reserve("job-90-g0")
    jobs.append(job("job-64-g0"))
    with pytest.raises(admission.GPUAllowanceFull):
        gate.before_copy(["sco", "acp", "jobs", "copy", "--name=job-90-g0"])
    jobs[-1] = job("job-90-g0")
    with pytest.raises(AssertionError, match="already active"):
        gate.before_copy(["sco", "acp", "jobs", "copy", "--name=job-90-g0"])


def test_unrelated_campaign_not_charged_and_uid_or_gpu_changes_rejected(tmp_path):
    jobs = [job("other-campaign"), job("job-0-g0")]
    gate = make_gate(tmp_path, jobs)
    assert gate.snapshot()["occupied_gpus"] == 1
    (tmp_path / "spot_000_g0.json").write_text(json.dumps(dict(name="job-0-g0", uid="wrong", gpu_count=1, quota_type="spot")))
    with pytest.raises(AssertionError, match="UID changed"):
        gate.snapshot()
    (tmp_path / "spot_000_g0.json").unlink()
    jobs[-1] = job("job-0-g0", gpu=2)
    with pytest.raises(AssertionError):
        gate.snapshot()


def test_paginated_query_counts_once_across_forward_state_transition(monkeypatch):
    def run(command, **kwargs):
        state = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--state="))
        rows = [job("job-0-g0", state)] if state in ("STARTING", "RUNNING") else []
        return SimpleNamespace(stdout=json.dumps(rows))
    monkeypatch.setattr(subprocess, "run", run)
    jobs = admission.list_active("sco", {}, "owner", "share-space")
    assert len(jobs) == 1 and jobs[0]["state"] == "RUNNING"


def test_repeated_page_or_failed_list_is_not_free_capacity(monkeypatch):
    rows = [job(f"job-{i}-g0", "WAITING") for i in range(20)]
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(rows)))
    with pytest.raises(AssertionError, match="Repeated"):
        admission.list_active("sco", {}, "owner", "share-space")
