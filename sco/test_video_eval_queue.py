import json
from pathlib import Path

import pytest

from video_eval_queue import Controller, pending_gpu_count, TERMINAL_STATES


class FakeAdapter:
    def __init__(self):
        self.jobs = {}
        self.inspections = {}
        self.created = []
        self.described = []
        self.create_state = "RUNNING"
        self.on_create = None
        self.describe_error = None
        self.inspect_error = None
        self.on_describe = None

    def create(self, record):
        self.created.append(record["index"])
        if self.on_create:
            self.on_create(record)
        reply = dict(uid=f'uid-{record["index"]}', name=record["name"], state=self.create_state)
        self.jobs[record["name"]] = reply
        return reply

    def describe(self, name):
        self.described.append(name)
        if self.on_describe:
            self.on_describe(name)
        if self.describe_error:
            raise self.describe_error
        return self.jobs.get(name, dict(not_found=True, query_complete=True))

    def inspect(self, record):
        if self.inspect_error:
            raise self.inspect_error
        return self.inspections.get(record["index"])


@pytest.fixture
def setup(tmp_path):
    plan = dict(checkpoint_step=2_000_000, run_id="b1k-2m-video-test", episodes=[
        dict(index=t * 10 + i, task=f"task_{t}", instance_id=301 + i)
        for t in range(100) for i in range(10)
    ])
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    adapter = FakeAdapter()
    return tmp_path, adapter, Controller(tmp_path, adapter)


def accept_pilot(adapter, controller):
    controller.run_once()
    name = controller._name(0)
    adapter.jobs[name]["state"] = "SUCCEEDED"
    adapter.inspections[0] = dict(verified=True, model_success=False)


def load_record(root, index):
    return json.loads((root / "submissions" / f"{index:04d}.json").read_text())


@pytest.mark.parametrize("state", [
    "NEW", "QUEUEING", "STARTING", "INIT", "CREATING", "WAITING", "RESTARTING",
    "STOPPING", "SUSPENDING", "SUSPENDED", "UNKNOWN", "INTENT", "UNRECOGNIZED", "",
])
def test_every_nonterminal_nonrunning_state_reserves_one_gpu(state):
    assert pending_gpu_count([dict(state=state, gpu_count=1)]) == 1
    assert pending_gpu_count([dict(state="RUNNING"), dict(state=state)]) == 1


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES | {"RUNNING"}))
def test_running_and_terminal_states_do_not_consume_pending_budget(state):
    assert pending_gpu_count([dict(state=state)]) == 0


def test_only_pilot_submitted_until_terminal_and_verified(setup):
    root, adapter, controller = setup
    first = controller.run_once()
    assert first["submitted_this_pass"] == [0]
    assert adapter.created == [0]
    adapter.inspections[0] = dict(verified=True)
    second = controller.run_once()
    assert second["submitted_this_pass"] == []  # verified output while still RUNNING is insufficient
    adapter.jobs[controller._name(0)]["state"] = "SUCCEEDED"
    third = controller.run_once()
    assert third["submitted_this_pass"] == [1, 2, 3, 4, 5]
    assert third["reserved_this_pass"] == 5


def test_five_per_pass_even_when_each_create_immediately_running(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    for expected in (list(range(1, 6)), list(range(6, 11)), list(range(11, 16))):
        status = controller.run_once()
        assert status["submitted_this_pass"] == expected
        assert status["pending_gpus"] == 0
        assert status["reserved_this_pass"] == 5
    assert status["running_gpus"] == 15
    assert status["model_successes"] == 0  # successful pilot job need not mean successful model


def test_pending_budget_fills_only_available_slots(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    adapter.create_state = "QUEUEING"
    assert controller.run_once()["pending_gpus"] == 5
    assert controller.run_once()["submitted_this_pass"] == []
    for index in (1, 3):
        adapter.jobs[controller._name(index)]["state"] = "RUNNING"
    status = controller.run_once()
    assert status["submitted_this_pass"] == [6, 7]
    assert status["pending_gpus"] == 5
    assert status["reserved_this_pass"] == 5


def test_intent_is_durable_before_call_and_unknown_is_never_retried(setup):
    root, adapter, controller = setup
    def fail(record):
        saved = load_record(root, record["index"])
        assert saved["name"] == record["name"]
        assert saved["create_outcome"] == "UNKNOWN"
        raise TimeoutError("SECRET_TOKEN and signed://private-url")
    adapter.on_create = fail
    status = controller.run_once()
    assert status["pending_gpus"] == 1
    assert status["block_reason"] == "submission_outcome_unknown"
    assert "SECRET_TOKEN" not in (root / "submissions" / "0000.json").read_text()
    for _ in range(2):
        status = Controller(root, adapter).run_once()
        assert status["query_complete"] is False
        assert status["pending_gpus"] == 1
    assert adapter.created == [0]
    assert adapter.described == [controller._name(0)] * 2
    # Eventual exact-name reconciliation binds the original UID without create().
    adapter.jobs[controller._name(0)] = dict(uid="eventual-original-uid", state="SUCCEEDED")
    adapter.inspections[0] = dict(verified=True)
    adapter.on_create = None
    status = controller.run_once()
    assert load_record(root, 0)["uid"] == "eventual-original-uid"
    assert status["submitted_this_pass"] == [1, 2, 3, 4, 5]
    assert adapter.created.count(0) == 1


def test_one_incomplete_observation_blocks_all_new_submissions(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    controller.run_once()
    adapter.jobs[controller._name(3)]["query_complete"] = False
    adapter.described.clear()
    status = controller.run_once()
    assert status["query_complete"] is False
    assert status["submitted_this_pass"] == []
    assert len(adapter.described) == 5  # do not skip the remaining UID checks after an error
    assert status["pending_gpus"] == 1


def test_mismatching_uid_blocks_new_submissions(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    controller.run_once()
    adapter.jobs[controller._name(1)]["uid"] = "replacement-uid"
    status = controller.run_once()
    assert status["submitted_this_pass"] == []
    assert load_record(root, 1)["uid"] == "uid-1"
    assert load_record(root, 1)["state"] == "UNKNOWN"


def test_failed_pilot_never_opens_gate_or_retries(setup):
    root, adapter, controller = setup
    controller.run_once()
    adapter.jobs[controller._name(0)]["state"] = "FAILED"
    adapter.inspections[0] = dict(verified=True, model_success=True)
    for _ in range(2):
        status = controller.run_once()
        assert status["block_reason"] == "pilot_not_accepted"
    assert adapter.created == [0]


def test_failed_production_job_preserved_without_retry(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    controller.run_once()
    adapter.jobs[controller._name(1)]["state"] = "FAILED"
    status = controller.run_once()
    assert status["states"]["FAILED"] == 1
    assert adapter.created.count(1) == 1
    assert status["submitted_this_pass"] == [6, 7, 8, 9, 10]


def test_stop_file_prevents_queries_and_submissions(setup):
    root, adapter, controller = setup
    (root / "STOP").touch()
    status = controller.run_once()
    assert status["stopped"] is True
    assert adapter.created == adapter.described == []


def test_stop_during_pass_blocks_remaining_submissions(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    adapter.on_create = lambda record: (root / "STOP").touch()
    status = controller.run_once()
    assert status["submitted_this_pass"] == [1]
    assert status["stopped"] is True


def test_exclusive_flock_prevents_second_controller(setup):
    root, adapter, controller = setup
    with controller.lock():
        with pytest.raises(RuntimeError, match="another video"):
            Controller(root, adapter).run_once()
    assert adapter.created == []


def test_plan_changes_fail_before_any_new_api_calls(setup):
    root, adapter, controller = setup
    controller.run_once()
    plan = json.loads((root / "plan.json").read_text())
    plan["checkpoint_path"] = "a-different-checkpoint"
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="plan changed"):
        controller.run_once()
    with pytest.raises(RuntimeError, match="durable frozen"):
        Controller(root, adapter).run_once()
    assert adapter.created == [0]


def test_plan_mutated_during_query_prevents_creation(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    def mutate(name):
        plan = json.loads((root / "plan.json").read_text())
        plan["checkpoint_path"] = "wrong"
        (root / "plan.json").write_text(json.dumps(plan))
    adapter.on_describe = mutate
    with pytest.raises(RuntimeError, match="plan changed"):
        controller.run_once()
    assert adapter.created == [0]


def test_inspection_error_recorded_but_existing_running_jobs_continue(setup):
    root, adapter, controller = setup
    accept_pilot(adapter, controller)
    controller.run_once()
    adapter.inspect_error = ValueError("private-secret")
    status = controller.run_once()
    assert status["submitted_this_pass"] == [6, 7, 8, 9, 10]
    assert status["inspection_errors"] == [1, 2, 3, 4, 5]
    assert "private-secret" not in (root / "submissions" / "0001.json").read_text()


def test_create_response_only_allowed_fields_persisted(setup):
    root, adapter, controller = setup
    original = adapter.create
    def create(record):
        return dict(original(record), access_token="do-not-save", start_command="also-private")
    adapter.create = create
    controller.run_once()
    saved = (root / "submissions" / "0000.json").read_text()
    assert "do-not-save" not in saved and "also-private" not in saved


def test_invalid_scope_rejected_without_side_effects(setup):
    root, adapter, controller = setup
    plan = json.loads((root / "plan.json").read_text())
    plan["episodes"][9]["instance_id"] = 311
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="301..310"):
        Controller(root, adapter)
    assert adapter.created == []


def test_poll_delay_starts_after_entire_pass_finishes(setup, monkeypatch):
    root, adapter, controller = setup
    import video_eval_queue
    clock = [0.0]
    started = []
    sleeps = []
    def fake_pass():
        started.append(clock[0])
        clock[0] += 11  # querying, verification and submissions take eleven seconds
        return dict(stopped=False, finished=len(started) == 2)
    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(controller, "_run_once", fake_pass)
    monkeypatch.setattr(video_eval_queue.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(video_eval_queue.time, "sleep", fake_sleep)
    controller.run(poll_seconds=45)
    assert started == [0, 56]
    assert sum(sleeps) == 45


def test_damaged_terminal_receipt_without_uid_does_not_skip_identity_check(setup):
    root, adapter, controller = setup
    controller.run_once()
    path = root / "submissions" / "0000.json"
    record = json.loads(path.read_text())
    record.update(state="SUCCEEDED", inspection=dict(verified=True))
    record.pop("uid")
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="frozen episode"):
        controller.run_once()
    assert adapter.created == [0]


def test_all_terminal_missing_artifacts_finishes_with_failures(setup):
    root, adapter, controller = setup
    rows = {i: dict(index=i, state="SUCCEEDED", terminal_observed_epoch=0,
                    inspection={"verified": i != 42}) for i in range(1000)}
    status = controller._status(rows, "test", [], True, None, 0)
    assert status["finished"] is True
    assert status["completion_state"] == "completed_with_failures"
    assert status["verified_episodes"] == 999


def test_model_failures_do_not_make_verified_evaluation_failed(setup):
    root, adapter, controller = setup
    rows = {i: dict(index=i, state="SUCCEEDED", inspection={"verified": True, "model_success": False})
            for i in range(1000)}
    status = controller._status(rows, "test", [], True, None, 0)
    assert status["finished"] is True
    assert status["completion_state"] == "completed"
    assert status["model_successes"] == 0


def test_never_finishes_with_unsubmitted_or_nonterminal_work(setup):
    root, adapter, controller = setup
    rows = {i: dict(index=i, state="SUCCEEDED", inspection={"verified": True}) for i in range(999)}
    assert controller._status(rows, "test", [], True, None, 0)["finished"] is False
    rows[999] = dict(index=999, state="UNKNOWN")
    assert controller._status(rows, "test", [], False, None, 1)["finished"] is False


def test_begin_round_receives_all_known_records_and_runs_before_observation(setup):
    root, adapter, controller = setup
    observed = []
    adapter.begin_round = lambda records: observed.append([r['index'] for r in records])
    controller.run_once()
    adapter.on_describe = lambda name: observed.append('describe')
    controller.run_once()
    assert observed == [[], [0], 'describe']


def test_begin_round_failure_prevents_new_submissions(setup):
    root, adapter, controller = setup
    def fail(records):
        raise TimeoutError('secret')
    adapter.begin_round = fail
    status = controller.run_once()
    assert not status['query_complete']
    assert status['round_preparation_error'] == 'TimeoutError'
    assert status['submitted_this_pass'] == []
    assert 'secret' not in (root / 'status.json').read_text()


def test_terminal_results_have_five_minute_arrival_grace(setup, monkeypatch):
    root, adapter, controller = setup
    import video_eval_queue
    clock = [1000]
    monkeypatch.setattr(video_eval_queue.time, 'time', lambda: clock[0])
    rows = {i: dict(index=i, state='SUCCEEDED', terminal_observed_epoch=1000,
                    inspection={'verified': i != 42}) for i in range(1000)}
    status = controller._status(rows, 'test', [], True, None, 0)
    assert not status['finished']
    assert status['completion_state'] == 'awaiting_artifacts'
    assert status['awaiting_terminal_artifacts'] == [42]
    clock[0] = 1299
    assert not controller._status(rows, 'test', [], True, None, 0)['finished']
    clock[0] = 1300
    status = controller._status(rows, 'test', [], True, None, 0)
    assert status['finished']
    assert status['completion_state'] == 'completed_with_failures'


def test_terminal_timestamp_persists_across_restarts(setup, monkeypatch):
    root, adapter, controller = setup
    import video_eval_queue
    clock = [1000]
    monkeypatch.setattr(video_eval_queue.time, 'time', lambda: clock[0])
    controller.run_once()
    adapter.jobs[controller._name(0)]['state'] = 'FAILED'
    controller.run_once()
    assert load_record(root, 0)['terminal_observed_epoch'] == 1000
    clock[0] = 1234
    Controller(root, adapter).run_once()
    assert load_record(root, 0)['terminal_observed_epoch'] == 1000


def test_adapter_round_caches_with_six_reader_bound_and_preserves_errors():
    from video_eval_adapter import Adapter
    import threading
    import time
    adapter = Adapter.__new__(Adapter)
    mutex = threading.Lock()
    calls, counts = [], {'active': 0, 'peak': 0}
    def request(kind, key):
        with mutex:
            counts['active'] += 1
            counts['peak'] = max(counts['peak'], counts['active'])
            calls.append((kind, key))
        try:
            time.sleep(.005)
            if kind == 'describe' and key == 'job-3':
                raise TimeoutError('private')
            return {'uid': key, 'name': key, 'state': 'RUNNING'} if kind == 'describe' else None
        finally:
            with mutex:
                counts['active'] -= 1
    adapter._describe_live = lambda name: request('describe', name)
    adapter._receipt_live = lambda record: request('receipt', record['index'])
    records = [dict(index=i, name=f'job-{i}', state='RUNNING') for i in range(12)]
    records.append(dict(index=12, name='job-12', state='SUCCEEDED', inspection={'verified': True}))
    adapter.begin_round(records)
    assert 1 < counts['peak'] <= 6
    assert len(calls) == 24
    assert adapter.describe('job-2')['uid'] == 'job-2'
    with pytest.raises(TimeoutError):
        adapter.describe('job-3')
    adapter.keys = lambda record: {'receipt': 'unused'}
    assert adapter.inspect(records[2]) is None
    assert len(calls) == 24  # inspect/describe consume caches, without an extra request
    assert adapter.describe('new-job')['uid'] == 'new-job'  # freshly created job after pass snapshot
    assert len(calls) == 25
    adapter.begin_round([])
    assert adapter._description_cache == adapter._receipt_cache == {}


def test_auxiliary_pending_request_shares_same_limit(setup):
    _, adapter, controller = setup
    accept_pilot(adapter, controller)
    adapter.auxiliary_gpu_step = lambda allow_submit=False: dict(state='QUEUEING', pending_gpus=1, query_complete=True)
    status = controller.run_once()
    assert len(status['submitted_this_pass']) == 4
    assert status['reserved_this_pass'] == 5
    assert status['pending_gpus'] == 1


def test_new_auxiliary_running_job_reserves_current_pass(setup):
    _, adapter, controller = setup
    accept_pilot(adapter, controller)
    state = {'created': False}
    def auxiliary(allow_submit=False):
        if allow_submit:
            state['created'] = True
            return dict(state='RUNNING', pending_gpus=1, query_complete=True)
        return dict(state='RUNNING' if state['created'] else 'PLANNED', pending_gpus=0, query_complete=True)
    adapter.auxiliary_gpu_step = auxiliary
    assert len(controller.run_once()['submitted_this_pass']) == 4
    assert len(controller.run_once()['submitted_this_pass']) == 5


def test_unknown_auxiliary_submission_blocks_further_admission(setup):
    _, adapter, controller = setup
    accept_pilot(adapter, controller)
    adapter.auxiliary_gpu_step = lambda allow_submit=False: dict(state='UNKNOWN', pending_gpus=1, query_complete=False)
    status = controller.run_once()
    assert status['submitted_this_pass'] == []
    assert status['query_complete'] is False
    assert status['pending_gpus'] == 1
