import argparse
import copy
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import eval_b1k_dynamic as dynamic
import eval_b1k_queue as queue_runner
from pi05_eval_queue import ArtifactValidationError, EvaluationQueue, LeaseLostError

DYNAMIC_PATH = SCRIPTS / "eval_b1k_dynamic.py"


def result(task: str, instance_id: int, rollout_id: int) -> dict:
    return {
        "task": task,
        "instance_id": instance_id,
        "rollout_id": rollout_id,
        "steps": 1,
        "success": rollout_id == 0,
        "q_score": {"final": 0.25},
    }


def write_attempt(lease, *, rollouts=3, videos=True, complete=True):
    lease.json_dir.mkdir(parents=True, exist_ok=True)
    lease.video_dir.mkdir(parents=True, exist_ok=True)
    count = rollouts if complete else rollouts - 1
    for rollout_id in range(count):
        lease.result_path(rollout_id).write_text(
            json.dumps(result(lease.task, lease.instance_id, rollout_id))
        )
        if videos:
            lease.video_path(rollout_id).write_bytes(b"video")


def make_queue(tmp_path: Path, jobs=None, max_attempts=3):
    queue = EvaluationQueue(tmp_path / "queue.sqlite3")
    queue.initialize(
        jobs or [("carrying_in_groceries", 301)],
        max_attempts=max_attempts,
        metadata={"run_id": "run", "num_rollouts": 3, "require_videos": True},
    )
    return queue


def test_concurrent_claims_are_unique_and_first_attempts_precede_retries(tmp_path):
    jobs = [("carrying_in_groceries", instance_id) for instance_id in range(301, 321)]
    queue = make_queue(tmp_path, jobs)
    first = queue.claim("carrying_in_groceries", "failed", tmp_path / "attempts")
    queue.fail(first, "retry")
    barrier = threading.Barrier(8)
    claimed = []
    lock = threading.Lock()

    def claim_all(worker):
        barrier.wait()
        while True:
            lease = queue.claim("carrying_in_groceries", worker, tmp_path / "attempts")
            if lease is None:
                return
            with lock:
                claimed.append(lease)

    threads = [
        threading.Thread(target=claim_all, args=(f"worker-{index}",))
        for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed) == 20
    assert len({lease.instance_id for lease in claimed}) == 20
    with sqlite3.connect(queue.db_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    retried = next(lease for lease in claimed if lease.instance_id == first.instance_id)
    assert retried.attempt == 2
    with sqlite3.connect(queue.db_path) as connection:
        starts = connection.execute(
            "SELECT attempt, started_at FROM attempts WHERE lease_token != ?",
            (first.lease_token,),
        ).fetchall()
    retry_started = next(started_at for attempt, started_at in starts if attempt == 2)
    assert retry_started >= max(
        started_at for attempt, started_at in starts if attempt == 1
    )


def test_retries_exhaustion_and_lease_fencing(tmp_path):
    queue = make_queue(tmp_path, max_attempts=2)
    first = queue.claim("carrying_in_groceries", "worker", tmp_path / "attempts")
    assert queue.fail(first, "one") == "pending"
    second = queue.claim("carrying_in_groceries", "worker", tmp_path / "attempts")
    assert second.attempt == 2
    with pytest.raises(LeaseLostError):
        queue.fail(first, "stale")
    assert queue.fail(second, "two") == "exhausted"
    assert queue.claim("carrying_in_groceries", "worker", tmp_path / "attempts") is None
    assert queue.snapshot()["counts"]["exhausted"] == 1


def test_partial_staging_is_not_published(tmp_path):
    queue = make_queue(tmp_path)
    lease = queue.claim("carrying_in_groceries", "worker", tmp_path / "attempts")
    write_attempt(lease, complete=False)
    canonical = tmp_path / "results" / "pi05" / lease.task

    with pytest.raises(ArtifactValidationError):
        queue.publish(lease, canonical, num_rollouts=3, require_videos=True)

    assert not canonical.exists()
    assert queue.snapshot()["counts"]["running"] == 1


def test_publication_writes_videos_before_json(tmp_path, monkeypatch):
    queue = make_queue(tmp_path)
    lease = queue.claim("carrying_in_groceries", "worker", tmp_path / "attempts")
    write_attempt(lease)
    canonical = tmp_path / "results" / "pi05" / lease.task
    published = []
    publish_one = queue._publish_one

    def record_publish(source, destination):
        published.append(destination.parent.name)
        publish_one(source, destination)

    monkeypatch.setattr(queue, "_publish_one", record_publish)
    queue.publish(lease, canonical, num_rollouts=3, require_videos=True)

    assert published == ["videos"] * 3 + ["json"] * 3


def test_publication_never_overwrites_existing_result(tmp_path):
    queue = make_queue(tmp_path)
    lease = queue.claim("carrying_in_groceries", "worker", tmp_path / "attempts")
    write_attempt(lease)
    canonical = tmp_path / "results" / "pi05" / lease.task
    canonical_json = canonical / "json" / f"{lease.task}_{lease.instance_id}_0.json"
    canonical_json.parent.mkdir(parents=True)
    canonical_json.write_text("different")

    with pytest.raises(ArtifactValidationError, match="refusing to overwrite"):
        queue.publish(lease, canonical, num_rollouts=3, require_videos=True)

    assert canonical_json.read_text() == "different"


def test_publication_reconciliation_and_manifest(tmp_path):
    jobs = [
        (task, instance_id)
        for task in dynamic.TASKS
        for instance_id in dynamic.INSTANCE_IDS
    ]
    queue = make_queue(tmp_path, jobs)
    results = tmp_path / "results"
    for task, instance_id in jobs:
        lease = queue.claim(task, "worker", tmp_path / "attempts")
        write_attempt(lease)
        canonical = results / "pi05" / task
        if task == jobs[0][0] and instance_id == jobs[0][1]:
            for source, relative in queue.validate_artifacts(
                lease.attempt_dir,
                task=task,
                instance_id=instance_id,
                num_rollouts=3,
                require_videos=True,
            ):
                queue._publish_one(source, canonical / relative)
            assert queue.reconcile(
                lease, canonical, num_rollouts=3, require_videos=True
            )
        else:
            queue.publish(lease, canonical, num_rollouts=3, require_videos=True)

    manifest = queue.create_result_manifest(
        results,
        tmp_path / "result_manifest.json",
        model="pi05",
        num_rollouts=3,
        require_videos=True,
    )
    keys = [
        (item["task"], item["instance_id"], item["rollout_id"])
        for item in manifest["rollouts"]
    ]
    assert manifest["rollout_count"] == 120
    assert keys == sorted(keys)
    assert len(set(keys)) == 120
    assert all(
        item["json"]["size"] > 0 and len(item["json"]["sha256"]) == 64
        for item in manifest["rollouts"]
    )
    assert all(
        item["video"]["size"] > 0 and len(item["video"]["sha256"]) == 64
        for item in manifest["rollouts"]
    )


def test_rollout_specs_require_dense_ids_exact_samples_and_eligible_video(tmp_path):
    task = "carrying_in_groceries"
    root = tmp_path / "artifacts"
    json_dir = root / "json"
    json_dir.mkdir(parents=True)
    path = json_dir / f"{task}_301_0.json"
    value = {
        **result(task, 301, 0),
        "sample_id": "wrong-sample",
        "metric_eligible": False,
        "result_state": "setup_excluded",
    }
    path.write_text(json.dumps(value))
    specs = [
        {
            "rollout_id": 0,
            "sample_id": "expected-sample",
            "video": "required_if_eligible",
        }
    ]

    with pytest.raises(ArtifactValidationError, match="sample_id"):
        EvaluationQueue.validate_artifacts(
            root, task=task, instance_id=301, rollout_specs=specs
        )

    value["sample_id"] = "expected-sample"
    value["metric_eligible"] = True
    value["result_state"] = "evaluated"
    path.write_text(json.dumps(value))
    with pytest.raises(ArtifactValidationError, match="video set mismatch"):
        EvaluationQueue.validate_artifacts(
            root, task=task, instance_id=301, rollout_specs=specs
        )

    value["metric_eligible"] = False
    value["result_state"] = "setup_excluded"
    path.write_text(json.dumps(value))
    EvaluationQueue.validate_artifacts(
        root, task=task, instance_id=301, rollout_specs=specs
    )

    value["result_state"] = "evaluated"
    path.write_text(json.dumps(value))
    with pytest.raises(ArtifactValidationError, match="conflicts"):
        EvaluationQueue.validate_artifacts(
            root, task=task, instance_id=301, rollout_specs=specs
        )
    value["result_state"] = "setup_excluded"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="ordered dense"):
        EvaluationQueue.validate_artifacts(
            root,
            task=task,
            instance_id=301,
            rollout_specs=[{"rollout_id": 1, "video": "required"}],
        )


def test_variable_rollout_specs_publish_reconcile_and_manifest(tmp_path):
    task = "carrying_in_groceries"
    queue = make_queue(tmp_path, [(task, 301), (task, 302)])
    results_root = tmp_path / "results"
    canonical = results_root / "pi05" / task
    specs_by_job = {
        (task, 301): [
            {"rollout_id": 0, "sample_id": "sample-a", "video": "required"},
            {
                "rollout_id": 1,
                "sample_id": "sample-b",
                "video": "required_if_eligible",
            },
        ],
        (task, 302): [
            {
                "rollout_id": 0,
                "sample_id": "sample-c",
                "video": "required_if_eligible",
            }
        ],
    }

    first = queue.claim(task, "worker", tmp_path / "attempts")
    first.json_dir.mkdir(parents=True)
    first.video_dir.mkdir(parents=True)
    for spec in specs_by_job[(task, 301)]:
        value = {
            **result(task, 301, spec["rollout_id"]),
            "sample_id": spec["sample_id"],
            "metric_eligible": True,
            "result_state": "evaluated",
        }
        first.result_path(spec["rollout_id"]).write_text(json.dumps(value))
    first.video_path(0).write_bytes(b"video-0")

    with pytest.raises(ArtifactValidationError, match="video set mismatch"):
        queue.publish(first, canonical, rollout_specs=specs_by_job[(task, 301)])
    assert not canonical.exists()

    first.video_path(1).write_bytes(b"video-1")
    queue.publish(first, canonical, rollout_specs=specs_by_job[(task, 301)])

    second = queue.claim(task, "worker", tmp_path / "attempts")
    second.json_dir.mkdir(parents=True)
    second.result_path(0).write_text(
        json.dumps(
            {
                **result(task, 302, 0),
                "sample_id": "sample-c",
                "metric_eligible": False,
                "result_state": "setup_excluded",
            }
        )
    )
    staged = queue.validate_artifacts(
        second.attempt_dir,
        task=task,
        instance_id=302,
        rollout_specs=specs_by_job[(task, 302)],
    )
    for source, relative in staged:
        queue._publish_one(source, canonical / relative)
    assert queue.reconcile(
        second, canonical, rollout_specs=specs_by_job[(task, 302)]
    )

    manifest = queue.create_result_manifest(
        results_root,
        tmp_path / "result_manifest.json",
        model="pi05",
        rollout_specs_by_job=specs_by_job,
    )
    assert manifest["rollout_count"] == 3
    assert [item["sample_id"] for item in manifest["rollouts"]] == [
        "sample-a",
        "sample-b",
        "sample-c",
    ]
    assert [item["video"] is not None for item in manifest["rollouts"]] == [
        True,
        True,
        False,
    ]

    count_manifest = queue.create_result_manifest(
        results_root,
        tmp_path / "count_result_manifest.json",
        model="pi05",
        require_videos=False,
        rollout_specs_by_job={(task, 301): 2, (task, 302): 1},
    )
    assert count_manifest["rollout_count"] == 3
    assert queue.snapshot()["counts"]["succeeded"] == 2


def manifest_args(tmp_path: Path, *extra: str):
    return dynamic.parse_args(
        [
            "start",
            "--run-root",
            str(tmp_path / "run"),
            "--run-id",
            "run",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--repo-id",
            "repo",
            "--gpus",
            *[str(index) for index in range(8)],
            "--base-port",
            "9000",
            "--max-attempts",
            "3",
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--eval-env",
            str(tmp_path / "eval-env"),
            "--pi05-env",
            str(tmp_path / "pi05-env"),
            "--omnigibson-data-path",
            str(tmp_path / "data"),
            *extra,
        ]
    )


def test_fresh_resume_and_fingerprint_validation(tmp_path):
    args = manifest_args(tmp_path)
    manifest = dynamic.create_manifest(args)
    dynamic.validate_manifest(manifest)
    (args.run_root).mkdir()
    dynamic.atomic_write_json(args.run_root / "manifest.json", manifest)
    assert dynamic._load_manifest(args.run_root) == manifest

    changed = copy.deepcopy(manifest)
    changed["base_port"] += 1
    dynamic.atomic_write_json(args.run_root / "manifest.json", changed)
    with pytest.raises(ValueError, match="fingerprint"):
        dynamic._load_manifest(args.run_root)
    with pytest.raises(ValueError, match="fresh"):
        dynamic._run_start(args)


def test_resume_requeues_orphan_without_rerunning_success(tmp_path):
    jobs = [(dynamic.TASKS[0], 301), (dynamic.TASKS[0], 302)]
    queue = make_queue(tmp_path, jobs)
    succeeded = queue.claim(
        dynamic.TASKS[0], "run:slot-0:generation-0", tmp_path / "attempts"
    )
    write_attempt(succeeded)
    manifest = {
        "run_id": "run",
        "results_root": str(tmp_path / "results"),
        "num_rollouts": 3,
        "require_videos": True,
    }
    queue.publish(
        succeeded,
        dynamic._canonical_task_dir(manifest, succeeded.task),
        num_rollouts=3,
        require_videos=True,
    )
    orphan = queue.claim(
        dynamic.TASKS[0], "run:slot-1:generation-0", tmp_path / "attempts"
    )

    dynamic.requeue_orphaned_leases(queue, manifest)

    snapshot = queue.snapshot()
    assert snapshot["counts"] == {
        "pending": 1,
        "running": 0,
        "succeeded": 1,
        "exhausted": 0,
    }
    assert (
        next(
            job for job in snapshot["jobs"] if job["instance_id"] == orphan.instance_id
        )["attempts"]
        == 1
    )


def test_worker_allocation_and_child_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PI05_MEM_FRACTION", "0.99")
    monkeypatch.setenv("SUBTASK_PREDICTION", "1")
    manifest = dynamic.create_manifest(manifest_args(tmp_path))
    specs = dynamic.worker_specs(manifest["gpus"], manifest["base_port"])
    assert len(specs) == 8
    assert all(spec.tasks == dynamic.TASKS for spec in specs[:4])
    assert all(spec.tasks == tuple(reversed(dynamic.TASKS)) for spec in specs[4:])
    assert len({spec.gpu for spec in specs}) == 8
    assert len({spec.port for spec in specs}) == 8
    environment = dynamic.child_environment(manifest, specs[0], 2)
    assert environment["PI05_EVAL_QUEUE_GENERATION"] == "2"
    assert environment["SERVER_GPU"] == environment["EVAL_GPU"] == "0"
    assert environment["PI05_MEM_FRACTION"] == "0.35"
    assert environment["SUBTASK_PREDICTION"] == "0"
    assert "AWS_SECRET_ACCESS_KEY" not in environment


def test_conditioning_cli_is_validated_and_recorded(tmp_path):
    args = manifest_args(
        tmp_path,
        "--subtask-prediction",
        "--external-subtask-selector",
        "state_machine_v1",
        "--max-token-len",
        "512",
    )
    manifest = dynamic.create_manifest(args)
    assert manifest["conditioning"] == {
        "subtask_prediction": True,
        "external_subtask_selector": "state_machine_v1",
        "max_token_len": 512,
    }
    assert manifest["child_environment"]["SUBTASK_PREDICTION"] == "1"
    assert dynamic.validate_manifest(manifest) == manifest


@pytest.mark.parametrize(
    "settings, message",
    [
        (
            {
                "subtask_prediction": False,
                "external_subtask_selector": "state_machine_v1",
                "max_token_len": 512,
            },
            "requires a subtask conditioning mode",
        ),
        (
            {
                "subtask_prediction": True,
                "external_subtask_selector": "other",
                "max_token_len": 512,
            },
            "must be state_machine_v1",
        ),
    ],
)
def test_invalid_conditioning_settings_are_rejected(settings, message):
    with pytest.raises((TypeError, ValueError), match=message):
        dynamic.validate_conditioning(settings)


def test_gpu_compute_occupancy_maps_uuid_to_index(monkeypatch):
    responses = {
        "gpu=index,uuid": [["3", "GPU-three"], ["7", "GPU-seven"]],
        "compute-apps=gpu_uuid,pid,process_name": [["GPU-seven", "4321", "python"]],
    }
    monkeypatch.setattr(dynamic, "_query_nvidia_smi", lambda query: responses[query])

    available, processes = dynamic._gpu_compute_occupancy()

    assert available == {3, 7}
    assert processes == {3: [], 7: [{"pid": 4321, "process_name": "python"}]}
    dynamic._assert_selected_gpus_idle([3], processes)
    with pytest.raises(ValueError, match="4321"):
        dynamic._assert_selected_gpus_idle([7], processes)


def test_next_generation_accounts_for_orphan_runtime_directory(tmp_path):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "slot-0" / "generation-4").mkdir(parents=True)
    assert dynamic._next_generation(tmp_path / "run", runtime_root, 0) == 5


def test_process_group_cleanup_uses_term_then_kill(monkeypatch):
    class FakeProcess:
        pid = 1234
        returncode = None

        def __init__(self):
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls < 4 else -signal.SIGKILL

        def wait(self):
            self.returncode = -signal.SIGKILL
            return self.returncode

    class FakeLog:
        def close(self):
            pass

    process = FakeProcess()
    worker = dynamic.WorkerProcess(
        dynamic.WorkerSpec(0, 0, 9000, dynamic.TASKS), 0, process, FakeLog(), 0
    )
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(dynamic.time, "sleep", lambda _: None)

    dynamic.terminate_process_group(worker, grace_seconds=0)

    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]


def test_status_reports_queue_and_canonical_progress_without_preflight(tmp_path):
    args = manifest_args(tmp_path)
    manifest = dynamic.create_manifest(args)
    run_root = args.run_root
    run_root.mkdir()
    dynamic.atomic_write_json(run_root / "manifest.json", manifest)
    queue = EvaluationQueue(manifest["queue_db"])
    queue.initialize(
        (
            (task, instance_id)
            for task in dynamic.TASKS
            for instance_id in dynamic.INSTANCE_IDS
        ),
        max_attempts=3,
        metadata={"run_id": "run", "fingerprint": manifest["fingerprint"]},
    )

    completed = subprocess.run(
        [sys.executable, str(DYNAMIC_PATH), "status", "--run-root", str(run_root)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["queue_counts"]["pending"] == 40
    assert status["completed_rollouts"] == 0
    assert status["expected_rollouts"] == 120
    assert status["result_manifest"]["present"] is False


def fake_base_module(recorded=None):
    recorded = recorded if recorded is not None else {}

    def parse_base_args(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task-name", required=True)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--robot-config")
        parser.add_argument("--instance-indices", type=int, nargs="+", default=[0])
        parser.add_argument("--mode", default="public_test")
        parser.add_argument("--num-rollouts", type=int, default=1)
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--env-wrapper", default="wrapper")
        parser.add_argument("--policy", default="websocket")
        parser.add_argument("--prompt-mode", default="whole_task")
        parser.add_argument("--external-subtask")
        parser.add_argument("--external-subtask-selector")
        parser.add_argument("--output-dir", default="/tmp/eval")
        parser.add_argument(
            "--write-video", action=argparse.BooleanOptionalAction, default=False
        )
        parser.add_argument("--video-fps", type=int, default=30)
        parser.add_argument(
            "--headless", action=argparse.BooleanOptionalAction, default=True
        )
        return parser.parse_args(argv)

    class FakeOmegaConf:
        @staticmethod
        def create(value):
            return value

        @staticmethod
        def load(path):
            return {"loaded": path}

    class FakeEvaluator:
        def __init__(self, cfg):
            recorded["cfg"] = cfg

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def run_task_instance(_evaluator, args, instance_id, output_dir):
        recorded.setdefault("instances", []).append(instance_id)
        output_dir = Path(output_dir)
        (output_dir / "json").mkdir(parents=True)
        results = []
        for rollout_id in range(args.num_rollouts):
            value = result(args.task_name, instance_id, rollout_id)
            path = (
                output_dir
                / "json"
                / f"{args.task_name}_{instance_id}_{rollout_id}.json"
            )
            path.write_text(json.dumps(value))
            results.append(value)
        return results

    logger = types.SimpleNamespace(
        info=lambda *_args: None, warning=lambda *_args: None, error=lambda *_args: None
    )
    logger.exception = lambda *_args: None
    return types.SimpleNamespace(
        parse_args=parse_base_args,
        OmegaConf=FakeOmegaConf,
        Evaluator=FakeEvaluator,
        run_task_instance=run_task_instance,
        gm=types.SimpleNamespace(HEADLESS=None),
        DEFAULT_EVAL_SEED=0,
        seed_everything=lambda seed: seed,
        verify_external_subtask_rollout=lambda *_args, **_kwargs: None,
        logger=logger,
    )


def queue_runner_argv(tmp_path, *extra):
    return [
        "--task-name",
        dynamic.TASKS[0],
        "--num-rollouts",
        "2",
        "--output-dir",
        str(tmp_path / "results" / "pi05" / dynamic.TASKS[0]),
        "--queue-db",
        str(tmp_path / "queue.sqlite3"),
        "--queue-worker",
        "slot-0",
        "--queue-run-id",
        "run",
        "--queue-generation",
        "0",
        "--queue-attempt-root",
        str(tmp_path / "attempts"),
        *extra,
    ]


def test_tracked_queue_runner_parser_validates_queue_args_and_rollouts(tmp_path):
    base = fake_base_module()
    with pytest.raises(SystemExit):
        queue_runner.parse_args(["--task-name", dynamic.TASKS[0]], base_module=base)
    with pytest.raises(SystemExit):
        queue_runner.parse_args(
            [
                "--task-name",
                dynamic.TASKS[0],
                "--queue-db",
                str(tmp_path / "queue.sqlite3"),
            ],
            base_module=base,
        )
    argv = queue_runner_argv(tmp_path)
    argv[argv.index("2")] = "0"
    with pytest.raises(SystemExit):
        queue_runner.parse_args(argv, base_module=base)

    args = queue_runner.parse_args(
        queue_runner_argv(tmp_path, "--instance-indices", "4", "5"),
        base_module=base,
    )
    assert args.num_rollouts == 2
    assert args.instance_indices == [4, 5]
    assert args.queue_generation == 0


def test_tracked_queue_runner_owns_instance_execution(tmp_path):
    args = queue_runner.parse_args(
        queue_runner_argv(tmp_path), base_module=fake_base_module()
    )
    base = fake_base_module()

    class Metric:
        def aggregate(self, _env):
            return {"q_score": {"final": 0.5}}

    class Evaluator:
        def __init__(self):
            self.env = types.SimpleNamespace(task=types.SimpleNamespace(success=True))
            self.metrics = [Metric()]
            self.conditioning_metadata = {}
            self.prompt_events = []
            self.subtask_events = []
            self.selector_events = []
            self.selector_summary = None
            self.final_goal_state = {}
            self.instances = []

        def reset(self, **_kwargs):
            pass

        def load_task_instance(self, instance_id):
            self.instances.append(instance_id)

        def step(self):
            return True, False

    evaluator = Evaluator()
    results = queue_runner.run_task_instance(
        base, evaluator, args, 301, tmp_path / "attempt"
    )

    assert evaluator.instances == [301]
    assert [item["rollout_id"] for item in results] == [0, 1]
    assert all(item["steps"] == 1 and item["success"] for item in results)
    assert len(list((tmp_path / "attempt" / "json").glob("*.json"))) == 2


def test_tracked_queue_runner_claims_executes_and_publishes(tmp_path):
    task = dynamic.TASKS[0]
    queue = EvaluationQueue(tmp_path / "queue.sqlite3")
    queue.initialize(
        [(task, 301), (task, 302)],
        max_attempts=2,
        metadata={
            "run_id": "run",
            "mode": "public_test",
            "num_rollouts": 2,
            "require_videos": False,
        },
    )
    recorded = {}
    base = fake_base_module(recorded)
    args = queue_runner.parse_args(queue_runner_argv(tmp_path), base_module=base)

    results = queue_runner.run(
        args,
        base_module=base,
        instance_runner=lambda base, evaluator, args, instance_id, output_dir: (
            base.run_task_instance(evaluator, args, instance_id, output_dir)
        ),
    )

    assert len(results) == 4
    assert recorded["instances"] == [301, 302]
    assert recorded["cfg"]["task"] == {"name": task}
    assert queue.snapshot()["counts"] == {
        "pending": 0,
        "running": 0,
        "succeeded": 2,
        "exhausted": 0,
    }
    canonical = Path(args.output_dir) / "json"
    assert len(list(canonical.glob("*.json"))) == 4


def test_tracked_queue_runner_reconciles_complete_canonical_output(tmp_path):
    task = dynamic.TASKS[0]
    queue = EvaluationQueue(tmp_path / "queue.sqlite3")
    queue.initialize(
        [(task, 301)],
        max_attempts=2,
        metadata={
            "run_id": "run",
            "mode": "public_test",
            "num_rollouts": 2,
            "require_videos": False,
        },
    )
    canonical = tmp_path / "results" / "pi05" / task / "json"
    canonical.mkdir(parents=True)
    for rollout_id in range(2):
        (canonical / f"{task}_301_{rollout_id}.json").write_text(
            json.dumps(result(task, 301, rollout_id))
        )
    base = fake_base_module()
    args = queue_runner.parse_args(queue_runner_argv(tmp_path), base_module=base)

    def fail_after_publish(*_args):
        raise RuntimeError("interrupted after publish")

    assert (
        queue_runner.run(args, base_module=base, instance_runner=fail_after_publish)
        == []
    )
    assert queue.snapshot()["counts"]["succeeded"] == 1


def test_exact_summary_contract_accepts_120_rollouts(tmp_path):
    args = manifest_args(tmp_path)
    manifest = dynamic.create_manifest(args)
    results_root = Path(manifest["results_root"])
    for task in dynamic.TASKS:
        json_dir = results_root / "pi05" / task / "json"
        video_dir = results_root / "pi05" / task / "videos"
        json_dir.mkdir(parents=True)
        video_dir.mkdir(parents=True)
        for instance_id in dynamic.INSTANCE_IDS:
            for rollout_id in dynamic.ROLLOUT_IDS:
                stem = f"{task}_{instance_id}_{rollout_id}"
                (json_dir / f"{stem}.json").write_text(
                    json.dumps(result(task, instance_id, rollout_id))
                )
                (video_dir / f"{stem}.mp4").write_bytes(b"video")
    completed = subprocess.run(
        [
            sys.executable,
            str(dynamic.REPO_ROOT / "nebula" / "summarize_pi05_eval.py"),
            "--results-root",
            str(results_root),
            "--output",
            str(tmp_path / "summary.json"),
            "--run-id",
            "run",
            "--profile",
            "two_tasks",
            "--checkpoint",
            manifest["checkpoint"],
            "--expected-rollouts",
            "120",
            "--expected-tasks",
            " ".join(dynamic.TASKS),
            "--expected-instance-ids",
            " ".join(str(value) for value in dynamic.INSTANCE_IDS),
            "--expected-rollout-ids",
            "0 1 2",
            "--require-videos",
            "--require-complete",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_rollouts"] == 120
    assert summary["missing_rollout_keys"] == []
