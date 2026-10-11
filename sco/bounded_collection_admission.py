"""Count this finite campaign's requested GPUs, including queued and stopping jobs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time


ACTIVE_STATES = ("WAITING", "INIT", "QUEUEING", "CREATING", "STARTING", "RESTARTING",
                 "RECOVERING", "RUNNING", "SUSPENDING", "DELETING")
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "SUSPENDED", "DELETED", "STOPPED", "CANCELED", "CANCELLED"}


class GPUAllowanceFull(RuntimeError):
    pass


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_policy(path, original, output, campaign):
    policy = json.loads(Path(path).read_text())
    assert policy["kind"] == "bounded_subtask_collection_authorization" and policy["schema_version"] == 1
    assert type(policy["max_active_gpus"]) is int and 1 <= policy["max_active_gpus"] <= 64
    assert policy["owner"] == "260010165" and policy["workspace"] == "share-space"
    assert policy["campaign_fingerprint"] == campaign["fingerprint"]
    assert policy["deadline_epoch"] == campaign["deadline_epoch"]
    assert policy["original_root"] == str(Path(original).resolve())
    assert policy["output_root"] == str(Path(output).resolve())
    assert policy["count_queued_and_stopping"] is True and policy["preserve_existing_running"] is True
    for root in (original, output):
        marker = Path(root) / "USER_CANCEL_NONRUNNING.json"
        assert marker.is_file(), "Retain the cancellation marker to block old unbounded entry points"
        assert policy["cancelled_markers_sha256"][str(Path(root).resolve())] == hashlib.sha256(marker.read_bytes()).hexdigest()
    return policy


def requested_gpus(job):
    total = 0
    for role in job.get("roles", []):
        specs = role.get("resource_spec", [])
        if isinstance(specs, dict):
            specs = [specs]
        for spec in specs:
            replicas = max(1, int(spec.get("replicas", 1)))
            if len(specs) == 1:
                replicas = max(replicas, int(role.get("total_replicas", 1)))
            total += int(spec.get("limits", {}).get("nvidia.com/gpu", 0)) * replicas
    return total


def list_active(cli, env, owner, workspace):
    jobs = {}
    for state in ACTIVE_STATES:
        state_uids = set()
        for page in range(1, 101):
            reply = subprocess.run([cli, "acp", "jobs", "list", "--workspace-name=" + workspace,
                "--user-name=" + owner, "--state=" + state, "--page-size=20", "--page-token=" + str(page),
                "-o", "json"], env=env, text=True, capture_output=True, check=True, timeout=30)
            batch = [] if reply.stdout.strip() == "No jobs found" else json.loads(reply.stdout)
            assert isinstance(batch, list)
            for job in batch:
                assert job["ownership"]["user_name"] == owner and job["state"] == state
                assert job["uid"] not in state_uids, "Repeated platform page; no admission"
                state_uids.add(job["uid"])
                jobs[job["uid"]] = job
            if len(batch) < 20:
                break
        else:
            raise RuntimeError("Active-job pagination incomplete; no admission")
    return list(jobs.values())


class GPUAdmission:
    def __init__(self, policy, root, specs, helper, cli, env, *, query=None, clock=time.time):
        self.policy, self.root, self.helper = policy, Path(root), helper
        self.cli, self.env, self.clock = cli, env, clock
        self.limit = policy["max_active_gpus"]
        self.allowed = set(policy.get("additional_known_job_names", []))
        self.directories = {Path(spec["directory"]) for spec in specs}
        for spec in specs:
            self.allowed.update(spec["prefix"] + f"-g{generation}" for generation in range(3))
        self.query = query or (lambda: list_active(cli, env, policy["owner"], policy["workspace"]))
        self.reservations_path = self.root / "gpu_reservations.json"
        self.reservations = json.loads(self.reservations_path.read_text()) if self.reservations_path.exists() else {}
        assert set(self.reservations) <= self.allowed
        self.blocked_for_pass = False
        self.last = None

    def receipts(self):
        found = {}
        for directory in self.directories:
            for path in directory.glob("spot_*_g*.json"):
                receipt = json.loads(path.read_text())
                assert receipt["name"] in self.allowed
                assert receipt["gpu_count"] == 1 and receipt["quota_type"] == "spot"
                found[receipt["name"]] = receipt
        return found

    def snapshot(self, current_candidate=None):
        jobs = [job for job in self.query() if job["name"] in self.allowed]
        by_name = {}
        receipts = self.receipts()
        for job in jobs:
            assert job["ownership"]["user_name"] == self.policy["owner"]
            assert job["state"] in ACTIVE_STATES and requested_gpus(job) == 1
            assert job["scheduling"]["quota_type"] == "SPOT"
            assert job["name"] not in by_name, "Multiple UIDs for one collection job"
            prior = receipts.get(job["name"])
            if prior and prior.get("uid"):
                assert prior["uid"] == job["uid"], "Active collection UID changed"
            by_name[job["name"]] = job
        # Unknown RPC outcomes retain one slot. Reservations survive a controller
        # restart and hide eventual consistency delays after actual cloud creation.
        for name, receipt in receipts.items():
            if not receipt.get("uid"):
                self.reservations.setdefault(name, {"reserved_at": self.clock(), "reason": "unknown_intent"})
        for name in list(self.reservations):
            if name in by_name or name == current_candidate:
                continue
            try:
                live = self.helper.describe(self.cli, self.env, name)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, ConnectionError, TimeoutError):
                continue  # keep its slot; never infer absence or stopped state
            assert live["ownership"]["user_name"] == self.policy["owner"]
            if receipts.get(name, {}).get("uid"):
                assert live["uid"] == receipts[name]["uid"]
            if live["state"] in TERMINAL_STATES:
                del self.reservations[name]
            else:
                assert live["state"] in ACTIVE_STATES and requested_gpus(live) == 1
                by_name[name] = live
        save(self.reservations_path, self.reservations)
        unknown = sorted(set(self.reservations) - set(by_name))
        value = dict(observed_at=self.clock(), max_active_gpus=self.limit,
                     occupied_gpus=len(by_name) + len(unknown), active_job_states=dict(Counter(j["state"] for j in by_name.values())),
                     unknown_or_reserved_jobs=unknown, counted_job_names=sorted(by_name),
                     count_queued_and_stopping=True, scope="this_finite_collection_campaign")
        save(self.root / "gpu_admission_status.json", value)
        self.last = value
        return value

    def begin_pass(self):
        self.blocked_for_pass = False
        try:
            self.blocked_for_pass = self.snapshot()["occupied_gpus"] >= self.limit
        except Exception:
            self.blocked_for_pass = True
            raise

    def reserve(self, name):
        assert name in self.allowed
        if self.blocked_for_pass:
            raise GPUAllowanceFull("GPU allowance occupied; preserve current jobs and wait")
        snapshot = self.snapshot()
        if snapshot["occupied_gpus"] >= self.limit:
            self.blocked_for_pass = True
            raise GPUAllowanceFull("GPU allowance occupied; no new submission")
        assert name not in self.reservations and name not in snapshot["counted_job_names"]
        self.reservations[name] = {"reserved_at": self.clock(), "reason": "bounded_submission"}
        save(self.reservations_path, self.reservations)

    def release_before_intent(self, name):
        self.reservations.pop(name, None)
        save(self.reservations_path, self.reservations)

    def before_copy(self, command):
        name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--name="))
        assert name in self.reservations, "Cloud creation needs an exclusive persistent GPU reservation"
        snapshot = self.snapshot(current_candidate=name)
        assert name not in snapshot["counted_job_names"], "Exact name already active; reconcile instead of copying"
        if snapshot["occupied_gpus"] > self.limit:
            self.blocked_for_pass = True
            raise GPUAllowanceFull("GPU allowance changed before actual cloud creation")
