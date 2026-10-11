"""One finite writer for existing collection jobs and an explicit frozen extension.

The original GPU drivers, prepared plans and receipts are never replaced. This
controller uses the original dispatcher's lock, resumes the highest saved job
generation, and treats failed platform reads as unknown, never as stopped jobs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DeadlineReached(RuntimeError):
    pass


class UserCancelledAutomaticCollection(RuntimeError):
    pass


def require_automatic_collection_authorized(*roots, authorization_policy=None):
    for root in roots:
        marker = Path(root) / "USER_CANCEL_NONRUNNING.json"
        if marker.exists():
            if (authorization_policy is not None
                    and authorization_policy.get("kind") == "bounded_subtask_collection_authorization"
                    and type(authorization_policy.get("max_active_gpus")) is int
                    and 1 <= authorization_policy["max_active_gpus"] <= 64
                    and authorization_policy.get("cancelled_markers_sha256", {}).get(str(Path(root).resolve()))
                    == hashlib.sha256(marker.read_bytes()).hexdigest()):
                continue
            raise UserCancelledAutomaticCollection(
                f"Automatic collection submission and recovery cancelled by the user: {marker}")


class DeadlineSubprocess:
    """Recheck the fixed deadline at the helper's actual cloud-create call."""
    def __init__(self, deadline, clock=time.time, cancellation_roots=(), authorization_policy=None):
        self.deadline, self.clock = deadline, clock
        self.cancellation_roots = cancellation_roots
        self.authorization_policy = authorization_policy
        self.admission_guard = None

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, command, **kwargs):
        if isinstance(command, list) and command[1:4] == ["acp", "jobs", "copy"]:
            require_automatic_collection_authorized(*self.cancellation_roots, authorization_policy=self.authorization_policy)
            if self.clock() >= self.deadline:
                raise DeadlineReached("Original collection deadline reached before cloud creation")
            if self.authorization_policy is not None:
                require(self.admission_guard is not None, "Bounded authorization requires a live GPU admission guard")
                self.admission_guard(command)
        return subprocess.run(command, **kwargs)


def load_helper(directory, deadline, cancellation_roots=(), authorization_policy=None):
    directory = Path(directory).resolve()
    sys.path.insert(0, str(directory))
    sys.path.insert(0, str(directory.parent / "scripts"))
    spec = importlib.util.spec_from_file_location("extension_submission_helper", directory / "submit_full_collection_shard.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.subprocess = DeadlineSubprocess(deadline, cancellation_roots=cancellation_roots,
                                          authorization_policy=authorization_policy)
    return helper


def transient(error):
    return isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError,
                              ConnectionError, TimeoutError))


def error_record(error):
    # Never serialize a submission argv, startup payload or signed URL.
    return {"error_type": type(error).__name__, "exit_code": getattr(error, "returncode", None)}


def highest_receipt(directory, index, prepared):
    directory = Path(directory)
    matches = []
    for path in directory.glob(f"spot_{index:03d}_g*.json"):
        match = re.fullmatch(rf"spot_{index:03d}_g([012])\.json", path.name)
        require(match is not None, "Unsupported execution generation: " + path.name)
        generation = int(match[1])
        receipt = read(path)
        require(receipt.get("generation") == generation, "Receipt generation mismatch")
        require(receipt.get("plan_fingerprint") == prepared["plan"]["fingerprint"], "Receipt belongs to another frozen plan")
        require(receipt.get("source_manifest_sha256") == prepared["source_manifest_sha256"], "Receipt source identity changed")
        require(receipt.get("quota_type") == "spot" and receipt.get("gpu_count") == 1
                and receipt.get("timeout_hours") == 24, "Receipt exceeds the authorized Spot scope")
        require(receipt.get("name", "").endswith(f"-g{generation}"), "Receipt name/generation mismatch")
        matches.append((generation, path, receipt))
    if not matches:
        return None
    matches.sort()
    require([r[0] for r in matches] == list(range(matches[-1][0] + 1)), "Missing prior generation receipt")
    require(len({r[2]["object_prefix"] for r in matches}) == 1, "Checkpoint namespace changed between generations")
    return matches[-1][1], matches[-1][2]


class Controller:
    def __init__(self, args, helper, *, client=None, clock=time.time, sleeper=time.sleep):
        self.args, self.helper, self.client = args, helper, client
        self.clock, self.sleep = clock, sleeper
        self.authorization_policy = None
        self.admission = None
        self.admission_error = None
        self.root, self.original = Path(args.output_dir), Path(args.original_root)
        self.campaign_path = Path(args.campaign)
        campaign = self.campaign = read(self.campaign_path)
        require(campaign.get("kind") == "low_subtask_collection_extension" and campaign.get("schema_version") == 1
                and campaign.get("state") == "complete", "Extension is not fully prepared")
        require(campaign.get("fingerprint") == digest({k: v for k, v in campaign.items() if k != "fingerprint"}), "Extension fingerprint changed")
        old_status = read(self.original / "status.json")
        original_deadline = old_status["started_at"] + 144 * 3600
        self.deadline = campaign["deadline_epoch"]
        require(self.deadline <= original_deadline, "Extension must not extend the original collection deadline")
        require(campaign.get("source_manifest_sha256") == old_status["source_manifest_sha256"], "Extension runtime differs from the original gates")
        require(campaign.get("shard_count") == len(campaign["groups"]) == 36
                and campaign.get("expected_trials") == 1080, "Authorized extension is exactly 36 shards / 1080 trials")
        require({e["index"] for e in campaign["groups"]} == set(range(299, 335)), "Extension indices must be 299..334")
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,44}", args.name_prefix), "Invalid stable job name prefix")
        require(args.object_prefix.startswith("behavior1k/subtask-collection/") and ".." not in args.object_prefix.split("/"), "Invalid collection object prefix")
        self.env = helper.environment()
        self.new_specs = []
        new_trial_ids, old_trial_ids = set(), set()
        original_paths = sorted((self.original / "prepared").glob("*.json"))
        require({path.stem for path in original_paths} == {f"{index:03d}" for index in range(299)}, "Original frozen plans must contain all 299 shards")
        for path in original_paths:
            old_trial_ids.update(t["trial_id"] for t in read(path)["plan"]["trials"])
        for entry in sorted(campaign["groups"], key=lambda row: row["index"]):
            path = self.campaign_path.parent / "groups" / f"{entry['index']:03d}" / "prepared.json"
            prepared = read(path)
            require(prepared == entry, "Local prepared entry differs from the frozen extension")
            require(prepared["state"] == "complete" and prepared["source_manifest_sha256"] == campaign["source_manifest_sha256"], "Incomplete source preparation")
            require(prepared["plan"]["expected_trials"] == len(prepared["plan"]["trials"]) == 30, "Each extension shard must contain exactly 30 frozen trials")
            ids = {t["trial_id"] for t in prepared["plan"]["trials"]}
            require(len(ids) == 30 and not ids & (new_trial_ids | old_trial_ids), "Extension reuses a prior trial identity")
            new_trial_ids.update(ids)
            self.new_specs.append(dict(index=entry["index"], path=path, prepared=prepared, directory=self.root,
                                       prefix=args.name_prefix + f"-a{entry['index']:03d}",
                                       object_prefix=args.object_prefix.rstrip("/") + f"/shard-{entry['index']:03d}"))
        self.root.mkdir(parents=True, exist_ok=True)
        scope = dict(campaign_fingerprint=campaign["fingerprint"], original_root=str(self.original.resolve()),
                     source_manifest_sha256=campaign["source_manifest_sha256"], deadline_epoch=self.deadline,
                     name_prefix=args.name_prefix, object_prefix=args.object_prefix.rstrip("/"),
                     maximum_executions_per_shard=3, timeout_hours=24, initial_shards=36, initial_trials=1080)
        path = self.root / "dispatcher_scope.json"
        if path.exists():
            require(read(path) == scope, "Existing extension dispatcher scope differs")
        else:
            with path.open("x") as stream:
                json.dump(scope, stream, indent=2)
        self.write_manifest()

    def check_deadline(self):
        if self.clock() >= self.deadline:
            raise DeadlineReached("Original collection deadline reached; jobs and data preserved")

    def retry_read(self, callback):
        for attempt in range(3):
            self.check_deadline()
            try:
                return callback()
            except Exception as error:
                if not transient(error) or attempt == 2:
                    raise
                self.sleep(min(2**attempt, max(0, self.deadline - self.clock())))

    def describe(self, receipt):
        live = self.retry_read(lambda: self.helper.describe(self.args.cli, self.env, receipt["name"]))
        require(live["ownership"]["user_name"] == self.helper.BUCKET, "Platform owner mismatch")
        require(live["scheduling"]["quota_type"] == "SPOT", "Platform quota differs from Spot")
        if receipt.get("uid"):
            require(live["uid"] == receipt["uid"], "Exact submitted UID changed")
        return live

    def existing(self, spec):
        saved = highest_receipt(spec["directory"], spec["index"], spec["prepared"])
        if saved:
            require(saved[1]["name"] == spec["prefix"] + f"-g{saved[1]['generation']}"
                    and saved[1]["object_prefix"] == spec["object_prefix"], "Receipt namespace differs from the frozen dispatcher scope")
        return saved

    def reconcile(self, path, receipt):
        live = self.describe(receipt)
        if not receipt.get("uid"):
            receipt.update(uid=live["uid"], state="submitted", platform_state=live["state"])
            write(path, receipt)
        return live

    def create(self, spec, previous=None):
        require_automatic_collection_authorized(self.original, self.root, authorization_policy=self.authorization_policy)
        require(self.authorization_policy is None or self.admission is not None, "Bounded authorization needs GPU accounting")
        self.check_deadline()
        generation = 0 if previous is None else previous[1]["generation"] + 1
        require(generation <= 2, "At most three executions per shard")
        path = spec["directory"] / f"spot_{spec['index']:03d}_g{generation}.json"
        # The shared writer lock and on-disk generation inventory protect both
        # original and extension namespaces, even after a controller restart.
        require(not path.exists(), "Existing intent must be reconciled, never overwritten")
        name = spec["prefix"] + f"-g{generation}"
        if self.admission is not None:
            self.admission.reserve(name)
        try:
            return self.helper.submit(SimpleNamespace(
                receipt=str(path), prepared=str(spec["path"]), name=name,
                quota_type="spot", gates=str(self.args.gates), timeout_hours=24,
                object_prefix=spec["object_prefix"], cli=self.args.cli,
                aoss_client_module=self.args.aoss_client_module,
                recover_receipt=None if previous is None else str(previous[0])))
        except Exception:
            if self.admission is not None and not path.exists():
                self.admission.release_before_intent(name)
            raise

    def publish(self, state, observations):
        value = dict(state=state, updated_at=self.clock(), deadline_epoch=self.deadline,
                     campaign_fingerprint=self.campaign["fingerprint"], original_data_modified=False,
                     requested_new_shards=36, expected_new_trials=1080, observations=observations,
                     submitted_new_shards=sum(bool(self.existing(s) and self.existing(s)[1].get("uid")) for s in self.new_specs))
        if self.admission is not None:
            value.update(max_active_gpus=self.admission.limit, gpu_admission=self.admission.last,
                         gpu_admission_error=self.admission_error)
        write(self.root / "extension_status.json", value)
        return value

    def submit_all(self):
        observations = []
        self.publish("submitting_extension", observations)
        for spec in self.new_specs:
            row = {"index": spec["index"]}
            try:
                self.check_deadline()
                saved = self.existing(spec)
                if saved:
                    if saved[1].get("uid"):
                        row.update(state="already_submitted", name=saved[1]["name"], uid=saved[1]["uid"], generation=saved[1]["generation"], platform_state=saved[1].get("platform_state"))
                    else:
                        live = self.reconcile(*saved)
                        row.update(state="reconciled", name=saved[1]["name"], uid=live["uid"], platform_state=live["state"])
                else:
                    receipt = None
                    for attempt in range(3):
                        try:
                            receipt = self.create(spec)
                            break
                        except Exception as error:
                            # Only retry before an intent exists. An RPC error
                            # after creation is handled by exact-name reads.
                            saved = self.existing(spec)
                            if saved:
                                live = self.reconcile(*saved)
                                receipt = saved[1]
                                receipt["uid"] = live["uid"]
                                break
                            if not transient(error) or attempt == 2:
                                raise
                            self.sleep(min(2**attempt, max(0, self.deadline - self.clock())))
                    row.update(state="submitted", name=receipt["name"], uid=receipt["uid"], generation=receipt["generation"], platform_state=receipt.get("platform_state"))
            except DeadlineReached:
                return self.publish("deadline_reached", observations)
            except Exception as error:
                row.update(state="unknown" if transient(error) else "needs_attention", **error_record(error))
            observations.append(row)
            self.publish("submitting_extension", observations)
        complete = all(self.existing(spec) and self.existing(spec)[1].get("uid") for spec in self.new_specs)
        return self.publish("submitted" if complete else "submission_incomplete", observations)

    def original_specs(self):
        specs = []
        for path in sorted((self.original / "prepared").glob("*.json")):
            index = int(path.stem)
            require(0 <= index < 299, "Unexpected original shard")
            prepared = read(path)
            saved = highest_receipt(self.original, index, prepared)
            require(saved is not None, "Missing original submission receipt")
            receipt = saved[1]
            specs.append(dict(index=index, path=path, prepared=prepared, directory=self.original,
                              prefix=receipt["name"].rsplit("-g", 1)[0], object_prefix=receipt["object_prefix"]))
        require(len(specs) == 299, "Unified monitoring requires all 299 original shards")
        return specs

    def report(self, receipt):
        if not receipt.get("final_report_key"):
            return None
        try:
            result = read_body(self.client.get_object(Bucket=self.helper.BUCKET, Key=receipt["final_report_key"]))
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404"}:
                return None
            raise
        require(result["source_manifest_sha256"] == receipt["source_manifest_sha256"]
                and result["plan_fingerprint"] == receipt["plan_fingerprint"]
                and result["name"] == receipt["name"], "Object acceptance identity mismatch")
        require(result["artifact_acceptance_complete"] and result["object_checkpoints_complete"]
                and result["collection_summary"]["complete"]
                and result["collection_summary"]["completed_trials"] == receipt["expected_trials"], "Incomplete object acceptance")
        if result["collection_summary"]["accepted_trials"]:
            require(result["actual_training_reader"]["complete"] and result["actual_training_reader"]["actual_openpi_training_reader"], "Missing real training reader evidence")
        require(result["archive"]["object_key"] == receipt["archive_key"], "Archive belongs to another execution")
        head = self.client.head_object(Bucket=self.helper.BUCKET, Key=result["archive"]["object_key"])
        require(head["ContentLength"] == result["archive"]["bytes"] and head["ETag"] == result["archive"]["etag"], "Archive changed")
        return result

    def validate_saved_acceptance(self, report, receipt, *, gate=False):
        require(report["source_manifest_sha256"] == receipt["source_manifest_sha256"] == self.campaign["source_manifest_sha256"]
                and report["plan_fingerprint"] == receipt["plan_fingerprint"]
                and report["name"] == receipt["name"], "Saved acceptance belongs to another frozen execution")
        if not gate:
            require(receipt.get("uid") and report.get("job_uid") == receipt["uid"]
                    and report.get("platform_state") in {"SUCCEEDED", "FAILED", "SUSPENDED"}, "Saved acceptance lacks exact terminal platform evidence")
        require(report["artifact_acceptance_complete"] and report["object_checkpoints_complete"]
                and report["collection_summary"]["complete"]
                and report["collection_summary"]["completed_trials"] == receipt["expected_trials"], "Saved acceptance is incomplete")
        archive = report["archive"]
        require(archive["object_key"] == receipt["archive_key"] and archive["bytes"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]), "Saved archive identity is incomplete")
        ids = report["collection_summary"]["completed_trial_ids"]
        require(len(ids) == len(set(ids)) == receipt["expected_trials"], "Saved completed trial identities are incomplete")
        if report["collection_summary"]["accepted_trials"]:
            require(report["actual_training_reader"]["complete"] and report["actual_training_reader"]["actual_openpi_training_reader"]
                    and 0 < report["export"]["episodes"] <= report["collection_summary"]["accepted_trials"]
                    and report["export"]["new_frames"] > 0, "Saved acceptance lacks verified training episodes")

    def saved_acceptance(self, spec, receipt):
        candidates = [self.root / "accepted" / f"{spec['index']:03d}.json"]
        if spec["index"] < 299:
            candidates.append(self.original / f"accepted_{spec['index']:03d}.json")
        for path in candidates:
            if not path.exists():
                continue
            report = read(path)
            # Legacy files without exact terminal UID evidence are re-observed
            # from the object store and platform by monitor_once instead.
            if not report.get("job_uid") or not report.get("platform_state"):
                continue
            self.validate_saved_acceptance(report, receipt)
            return report
        return None

    def write_manifest(self):
        """Merge verified receipts locally; deduplicate whole archives only."""
        reports = []
        for spec in self.original_specs() + self.new_specs:
            saved = self.existing(spec)
            if saved:
                report = self.saved_acceptance(spec, saved[1])
                if report:
                    reports.append(("groups", report))
        gates = read(self.args.gates)
        require(gates["state"] == "complete" and gates["source_manifest_sha256"] == self.campaign["source_manifest_sha256"]
                and {gate["kind"] for gate in gates["gates"]} == {"native", "history", "additional"}, "Original standard gates are incomplete")
        for gate in gates["gates"]:
            report = read(Path(self.args.gates).parent / f"gate_{gate['kind']}_acceptance.json")
            require(gate.get("uid") and report["checkpoint_restore_verified"], "Original standard gate lacks verified restore evidence")
            self.validate_saved_acceptance(report, gate, gate=True)
            reports.append(("standard_gate_datasets", report))
        unique = {"groups": [], "standard_gate_datasets": []}
        archive_hashes, object_hashes, trial_ids, duplicates = {}, {}, set(), []
        for category, report in reports:
            archive = report["archive"]
            sha, key = archive["sha256"], archive["object_key"]
            require(key not in object_hashes or object_hashes[key] == sha, "One archive object key has conflicting content hashes")
            object_hashes[key] = sha
            if sha in archive_hashes:
                require(archive_hashes[sha]["archive"]["bytes"] == archive["bytes"], "One archive content hash has conflicting byte counts")
                duplicates.append(dict(name=report["name"], kept_name=archive_hashes[sha]["name"], archive_sha256=sha))
                continue
            ids = set(report["collection_summary"]["completed_trial_ids"])
            require(not ids & trial_ids, "Different archives reuse completed trial identities; manual dataset review required")
            trial_ids.update(ids)
            archive_hashes[sha] = report
            unique[category].append(report)
        groups, all_reports = unique["groups"], unique["groups"] + unique["standard_gate_datasets"]
        count = lambda rows, key: sum((report.get("export") or {}).get(key, 0) for report in rows)
        manifest = dict(kind="verified_low_subtask_success_training_collection", source_manifest_sha256=self.campaign["source_manifest_sha256"],
                        extension_campaign_fingerprint=self.campaign["fingerprint"], requested_groups=335,
                        original_requested_groups=299, extension_requested_shards=36, requested_unique_subtask_groups=299,
                        accepted_groups=len(groups), accepted_trials=sum(report["collection_summary"]["accepted_trials"] for report in groups),
                        completed_trials=sum(report["collection_summary"]["completed_trials"] for report in groups),
                        training_episodes=count(groups, "episodes"), training_frames=count(groups, "new_frames"),
                        total_training_episodes_including_standard_gates=count(all_reports, "episodes"),
                        total_training_frames_including_standard_gates=count(all_reports, "new_frames"),
                        deduplication=dict(basis="archive_sha256", cross_archive_episode_content_verified=False,
                                           duplicate_archives_omitted=duplicates), updated_at=self.clock(), **unique)
        write(self.root / "dataset_manifest.json", manifest)
        return manifest

    def monitor_once(self):
        from bounded_collection_admission import GPUAllowanceFull

        observations = []
        if self.admission is not None:
            self.admission_error = None
            try:
                self.admission.begin_pass()
            except Exception as error:
                self.admission_error = error_record(error)
        for spec in self.original_specs() + self.new_specs:
            row = {"index": spec["index"]}
            try:
                self.check_deadline()
                saved = self.existing(spec)
                if saved is None:
                    require(spec["index"] >= 299, "Original shard lost its persistent receipt")
                    receipt = self.create(spec)
                    row.update(state="submitted", name=receipt["name"], uid=receipt["uid"],
                               generation=receipt["generation"], platform_state=receipt.get("platform_state"))
                else:
                    path, receipt = saved
                    row.update(name=receipt["name"], uid=receipt.get("uid"), generation=receipt["generation"])
                    accepted_path = self.root / "accepted" / f"{spec['index']:03d}.json"
                    prior = self.saved_acceptance(spec, receipt)
                    if prior:
                        row.update(state="accepted", accepted_trials=prior["collection_summary"]["accepted_trials"],
                                   platform_state=prior["platform_state"])
                    else:
                        # Probe the final object first; no report is accepted
                        # until its exact job UID has an observed terminal state.
                        report = self.retry_read(lambda: self.report(receipt))
                        live = self.reconcile(path, receipt)
                        row["platform_state"] = live["state"]
                        if live["state"] in {"SUCCEEDED", "FAILED", "SUSPENDED"} and report:
                            report.update(job_uid=live["uid"], platform_state=live["state"], index=spec["index"])
                            write(accepted_path, report)
                            row.update(state="accepted", accepted_trials=report["collection_summary"]["accepted_trials"])
                        elif live["state"] in {"FAILED", "SUSPENDED"}:
                            if receipt["generation"] < 2:
                                next_receipt = self.create(spec, saved)
                                row.update(state="recovered", uid=next_receipt["uid"], name=next_receipt["name"], generation=next_receipt["generation"])
                            else:
                                row["state"] = "exhausted"
                        elif live["state"] == "SUCCEEDED":
                            row["state"] = "missing_acceptance"
                        else:
                            row["state"] = "active"
            except DeadlineReached:
                self.write_manifest()
                return self.publish("deadline_reached", observations)
            except GPUAllowanceFull:
                row["state"] = "waiting_for_gpu_allowance"
            except Exception as error:
                row.update(state="unknown" if transient(error) else "needs_attention", **error_record(error))
            observations.append(row)
        state = "complete" if all(r["state"] in {"accepted", "exhausted"} for r in observations) else "monitoring"
        self.write_manifest()
        return self.publish(state, observations)


def read_body(response):
    return json.loads(response["Body"].read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("campaign", "original-root", "output-dir", "gates", "submit-module-dir"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("name-prefix", "object-prefix"):
        parser.add_argument("--" + field, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--submit", action="store_true")
    modes.add_argument("--monitor", action="store_true")
    parser.add_argument("--once", action="store_true", help="One monitor pass, never extend its deadline")
    parser.add_argument("--cli", default="/private/tmp/behavior-sco-runtime/bin/sco")
    parser.add_argument("--aoss-client-module", default="/private/tmp/behavior_aoss_client.py")
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--admission-policy", type=Path, help="Explicit user reauthorization with a finite GPU limit")
    args = parser.parse_args()
    policy = None
    if args.admission_policy:
        from bounded_collection_admission import load_policy
        policy = load_policy(args.admission_policy, args.original_root, args.output_dir, read(args.campaign))
    require_automatic_collection_authorized(args.original_root, args.output_dir, authorization_policy=policy)
    require(1 <= args.poll_seconds <= 60, "Poll interval must be 1..60 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (args.original_root / ".dispatcher.lock", args.output_dir / "extension.lock"):
            stream = stack.enter_context(path.open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(policy is not None, "Automatic collection requires an explicit bounded admission policy")
        helper = load_helper(args.submit_module_dir, read(args.campaign)["deadline_epoch"],
                             cancellation_roots=(args.original_root, args.output_dir), authorization_policy=policy)
        client = None
        if args.monitor:
            spec = importlib.util.spec_from_file_location("extension_private_aoss", args.aoss_client_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            client = module.client()
        controller = Controller(args, helper, client=client)
        from bounded_collection_admission import GPUAdmission
        controller.authorization_policy = policy
        controller.admission = GPUAdmission(policy, args.output_dir, controller.original_specs() + controller.new_specs,
                                           helper, args.cli, controller.env)
        helper.subprocess.admission_guard = controller.admission.before_copy
        if args.submit:
            result = controller.submit_all()
        else:
            while True:
                result = controller.monitor_once()
                if args.once or result["state"] in {"complete", "deadline_reached"}:
                    break
                time.sleep(min(args.poll_seconds, max(0, controller.deadline - time.time())))
        print(json.dumps({k: result[k] for k in ("state", "submitted_new_shards", "deadline_epoch")}))


if __name__ == "__main__":
    main()
