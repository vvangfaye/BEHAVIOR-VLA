"""CPU-only contract tests: no SCO executable, credentials, or object writes."""
from io import BytesIO
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import continue_collection_extension as dispatcher


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt(prepared, name, prefix, generation=0, uid=True):
    return dict(name=name, generation=generation, state="submitted" if uid else "submission_unknown",
                source_manifest_sha256=prepared["source_manifest_sha256"],
                plan_fingerprint=prepared["plan"]["fingerprint"], expected_trials=prepared["plan"]["expected_trials"],
                quota_type="spot", gpu_count=1, timeout_hours=24, object_prefix=prefix,
                **({"uid": "uid-" + name, "platform_state": "STARTING"} if uid else {}))


def acceptance(item, trial_ids):
    return dict(name=item["name"], job_uid=item["uid"], platform_state="SUCCEEDED",
                source_manifest_sha256=item["source_manifest_sha256"], plan_fingerprint=item["plan_fingerprint"],
                artifact_acceptance_complete=True, object_checkpoints_complete=True, checkpoint_restore_verified=True,
                collection_summary=dict(complete=True, completed_trials=item["expected_trials"], accepted_trials=1,
                                        completed_trial_ids=trial_ids),
                actual_training_reader=dict(complete=True, actual_openpi_training_reader=True),
                export=dict(episodes=1, new_frames=100),
                archive=dict(object_key=item["archive_key"], bytes=123, etag="etag", sha256=hashlib.sha256(item["name"].encode()).hexdigest()))


def seed_new_receipts(controller):
    for spec in controller.new_specs:
        save(spec["directory"] / f"spot_{spec['index']:03d}_g0.json",
             receipt(spec["prepared"], spec["prefix"] + "-g0", spec["object_prefix"]))


class FakeHelper:
    BUCKET = "owner"

    def __init__(self):
        self.created = []
        self.queries = []
        self.fail_reads = set()
        self.fail_before_intent = {}
        self.fail_after_intent = set()
        self.states = {}

    def environment(self):
        return {}

    def describe(self, cli, env, name):
        self.queries.append(name)
        if name in self.fail_reads:
            raise subprocess.CalledProcessError(65, ["redacted"])
        return dict(uid="uid-" + name, state=self.states.get(name, "STARTING"),
                    ownership=dict(user_name=self.BUCKET), scheduling=dict(quota_type="SPOT"))

    def submit(self, args):
        self.created.append(args)
        remaining = self.fail_before_intent.get(args.name, 0)
        if remaining:
            self.fail_before_intent[args.name] -= 1
            raise subprocess.CalledProcessError(65, ["redacted"])
        prepared = dispatcher.read(args.prepared)
        generation = 0 if args.recover_receipt is None else dispatcher.read(args.recover_receipt)["generation"] + 1
        item = receipt(prepared, args.name, args.object_prefix, generation, uid=args.name not in self.fail_after_intent)
        save(Path(args.receipt), item)
        if args.name in self.fail_after_intent:
            raise subprocess.CalledProcessError(65, ["redacted"])
        return item


@pytest.fixture
def setup(tmp_path):
    original = tmp_path / "original"
    root = tmp_path / "extension"
    save(original / "status.json", dict(started_at=1000, source_manifest_sha256="source"))
    for index in range(299):
        prepared = dict(state="complete", source_manifest_sha256="source",
                        plan=dict(fingerprint=f"old-plan-{index}", expected_trials=1,
                                  trials=[dict(trial_id=f"old-{index}")]))
        save(original / "prepared" / f"{index:03d}.json", prepared)
        save(original / f"spot_{index:03d}_g0.json", receipt(prepared, f"old-a{index:03d}-g0", f"old/shard-{index}"))
    groups = []
    for index in range(299, 335):
        entry = dict(index=index, state="complete", source_manifest_sha256="source",
                     plan=dict(fingerprint=f"new-plan-{index}", expected_trials=30,
                               trials=[dict(trial_id=f"new-{index}-{trial}") for trial in range(30)]))
        groups.append(entry)
        save(root / "groups" / f"{index:03d}" / "prepared.json", entry)
    campaign = dict(schema_version=1, kind="low_subtask_collection_extension", state="complete",
                    source_manifest_sha256="source", deadline_epoch=1000 + 144 * 3600,
                    shard_count=36, expected_trials=1080, groups=groups)
    campaign["fingerprint"] = dispatcher.digest(campaign)
    save(root / "campaign.json", campaign)
    gates = []
    for kind in ("native", "history", "additional"):
        gate = receipt(dict(source_manifest_sha256="source", plan=dict(fingerprint=kind, expected_trials=1)),
                       "gate-" + kind, "gates/" + kind)
        gate.update(kind=kind, archive_key="gate-archive-" + kind)
        gates.append(gate)
        save(original / f"gate_{kind}_acceptance.json", acceptance(gate, ["gate-trial-" + kind]))
    save(original / "standard_gates.json", dict(state="complete", source_manifest_sha256="source", gates=gates))
    args = SimpleNamespace(campaign=root / "campaign.json", original_root=original, output_dir=root,
                           name_prefix="extension", object_prefix="behavior1k/subtask-collection/extension",
                           cli="no-cloud", gates=original / "standard_gates.json", aoss_client_module="unused")
    helper = FakeHelper()
    controller = dispatcher.Controller(args, helper, clock=lambda: 2000, sleeper=lambda delay: None)
    return controller, helper, args


def by_index(result, index):
    return next(row for row in result["observations"] if row["index"] == index)


def test_submit_36_then_repeat_is_idempotent(setup):
    controller, helper, _ = setup
    first = controller.submit_all()
    assert first["state"] == "submitted" and first["submitted_new_shards"] == 36
    assert len(helper.created) == 36
    assert all(row["uid"] and row["platform_state"] == "STARTING" for row in first["observations"])
    assert all(call.timeout_hours == 24 and call.quota_type == "spot" and call.recover_receipt is None for call in helper.created)
    assert controller.submit_all()["state"] == "submitted"
    assert len(helper.created) == 36


def test_unknown_submission_continues_other_shards_and_reconciles_without_duplicate(setup):
    controller, helper, _ = setup
    name = "extension-a299-g0"
    helper.fail_after_intent.add(name)
    helper.fail_reads.add(name)
    result = controller.submit_all()
    assert result["state"] == "submission_incomplete" and result["submitted_new_shards"] == 35
    assert by_index(result, 299)["state"] == "unknown"
    assert len(helper.created) == 36
    helper.fail_reads.clear()
    result = controller.submit_all()
    assert result["state"] == "submitted" and by_index(result, 299)["state"] == "reconciled"
    assert len(helper.created) == 36


def test_transient_pre_intent_read_can_retry(setup):
    controller, helper, _ = setup
    helper.fail_before_intent["extension-a299-g0"] = 2
    assert controller.submit_all()["submitted_new_shards"] == 36
    assert len(helper.created) == 38


def test_monitor_uses_latest_generation_and_recovers_only_terminal(setup):
    controller, helper, args = setup
    seed_new_receipts(controller)
    path = args.original_root / "prepared/000.json"
    prepared = dispatcher.read(path)
    save(args.original_root / "spot_000_g1.json", receipt(prepared, "old-a000-g1", "old/shard-0", 1))
    helper.states.update({"old-a000-g0": "FAILED", "old-a000-g1": "RUNNING", "old-a001-g0": "FAILED",
                          "old-a002-g0": "SUCCEEDED", "old-a004-g0": "SUSPENDED"})
    helper.fail_reads.add("old-a003-g0")
    result = controller.monitor_once()
    assert "old-a000-g0" not in helper.queries
    assert by_index(result, 0)["state"] == "active" and by_index(result, 0)["generation"] == 1
    assert by_index(result, 1)["state"] == by_index(result, 4)["state"] == "recovered"
    assert by_index(result, 2)["state"] == "missing_acceptance"
    assert by_index(result, 3)["state"] == "unknown"
    assert len(helper.created) == 2
    assert {Path(call.recover_receipt).name for call in helper.created} == {"spot_001_g0.json", "spot_004_g0.json"}


def test_never_more_than_three_generations(setup):
    controller, helper, args = setup
    seed_new_receipts(controller)
    prepared = dispatcher.read(args.original_root / "prepared/000.json")
    for generation in (1, 2):
        save(args.original_root / f"spot_000_g{generation}.json",
             receipt(prepared, f"old-a000-g{generation}", "old/shard-0", generation))
    helper.states["old-a000-g2"] = "FAILED"
    result = controller.monitor_once()
    assert by_index(result, 0)["state"] == "exhausted"
    assert not helper.created


def test_report_requires_terminal_exact_platform_uid(setup):
    controller, helper, args = setup
    seed_new_receipts(controller)
    item_path = args.original_root / "spot_000_g0.json"
    item = dispatcher.read(item_path)
    item.update(final_report_key="report", archive_key="archive")
    save(item_path, item)
    report = acceptance(item, ["old-0"])
    controller.client = SimpleNamespace(get_object=lambda **kw: {"Body": BytesIO(json.dumps(report).encode())},
                                        head_object=lambda **kw: dict(ContentLength=123, ETag="etag"))
    assert by_index(controller.monitor_once(), 0)["state"] == "active"
    assert not (args.output_dir / "accepted/000.json").exists()
    helper.states[item["name"]] = "SUCCEEDED"
    assert by_index(controller.monitor_once(), 0)["state"] == "accepted"
    assert dispatcher.read(args.output_dir / "accepted/000.json")["job_uid"] == item["uid"]
    helper.fail_reads.add(item["name"])
    assert by_index(controller.monitor_once(), 0)["state"] == "accepted"


def test_invalid_archive_or_report_cannot_trigger_recovery(setup):
    controller, helper, args = setup
    seed_new_receipts(controller)
    item_path = args.original_root / "spot_000_g0.json"
    item = dispatcher.read(item_path)
    item.update(final_report_key="report", archive_key="archive")
    save(item_path, item)
    helper.states[item["name"]] = "FAILED"
    controller.client = SimpleNamespace(get_object=lambda **kw: {"Body": BytesIO(b'{"plan_fingerprint":"foreign"}')})
    assert by_index(controller.monitor_once(), 0)["state"] == "needs_attention"
    assert not helper.created


def test_deadline_prevents_submit_and_actual_helper_copy(setup, monkeypatch):
    controller, helper, _ = setup
    controller.clock = lambda: controller.deadline
    assert controller.submit_all()["state"] == "deadline_reached"
    assert not helper.created
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    proxy = dispatcher.DeadlineSubprocess(10, clock=lambda: 10)
    with pytest.raises(dispatcher.DeadlineReached):
        proxy.run(["sco", "acp", "jobs", "copy", "--name=bound"])
    assert not calls
    proxy.run(["sco", "acp", "jobs", "describe", "bound"])
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["fingerprint", "source", "deadline", "collision", "missing_original"])
def test_changed_scope_is_rejected_before_submission(setup, mutation):
    _, helper, args = setup
    campaign = dispatcher.read(args.campaign)
    if mutation == "fingerprint":
        campaign["fingerprint"] = "changed"
    elif mutation == "source":
        campaign["source_manifest_sha256"] = "changed"
    elif mutation == "deadline":
        campaign["deadline_epoch"] += 1
    elif mutation == "collision":
        campaign["groups"][0]["plan"]["trials"][0]["trial_id"] = "old-0"
        save(args.output_dir / "groups/299/prepared.json", campaign["groups"][0])
    else:
        (args.original_root / "prepared/000.json").unlink()
    if mutation != "fingerprint":
        campaign["fingerprint"] = dispatcher.digest({k: v for k, v in campaign.items() if k != "fingerprint"})
    save(args.campaign, campaign)
    with pytest.raises(ValueError):
        dispatcher.Controller(args, helper, clock=lambda: 2000)
    assert not helper.created


def test_receipt_namespace_or_skipped_generation_is_rejected(setup):
    controller, _, args = setup
    entry = controller.new_specs[0]["prepared"]
    save(args.output_dir / "spot_299_g1.json", receipt(entry, "extension-a299-g1", controller.new_specs[0]["object_prefix"], 1))
    with pytest.raises(ValueError, match="Missing prior"):
        controller.existing(controller.new_specs[0])
    (args.output_dir / "spot_299_g1.json").unlink()
    save(args.output_dir / "spot_299_g0.json", receipt(entry, "wrong-g0", "wrong"))
    with pytest.raises(ValueError, match="namespace"):
        controller.existing(controller.new_specs[0])


@pytest.mark.parametrize("lock_name", ["original", "extension"])
def test_shared_writer_lock_blocks_before_loading_cloud_helper(setup, monkeypatch, lock_name):
    _, helper, args = setup
    argv = ["dispatcher", "--submit", "--campaign", str(args.campaign), "--original-root", str(args.original_root),
            "--output-dir", str(args.output_dir), "--gates", str(args.gates), "--submit-module-dir", "unused",
            "--name-prefix", args.name_prefix, "--object-prefix", args.object_prefix]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(dispatcher, "load_helper", lambda *args: pytest.fail("Cloud helper must not load without both locks"))
    path = args.original_root / ".dispatcher.lock" if lock_name == "original" else args.output_dir / "extension.lock"
    with path.open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            dispatcher.main()
    assert not helper.created


def test_monitor_submits_only_missing_extension_intents_and_reconciles_unknown(setup):
    controller, helper, _ = setup
    name = "extension-a299-g0"
    helper.fail_after_intent.add(name)
    helper.fail_reads.add(name)
    result = controller.monitor_once()
    assert len(helper.created) == 36
    assert by_index(result, 299)["state"] == "unknown"
    assert by_index(result, 300)["state"] == "submitted"
    helper.fail_reads.clear()
    result = controller.monitor_once()
    assert len(helper.created) == 36
    assert by_index(result, 299)["state"] == "active"
    assert result["submitted_new_shards"] == 36


def test_combined_manifest_contains_old_new_and_gates_without_modifying_original(setup):
    controller, _, args = setup
    seed_new_receipts(controller)
    old_manifest = args.original_root / "dataset_manifest.json"
    save(old_manifest, {"original": "unchanged"})
    for index, directory, destination in [(0, args.original_root, args.original_root / "accepted_000.json"),
                                          (299, args.output_dir, args.output_dir / "accepted/299.json")]:
        path = directory / f"spot_{index:03d}_g0.json"
        item = dispatcher.read(path)
        item["archive_key"] = f"archive-{index}"
        save(path, item)
        ids = ["old-0"] if index == 0 else [f"new-299-{trial}" for trial in range(30)]
        save(destination, acceptance(item, ids))
    for _ in range(2):
        manifest = controller.write_manifest()
        assert manifest["requested_groups"] == 335
        assert manifest["accepted_groups"] == 2
        assert manifest["training_episodes"] == 2 and manifest["training_frames"] == 200
        assert len(manifest["standard_gate_datasets"]) == 3
        assert manifest["total_training_episodes_including_standard_gates"] == 5
        assert manifest["total_training_frames_including_standard_gates"] == 500
        assert manifest["deduplication"]["cross_archive_episode_content_verified"] is False
    assert dispatcher.read(old_manifest) == {"original": "unchanged"}


def test_combined_manifest_deduplicates_same_archive_at_different_keys(setup):
    controller, _, args = setup
    native = dispatcher.read(args.original_root / "gate_native_acceptance.json")
    path = args.original_root / "gate_history_acceptance.json"
    history = dispatcher.read(path)
    history["archive"]["sha256"] = native["archive"]["sha256"]
    save(path, history)
    manifest = controller.write_manifest()
    assert len(manifest["standard_gate_datasets"]) == 2
    assert manifest["total_training_episodes_including_standard_gates"] == 2
    assert len(manifest["deduplication"]["duplicate_archives_omitted"]) == 1


def test_distinct_archives_with_reused_trial_ids_are_not_silently_combined(setup):
    controller, _, args = setup
    path = args.original_root / "gate_history_acceptance.json"
    history = dispatcher.read(path)
    history["collection_summary"]["completed_trial_ids"] = ["gate-trial-native"]
    save(path, history)
    with pytest.raises(ValueError, match="reuse completed trial"):
        controller.write_manifest()


def test_legacy_acceptance_without_terminal_uid_is_not_inherited(setup):
    controller, _, args = setup
    item = dispatcher.read(args.original_root / "spot_000_g0.json")
    item["archive_key"] = "old-archive"
    old = acceptance(item, ["old-0"])
    del old["job_uid"]
    save(args.original_root / "accepted_000.json", old)
    assert controller.write_manifest()["accepted_groups"] == 0


def test_accepted_attempts_can_exceed_deduplicated_exported_episodes(setup):
    controller, _, args = setup
    seed_new_receipts(controller)
    path = args.output_dir / "spot_299_g0.json"
    item = dispatcher.read(path)
    item["archive_key"] = "deduplicated-new-archive"
    save(path, item)
    report = acceptance(item, [f"new-299-{trial}" for trial in range(30)])
    report["collection_summary"]["accepted_trials"] = 2
    save(args.output_dir / "accepted/299.json", report)
    manifest = controller.write_manifest()
    assert manifest["accepted_trials"] == 2 and manifest["training_episodes"] == 1


@pytest.mark.parametrize("mode", ["--submit", "--monitor"])
@pytest.mark.parametrize("marker_root", ["original_root", "output_dir"])
def test_user_cancel_marker_blocks_entry_before_cloud_helper(setup, monkeypatch, mode, marker_root):
    _, helper, args = setup
    save(getattr(args, marker_root) / "USER_CANCEL_NONRUNNING.json", {"user_cancelled": True})
    monkeypatch.setattr(sys, "argv", ["dispatcher", mode, "--campaign", str(args.campaign),
        "--original-root", str(args.original_root), "--output-dir", str(args.output_dir),
        "--gates", str(args.gates), "--submit-module-dir", "unused",
        "--name-prefix", args.name_prefix, "--object-prefix", args.object_prefix])
    monkeypatch.setattr(dispatcher, "load_helper", lambda *a, **kw: pytest.fail("Cancelled collection must not load cloud helper"))
    with pytest.raises(dispatcher.UserCancelledAutomaticCollection, match="cancelled by the user"):
        dispatcher.main()
    assert not helper.created and not helper.queries


@pytest.mark.parametrize("marker_root", ["original_root", "output_dir"])
def test_user_cancel_after_start_blocks_initial_recovery_and_actual_copy(setup, monkeypatch, marker_root):
    controller, helper, args = setup
    spec = controller.new_specs[0]
    prior_path = args.output_dir / "spot_299_g0.json"
    prior_receipt = receipt(spec["prepared"], spec["prefix"] + "-g0", spec["object_prefix"])
    save(prior_path, prior_receipt)
    save(getattr(args, marker_root) / "USER_CANCEL_NONRUNNING.json", {})
    for previous in (None, (prior_path, prior_receipt)):
        with pytest.raises(dispatcher.UserCancelledAutomaticCollection):
            controller.create(spec, previous)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Cancelled collection must not create cloud jobs"))
    proxy = dispatcher.DeadlineSubprocess(controller.deadline, clock=lambda: 2000,
                                         cancellation_roots=(args.original_root, args.output_dir))
    with pytest.raises(dispatcher.UserCancelledAutomaticCollection):
        proxy.run(["sco", "acp", "jobs", "copy", "--name=cancelled"])
    assert not helper.created


def test_user_cancel_blocks_legacy_entry_before_reading_campaign(tmp_path, monkeypatch):
    import continue_full_subtask_collection as legacy

    root = tmp_path / "original"
    save(root / "USER_CANCEL_NONRUNNING.json", {})
    monkeypatch.setattr(sys, "argv", ["legacy", "--output-dir", str(root), "--campaign", "missing-campaign",
                                     "--gate-receipts", "missing-gates", "--aoss-client-module", "missing-client"])
    with pytest.raises(RuntimeError, match="cancelled by the user"):
        legacy.main()
    assert not (root / ".dispatcher.lock").exists()


def test_bounded_monitor_keeps_reading_but_defers_recovery_and_initial_jobs(setup):
    from bounded_collection_admission import GPUAllowanceFull
    controller, helper, args = setup
    helper.states["old-a000-g0"] = "SUSPENDED"
    attempts = []
    class FullAdmission:
        limit = 64
        last = {"occupied_gpus": 176}
        def begin_pass(self):
            pass
        def reserve(self, name):
            attempts.append(name)
            raise GPUAllowanceFull("full")
    controller.admission = FullAdmission()
    result = controller.monitor_once()
    assert result["max_active_gpus"] == 64
    assert by_index(result, 0)["state"] == "waiting_for_gpu_allowance"
    assert by_index(result, 1)["state"] == "active"
    assert by_index(result, 299)["state"] == "waiting_for_gpu_allowance"
    assert "old-a000-g1" in attempts and "extension-a299-g0" in attempts
    assert not helper.created and helper.queries


def test_reauthorized_policy_must_match_current_cancellation_marker(setup, monkeypatch):
    controller, helper, args = setup
    save(args.original_root / "USER_CANCEL_NONRUNNING.json", {"request": "old cancellation"})
    policy = dict(kind="bounded_subtask_collection_authorization", max_active_gpus=64,
                  cancelled_markers_sha256={str(args.original_root.resolve()): hashlib.sha256(
                      (args.original_root / "USER_CANCEL_NONRUNNING.json").read_bytes()).hexdigest()})
    dispatcher.require_automatic_collection_authorized(args.original_root, args.output_dir, authorization_policy=policy)
    proxy = dispatcher.DeadlineSubprocess(controller.deadline, clock=lambda: 2000,
        cancellation_roots=(args.original_root, args.output_dir), authorization_policy=policy)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("No copy without accounting"))
    with pytest.raises(ValueError, match="admission guard"):
        proxy.run(["sco", "acp", "jobs", "copy", "--name=bounded"])
    save(args.original_root / "USER_CANCEL_NONRUNNING.json", {"request": "new stop request"})
    with pytest.raises(dispatcher.UserCancelledAutomaticCollection):
        dispatcher.require_automatic_collection_authorized(args.original_root, authorization_policy=policy)


def test_pre_intent_failure_releases_slot_but_unknown_intent_keeps_it(setup):
    controller, helper, _ = setup
    reserved, released = [], []
    controller.admission = SimpleNamespace(reserve=reserved.append, release_before_intent=released.append)
    spec = controller.new_specs[0]
    name = spec["prefix"] + "-g0"
    helper.fail_before_intent[name] = 1
    with pytest.raises(subprocess.CalledProcessError):
        controller.create(spec)
    assert reserved == released == [name]
    helper.fail_after_intent.add(name)
    with pytest.raises(subprocess.CalledProcessError):
        controller.create(spec)
    assert reserved == [name, name] and released == [name]
