"""Controller acceptance tests use synthetic bytes, never real pilot evidence."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import full_episode_acceptance as acceptance
from full_episode_collection import build_plan


def tar_bytes(raw, result, *, extra=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, body in (("episode.hdf5", raw), ("result.json", json.dumps(result, sort_keys=True).encode())):
            info = tarfile.TarInfo(name); info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        if extra:
            info = tarfile.TarInfo("../escape"); info.size = 4
            archive.addfile(info, io.BytesIO(b"fake"))
    return output.getvalue()


def archive_receipt(key, body):
    etag = hashlib.md5(hashlib.md5(body).digest()).hexdigest() + "-1"
    return dict(object_key=key, bytes=len(body), sha256=acceptance.digest(body), etag='"' + etag + '"',
                parts=1, part_content_md5_verified=True)


class Client:
    def __init__(self, objects, archive_key):
        self.objects, self.archive_key = objects, archive_key
        self.streams, self.read_sizes = [], []

    def get_object(self, *, Bucket, Key):
        assert Bucket == "260010165"
        client = self

        class Body(io.BytesIO):
            def read(self, size=-1):
                if Key == client.archive_key:
                    client.read_sizes.append(size)
                    assert size != -1, "Archive must be streamed, never read whole"
                return super().read(size)

        body = Body(self.objects[Key])
        self.streams.append(body)
        return dict(Body=body, ContentLength=len(self.objects[Key]))

    def head_object(self, *, Bucket, Key):
        assert Bucket == "260010165"
        body = self.objects[Key]
        value = archive_receipt(Key, body)
        return dict(ContentLength=len(body), ETag=value["etag"])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    tasks = ["turning_on_radio"] + [f"task_{index:03d}" for index in range(99)]
    ready = dict(state="ready", checkpoint_step=1688977, checkpoint="/immutable/checkpoint", code="/immutable/code",
                 source_manifest_sha256="a" * 64, scope=dict(tasks=tasks,
                     checkpoint_files=[dict(relative_path="params/a", bytes=42)]))
    plan = build_plan(ready, {task: task for task in tasks}, [["turning_on_radio", 123]], run_id="fixture-full-ep")
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    prefix = f"behavior1k/full-episode-collection/{plan['run_id']}/{plan['fingerprint'][:12]}"
    completion_key = prefix + "/shards/0000.json"
    raw = b"\x89HDF\r\n\x1a\n" + b"synthetic HDF5-shaped fixture; not simulator data" * 400
    result = dict(task="turning_on_radio", instance_id=301, rollout_id=0, steps=70, success=True,
                  task_success=True, final_goal_state={"success": True}, terminated=True, truncated=False,
                  official_benchmark=False, success_source="environment_task_and_final_goal",
                  conditioning=dict(mode="whole_task", oracle=False, privileged=False))
    archive_key = prefix + "/episodes/turning_on_radio/301/attempt-1.tar"
    body = tar_bytes(raw, result)
    receipt = dict(kind="native_full_episode_object_checkpoint", run_id=plan["run_id"] + "-a0000",
                   source_manifest_sha256="b" * 64, attempt_id="turning_on_radio_301/attempt-1/rollout-0",
                   task="turning_on_radio", instance_id=301, rollout_id=0, steps=70, success=True,
                   trajectory_sha256=acceptance.digest(raw), content_sha256="c" * 64, subtask_sha256="d" * 64,
                   result=result, archive=archive_receipt(archive_key, body))
    codec = dict(kind="native_full_episode_real_codec_probe", real_lerobot_reader=True,
                 native_frame_action_alignment_verified=True, decoded_frames=64, source_steps=70,
                 source_trajectory_sha256=receipt["trajectory_sha256"], source_content_sha256="c" * 64,
                 source_success=True, maximum_image_mae=1.5, fps=30, ready_for_training=False,
                 format_probe_only=True, official_benchmark=False)
    completion = dict(kind="native_full_episode_collection_complete", complete=True, official_benchmark=False,
                      run_id=plan["run_id"] + "-a0000", checkpoint=plan["checkpoint_identity"],
                      source_manifest_sha256="b" * 64, episodes=1, frames=70, successful_episodes=1,
                      object_checkpoints=[receipt], real_codec_reader_verified=True,
                      codec_verification=codec, ready_for_training=False)
    submission = dict(name="behavior-full-ep-fixture-a0000-g0", uid="actual-pilot-uid", index=0,
                      plan_fingerprint=plan["fingerprint"], task="turning_on_radio", instance_ids=[301],
                      completion_key=completion_key)
    (tmp_path / "submissions").mkdir()
    (tmp_path / "submissions/0000.json").write_text(json.dumps(submission))
    receipt_key = prefix + "/episodes/turning_on_radio/301/attempt-1.json"
    objects = {archive_key: body, completion_key: json.dumps(completion).encode(),
               receipt_key: json.dumps(receipt).encode()}
    client = Client(objects, archive_key)
    job = dict(uid="actual-pilot-uid", state="SUCCEEDED")
    monkeypatch.setattr(acceptance, "describe", lambda *args: job)
    return dict(root=tmp_path, plan=plan, submission=submission, completion=completion, receipt=receipt,
                codec=codec, client=client, job=job, archive_key=archive_key, completion_key=completion_key,
                receipt_key=receipt_key, raw=raw)


def sync(value, *, include_receipt=True):
    value["client"].objects[value["completion_key"]] = json.dumps(value["completion"]).encode()
    if include_receipt:
        value["client"].objects[value["receipt_key"]] = json.dumps(value["receipt"]).encode()


def replace_archive(value, body):
    value["client"].objects[value["archive_key"]] = body
    value["receipt"]["archive"] = archive_receipt(value["archive_key"], body)
    sync(value)


def test_exact_pilot_streams_whole_archive_raw_and_result_then_binds_local_gate(fixture):
    value = fixture
    gate = acceptance.accept_pilot(value["root"], "fixture-cli", value["client"])
    assert gate["pilot_uid"] == "actual-pilot-uid"
    assert gate["full_episode_raw_verified"] and gate["real_codec_reader_verified"]
    assert gate["archive_proof"]["trajectory_sha256"] == acceptance.digest(value["raw"])
    assert acceptance.validate_gate(value["root"], value["plan"]) == gate
    assert value["client"].read_sizes and all(size > 0 for size in value["client"].read_sizes)
    assert all(stream.closed for stream in value["client"].streams)
    for file in value["root"].glob("pilot_*.json"):
        assert "https://" not in file.read_text()
    assert acceptance.accept_pilot(value["root"], "fixture-cli", value["client"]) == gate


def test_real_format_pilot_may_be_a_completed_policy_failure(fixture):
    value = fixture
    result = value["receipt"]["result"]
    result.update(success=False, task_success=False, final_goal_state={"success": False},
                  terminated=False, truncated=True)
    value["receipt"]["success"] = False
    value["codec"]["source_success"] = False
    value["completion"]["successful_episodes"] = 0
    replace_archive(value, tar_bytes(value["raw"], result))
    gate = acceptance.accept_pilot(value["root"], "fixture-cli", value["client"])
    assert gate["success_required"] is False


@pytest.mark.parametrize("field,change", [("state", "RUNNING"), ("state", "FAILED"), ("uid", "wrong-uid")])
def test_platform_must_match_exact_successful_uid(fixture, field, change):
    fixture["job"][field] = change
    with pytest.raises(ValueError, match="Exact pilot"):
        acceptance.accept_pilot(fixture["root"], "cli", fixture["client"])
    assert not fixture["client"].streams


@pytest.mark.parametrize("damage", ["run_id", "checkpoint", "episodes", "frames", "task", "result_success",
                                    "source_receipt", "raw_sha", "archive_key"])
def test_completion_identity_outcome_and_scope_cannot_change(fixture, damage):
    value = fixture
    if damage == "run_id": value["completion"]["run_id"] = "other-run"
    elif damage == "checkpoint": value["completion"]["checkpoint"] = {"checkpoint_step": 1375000}
    elif damage == "episodes": value["completion"]["episodes"] = 2
    elif damage == "frames": value["completion"]["frames"] = 71
    elif damage == "task": value["receipt"]["task"] = "different_task"
    elif damage == "result_success": value["receipt"]["result"]["task_success"] = False
    elif damage == "source_receipt": value["receipt"]["source_manifest_sha256"] = "e" * 64
    elif damage == "raw_sha": value["receipt"]["trajectory_sha256"] = "f" * 64
    elif damage == "archive_key": value["receipt"]["archive"]["object_key"] = "outside/other.tar"
    sync(value)
    with pytest.raises(ValueError):
        acceptance.accept_pilot(value["root"], "cli", value["client"])
    assert not (value["root"] / "pilot_accepted.json").exists()


@pytest.mark.parametrize("field,change", [("real_lerobot_reader", False),
                                          ("native_frame_action_alignment_verified", False),
                                          ("source_content_sha256", "a" * 64),
                                          ("source_trajectory_sha256", "a" * 64),
                                          ("source_steps", 69), ("decoded_frames", 0),
                                          ("kind", "synthetic_probe")])
def test_codec_proof_must_bind_actual_pilot_bytes(fixture, field, change):
    fixture["codec"][field] = change
    sync(fixture)
    with pytest.raises(ValueError, match="codec/frame-action"):
        acceptance.accept_pilot(fixture["root"], "cli", fixture["client"])


def test_remote_receipt_must_equal_the_completion_record(fixture):
    fixture["receipt"]["attempt_id"] = "other-attempt"
    sync(fixture, include_receipt=False)
    with pytest.raises(ValueError, match="Object receipt differs"):
        acceptance.accept_pilot(fixture["root"], "cli", fixture["client"])


@pytest.mark.parametrize("damage", ["trailing_bytes", "raw_member", "result_member", "unexpected_member", "missing_result"])
def test_archive_sha_whole_raw_and_result_are_independently_checked(fixture, damage):
    value = fixture
    if damage == "trailing_bytes":
        body = bytearray(value["client"].objects[value["archive_key"]]); body[-1] = 1
        value["client"].objects[value["archive_key"]] = bytes(body)
        # HEAD ETag changes too, but keep that header consistent to reach SHA verification.
        value["receipt"]["archive"]["etag"] = archive_receipt(value["archive_key"], bytes(body))["etag"]
        sync(value)
    elif damage == "raw_member":
        replace_archive(value, tar_bytes(value["raw"] + b"tampered", value["receipt"]["result"]))
    elif damage == "result_member":
        changed = copy.deepcopy(value["receipt"]["result"]); changed["steps"] += 1
        replace_archive(value, tar_bytes(value["raw"], changed))
    elif damage == "unexpected_member":
        replace_archive(value, tar_bytes(value["raw"], value["receipt"]["result"], extra=True))
    else:
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            member = tarfile.TarInfo("episode.hdf5"); member.size = len(value["raw"])
            archive.addfile(member, io.BytesIO(value["raw"]))
        replace_archive(value, output.getvalue())
    with pytest.raises(ValueError):
        acceptance.accept_pilot(value["root"], "cli", value["client"])
    assert not (value["root"] / "pilot_accepted.json").exists()
    assert all(stream.closed for stream in value["client"].streams)


@pytest.mark.parametrize("file", ["pilot_completion.json", "pilot_codec_verification.json",
                                 "pilot_archive_verification.json", "pilot_accepted.json", "submissions/0000.json"])
def test_local_gate_revalidates_each_saved_evidence_and_exact_submission(fixture, file):
    value = fixture
    acceptance.accept_pilot(value["root"], "cli", value["client"])
    path = value["root"] / file
    saved = json.loads(path.read_text())
    if file.endswith("0000.json"): saved["uid"] = "replaced-uid"
    elif file == "pilot_accepted.json": saved["trajectory_sha256"] = "a" * 64
    else: saved["tampered"] = True
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        acceptance.validate_gate(value["root"], value["plan"])
