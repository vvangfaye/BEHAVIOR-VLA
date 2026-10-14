"""Accept the one real episode/codec pilot before enabling production shards.

This controller-side verifier uses only the standard library and the caller's
object-store client. Large archives are hashed and inspected as streams; signed
transport URLs are neither required nor written into the acceptance evidence.
"""
import hashlib
import json
from pathlib import Path
import tarfile

from submit_full_collection_shard import describe, environment

BUCKET = "260010165"
GATE_KIND = "native_full_episode_pilot_acceptance"
COMPLETION_FILE = "pilot_completion.json"
CODEC_FILE = "pilot_codec_verification.json"
ARCHIVE_FILE = "pilot_archive_verification.json"
CHUNK_BYTES = 8 * 1024**2


def digest(body):
    return hashlib.sha256(body).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_value(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _load_inputs(root, plan=None):
    root = Path(root)
    saved_plan = json.loads((root / "plan.json").read_bytes())
    if plan is not None:
        ensure(plan == saved_plan, "Submission plan differs from the frozen local plan")
    plan = saved_plan
    computed = digest(json.dumps({key: value for key, value in plan.items() if key != "fingerprint"},
                                 sort_keys=True).encode())
    ensure(plan.get("kind") == "native_full_episode_collection_plan" and plan.get("fingerprint") == computed,
           "Frozen full-episode plan fingerprint differs")
    if plan.get("expected_episodes") == 1000:
        from full_episode_scope import validate_final1000
        validate_final1000(plan)
    ensure(plan.get("expected_episodes") in (1000, 2000) and plan.get("official_benchmark") is False,
           "Unexpected collection scope or benchmark status")
    ensure(plan["shards"][0] == dict(index=0, task="turning_on_radio", instance_ids=[301], pilot=True),
           "Pilot scope differs from radio instance 301")
    submission = json.loads((root / "submissions/0000.json").read_bytes())
    ensure(submission.get("index") == 0 and submission.get("plan_fingerprint") == plan["fingerprint"]
           and submission.get("task") == "turning_on_radio" and submission.get("instance_ids") == [301]
           and isinstance(submission.get("uid"), str) and bool(submission["uid"])
           and isinstance(submission.get("name"), str) and bool(submission["name"]),
           "Pilot submission identity is incomplete or differs from the plan")
    prefix = f"behavior1k/full-episode-collection/{plan['run_id']}/{plan['fingerprint'][:12]}"
    ensure(submission.get("completion_key") == prefix + "/shards/0000.json",
           "Pilot completion object is outside the frozen scope")
    return root, plan, submission, prefix


def _validate_completion(completion, plan, prefix):
    run_id = plan["run_id"] + "-a0000"
    ensure(completion.get("kind") == "native_full_episode_collection_complete"
           and completion.get("complete") is True and completion.get("official_benchmark") is False
           and completion.get("run_id") == run_id and completion.get("checkpoint") == plan["checkpoint_identity"],
           "Pilot completion policy or run identity differs")
    source_sha = completion.get("source_manifest_sha256")
    ensure(sha256_value(source_sha), "Pilot completion lacks its exact source SHA256")
    ensure(type(completion.get("episodes")) is int and completion["episodes"] == 1
           and isinstance(completion.get("object_checkpoints"), list)
           and len(completion["object_checkpoints"]) == 1, "Pilot must contain exactly one completed episode")
    receipt = completion["object_checkpoints"][0]
    ensure(receipt.get("kind") == "native_full_episode_object_checkpoint"
           and receipt.get("run_id") == run_id and receipt.get("source_manifest_sha256") == source_sha
           and (receipt.get("task"), receipt.get("instance_id"), receipt.get("rollout_id"))
               == ("turning_on_radio", 301, 0)
           and isinstance(receipt.get("attempt_id"), str) and bool(receipt["attempt_id"]),
           "Pilot object receipt identity differs")
    steps = receipt.get("steps")
    ensure(type(steps) is int and steps > 0 and type(completion.get("frames")) is int
           and completion["frames"] == steps, "Pilot frame count is not its executed episode length")
    for key in ("trajectory_sha256", "content_sha256", "subtask_sha256"):
        ensure(sha256_value(receipt.get(key)), "Pilot trajectory hash is incomplete: " + key)
    result = receipt.get("result")
    ensure(isinstance(result, dict), "Pilot receipt lacks its canonical environment result")
    ensure((result.get("task"), result.get("instance_id"), result.get("rollout_id"), result.get("steps"))
           == ("turning_on_radio", 301, 0, steps)
           and all(type(result.get(key)) is int for key in ("instance_id", "rollout_id", "steps")),
           "Pilot result identity differs from its raw receipt")
    success = result.get("success")
    ensure(type(success) is bool and receipt.get("success") is success
           and type(result.get("task_success")) is bool and result["task_success"] is success
           and isinstance(result.get("final_goal_state"), dict)
           and result["final_goal_state"].get("success") is success,
           "Pilot success differs from the environment task and final goal")
    ensure(type(result.get("terminated")) is bool and type(result.get("truncated")) is bool
           and (result["terminated"] or result["truncated"])
           and (not success or result["terminated"]), "Pilot is not a complete environment-terminal episode")
    condition = result.get("conditioning", {})
    ensure(isinstance(condition, dict) and condition.get("mode") == "whole_task" and condition.get("oracle") is False
           and condition.get("privileged") is False and result.get("official_benchmark") is False
           and result.get("success_source") == "environment_task_and_final_goal",
           "Pilot is not native whole-task collection")
    ensure(type(completion.get("successful_episodes")) is int
           and completion["successful_episodes"] == int(success), "Pilot success count differs")
    archive = receipt.get("archive", {})
    ensure(archive.get("object_key") == prefix + "/episodes/turning_on_radio/301/attempt-1.tar"
           and type(archive.get("bytes")) is int and 0 < archive["bytes"] <= 64 * 1024**3
           and sha256_value(archive.get("sha256")) and isinstance(archive.get("etag"), str)
           and bool(archive["etag"]) and type(archive.get("parts")) is int and 1 <= archive["parts"] <= plan.get('maximum_archive_parts', 512)
           and archive.get("part_content_md5_verified") is True, "Pilot archive receipt is incomplete")
    codec = completion.get("codec_verification", {})
    ensure(isinstance(codec, dict) and codec.get("kind") == "native_full_episode_real_codec_probe"
           and completion.get("real_codec_reader_verified") is True
           and codec.get("real_lerobot_reader") is True
           and codec.get("native_frame_action_alignment_verified") is True
           and codec.get("source_trajectory_sha256") == receipt["trajectory_sha256"]
           and codec.get("source_content_sha256") == receipt["content_sha256"]
           and type(codec.get("source_steps")) is int and codec["source_steps"] == steps
           and type(codec.get("decoded_frames")) is int and codec["decoded_frames"] == min(64, steps)
           and codec.get("source_success") is success and codec.get("official_benchmark") is False
           and codec.get("format_probe_only") is True and codec.get("ready_for_training") is False,
           "Real codec/frame-action proof does not describe the collected pilot")
    return receipt, codec


def _get_json(client, key):
    response = client.get_object(Bucket=BUCKET, Key=key)
    stream = response["Body"]
    try:
        raw = stream.read()
    finally:
        stream.close()
    return raw, json.loads(raw)


class HashingReader:
    def __init__(self, stream):
        self.stream, self.hash, self.count = stream, hashlib.sha256(), 0
    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data); self.count += len(data)
        return data


def _verify_archive(client, receipt):
    archive = receipt["archive"]
    head = client.head_object(Bucket=BUCKET, Key=archive["object_key"])
    ensure(head.get("ContentLength") == archive["bytes"] and head.get("ETag") == archive["etag"],
           "Pilot archive HEAD differs from its receipt")
    response = client.get_object(Bucket=BUCKET, Key=archive["object_key"])
    stream = response["Body"]
    hashed = HashingReader(stream)
    members, raw_sha, raw_bytes, result_sha = [], None, None, None
    try:
        ensure(response.get("ContentLength", archive["bytes"]) == archive["bytes"],
               "Pilot archive download size differs")
        with tarfile.open(fileobj=hashed, mode="r|") as tar:
            for member in tar:
                ensure(member.isfile() and member.name in ("episode.hdf5", "result.json")
                       and member.name not in members, "Pilot archive contains an unexpected or repeated member")
                members.append(member.name)
                body = tar.extractfile(member)
                ensure(body is not None, "Pilot archive member cannot be read")
                if member.name == "episode.hdf5":
                    value, length, first = hashlib.sha256(), 0, b""
                    for block in iter(lambda: body.read(CHUNK_BYTES), b""):
                        if not first:
                            first = block[:8]
                        value.update(block); length += len(block)
                    ensure(length == member.size and first == b"\x89HDF\r\n\x1a\n",
                           "Pilot raw member is not a whole HDF5 trajectory")
                    raw_sha, raw_bytes = value.hexdigest(), length
                else:
                    ensure(member.size <= 64 * 1024**2, "Pilot result JSON exceeds its expected metadata budget")
                    raw_result = body.read()
                    ensure(len(raw_result) == member.size and json.loads(raw_result) == receipt["result"],
                           "Archived result differs from its completed trajectory receipt")
                    result_sha = digest(raw_result)
        # tarfile stops at its zero-block terminator. Hash remaining padding too.
        for _ in iter(lambda: hashed.read(CHUNK_BYTES), b""):
            pass
    finally:
        stream.close()
    ensure(set(members) == {"episode.hdf5", "result.json"}, "Pilot archive is missing raw trajectory or result")
    ensure(hashed.count == archive["bytes"] and hashed.hash.hexdigest() == archive["sha256"],
           "Pilot archive bytes changed after upload")
    ensure(raw_sha == receipt["trajectory_sha256"], "Archived whole HDF5 differs from its codec-verified source")
    return dict(kind="native_full_episode_streamed_archive_verification", archive_key=archive["object_key"],
                archive_bytes=hashed.count, archive_sha256=hashed.hash.hexdigest(), archive_etag=head["ETag"],
                trajectory_bytes=raw_bytes, trajectory_sha256=raw_sha, result_sha256=result_sha,
                complete_archive_stream_sha256_verified=True, whole_hdf5_stream_sha256_verified=True,
                archived_result_matches_receipt=True, members=members)


def _save_once(path, raw):
    if path.exists():
        ensure(path.read_bytes() == raw, "Previously saved pilot evidence differs: " + path.name)
    else:
        with path.open("xb") as output:
            output.write(raw)


def accept_pilot(root, cli, client):
    root, plan, submission, prefix = _load_inputs(root)
    job = describe(cli, environment(), submission["name"])
    ensure(job.get("uid") == submission["uid"] and job.get("state") == "SUCCEEDED",
           "Exact pilot job UID has not succeeded")
    raw, completion = _get_json(client, submission["completion_key"])
    receipt, codec = _validate_completion(completion, plan, prefix)
    receipt_key = prefix + "/episodes/turning_on_radio/301/attempt-1.json"
    _, stored_receipt = _get_json(client, receipt_key)
    ensure(stored_receipt == receipt, "Object receipt differs from the pilot completion")
    proof = _verify_archive(client, receipt)
    codec_bytes, proof_bytes = canonical(codec), canonical(proof)
    gate = dict(kind=GATE_KIND, plan_fingerprint=plan["fingerprint"], pilot_name=submission["name"],
                pilot_uid=submission["uid"], platform_state="SUCCEEDED", run_id=completion["run_id"],
                checkpoint=completion["checkpoint"], completion_key=submission["completion_key"],
                completion_sha256=digest(raw), source_manifest_sha256=completion["source_manifest_sha256"],
                trajectory_sha256=receipt["trajectory_sha256"], content_sha256=receipt["content_sha256"],
                codec_report_sha256=digest(codec_bytes), archive_proof_sha256=digest(proof_bytes),
                full_episode_raw_verified=True, real_codec_reader_verified=True,
                native_frame_action_alignment_verified=True, success_required=False,
                official_benchmark=False, archive_proof=proof)
    for name, data in ((COMPLETION_FILE, raw), (CODEC_FILE, codec_bytes), (ARCHIVE_FILE, proof_bytes),
                       ("pilot_accepted.json", canonical(gate))):
        _save_once(root / name, data)
    return validate_gate(root, plan)


def validate_gate(root, plan):
    """Verify local evidence and its exact plan/submission binding before submit."""
    root, plan, submission, prefix = _load_inputs(root, plan)
    gate = json.loads((root / "pilot_accepted.json").read_bytes())
    raw = (root / COMPLETION_FILE).read_bytes()
    codec_raw, proof_raw = (root / CODEC_FILE).read_bytes(), (root / ARCHIVE_FILE).read_bytes()
    completion = json.loads(raw)
    receipt, codec = _validate_completion(completion, plan, prefix)
    proof = json.loads(proof_raw)
    expected = dict(kind=GATE_KIND, plan_fingerprint=plan["fingerprint"], pilot_name=submission["name"],
                    pilot_uid=submission["uid"], platform_state="SUCCEEDED", run_id=completion["run_id"],
                    checkpoint=plan["checkpoint_identity"], completion_key=submission["completion_key"],
                    completion_sha256=digest(raw), source_manifest_sha256=completion["source_manifest_sha256"],
                    trajectory_sha256=receipt["trajectory_sha256"], content_sha256=receipt["content_sha256"],
                    codec_report_sha256=digest(codec_raw), archive_proof_sha256=digest(proof_raw),
                    full_episode_raw_verified=True, real_codec_reader_verified=True,
                    native_frame_action_alignment_verified=True, success_required=False,
                    official_benchmark=False, archive_proof=proof)
    ensure(gate == expected and json.loads(codec_raw) == codec, "Pilot gate or codec evidence differs")
    ensure(proof.get("kind") == "native_full_episode_streamed_archive_verification"
           and proof.get("archive_key") == receipt["archive"]["object_key"]
           and proof.get("archive_bytes") == receipt["archive"]["bytes"]
           and proof.get("archive_sha256") == receipt["archive"]["sha256"]
           and proof.get("archive_etag") == receipt["archive"]["etag"]
           and proof.get("trajectory_sha256") == receipt["trajectory_sha256"]
           and sha256_value(proof.get("result_sha256"))
           and type(proof.get("trajectory_bytes")) is int and proof["trajectory_bytes"] > 0
           and proof.get("complete_archive_stream_sha256_verified") is True
           and proof.get("whole_hdf5_stream_sha256_verified") is True
           and proof.get("archived_result_matches_receipt") is True
           and set(proof.get("members", [])) == {"episode.hdf5", "result.json"},
           "Pilot archive proof does not describe the accepted raw episode")
    return gate
