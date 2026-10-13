"""Read one finished assisted subtask, verify provenance, and export its artifacts."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def put_file(path, url):
    uri = urlsplit(url)
    if uri.scheme != "https" or not uri.hostname or uri.username or uri.password:
        raise ValueError("Artifact transport requires private HTTPS")
    connection = http.client.HTTPSConnection(uri.hostname, uri.port or 443, timeout=120)
    try:
        with Path(path).open("rb") as stream:
            connection.request("PUT", uri.path + ("?" + uri.query if uri.query else ""), body=stream,
                               headers={"Content-Length": str(Path(path).stat().st_size)})
            response = connection.getresponse()
            status = response.status
            response.read(8192)
        if status not in (200, 201, 204):
            raise RuntimeError("Artifact upload failed with HTTP status " + str(status))
    finally:
        connection.close()


def run(config):
    code, root, output = (Path(config[key]) for key in ("code", "run_root", "output_root"))
    output.mkdir(parents=True, exist_ok=False)
    for name, digest in config["source_files_sha256"].items():
        if sha(code / name) != digest:
            raise ValueError("Collected code differs from its frozen source")
    sys.path[:0] = [str(code / "scripts"), str(code / "sco")]
    from codex_subtask_collection import inspect_attempt, load_plan
    plan = load_plan(config["plan_path"])
    if plan["fingerprint"] != config["plan_fingerprint"]:
        raise ValueError("Wrong collected plan")
    status = json.loads((root / "_COMPLETE").read_text())
    if status["state"] != "complete" or status["plan_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Worker completion differs")
    attempt = root / "attempt"
    info = inspect_attempt(attempt, require_success=bool(status["result"]["success"]))
    result = info["result"]
    if status["result"] != result or status["training_eligible"] != bool(result["success"] and result["metric_eligible"]):
        raise ValueError("Worker and HDF5 disagree")
    receipts = sorted((attempt / "teacher").glob("step-*/decision/*/receipt.json"))
    if len(receipts) != result["teacher_decisions"]:
        raise ValueError("Teacher decision evidence is incomplete")
    teachers, usage = [], {}
    for path in receipts:
        receipt = json.loads(path.read_text())
        decision = json.loads(path.with_name("response.json").read_text())
        from codex_collection_teacher import validate_teacher_configuration
        teacher_config = validate_teacher_configuration(plan.get('teacher_configuration'))
        if (receipt["status"] != "validated" or receipt["requested_model"] != teacher_config['model']
                or receipt["requested_reasoning_effort"] != teacher_config['reasoning_effort']
                or receipt.get('requested_service_tier') != teacher_config['service_tier']
                or receipt["tool_event_validation"]["status"] != "no_tool_events_observed"
                or receipt["request_id"] != decision["request_id"]
                or decision["subtask"] != plan["sample"]["subtask"]):
            raise ValueError("Teacher model, fixed target or tool audit differs")
        teachers.append(dict(step_directory=path.parents[2].name, receipt=receipt, decision=decision))
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
            value = (receipt.get("usage") or {}).get(key)
            if value is None or usage.get(key, 0) is None:
                usage[key] = None
            else:
                usage[key] = usage.get(key, 0) + value
    report = dict(kind="codex_subtask_readback_audit_v1", state="verified", success=result["success"],
        training_eligible=status["training_eligible"], official_benchmark=False, plan_fingerprint=plan["fingerprint"],
        checkpoint_identity=plan["checkpoint_identity"], sample=plan["sample"], steps=info["steps"],
        action_source_counts=info["action_source_counts"], result=result, teacher_usage=usage,
        teachers=teachers, content_sha256=info["content_sha256"], artifacts=[])
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata = output / "metadata.zip"
    selected = [attempt / name for name in ("completed.json", "action_sources.jsonl", "correction_rejections.jsonl")]
    for path in receipts:
        selected.extend(p for p in path.parent.iterdir() if p.is_file()
                        and (p.suffix in (".json", ".png") or p.name == "prompt.txt") and p.name != "command.json")
    if len(selected) > 1024 or sum(p.stat().st_size for p in selected) > 128 * 1024**2:
        raise ValueError("Teacher metadata exceeds bounded artifact size")
    with zipfile.ZipFile(metadata, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "audit.json", "audit.json")
        for path in selected:
            archive.write(path, path.relative_to(attempt).as_posix())
    files = dict(trajectory=attempt / "trajectory.hdf5", video=attempt / "rollout.mp4", metadata=metadata)
    for name, path in files.items():
        size = path.stat().st_size
        if not 0 < size <= 5 * 1024**3:
            raise ValueError("Collected artifact exceeds finite size bound")
        digest = sha(path)
        put_file(path, config["artifact_puts"][name])
        report["artifacts"].append(dict(name=name, bytes=size, sha256=digest, key=config["artifact_keys"][name]))
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
