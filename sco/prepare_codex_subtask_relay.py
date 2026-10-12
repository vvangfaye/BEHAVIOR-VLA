"""Read verified AOSS replicas using the unchanged original-demonstration reader."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def run(config):
    plan = config["preparation_plan"]
    relay = config["relay_provenance"]
    signed = config["signed_files"]
    root = Path(config["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    output = Path(config["output_root"])
    if output.exists():
        raise ValueError("Relay demonstration output must be fresh")
    expected = {r["relative_path"]: r for r in relay["accepted"]}
    if relay["state"] != "complete" or len(expected) != 4 or len(signed) != 4:
        raise ValueError("Relay must contain exactly four completely verified objects")
    for row in signed:
        proof = expected[row["relative_path"]]
        if (proof["full_readback"] is not True or row["bytes"] != proof["bytes"]
                or row["sha256"] != proof["sha256"] or row["etag"] != proof["aoss_etag"]):
            raise ValueError("Relay object identity differs")
    if {r["relative_path"] for r in plan["signed_objects"]} != set(expected):
        raise ValueError("Relay does not cover the original selected files")
    for original in plan["signed_objects"]:
        proof = expected[original["relative_path"]]
        if original["bytes"] != proof["bytes"] or original["etag"] != proof["upstream_etag"]:
            raise ValueError("Relay upstream identity differs")
    code, bundle = Path(plan["code_root"]), Path(plan["bundle_root"])
    if sha(Path(plan["base_code"]) / "_PUBLIC_TEST_SOURCE_READY") != plan["base_source_sha256"]:
        raise ValueError("Original base changed")
    inventory = json.loads((bundle / "manifest.json").read_text())
    for name, item in inventory.items():
        path = bundle / name
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Frozen preparation bundle changed")
        if name.startswith("code/") and sha(code / name.removeprefix("code/")) != item["sha256"]:
            raise ValueError("Frozen preparation code changed")
    payload = {**plan["payload"], "output_root": str(output), "signed_files": signed,
               "max_seconds": 1500, "transport": "verified_original_OSS_to_AOSS_relay",
               "transport_provenance": relay}
    private = root / "private_payload.json"
    with os.fdopen(os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(payload, stream)
    env = dict(os.environ, PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", PYTHONPATH=":".join(str(code / name) for name in (
                   "scripts", "openpi/src", "openpi/packages/openpi-client/src")))
    report = dict(kind="codex_subtask_relay_preparation_v1", state="starting",
                  output_root=str(output), original_episode=10020, model_calls=0)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def publish():
        (root / "status.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            request = urllib.request.Request(os.environ["CODEX_PILOT_REPORT_PUT"],
                data=json.dumps(report).encode(), method="PUT")
            with opener.open(request, timeout=20):
                pass
        except Exception as error:
            print(json.dumps(dict(report_error_type=type(error).__name__)), flush=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Owned relay preparation interrupted")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    process = None
    started = time.monotonic()
    try:
        publish()
        with (root / "staging.log").open("x") as log:
            process = subprocess.Popen([plan["runtime_python"], str(code / "sco/prepare_codex_original_data.py"),
                "--payload", str(private)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                try:
                    rc = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    rc = None
                if (output / "status.json").is_file():
                    report["preparation"] = json.loads((output / "status.json").read_text())
                report.update(state="running" if rc is None else "worker_exited", worker_exit_code=rc,
                              elapsed_seconds=time.monotonic() - started)
                publish()
                if rc is not None:
                    break
                if time.monotonic() - started > 1560:
                    raise TimeoutError("Bounded relay preparation expired")
        if rc:
            raise RuntimeError("Real demonstration preparation failed")
        ready = json.loads((output / "_ORIGINAL_DATA_READY.json").read_text())
        demo_path = output / "demonstrations/10020/manifest.json"
        demo = json.loads(demo_path.read_text())
        if (ready["selected_episodes"] != [10020] or ready["original_reader_verified"] is not True
                or ready["raw_success_verified"] is not True or demo["required_subtask"] != plan["target_subtask"]
                or not any(row["subtask"] == plan["target_subtask"] for row in demo["examples"])):
            raise ValueError("Original successful target demonstration was not verified")
        report.update(state="original_episode10020_verified", demonstration_manifest=str(demo_path),
                      demonstration_fingerprint=demo["fingerprint"], bytes_staged=ready["bytes_staged"])
    except BaseException as error:
        report.update(state="failed", error_type=type(error).__name__)
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        publish()
    return report
