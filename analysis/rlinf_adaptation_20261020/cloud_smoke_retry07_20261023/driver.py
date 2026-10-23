"""Bounded sequential acceptance stages; no training after a failed gate."""

import json
import hashlib
import shutil
import os
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import write


def upload_file(url, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(4):
        try:
            with Path(path).open("rb") as stream:
                request = urllib.request.Request(
                    url,
                    data=stream,
                    method="PUT",
                    headers={"Content-Length": str(Path(path).stat().st_size)},
                )
                with opener.open(request, timeout=180) as response:
                    assert response.status in (200, 201, 204)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
            time.sleep(min(2**attempt, 4))


def status_upload(url, path):
    try:
        upload_file(url, path)
    except Exception as error:
        print("Status upload unavailable: " + type(error).__name__, flush=True)


def run_stage(plan, name, arguments, seconds):
    root = Path(plan["output"])
    write(
        root / "status.json",
        {"stage": name, "state": "running", "updated_at": time.time()},
    )
    target = os.environ.get("B1K_STATUS_PUT")
    if target:
        status_upload(target, root / "status.json")
    with (root / (name + ".log")).open("x") as log:
        stage_env = os.environ.copy()
        if name in {"baseline", "reference_inputs"}:
            stage_env["JAX_PLATFORMS"] = "cpu"
        process = subprocess.Popen(
            [sys.executable, *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=stage_env,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=seconds)
        finally:
            # This group contains only the current stage and its own simulator.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if returncode:
            raise RuntimeError(f"{name} exited with code {returncode}")


def main():
    here = Path(__file__).resolve().parent
    planpath = here / "plan.json"
    plan = json.loads(planpath.read_text())
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = {"state": "running", "training_started": False, "started_at": started}
    try:
        source = Path(plan["reuse_source_root"])
        for relative, expected in plan["reuse_files_sha256"].items():
            item = source / relative
            assert hashlib.sha256(item.read_bytes()).hexdigest() == expected
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, destination)
        write(root / "reuse_receipt.json", {"source": str(source), "files": plan["reuse_files_sha256"]})
        run_stage(plan, "precision_gate", [str(here / "precision_gate.py"), "--plan", str(planpath)], 1800)
        run_stage(
            plan,
            "baseline",
            [str(here / "rollout.py"), "--plan", str(planpath), "--backend", "jax"],
            plan["baseline_stage_seconds"],
        )
        run_stage(
            plan,
            "adapted",
            [str(here / "rollout.py"), "--plan", str(planpath), "--backend", "torch"],
            7200,
        )
        run_stage(plan, "gate", [str(here / "gate.py"), "--plan", str(planpath)], 1800)
        assert json.loads((root / "gate.json").read_text())["passed"]
        status["training_started"] = True
        run_stage(plan, "ppo", [str(here / "ppo.py"), "--plan", str(planpath)], 3600)
        assert json.loads((root / "ppo/result.json").read_text())["passed"]
        status["state"] = "complete"
    except BaseException as error:
        status.update(
            state="failed", error_type=type(error).__name__, error=str(error)[:800]
        )
    finally:
        status["elapsed_seconds"] = time.time() - started
        write(root / "status.json", status)
        if os.environ.get("B1K_STATUS_PUT"):
            status_upload(os.environ["B1K_STATUS_PUT"], root / "status.json")
        archive = root.parent / "result.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in root.rglob("*"):
                relative = path.relative_to(root)
                if path.is_file() and not any(
                    part in {"appdata", "cache", "config", "warp"}
                    for part in relative.parts
                ):
                    tar.add(path, arcname=str(relative), recursive=False)
        if os.environ.get("B1K_RESULT_PUT"):
            upload_file(os.environ["B1K_RESULT_PUT"], archive)
    print(json.dumps(status), flush=True)
    raise SystemExit(0 if status["state"] == "complete" else 1)


if __name__ == "__main__":
    main()
