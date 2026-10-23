"""Read-only, bounded local observer of the existing single GPU worker."""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
deadline = time.monotonic() + 6.5 * 3600
index = 1
previous = None
while time.monotonic() < deadline:
    status_file = HERE / "latest_job_status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status.get("state") in {"FAILED", "SUCCEEDED", "TERMINATED", "CANCELED"}:
            print(json.dumps({"terminal": status}), flush=True)
            break
    output = HERE / f"live_{index:02d}.json"
    with (HERE / f"live_{index:02d}_transport.log").open("w") as log:
        completed = subprocess.run(
            [sys.executable, str(HERE.parent / "forensics_20261023/exec_probe.py"),
             "--job", "b1k-rlinf-1023-r05", "--uid", "22d75ecc060a4ff6b43262d1062d4465",
             "--script", str(HERE / "live_probe.py"), "--output", str(output), "--timeout", "60"],
            stdout=log, stderr=subprocess.STDOUT, timeout=110, cwd=ROOT,
        )
    if output.exists():
        receipt = json.loads(output.read_text())
        summary = {"gpu": receipt.get("gpu"), "files": {
            name: value for name, value in receipt["files"].items() if not name.endswith(".log")
        }}
        text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        if text != previous:
            print(json.dumps({"probe": index, "observed_at": receipt["observed_at"], **summary}, ensure_ascii=False), flush=True)
            previous = text
    elif completed.returncode:
        print(json.dumps({"probe": index, "observation_error": completed.returncode}), flush=True)
    index += 1
    time.sleep(30)
