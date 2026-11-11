#!/usr/bin/env python3
"""Read bounded status from the already-approved pilot; never start a job."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pty
import re
import select
import subprocess
import time
import zlib


CLI = "/home/faye/.local/bin/sco"
JOB = "b1k-boundary-probe-20261019-pilot01"
UID = "7f68ca4e90e047adbe7975dd03049fac"
ROOT = "/mnt/afs/260010165/behavior_1k/training_boundary_audit_20261019/pilot01/results"


def peek(timeout=60):
    response = subprocess.run(
        [CLI, "acp", "jobs", "describe", JOB, "--workspace-name=share-space", "-o", "json"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    job = json.loads(response.stdout)
    if job["uid"] != UID or job["ownership"]["user_name"] != "260010165":
        raise ValueError("Unexpected job identity")
    result = {"job_name": JOB, "uid": UID, "state": job["state"],
              "observed_at_utc": datetime.now(timezone.utc).isoformat()}
    if job["state"] != "RUNNING":
        return result
    remote = '''import json,re,subprocess
from pathlib import Path
p=Path(ROOT)
d={"files":{f.name:f.stat().st_size for f in p.iterdir() if f.is_file()}} if p.exists() else {"files":{}}
for name in ("progress.json","failure.json"):
 f=p/name
 if f.exists():
  if f.stat().st_size>16384:raise ValueError("Oversized status")
  d[name]=json.loads(f.read_text())
d["logs"]={}
if p.exists():
 for f in sorted(p.glob("*.log")):
  with f.open("rb") as stream:
   stream.seek(-min(f.stat().st_size,3000),2)
   d["logs"][f.name]=re.sub(r"https?://\\S+","<redacted URL>",stream.read().decode(errors="replace"))
r=subprocess.run(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True,timeout=10)
d["gpu"]=r.stdout.strip()
print("B1K_PEEK_JSON="+json.dumps(d),flush=True)
'''.replace("Path(ROOT)", "Path(" + repr(ROOT) + ")")
    encoded = base64.b64encode(zlib.compress(remote.encode())).decode()
    wire = ("python3 - <<'B1K_PEEK_SCRIPT'\nimport base64,zlib\nexec(zlib.decompress(base64.b64decode("
            + repr(encoded) + ")))\nB1K_PEEK_SCRIPT\nexit\n").encode()
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    process = subprocess.Popen(
        [CLI, "acp", "jobs", "exec", JOB, "--workspace-name=share-space", "--worker-name=pt-" + UID + "-worker-0"],
        stdin=slave, stdout=slave, stderr=slave,
    )
    os.close(slave)
    pending, prompt, offset, ready = b"", b"", 0, False
    started = time.monotonic()
    try:
        while time.monotonic() - started < timeout:
            readable, writable, _ = select.select([master], [master] if ready and offset < len(wire) else [], [], .2)
            if readable:
                try:
                    data = os.read(master, 65536)
                except BlockingIOError:
                    continue
                except OSError:
                    break
                if not data:
                    break
                pending += data
                prompt = (prompt + data)[-8192:]
                clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", prompt.decode(errors="replace").replace("\r", ""))
                ready = ready or bool(re.search(r"(?:^|\n)[^\n]*[#$]\s*$", clean))
            if writable:
                offset += os.write(master, wire[offset:offset + 512])
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                line = line.replace(b"\r", b"").replace(b"\x01", b"")
                marker = b"B1K_PEEK_JSON="
                if marker in line:
                    payload = line.split(marker, 1)[1]
                    if len(payload) > 65536:
                        raise ValueError("Oversized status response")
                    result["worker"] = json.loads(payload)
                    result["worker_observed_at_utc"] = datetime.now(timezone.utc).isoformat()
                    return result
            if len(pending) > 131072:
                raise ValueError("Oversized observer transport")
        result["observation"] = "worker_read_unavailable_or_timed_out; do not infer job failure"
        return result
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        os.close(master)


if __name__ == "__main__":
    print(json.dumps(peek(), ensure_ascii=False, indent=2))
