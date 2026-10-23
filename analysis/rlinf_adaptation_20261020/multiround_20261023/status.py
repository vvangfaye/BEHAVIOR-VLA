"""Bounded cloud status retrieval without exposing startup URLs."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JOB = "b1k-ppo5-1023-p01"
cli = str(Path.home() / ".local/bin/sco")
r = subprocess.run(
    [cli, "acp", "jobs", "describe", JOB, "--workspace-name=share-space", "-o", "json"],
    text=True,
    capture_output=True,
    timeout=45,
)
if r.returncode == 0:
    result = json.loads(r.stdout)
    safe = {
        k: result.get(k)
        for k in [
            "uid",
            "name",
            "state",
            "create_time",
            "start_time",
            "complete_time",
            "resource_pool",
        ]
    }
    (HERE / "latest_job_status.json").write_text(json.dumps(safe, indent=2) + "\n")
    print(json.dumps(safe))
else:
    print(json.dumps({"job_query_returncode": r.returncode}))
sys.path.insert(0, str(ROOT / "BEHAVIOR-VLAs/sco"))
from video_eval_adapter import aoss_client

client = aoss_client()
try:
    key = "behavior1k/rlinf-multiround-20261023/" + JOB + "/status.json"
    try:
        obj = client.get_object(Bucket="260010165", Key=key)
        raw = obj["Body"].read(65537)
        assert len(raw) < 65537
        status = json.loads(raw)
        (HERE / "latest_worker_status.json").write_text(
            json.dumps(status, indent=2) + "\n"
        )
        print(json.dumps(status))
    except client.exceptions.NoSuchKey:
        print(json.dumps({"worker_status": "not_published"}))
finally:
    client.close()
