"""Fetch this probe's compact output from the user's existing AOSS bucket."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "analysis/cluster_probe_20261104"))
import aoss_io

PREFIX = "behavior1k/supermimic-feasibility-20261106/export-a1/"


def main():
    client = aoss_io.local()
    try:
        status = json.loads(client.get_object(Bucket="260010165", Key=PREFIX+"status.json")["Body"].read())
    except client.exceptions.NoSuchKey:
        print(json.dumps({"state": "no_completion_artifact_yet"}))
        return
    (HERE / "data/export_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({k: v for k, v in status.items() if k != "metadata"}, indent=2))
    if status["state"] != "complete":
        return
    receipts = []
    for name in ("transitions.npz", "transitions.json", "worker.log"):
        target = HERE / "data" / name
        if target.exists():
            continue
        key = PREFIX + name
        response = client.get_object(Bucket="260010165", Key=key)
        if response["ContentLength"] > 256 * 1024**2:
            raise RuntimeError("Artifact exceeds declared download cap")
        data = response["Body"].read()
        temporary = target.with_suffix(target.suffix+".part")
        temporary.write_bytes(data)
        temporary.replace(target)
        receipts.append({"key": key, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (HERE / "data/download_receipts.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
