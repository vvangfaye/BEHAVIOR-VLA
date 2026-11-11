"""Read six existing episode JSON receipts; never upload or create cloud jobs."""
from pathlib import Path
import hashlib
import json
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "BEHAVIOR-VLAs/sco"))
from video_eval_adapter import aoss_client


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    plan_bytes = (HERE / "read_plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    rows = plan["receipts"]
    if plan["bucket"] != "260010165" or len(rows) != 6:
        raise ValueError("Unexpected read scope")
    client = aoss_client()
    evidence = []
    for row in rows:
        key = row["receipt_key"]
        if not key.startswith(plan["allowed_key_prefix"]) or not key.endswith("/attempt-1.json"):
            raise ValueError("Receipt key outside frozen scope")
        path = HERE / f"{row['task']}_{row['instance_id']}_{row['role']}.json"
        if path.exists():
            payload = path.read_bytes()
            origin = "existing_local_receipt"
        else:
            head = client.head_object(Bucket=plan["bucket"], Key=key)
            size = head["ContentLength"]
            if size != row["expected_object_bytes"] or size > plan["max_bytes_per_receipt"]:
                raise ValueError("Receipt size differs from frozen inventory")
            response = client.get_object(Bucket=plan["bucket"], Key=key, IfMatch=head["ETag"])
            body = response["Body"]
            try:
                payload = body.read(plan["max_bytes_per_receipt"] + 1)
            finally:
                body.close()
            if len(payload) != size:
                raise ValueError("Incomplete or oversized receipt")
            origin = "existing_private_object_read"
        receipt = json.loads(payload)
        canonical = digest(json.dumps(receipt, sort_keys=True).encode())
        if canonical != row["receipt_sha256"]:
            raise ValueError("Receipt differs from historical content hash")
        if (receipt["task"] != row["task"] or receipt["instance_id"] != row["instance_id"]
                or receipt["success"] is not row["expected_success"]
                or receipt["trajectory_sha256"] != row["raw_sha256"]
                or receipt["archive"]["sha256"] != row["archive_sha256"]
                or receipt["steps"] != row["expected_steps"]):
            raise ValueError("Receipt identity or trajectory binding differs")
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(payload)
        evidence.append({"file": path.name, "bytes": len(payload), "sha256": digest(payload),
                         "canonical_sha256": canonical, "origin": origin})
    receipt = {"read_plan_sha256": digest(plan_bytes), "cloud_uploads": 0, "cloud_jobs": 0,
               "files": evidence, "total_bytes": sum(row["bytes"] for row in evidence)}
    (HERE / "read_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"verified_receipts": len(evidence), "total_bytes": receipt["total_bytes"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never print request headers, credential material or signed URLs.
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
