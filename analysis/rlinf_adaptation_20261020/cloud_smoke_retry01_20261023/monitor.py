"""Finite local observer; never restarts jobs or changes the frozen experiment."""

import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JOB = "b1k-rlinf-1023-retry01"
PREFIX = "behavior1k/rlinf-adapter-20261023/" + JOB + "/"
sys.path.insert(0, str(ROOT / "BEHAVIOR-VLAs/sco"))
from video_eval_adapter import aoss_client


def write(name, value):
    p = HERE / name
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    q.replace(p)


def main():
    intent = json.loads((HERE / "submission_intent.json").read_text())
    assert intent.get("uid")
    expected_uid = intent["uid"]
    client = aoss_client()
    deadline = time.monotonic() + 7 * 3600
    try:
        while time.monotonic() < deadline:
            result = subprocess.run(
                [
                    str(Path.home() / ".local/bin/sco"),
                    "acp",
                    "jobs",
                    "describe",
                    JOB,
                    "--workspace-name=share-space",
                    "-o",
                    "json",
                ],
                text=True,
                capture_output=True,
                timeout=45,
            )
            if result.returncode:
                write(
                    "monitor_state.json",
                    {
                        "state": "observation_failed",
                        "returncode": result.returncode,
                        "at": time.time(),
                    },
                )
                time.sleep(30)
                continue
            job = json.loads(result.stdout)
            assert job["uid"] == expected_uid
            safe = {
                k: job.get(k)
                for k in ["uid", "name", "state", "start_time", "complete_time"]
            }
            write("latest_job_status.json", safe)
            try:
                obj = client.get_object(Bucket="260010165", Key=PREFIX + "status.json")
                raw = obj["Body"].read(65537)
                assert len(raw) <= 65536
                write("latest_worker_status.json", json.loads(raw))
            except client.exceptions.NoSuchKey:
                pass
            if job["state"] in {"SUCCEEDED", "FAILED", "TERMINATED", "CANCELED"}:
                break
            time.sleep(30)
        else:
            write(
                "monitor_state.json",
                {"state": "observer_deadline", "job_not_modified": True},
            )
            return
        try:
            head = client.head_object(Bucket="260010165", Key=PREFIX + "result.tar.gz")
        except client.exceptions.ClientError:
            write(
                "monitor_state.json",
                {"state": "terminal_without_result_archive", "job": safe},
            )
            return
        assert head["ContentLength"] < 16 * 1024**3
        archive = HERE / "result.tar.gz"
        temporary = archive.with_suffix(".download")
        client.download_file("260010165", PREFIX + "result.tar.gz", str(temporary))
        assert temporary.stat().st_size == head["ContentLength"]
        temporary.replace(archive)
        retrieved = HERE / "retrieved"
        retrieved.mkdir(exist_ok=True)
        # Retrieve small reports/logs. Large videos and trained tensors remain in the archive/AOSS.
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                rel = Path(member.name)
                assert (
                    member.isfile() and not rel.is_absolute() and ".." not in rel.parts
                )
                if member.size > 32 * 1024**2 or rel.suffix not in {".json", ".log"}:
                    continue
                dest = retrieved / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, dest.open("wb") as target:
                    target.write(source.read())
        write(
            "monitor_state.json",
            {
                "state": "results_retrieved",
                "job": safe,
                "archive_bytes": archive.stat().st_size,
                "reports": str(retrieved),
            },
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
