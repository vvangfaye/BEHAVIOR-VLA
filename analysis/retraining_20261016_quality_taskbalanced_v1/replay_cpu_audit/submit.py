"""Submit exactly one CPU diagnostic from the existing Mac credentials."""
import configparser
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re

import dispatch_support as dispatch
from submit_oss import OSSClient

HERE = Path(__file__).resolve().parent


def load_credentials():
    if platform.system() != "Darwin":
        raise ValueError("Submit only from the configured Mac")
    values = {}
    for line in (Path.home() / ".ai-hub-cli/config.yaml").read_text().splitlines():
        match = re.fullmatch(r"(namespace|token):\s*(\S+)\s*", line)
        if match:
            values[match[1]] = match[2].strip("\"'")
    if set(values) != {"namespace", "token"}:
        raise ValueError("Unexpected AIHub configuration layout")
    parser = configparser.ConfigParser()
    parser.read(Path.home() / ".ossutilconfig")
    source = parser["Credentials"]
    os.environ.update(AI_HUB_NAMESPACE=values["namespace"], AI_HUB_TOKEN=values["token"],
        OSS_ACCESS_ID=source["accesskeyid"], OSS_ACCESS_KEY=source["accesskeysecret"],
        OSS_ENDPOINT=source["endpoint"], OSS_BUCKET="traffic-model")


def main():
    with (HERE / "submission.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / "intent.json").exists() or (HERE / "receipt.json").exists():
            raise ValueError("Existing submission evidence; reconcile instead of resubmitting")
        bundle = HERE / "bundle"
        validation = dispatch.read(HERE / "validation.json")
        if not validation["passed"]:
            raise ValueError("Local validation did not pass")
        for name, expected in dispatch.read(HERE / "public_bundle_manifest.json").items():
            if dispatch.digest(bundle / name) != expected:
                raise ValueError("Prepared public bundle changed")
        base = dispatch.read(HERE / "runtime_base.json")
        plan = dispatch.read(bundle / "column_plan.json")
        footer = dispatch.read(bundle / "footer_probe.json")
        if not footer["complete"] or plan["candidate_window_count"] != 1829:
            raise ValueError("Incomplete data-access preflight")
        if base["maximum_worker_seconds"] != 3600 or base["maximum_data_range_bytes"] != 48 * 1024**3:
            raise ValueError("Diagnostic budget changed")
        if dispatch.read(bundle / "cluster.json") != {"worker": {"cpu": 800, "gpu": 0, "memory": 64000}}:
            raise ValueError("CPU-only resource contract changed")
        load_credentials()
        client = OSSClient()
        client.unused(base["output_key"])
        import oss2
        download = oss2.Bucket(oss2.Auth(os.environ["OSS_ACCESS_ID"], os.environ["OSS_ACCESS_KEY"]),
                               base["download_endpoint"], "traffic-model")
        head = download.head_object(base["environment"]["key"])
        if head.content_length != base["environment"]["bytes"] or head.etag != base["environment"]["etag"]:
            raise ValueError("Frozen environment object changed")
        private = dict(base)
        private["environment"] = dict(base["environment"],
            get_url=download.sign_url("GET", base["environment"]["key"], 43200, slash_safe=True))
        private["objects"] = [{"relative_path": item["source"]["relative_path"],
            "get_url": download.sign_url("GET", item["source"]["key"], 43200, slash_safe=True)}
            for item in plan["objects"]]
        path = bundle / "runtime.private.json"
        dispatch.save(path, private)
        path.chmod(0o600)
        dispatch.create(client, "replay_cpu_action_audit", HERE, "worker.py", bundle / "cluster.json",
            "/data/oss_bucket_0/" + base["output_key"], 1, 0, 3600, base["run_id"])


if __name__ == "__main__":
    main()
