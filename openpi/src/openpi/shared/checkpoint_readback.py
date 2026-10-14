"""Read a committed checkpoint from a private OSS mount into a fresh local cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

OWNED_LOCAL_PREFIX = "/var/tmp/wangfaye/four-source-readback/"


def copy_committed_checkpoint(directory, step, local_root, tool, endpoint, tool_sha256):
    if type(step) is not int or step < 0:
        raise ValueError("Invalid committed checkpoint step")
    if endpoint not in ("https://oss-cn-zhangjiakou.aliyuncs.com", "https://cn-zhangjiakou.oss.aliyuncs.com"):
        raise ValueError("Readback must use the verified private bucket origin")
    source = str(directory).rstrip("/") + "/" + str(step)
    if not source.startswith("/data/oss_bucket_0/wangfaye/result/behavior-challenge/"):
        raise ValueError("Checkpoint readback must stay within this user's result prefix")
    root = Path(local_root)
    if (not root.is_absolute() or ".." in root.parts
            or not str(root).startswith(OWNED_LOCAL_PREFIX)):
        raise ValueError("Checkpoint readback requires a fresh owned local directory")
    if root.exists():
        raise ValueError("Never overwrite an existing checkpoint readback cache")
    if hashlib.sha256(Path(tool).read_bytes()).hexdigest() != tool_sha256:
        raise ValueError("Frozen OSS transfer tool changed")
    root.mkdir(parents=True, exist_ok=False)
    target = root / str(step)
    uri = "oss://traffic-model/" + source.removeprefix("/data/oss_bucket_0/")
    command = [str(tool), "cp", uri + "/", str(target) + "/", "--recursive", "--job", "16",
               "--parallel", "8", "--endpoint", endpoint, "--sign-version", "v1",
               "--checkpoint-dir", str(root / "transfer-checkpoints"),
               "--output-dir", str(root / "transfer-output"), "--retry-times", "10", "--no-progress"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
    if result.returncode:
        message = result.stdout[-2000:]
        for key in ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[REDACTED]")
        raise RuntimeError("Committed checkpoint local readback failed: " + message)
    metadata = json.loads((target / "_CHECKPOINT_METADATA").read_text())
    if not metadata.get("commit_timestamp_nsecs"):
        raise ValueError("Checkpoint is not committed; do not restore partial output")
    for relative in ("params/_METADATA", "train_state/_METADATA", "assets/learning_rate_contract.json",
                     "assets/subtask_mixture.json", "assets/subtask_mixture_execution.json"):
        if not (target / relative).is_file():
            raise ValueError("Committed checkpoint component missing: " + relative)
    return str(root)
