import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpi.shared import checkpoint_readback as readback


@pytest.mark.parametrize("committed,missing_optimizer", [(True, False), (False, False), (True, True)])
def test_readback_requires_committed_complete_owned_copy(tmp_path, monkeypatch, committed, missing_optimizer):
    root = tmp_path / "cache"
    tool = tmp_path / "ossutil"
    tool.write_bytes(b"pinned transfer tool")
    digest = hashlib.sha256(tool.read_bytes()).hexdigest()
    monkeypatch.setattr(readback, "OWNED_LOCAL_PREFIX", str(tmp_path) + "/")
    def transfer(command, **kwargs):
        target = root / "5000"
        target.mkdir(parents=True)
        (target / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 123 if committed else None}))
        for name in ("params/_METADATA", "train_state/_METADATA", "assets/learning_rate_contract.json",
                     "assets/subtask_mixture.json", "assets/subtask_mixture_execution.json"):
            if missing_optimizer and name == "train_state/_METADATA":
                continue
            p = target / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text("{}")
        assert command[2].startswith("oss://traffic-model/wangfaye/result/behavior-challenge/")
        assert "--recursive" in command and "--force" not in command
        return SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(readback.subprocess, "run", transfer)
    arguments = ("/data/oss_bucket_0/wangfaye/result/behavior-challenge/owned/checkpoints", 5000,
                 str(root), str(tool), "https://oss-cn-zhangjiakou.aliyuncs.com", digest)
    if committed and not missing_optimizer:
        assert readback.copy_committed_checkpoint(*arguments) == str(root)
        with pytest.raises(ValueError, match="Never overwrite"):
            readback.copy_committed_checkpoint(*arguments)
    else:
        with pytest.raises(ValueError, match="component missing" if committed else "not committed"):
            readback.copy_committed_checkpoint(*arguments)


def test_readback_rejects_unowned_source_before_running_transfer(tmp_path):
    with pytest.raises(ValueError, match="user's result prefix"):
        readback.copy_committed_checkpoint("/data/oss_bucket_0/other-user/result", 2,
            "/var/tmp/wangfaye/four-source-readback/x", tmp_path/"not-executed", "https://oss-cn-zhangjiakou.aliyuncs.com", "hash")
