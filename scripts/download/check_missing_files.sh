#!/usr/bin/env bash

set -euo pipefail

REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos}"
REVISION="${REVISION:-v3.0}"
LOCAL_DIR="${LOCAL_DIR:-/data/2026-challenge-demos}"

HF_HUB_DISABLE_XET=1 \
REPO_ID="$REPO_ID" \
REVISION="$REVISION" \
LOCAL_DIR="$LOCAL_DIR" \
python3 - <<'PY'
import os
from pathlib import Path

from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
revision = os.environ["REVISION"]
local_dir = Path(os.environ["LOCAL_DIR"])

remote_files = HfApi().dataset_info(
    repo_id,
    revision=revision,
    files_metadata=True,
).siblings or []

local_sizes: dict[str, int] = {}
if local_dir.exists():
    for base, dirs, files in os.walk(local_dir):
        base_path = Path(base)
        relative_base = base_path.relative_to(local_dir)

        if relative_base == Path("."):
            dirs[:] = [directory for directory in dirs if directory != ".cache"]
        if relative_base == Path(".cache") or ".cache" in relative_base.parts[:1]:
            continue

        for filename in files:
            path = base_path / filename
            relative_path = path.relative_to(local_dir).as_posix()
            try:
                local_sizes[relative_path] = path.stat().st_size
            except OSError:
                pass

matched = 0
missing = 0
size_mismatch = 0
remaining_bytes = 0

for remote in remote_files:
    actual_size = local_sizes.get(remote.rfilename)
    expected_size = remote.size

    if actual_size is None:
        missing += 1
        remaining_bytes += expected_size or 0
    elif expected_size is not None and actual_size != expected_size:
        size_mismatch += 1
        remaining_bytes += expected_size
    else:
        matched += 1

unfinished = missing + size_mismatch

print(f"远端文件总数：{len(remote_files)}")
print(f"大小匹配完成：{matched}")
print(f"本地缺失：{missing}")
print(f"大小不一致：{size_mismatch}")
print(f"尚未完整下载：{unfinished}")
print(f"待下载数据量：{remaining_bytes / 1_000_000_000:.2f} GB")
PY
