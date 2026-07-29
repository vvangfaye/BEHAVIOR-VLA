#!/usr/bin/env bash

set -u

REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos}"
REVISION="${REVISION:-v3.0}"
LOCAL_DIR="${LOCAL_DIR:-/data/2026-challenge-demos}"
MAX_WORKERS="${MAX_WORKERS:-4}"
RETRY_DELAY="${RETRY_DELAY:-10}"
MANIFEST="${MANIFEST:-/tmp/hf-download-needed.txt}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

if pgrep -f '(^|/)(hf|huggingface-cli) download ' >/dev/null 2>&1; then
  log "错误：检测到已有 hf download 进程。请先停止旧下载任务，再运行本脚本。"
  exit 1
fi

while true; do
  log "核算远端 ${REVISION} 与本地文件大小……"

  if ! HF_HUB_DISABLE_XET=1 \
    REPO_ID="$REPO_ID" \
    REVISION="$REVISION" \
    LOCAL_DIR="$LOCAL_DIR" \
    MANIFEST="$MANIFEST" \
    python3 - <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
revision = os.environ["REVISION"]
local_dir = Path(os.environ["LOCAL_DIR"])
manifest = Path(os.environ["MANIFEST"])
temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")

info = HfApi().dataset_info(
    repo_id,
    revision=revision,
    files_metadata=True,
)
remote_files = info.siblings or []

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

needed: list[tuple[str, int, str]] = []
matched = 0
matched_bytes = 0
needed_bytes = 0

for remote in remote_files:
    expected_size = remote.size
    actual_size = local_sizes.get(remote.rfilename)

    if expected_size is not None and actual_size == expected_size:
        matched += 1
        matched_bytes += expected_size
        continue

    reason = "missing" if actual_size is None else "size-mismatch"
    size = expected_size or 0
    needed.append((remote.rfilename, size, reason))
    needed_bytes += size

temporary_manifest.parent.mkdir(parents=True, exist_ok=True)
with temporary_manifest.open("w", encoding="utf-8") as output:
    for path, _, _ in needed:
        if "\n" in path:
            raise RuntimeError(f"Unsupported newline in repository path: {path!r}")
        output.write(path + "\n")
temporary_manifest.replace(manifest)

missing = sum(reason == "missing" for _, _, reason in needed)
mismatch = sum(reason == "size-mismatch" for _, _, reason in needed)

print(f"远端文件：{len(remote_files)}")
print(f"大小匹配：{matched}")
print(f"缺失文件：{missing}")
print(f"大小不符：{mismatch}")
print(f"待处理文件：{len(needed)}")
print(f"待下载数据：{needed_bytes / 1_000_000_000:.2f} GB")
print(f"清单位置：{manifest}")
PY
  then
    log "核算失败，${RETRY_DELAY} 秒后重试"
    sleep "$RETRY_DELAY"
    continue
  fi

  mapfile -t needed_files < "$MANIFEST"

  if [ "${#needed_files[@]}" -eq 0 ]; then
    log "所有文件均已存在且大小与 ${REVISION} 一致。"
    exit 0
  fi

  log "开始下载 ${#needed_files[@]} 个缺失或大小不符的文件"

  HF_HUB_DISABLE_XET=1 \
    REPO_ID="$REPO_ID" \
    REVISION="$REVISION" \
    LOCAL_DIR="$LOCAL_DIR" \
    MAX_WORKERS="$MAX_WORKERS" \
    MANIFEST="$MANIFEST" \
    python3 - <<'PY'
import os

from huggingface_hub import hf_hub_download
from tqdm.contrib.concurrent import thread_map

repo_id = os.environ["REPO_ID"]
revision = os.environ["REVISION"]
local_dir = os.environ["LOCAL_DIR"]
max_workers = int(os.environ["MAX_WORKERS"])
manifest = os.environ["MANIFEST"]

with open(manifest, encoding="utf-8") as source:
    files = [line.rstrip("\n") for line in source if line.rstrip("\n")]

def download_one(filename: str) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )

thread_map(
    download_one,
    files,
    max_workers=max_workers,
    desc="Downloading required files",
)
PY

  status=$?

  if [ "$status" -eq 0 ]; then
    log "本轮下载结束，重新核算完整性"
  else
    log "本轮下载失败，退出码 ${status}；${RETRY_DELAY} 秒后重新核算并续传"
    sleep "$RETRY_DELAY"
  fi
done
