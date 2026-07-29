#!/usr/bin/env bash

set -u

attempt=0

while true; do
  attempt=$((attempt + 1))
  echo "[$(date '+%F %T')] 开始第 ${attempt} 次下载"

  HF_HUB_DISABLE_XET=1 \
    hf download behavior-1k/2026-challenge-demos \
      --repo-type dataset \
      --revision v3.0 \
      --local-dir /data/2026-challenge-demos \
      --max-workers 4

  status=$?

  if [ "$status" -eq 0 ]; then
    echo "[$(date '+%F %T')] 下载完成"
    break
  fi

  echo "[$(date '+%F %T')] 下载失败，退出码 ${status}，10 秒后继续"
  sleep 10
done
