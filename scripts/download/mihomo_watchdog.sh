#!/usr/bin/env bash

set -u

MIHOMO_CONFIG="${MIHOMO_CONFIG:-/root/.config/mihomo/config.yaml}"
SPEED_THRESHOLD_MIB="${SPEED_THRESHOLD_MIB:-2}"
LOW_SPEED_SECONDS="${LOW_SPEED_SECONDS:-300}"
CHECK_INTERVAL="${CHECK_INTERVAL:-5}"
RESTART_GRACE_SECONDS="${RESTART_GRACE_SECONDS:-30}"
CONTROLLER="${CONTROLLER:-http://127.0.0.1:9090}"
LOG_FILE="${LOG_FILE:-/tmp/mihomo-watchdog.log}"
MIHOMO_LOG="${MIHOMO_LOG:-/tmp/mihomo-managed.log}"
LOCK_FILE="${LOCK_FILE:-/tmp/mihomo-watchdog.lock}"

mihomo_pid=""
previous_total=""
low_speed_elapsed=0

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"
}

stop_mihomo() {
  if [ -n "$mihomo_pid" ] && kill -0 "$mihomo_pid" 2>/dev/null; then
    kill -TERM "$mihomo_pid" 2>/dev/null || true

    local waited=0
    while kill -0 "$mihomo_pid" 2>/dev/null && [ "$waited" -lt 10 ]; do
      sleep 1
      waited=$((waited + 1))
    done

    if kill -0 "$mihomo_pid" 2>/dev/null; then
      log "Mihomo 未及时退出，发送 KILL"
      kill -KILL "$mihomo_pid" 2>/dev/null || true
    fi
  fi

  mihomo_pid=""
}

wait_for_controller() {
  local waited=0
  while [ "$waited" -lt 30 ]; do
    if curl -fsS --max-time 2 "$CONTROLLER/version" >/dev/null 2>&1; then
      return 0
    fi

    if ! kill -0 "$mihomo_pid" 2>/dev/null; then
      log "错误：Mihomo 启动后提前退出，请检查 $MIHOMO_LOG"
      return 1
    fi

    sleep 1
    waited=$((waited + 1))
  done

  log "错误：等待 Mihomo 控制器超时"
  return 1
}

verify_hf_group() {
  python3 - "$CONTROLLER" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

controller = sys.argv[1]
endpoint = controller + "/proxies/" + urllib.parse.quote("HF-Download", safe="")

with urllib.request.urlopen(endpoint, timeout=5) as response:
    data = json.load(response)

if data.get("type") != "Selector":
    raise SystemExit(f"HF-Download 类型不是 Selector：{data.get('type')}")

current = data.get("now")
expected = "🇸🇬 新加坡 A1 | 中转"
if current != expected:
    body = json.dumps({"name": expected}).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=5):
        pass

print(expected)
PY
}

start_mihomo() {
  if [ ! -f "$MIHOMO_CONFIG" ]; then
    log "错误：配置文件不存在：$MIHOMO_CONFIG"
    return 1
  fi

  if ! mihomo -t -f "$MIHOMO_CONFIG" >>"$MIHOMO_LOG" 2>&1; then
    log "错误：配置校验失败：$MIHOMO_CONFIG"
    return 1
  fi

  log "启动 Mihomo，固定配置：$MIHOMO_CONFIG"
  mihomo -f "$MIHOMO_CONFIG" >>"$MIHOMO_LOG" 2>&1 &
  mihomo_pid=$!

  wait_for_controller || return 1

  local hf_node
  if hf_node="$(verify_hf_group 2>>"$LOG_FILE")"; then
    log "HF 固定节点：$hf_node"
  else
    log "错误：无法确认 HF 固定节点"
    stop_mihomo
    return 1
  fi

  previous_total=""
  low_speed_elapsed=0
  log "Mihomo 已启动，PID=$mihomo_pid；进入 ${RESTART_GRACE_SECONDS}s 宽限期"
  sleep "$RESTART_GRACE_SECONDS"
}

restart_mihomo() {
  log "HF 数据连接连续 ${low_speed_elapsed}s 低于 ${SPEED_THRESHOLD_MIB} MiB/s，重启同一配置"
  stop_mihomo
  start_mihomo
}

read_stats() {
  curl -fsS --max-time 3 "$CONTROLLER/connections" |
    python3 -c '
import json
import sys

data = json.load(sys.stdin)
connections = data.get("connections", [])
data_hosts = (
    "cas-bridge.xethub.hf.co",
    "cas-server.xethub.hf.co",
    "transfer.xethub.hf.co",
    "cdn.hf.co",
)
has_hf_data = any(
    any(part in ((connection.get("metadata") or {}).get("host") or "") for part in data_hosts)
    for connection in connections
)
print(data.get("downloadTotal", 0), int(has_hf_data))
'
}

cleanup() {
  log "watchdog 退出，停止由它托管的 Mihomo"
  stop_mihomo
}

trap cleanup EXIT
trap 'exit 0' INT TERM

if ! command -v mihomo >/dev/null 2>&1; then
  log "错误：找不到 mihomo"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  log "错误：找不到 curl"
  exit 1
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "错误：已有另一个 Mihomo watchdog 正在运行"
  exit 1
fi

if curl -fsS --max-time 2 "$CONTROLLER/version" >/dev/null 2>&1; then
  log "错误：检测到已有 Mihomo。请先停止旧 watchdog/Mihomo，再运行本脚本。"
  exit 1
fi

start_mihomo || exit 1

while true; do
  if ! kill -0 "$mihomo_pid" 2>/dev/null; then
    log "检测到 Mihomo 已退出，重新启动固定配置"
    start_mihomo || {
      sleep 10
      continue
    }
  fi

  stats="$(read_stats 2>/dev/null || true)"
  if [ -z "$stats" ]; then
    log "无法读取控制器统计，本轮跳过"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  read -r total_download has_hf_data <<< "$stats"

  if [ -z "$previous_total" ]; then
    previous_total="$total_download"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  delta=$((total_download - previous_total))
  previous_total="$total_download"

  if [ "$delta" -lt 0 ]; then
    low_speed_elapsed=0
    sleep "$CHECK_INTERVAL"
    continue
  fi

  speed_mib="$(awk -v bytes="$delta" -v seconds="$CHECK_INTERVAL" \
    'BEGIN { printf "%.2f", bytes / seconds / 1024 / 1024 }')"

  hf_running=0
  if pgrep -f '(^|/)(hf|huggingface-cli) download ' >/dev/null 2>&1 ||
    pgrep -f '/download_needed\.sh' >/dev/null 2>&1; then
    hf_running=1
  fi

  if [ "$hf_running" -eq 1 ] && [ "$has_hf_data" -eq 1 ]; then
    if awk -v speed="$speed_mib" -v threshold="$SPEED_THRESHOLD_MIB" \
      'BEGIN { exit !(speed < threshold) }'; then
      low_speed_elapsed=$((low_speed_elapsed + CHECK_INTERVAL))
    else
      low_speed_elapsed=0
    fi

    log "速度=${speed_mib} MiB/s，低速累计=${low_speed_elapsed}s"
  else
    low_speed_elapsed=0
  fi

  if [ "$low_speed_elapsed" -ge "$LOW_SPEED_SECONDS" ]; then
    restart_mihomo || sleep 10
  fi

  sleep "$CHECK_INTERVAL"
done
