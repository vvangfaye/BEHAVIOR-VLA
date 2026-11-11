#!/bin/bash
# Cluster job bootstrap for RLinf runs (2026-11-04). Inputs (environment, set by the submit script):
#   B1K_TAG          run tag; outputs go to /mnt/afs/260010165/rlinf_runs_20261104/$B1K_TAG
#   B1K_PARTS        space-separated bundle parts to fetch (e.g. "sys_extra.tar.gz uv_python.tar.gz ...")
#   B1K_GET_<i>      presigned GET URL for part i (same order as B1K_PARTS); B1K_SHA_<i> its sha256
#   B1K_PUT_LOG / B1K_PUT_TB / B1K_PUT_STATUS   presigned PUT URLs for console.log, tensorboard tar, status json
#   B1K_VENV         .venv | .venv-calvin | .venv-robocasa365
#   B1K_ENTRY        train | eval
#   B1K_CONFIG       Hydra config name under /mnt/data1/faye/code/libero_rlinf_20261103/configs
#   B1K_OVERRIDES    extra Hydra overrides (space separated, no spaces inside an override)
set -uo pipefail
OUT=/mnt/afs/260010165/rlinf_runs_20261104/$B1K_TAG
mkdir -p "$OUT" /opt/b1k_dl
exec >> "$OUT/bootstrap.log" 2>&1
DIAG_PID=""
trap 'diag_rc=$?; [ -z "$DIAG_PID" ] || kill "$DIAG_PID" 2>/dev/null || true; printf "%s bootstrap_exit=%s\n" "$(date -Iseconds)" "$diag_rc" >> "$OUT/termination.log"; exit "$diag_rc"' EXIT
trap 'exit 143' TERM
status() { printf '{"tag":"%s","state":"%s","time":"%s","host":"%s"}\n' "$B1K_TAG" "$1" "$(date -Iseconds)" "$(hostname)" > /tmp/b1k_status.json
           curl -s -m 60 -X PUT --upload-file /tmp/b1k_status.json "$B1K_PUT_STATUS" >/dev/null || true; }
status bootstrap
# 1. fetch + verify bundle parts (cluster-internal AOSS endpoint)
i=0
for part in $B1K_PARTS; do
    url_var="B1K_GET_$i"; sha_var="B1K_SHA_$i"
    ( curl -s -L -m 3600 --retry 5 -o "/opt/b1k_dl/$part" "${!url_var}" && echo "${!sha_var}  /opt/b1k_dl/$part" | sha256sum -c --quiet ) &
    i=$((i + 1))
done
wait || true
for part in $B1K_PARTS; do [ -s "/opt/b1k_dl/$part" ] || { status "fetch_failed:$part"; exit 2; }; done
i=0; for part in $B1K_PARTS; do sha_var="B1K_SHA_$i"; echo "${!sha_var}  /opt/b1k_dl/$part" | sha256sum -c --quiet || { status "sha_failed:$part"; exit 2; }; i=$((i + 1)); done
status fetched
# 2. unpack to the same absolute paths as on the build machine
M="/mnt/data1/faye/code/Behavior-1k Challenge/analysis/libero_feasibility_20261103/models"
mkdir -p /mnt/data1/faye/code/libero_rlinf_20261103 "$M" /home/faye/.local/share/uv/python /root/.cache
for part in $B1K_PARTS; do
    case "$part" in
        sys_extra.tar.gz) tar -C /opt -xzf "/opt/b1k_dl/$part"; mkdir -p /root/.libero; cp /opt/sys_extra/home/.libero/config.yaml /root/.libero/
                          cp -r /opt/sys_extra/home/.cache/openpi /root/.cache/ ;;
        uv_python.tar.gz) tar -C /home/faye/.local/share/uv/python -xzf "/opt/b1k_dl/$part" ;;
        rlinf_code_venvs.tar.gz) tar -C /mnt/data1/faye/code/libero_rlinf_20261103 -I pigz -xf "/opt/b1k_dl/$part" 2>/dev/null || \
                                 tar -C /mnt/data1/faye/code/libero_rlinf_20261103 -xzf "/opt/b1k_dl/$part" ;;
        models_*.tar) tar -C "$M" -xf "/opt/b1k_dl/$part" ;;
        robocasa_assets.tar) tar -C /mnt/data1/faye/code/libero_rlinf_20261103 -xf "/opt/b1k_dl/$part" ;;
        hf_cache.tar.gz) mkdir -p /home/faye/.cache/huggingface/hub; tar -C /home/faye/.cache/huggingface/hub -xzf "/opt/b1k_dl/$part" ;;  # LIBERO assets (venv symlink target)
        overlay_*.tar.gz) tar -C /mnt/data1/faye/code/libero_rlinf_20261103 -xzf "/opt/b1k_dl/$part" ;;  # current configs + changed RLinf files, last
    esac
    rm -f "/opt/b1k_dl/$part"
done
status unpacked
# 3. runtime environment
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source "$B1K_VENV/bin/activate"
export LD_LIBRARY_PATH="/opt/sys_extra/lib:${LD_LIBRARY_PATH:-}"
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export NCCL_P2P_DISABLE="${B1K_NCCL_P2P_DISABLE:-1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ray kills a worker above this share of the cgroup limit (default 0.95). On 5090 (240 GiB) a kill during a rollout -> actor
# transfer left jobs hung (11/5: libero_seed2_arv2, pirl_repro); ~11 GB of the counted memory is reclaimable page cache.
export RAY_memory_usage_threshold="${B1K_RAY_MEM_THRESHOLD:-0.95}"
export ROBOCASA_ASSETS_PATH=/mnt/data1/faye/code/libero_rlinf_20261103/robocasa_assets
export RAY_TMPDIR=/tmp/ray_b1k; mkdir -p "$RAY_TMPDIR"
export HF_HOME=/home/faye/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no HF access from jobs
git config --global --add safe.directory '*'   # calvin_env logs its git commit; files are owned by the build uid
python -c "import ctypes; ctypes.CDLL('libEGL.so.1'); print('libEGL.so.1 ok')" > "$OUT/env_check.txt" 2>&1
nvidia-smi >> "$OUT/env_check.txt" 2>&1
# optional: evaluate a checkpoint of an earlier cluster run (B1K_EVAL_CKPT_TAG / B1K_EVAL_CKPT_STEP). RLinf's eval loads
# strictly and the eval policy has no value head, so a copy without value_head.* tensors is evaluated.
if [ -n "${B1K_EVAL_CKPT_TAG:-}" ]; then
    CK=$(find "/mnt/afs/260010165/rlinf_runs_20261104/$B1K_EVAL_CKPT_TAG" -path "*checkpoints/global_step_${B1K_EVAL_CKPT_STEP}/actor/model_state_dict/full_weights.pt" 2>/dev/null | head -1)
    [ -n "$CK" ] || { status "ckpt_missing:${B1K_EVAL_CKPT_TAG}:${B1K_EVAL_CKPT_STEP}"; exit 2; }
    python -c "import sys, torch; sd = torch.load(sys.argv[1], map_location='cpu'); d = [k for k in sd if k.startswith('value_head.')]; [sd.pop(k) for k in d]; torch.save(sd, sys.argv[2]); print('dropped', len(d), 'value_head tensors, kept', len(sd))" \
        "$CK" /opt/b1k_eval_novalue.pt > "$OUT/eval_ckpt_source.txt" 2>&1 || { status "ckpt_strip_failed"; exit 2; }
    echo "$CK" >> "$OUT/eval_ckpt_source.txt"
    B1K_OVERRIDES="$B1K_OVERRIDES runner.ckpt_path=/opt/b1k_eval_novalue.pt"
fi
# 4. periodic log sync to AOSS (console + tensorboard), every 5 min
# Host-memory trace (2026-11-04): the 5090x4 spec has 240 GiB, which bounds the env count and the actor prefix cache;
# one line per sync with the cgroup usage and the RSS summed by process kind (RSS counts shared pages per process).
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | sed 's/^/cgroup memory limit: /' >> "$OUT/env_check.txt"
touch "$OUT/mem.log"
mem_log() {
    local cg=""
    if [ -r /sys/fs/cgroup/memory.current ]; then cg=$(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))
    elif [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then cg=$(( $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) / 1048576 )); fi
    printf '%s cgroup_mib=%s %s\n' "$(date +%T)" "$cg" "$(ps -eo rss=,args= | awk '{k = "other"
        if ($0 ~ /Actor/) k = "actor"; else if ($0 ~ /Rollout/) k = "rollout"; else if ($0 ~ /multiprocessing.spawn|Env/) k = "env"
        s[k] += $1; n[k]++} END {for (k in s) printf "%s=%dx%.1fG ", k, n[k], s[k] / 1048576}')" >> "$OUT/mem.log"
}
sync_logs() {
    mem_log
    if [ "$B1K_ENTRY" = multi ]; then
        # multi-stage job: latest stage console as console.log; all stage consoles + tb + status as tensorboard.tgz.
        # Only existing files are archived (stages_status.json appears after the first stage; a missing name made tar fail).
        last=$(ls -t "$OUT"/*/console.log 2>/dev/null | head -1)
        [ -n "$last" ] && curl -s -m 120 -X PUT --upload-file "$last" "$B1K_PUT_LOG" >/dev/null
        (cd "$OUT" && files=$(ls -d stages_status.json mem.log bootstrap.log termination.log resource_health.log multi_driver.log */console.log 2>/dev/null; find . \( -name "events.out.tfevents.*" -o -name decisions.jsonl \) 2>/dev/null) \
            && [ -n "$files" ] && tar -czf /tmp/b1k_tb.tgz $files) && curl -s -m 300 -X PUT --upload-file /tmp/b1k_tb.tgz "$B1K_PUT_TB" >/dev/null
        # verifier frames of the stages so far, re-uploaded when new frames appeared (the external verifier can judge a
        # finished Ver stage before the whole job ends; stages_status.json says which stages are complete)
        if [ -n "${B1K_PUT_FRAMES:-}" ] && [ -d /opt/b1k_frames ]; then
            if [ ! -f /tmp/b1k_frames_mark ] || [ -n "$(find /opt/b1k_frames -type f -newer /tmp/b1k_frames_mark 2>/dev/null | head -1)" ]; then
                touch /tmp/b1k_frames_mark
                tar -C /opt/b1k_frames -czf /tmp/b1k_frames_part.tgz . 2>/dev/null \
                    && curl -s -m 600 -X PUT --upload-file /tmp/b1k_frames_part.tgz "$B1K_PUT_FRAMES" >/dev/null
            fi
        fi
        return
    fi
    [ -f "$OUT/console.log" ] && curl -s -m 120 -X PUT --upload-file "$OUT/console.log" "$B1K_PUT_LOG" >/dev/null
    tb=$(find "$OUT" -name "events.out.tfevents.*" 2>/dev/null | head -5)
    if [ -n "$tb" ]; then tar -czf /tmp/b1k_tb.tgz $tb -C "$OUT" mem.log 2>/dev/null && curl -s -m 120 -X PUT --upload-file /tmp/b1k_tb.tgz "$B1K_PUT_TB" >/dev/null; fi
}
( while true; do sleep 300; sync_logs; done ) &
SYNC_PID=$!
status running
diag_parent=$$
(
    while kill -0 "$diag_parent" 2>/dev/null; do
        date -Iseconds
        for diag_file in memory.current memory.events memory.max; do
            printf "%s " "$diag_file"
            cat "/sys/fs/cgroup/$diag_file" 2>/dev/null || true
        done
        sleep 10
    done
) >> "$OUT/resource_health.log" 2>&1 &
DIAG_PID=$!
if [ "$B1K_ENTRY" = multi ]; then
    echo "$B1K_MULTI_B64" | base64 -d > /tmp/b1k_multi.py
    B1K_OUT="$OUT" python /tmp/b1k_multi.py > "$OUT/multi_driver.log" 2>&1
    RC=$?
    [ -n "${B1K_PUT_FRAMES:-}" ] && [ -d /opt/b1k_frames ] && B1K_FRAMES_DIR=/opt/b1k_frames
else
    if [ "$B1K_ENTRY" = eval ]; then SCRIPT=evaluations/eval_embodied_agent.py; else SCRIPT=examples/embodiment/train_embodied_agent.py; fi
    python "$SCRIPT" --config-path /mnt/data1/faye/code/libero_rlinf_20261103/configs/ --config-name "$B1K_CONFIG" \
        "runner.logger.log_path=$OUT" $B1K_OVERRIDES > "$OUT/console.log" 2>&1
    RC=$?
fi
kill $SYNC_PID 2>/dev/null; sync_logs
for _ in $(seq 60); do pgrep -f b1k_frames_part.tgz >/dev/null 2>&1 || break; sleep 5; done  # let a periodic frames upload finish first
# optional: eval frame dump for the external verifier (B1K_FRAMES_DIR set by submit_rlinf.py --frames)
if [ -n "${B1K_FRAMES_DIR:-}" ] && [ -d "$B1K_FRAMES_DIR" ] && [ -n "${B1K_PUT_FRAMES:-}" ]; then
    tar -C "$B1K_FRAMES_DIR" -czf /tmp/b1k_frames.tgz . && curl -s -m 600 -X PUT --upload-file /tmp/b1k_frames.tgz "$B1K_PUT_FRAMES" >/dev/null
fi
status "finished_rc_$RC"
exit $RC
