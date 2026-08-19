#!/usr/bin/env bash
# Sweep a model over many tasks by running several eval_b1k.sh workers in
# parallel, one per GPU pair.
#
# Usage:
#   MODEL=pi05 ./scripts/eval_sweep.sh                      # all of scripts/b1k_tasks.txt
#   MODEL=pi05 WORKERS=2 ./scripts/eval_sweep.sh
#   MODEL=pi05 TASK_FILE=/tmp/subset.txt ./scripts/eval_sweep.sh
#   MODEL=groot GROOT_CKPT=/path/checkpoint-50000 ./scripts/eval_sweep.sh
#
# Each worker owns two GPUs (policy server + simulator) and its own port, and
# takes a disjoint slice of the task list, so workers never contend for a port,
# a GPU, or an output directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:?set MODEL=pi05 or MODEL=groot}"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/scripts/b1k_tasks.txt}"
WORKERS="${WORKERS:-4}"
BASE_PORT="${BASE_PORT:-8000}"
# Indices into the split's instance list. 5 of the 20 public_test instances
# covers every task at a fifth of the cost.
INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3 4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_results}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/$MODEL/sweep_logs}"

[[ -f "$TASK_FILE" ]] || { echo "task file not found: $TASK_FILE" >&2; exit 1; }

mapfile -t ALL_TASKS < <(grep -vE '^\s*(#|$)' "$TASK_FILE")
(( ${#ALL_TASKS[@]} )) || { echo "no tasks in $TASK_FILE" >&2; exit 1; }

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
if (( WORKERS * 2 > GPU_COUNT )); then
    echo "WORKERS=$WORKERS needs $((WORKERS * 2)) GPUs but only $GPU_COUNT present" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
echo "sweeping $MODEL over ${#ALL_TASKS[@]} tasks with $WORKERS workers"
echo "instances per task: $INSTANCE_INDICES"
echo "logs: $LOG_DIR"

PIDS=()
for ((w = 0; w < WORKERS; w++)); do
    # Round-robin so a worker that draws several slow tasks does not decide the
    # total wall time on its own.
    shard=()
    for ((i = w; i < ${#ALL_TASKS[@]}; i += WORKERS)); do
        shard+=("${ALL_TASKS[i]}")
    done
    (( ${#shard[@]} )) || continue

    log="$LOG_DIR/worker_$w.log"
    echo "worker $w: ${#shard[@]} tasks, gpus $((w * 2))/$((w * 2 + 1)), port $((BASE_PORT + w)), log $log"
    env MODEL="$MODEL" \
        PORT="$((BASE_PORT + w))" \
        SERVER_GPU="$((w * 2))" \
        EVAL_GPU="$((w * 2 + 1))" \
        INSTANCE_INDICES="$INSTANCE_INDICES" \
        OUTPUT_ROOT="$OUTPUT_ROOT" \
        "$REPO_ROOT/scripts/eval_b1k.sh" "${shard[@]}" > "$log" 2>&1 &
    PIDS+=($!)
done

status=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || status=1
done

echo "sweep finished (status $status); results under $OUTPUT_ROOT/$MODEL"
exit "$status"
