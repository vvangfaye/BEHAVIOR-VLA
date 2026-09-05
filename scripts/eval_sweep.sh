#!/usr/bin/env bash
# Sweep a model over many tasks by running several eval_b1k.sh workers in
# parallel, one per GPU pair.
#
# Usage:
#   MODEL=pi05 ./scripts/eval_sweep.sh                      # all of scripts/b1k_tasks.txt
#   MODEL=pi05 WORKERS=2 ./scripts/eval_sweep.sh
#   MODEL=pi05 TASK_FILE=scripts/b1k_tasks_reduced.txt ./scripts/eval_sweep.sh
#   MODEL=groot GROOT_CKPT=/path/checkpoint-50000 ./scripts/eval_sweep.sh
#
# Each worker owns two GPUs (policy server + simulator) and its own port, and
# takes a disjoint slice of the task list, so workers never contend for a port,
# a GPU, or an output directory.
#
# The task file is one task per line, optionally followed by two whitespace-
# separated columns: the number of instances to run for that task, and an
# estimated step count used to balance workers. Tasks without those columns fall
# back to INSTANCE_INDICES and to equal weight.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:?set MODEL=pi05 or MODEL=groot}"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/scripts/b1k_tasks.txt}"
WORKERS="${WORKERS:-4}"
BASE_PORT="${BASE_PORT:-8000}"
COLOCATE_GPUS="${COLOCATE_GPUS:-0}"
# Indices into the split's instance list. 5 of the 20 public_test instances
# covers every task at a fifth of the cost.
INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3 4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_results}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/$MODEL/sweep_logs}"

[[ -f "$TASK_FILE" ]] || { echo "task file not found: $TASK_FILE" >&2; exit 1; }

mapfile -t RAW_LINES < <(grep -vE '^\s*(#|$)' "$TASK_FILE")
(( ${#RAW_LINES[@]} )) || { echo "no tasks in $TASK_FILE" >&2; exit 1; }

# ninst=0 means "leave the instance selection to INSTANCE_INDICES".
ALL_TASKS=()
NINST=()
WEIGHT=()
for line in "${RAW_LINES[@]}"; do
    read -r task ninst weight _rest <<< "$line"
    ALL_TASKS+=("$task")
    NINST+=("${ninst:-0}")
    WEIGHT+=("${weight:-1}")
done

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
GPUS_PER_WORKER=2
(( COLOCATE_GPUS )) && GPUS_PER_WORKER=1
if (( WORKERS * GPUS_PER_WORKER > GPU_COUNT )); then
    echo "WORKERS=$WORKERS needs $((WORKERS * GPUS_PER_WORKER)) GPUs but only $GPU_COUNT present" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
echo "sweeping $MODEL over ${#ALL_TASKS[@]} tasks with $WORKERS workers"
echo "default instances per task: $INSTANCE_INDICES"
echo "co-located policy/simulator GPUs: $COLOCATE_GPUS"
echo "logs: $LOG_DIR"

# Longest-processing-time-first: hand the heaviest remaining task to whichever
# worker is currently least loaded. Round-robin over a list whose per-task cost
# spans 20x left one worker running 15 h after the others had finished.
mapfile -t ORDER < <(
    for i in "${!ALL_TASKS[@]}"; do printf '%s %s\n' "${WEIGHT[i]}" "$i"; done |
        sort -k1,1rn -k2,2n | awk '{print $2}'
)

declare -a LOAD SHARD
for ((w = 0; w < WORKERS; w++)); do LOAD[w]=0; SHARD[w]=""; done
for i in "${ORDER[@]}"; do
    best=0
    for ((w = 1; w < WORKERS; w++)); do
        (( LOAD[w] < LOAD[best] )) && best=$w
    done
    LOAD[best]=$(( LOAD[best] + WEIGHT[i] ))
    SHARD[best]+="${ALL_TASKS[i]} ${NINST[i]}"$'\n'
done

PIDS=()
for ((w = 0; w < WORKERS; w++)); do
    spec="${SHARD[w]}"
    [[ -n "$spec" ]] || continue
    ntask="$(printf '%s' "$spec" | grep -c .)"
    if (( COLOCATE_GPUS )); then
        server_gpu="$w"
        eval_gpu="$w"
    else
        server_gpu="$((w * 2))"
        eval_gpu="$((w * 2 + 1))"
    fi

    log="$LOG_DIR/worker_$w.log"
    echo "worker $w: $ntask tasks, weight ${LOAD[w]}, gpus $server_gpu/$eval_gpu, port $((BASE_PORT + w)), log $log"
    (
        # One eval_b1k.sh call per distinct instance count, since --instance-indices
        # applies to every task in a call.
        for n in $(printf '%s' "$spec" | awk 'NF {print $2}' | sort -un); do
            group=()
            while read -r task ninst; do
                [[ -n "$task" && "$ninst" == "$n" ]] && group+=("$task")
            done <<< "$spec"
            (( ${#group[@]} )) || continue

            if (( n == 0 )); then
                indices="$INSTANCE_INDICES"
            else
                indices="$(seq 0 $((n - 1)) | tr '\n' ' ')"
            fi
            echo "==> worker $w: ${#group[@]} tasks with instances [$indices]"
            env MODEL="$MODEL" \
                PORT="$((BASE_PORT + w))" \
                SERVER_GPU="$server_gpu" \
                EVAL_GPU="$eval_gpu" \
                INSTANCE_INDICES="$indices" \
                OUTPUT_ROOT="$OUTPUT_ROOT" \
                "$REPO_ROOT/scripts/eval_b1k.sh" "${group[@]}"
        done
    ) > "$log" 2>&1 &
    PIDS+=($!)
done

status=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || status=1
done

echo "sweep finished (status $status); results under $OUTPUT_ROOT/$MODEL"
exit "$status"
