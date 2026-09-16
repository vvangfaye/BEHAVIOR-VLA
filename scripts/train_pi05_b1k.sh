#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENPI_DIR="${OPENPI_DIR:-$CODE_ROOT/openpi}"
DATASET_ROOT="${DATASET_ROOT:-/data/2026-challenge-demos}"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
EXP_NAME="${EXP_NAME:-pi05_b1k_${MODE}_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Optional (default off = unchanged behavior):
#   SUBTASK_PROMPTS=<segment parquet>  -> condition on the current sub-task
#   FRAME_STRIDE=<int>                 -> subsample training frames by this stride
#   EPISODES_FILE=<newline list>       -> train only the listed original episode indices
SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
EPISODES_FILE="${EPISODES_FILE:-}"
SUBTASK_PREDICTION="${SUBTASK_PREDICTION:-0}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$OPENPI_DIR/outputs/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$OPENPI_DIR/outputs/checkpoints}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
KEEP_PERIOD="${KEEP_PERIOD:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
EXPECTED_TASKS="${EXPECTED_TASKS:-}"
EXPECTED_RETAINED_EPISODES="${EXPECTED_RETAINED_EPISODES:-}"
EXPECTED_RETAINED_FRAMES="${EXPECTED_RETAINED_FRAMES:-}"
EXPECTED_BATCHES_PER_EPOCH="${EXPECTED_BATCHES_PER_EPOCH:-}"
EXPECTED_TRAIN_STEPS="${EXPECTED_TRAIN_STEPS:-}"
if [[ "$SUBTASK_PREDICTION" != 0 && "$SUBTASK_PREDICTION" != 1 ]]; then
    echo "SUBTASK_PREDICTION must be 0 or 1" >&2
    exit 2
fi
if [[ "$SUBTASK_PREDICTION" == 1 ]]; then
    SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-$DATASET_ROOT/meta/subtask_segments.parquet}"
    SUBTASK_EMPTY_LABEL_POLICY="${SUBTASK_EMPTY_LABEL_POLICY:-drop_episode}"
    MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-512}"
    MAX_SUBTASK_LEN="${MAX_SUBTASK_LEN:-64}"
    if [[ "${WHOLE_TASK_PROB:-0}" != 0 && "${WHOLE_TASK_PROB:-0}" != 0.0 ]]; then
        echo "Subtask prediction preserves the whole task; WHOLE_TASK_PROB must be 0." >&2
        exit 2
    fi
fi

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
    echo "Dataset not found or incomplete: $DATASET_ROOT/meta/info.json" >&2
    exit 1
fi
if [[ ! -x "$OPENPI_DIR/.venv/bin/python" ]]; then
    echo "OpenPI environment not found: $OPENPI_DIR/.venv" >&2
    exit 1
fi

case "$MODE" in
    smoke)
        REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos-smoke}"
        EPISODES_INDEX=(0 1)
        TRAIN_STEPS="${TRAIN_STEPS:-2}"
        BATCH_SIZE="${BATCH_SIZE:-$NUM_GPUS}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_STEPS}"
        NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-256}"
        ;;
    full)
        REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos}"
        EPISODES_INDEX=()
        TRAIN_STEPS="${TRAIN_STEPS:-50000}"
        BATCH_SIZE="${BATCH_SIZE:-64}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
        NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-500000}"
        ;;
    *)
        echo "Usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac

if [[ -n "$EPISODES_FILE" ]]; then
    if [[ ! -f "$EPISODES_FILE" ]]; then
        echo "Episode index file not found: $EPISODES_FILE" >&2
        exit 1
    fi
    EPISODES_INDEX=()
    while IFS= read -r EPISODE_INDEX || [[ -n "$EPISODE_INDEX" ]]; do
        EPISODES_INDEX+=("$EPISODE_INDEX")
    done < "$EPISODES_FILE"
    if (( ${#EPISODES_INDEX[@]} == 0 )); then
        echo "Episode index file is empty: $EPISODES_FILE" >&2
        exit 1
    fi
    for EPISODE_INDEX in "${EPISODES_INDEX[@]}"; do
        if [[ ! "$EPISODE_INDEX" =~ ^[0-9]+$ ]]; then
            echo "Invalid episode index '$EPISODE_INDEX' in $EPISODES_FILE" >&2
            exit 1
        fi
    done
fi

if [[ "$NUM_GPUS" -ne "$(awk -F, '{print NF}' <<<"$CUDA_DEVICES")" ]]; then
    echo "NUM_GPUS=$NUM_GPUS does not match CUDA_DEVICES=$CUDA_DEVICES" >&2
    exit 2
fi
TOTAL_DEVICES="${OPENPI_EXPECTED_GLOBAL_DEVICES:-$NUM_GPUS}"
if (( BATCH_SIZE % TOTAL_DEVICES != 0 )); then
    echo "BATCH_SIZE=$BATCH_SIZE must be divisible by total devices=$TOTAL_DEVICES" >&2
    exit 2
fi

cd "$OPENPI_DIR"
set +u
source .venv/bin/activate
set -u

# openpi shards every batch across all devices jax can see, so the device set
# must be restricted before the norm-stats step too, not just for training.
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

if [[ "$SUBTASK_PREDICTION" == 1 ]]; then
    PREFLIGHT_OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-outputs/subtask_preflight}"
    mkdir -p "$PREFLIGHT_OUTPUT_DIR"
    VALIDATED_EPISODES_FILE="$PREFLIGHT_OUTPUT_DIR/$EXP_NAME.episodes.txt"
    PREFLIGHT_REPORT_FILE="$PREFLIGHT_OUTPUT_DIR/$EXP_NAME.json"
    PREFLIGHT_ARGS=(
        --dataset-root "$DATASET_ROOT"
        --segments-path "$SUBTASK_PROMPTS"
        --max-token-len "$MAX_TOKEN_LEN"
        --max-subtask-len "$MAX_SUBTASK_LEN"
        --empty-label-policy "$SUBTASK_EMPTY_LABEL_POLICY"
        --frame-stride "$FRAME_STRIDE"
        --report-file "$PREFLIGHT_REPORT_FILE"
        --write-episodes-file "$VALIDATED_EPISODES_FILE"
    )
    if [[ -n "$TRAIN_EPOCHS" ]]; then
        PREFLIGHT_ARGS+=(
            --global-batch-size "$BATCH_SIZE"
            --process-count "${OPENPI_PROCESS_COUNT:-1}"
            --epochs "$TRAIN_EPOCHS"
        )
    fi
    if [[ -n "$EXPECTED_TASKS" ]]; then
        PREFLIGHT_ARGS+=(--expected-tasks "$EXPECTED_TASKS")
    fi
    if (( ${#EPISODES_INDEX[@]} )); then
        PREFLIGHT_ARGS+=(--episodes-index "${EPISODES_INDEX[@]}")
    fi
    python scripts/b1k/check_subtask_data.py "${PREFLIGHT_ARGS[@]}"
    # Use exactly the validated subset for both norm stats and training. The
    # source segment table and the caller's EPISODES_FILE remain untouched.
    EPISODES_INDEX=()
    while IFS= read -r EPISODE_INDEX || [[ -n "$EPISODE_INDEX" ]]; do
        if [[ ! "$EPISODE_INDEX" =~ ^[0-9]+$ ]]; then
            echo "Invalid validated episode id '$EPISODE_INDEX' in $VALIDATED_EPISODES_FILE" >&2
            exit 2
        fi
        EPISODES_INDEX+=("$EPISODE_INDEX")
    done < "$VALIDATED_EPISODES_FILE"
    if (( ${#EPISODES_INDEX[@]} == 0 )); then
        echo "No fully annotated episodes remain; training will not start." >&2
        exit 2
    fi
    echo "Subtask preflight retained ${#EPISODES_INDEX[@]} episodes: $VALIDATED_EPISODES_FILE"
    if [[ -n "$TRAIN_EPOCHS" ]]; then
        read -r CONTRACT_EPISODES CONTRACT_FRAMES CONTRACT_BATCHES CONTRACT_STEPS CONTRACT_SHA < <(
            python - "$PREFLIGHT_REPORT_FILE" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
contract = report["training_contract"]
print(
    contract["retained_episodes"],
    contract["retained_frames"],
    contract["batches_per_epoch"],
    contract["train_steps"],
    report["training_contract_sha256"],
)
PY
        )
        for actual_expected in \
            "$CONTRACT_EPISODES:${EXPECTED_RETAINED_EPISODES:-$CONTRACT_EPISODES}:retained episodes" \
            "$CONTRACT_FRAMES:${EXPECTED_RETAINED_FRAMES:-$CONTRACT_FRAMES}:retained frames" \
            "$CONTRACT_BATCHES:${EXPECTED_BATCHES_PER_EPOCH:-$CONTRACT_BATCHES}:batches per epoch" \
            "$CONTRACT_STEPS:${EXPECTED_TRAIN_STEPS:-$CONTRACT_STEPS}:train steps"; do
            IFS=: read -r actual expected label <<< "$actual_expected"
            [[ "$actual" == "$expected" ]] || {
                echo "Training contract mismatch for $label: got $actual, expected $expected" >&2
                exit 2
            }
        done
        [[ "$TRAIN_STEPS" == "$CONTRACT_STEPS" ]] || {
            echo "TRAIN_STEPS=$TRAIN_STEPS does not match contract train_steps=$CONTRACT_STEPS" >&2
            exit 2
        }
        export OPENPI_EXPECTED_BATCHES_PER_EPOCH="$CONTRACT_BATCHES"
        export OPENPI_TRAIN_EPOCHS="$TRAIN_EPOCHS"
        export OPENPI_TRAINING_CONTRACT_SHA256="$CONTRACT_SHA"
    fi
    if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
        exit 0
    fi
fi

NORM_FILE="$ASSETS_BASE_DIR/pi05_b1k/$REPO_ID/norm_stats.json"
if [[ ! -f "$NORM_FILE" ]]; then
    echo "Computing normalization statistics..."
    NORM_ARGS=(
        --config-name pi05_b1k
        --dataset-root "$DATASET_ROOT"
        --repo-id "$REPO_ID"
        --batch-size "$BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --assets-base-dir "$ASSETS_BASE_DIR"
    )
    if [[ -n "$NORM_MAX_FRAMES" ]]; then
        NORM_ARGS+=(--max-frames "$NORM_MAX_FRAMES")
    fi
    if (( ${#EPISODES_INDEX[@]} )); then
        NORM_ARGS+=(--episodes-index "${EPISODES_INDEX[@]}")
    fi
    python scripts/compute_norm_stats.py "${NORM_ARGS[@]}"
fi

echo "Starting π0.5 $MODE run: $EXP_NAME"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# per-step loss goes through tqdm's pbar.write -> stdout, which block-buffers
# when redirected to a log file and would hide loss for ~12k steps at a time.
export PYTHONUNBUFFERED=1
TRAIN_DATA_ARGS=()
if [[ -n "${INIT_PARAMS:-}" ]]; then
    TRAIN_DATA_ARGS+=(--weight-loader.params-path "$INIT_PARAMS")
fi
if (( ${#EPISODES_INDEX[@]} )); then
    TRAIN_DATA_ARGS+=(--data.episodes-index "${EPISODES_INDEX[@]}")
fi
if [[ -n "${FSDP_DEVICES:-}" ]]; then
    TRAIN_DATA_ARGS+=(--fsdp-devices "$FSDP_DEVICES")
fi
if [[ -n "$SUBTASK_PROMPTS" ]]; then
    TRAIN_DATA_ARGS+=(--data.subtask-prompts-path "$SUBTASK_PROMPTS")
    if [[ "$SUBTASK_PREDICTION" == 1 ]]; then
        TRAIN_DATA_ARGS+=(
            --data.subtask-whole-task-prob 0
            --model.train-subtask-prediction
            --model.max-token-len "$MAX_TOKEN_LEN"
            --model.max-subtask-len "$MAX_SUBTASK_LEN"
            --model.subtask-loss-weight "${SUBTASK_LOSS_WEIGHT:-1.0}"
            --model.subtask-loss-chunk-size "${SUBTASK_LOSS_CHUNK_SIZE:-8}"
        )
    else
        TRAIN_DATA_ARGS+=(--data.subtask-whole-task-prob "${WHOLE_TASK_PROB:-0.15}")
    fi
fi
if (( FRAME_STRIDE > 1 )); then
    TRAIN_DATA_ARGS+=(--data.frame-stride "$FRAME_STRIDE")
fi

# CosineDecaySchedule defaults to decay_steps=30_000 regardless of
# --num-train-steps, so a longer run silently spends everything past 30k at the
# 2.5e-6 floor. Set LR_DECAY_STEPS to the real run length to stretch the cosine.
if [[ -n "${LR_DECAY_STEPS:-}" ]]; then
    if [[ -n "$TRAIN_EPOCHS" && "$LR_DECAY_STEPS" != "$TRAIN_STEPS" ]]; then
        echo "LR_DECAY_STEPS=$LR_DECAY_STEPS must equal exact-epoch TRAIN_STEPS=$TRAIN_STEPS" >&2
        exit 2
    fi
    TRAIN_DATA_ARGS+=(--lr-schedule.decay-steps "$LR_DECAY_STEPS")
fi
if [[ -n "$KEEP_PERIOD" ]]; then
    TRAIN_DATA_ARGS+=(--keep-period "$KEEP_PERIOD")
fi
if [[ -n "${LR_PEAK:-}" ]]; then
    TRAIN_DATA_ARGS+=(--lr-schedule.peak-lr "$LR_PEAK")
fi
if [[ -n "${LR_FLOOR:-}" ]]; then
    TRAIN_DATA_ARGS+=(--lr-schedule.decay-lr "$LR_FLOOR")
fi

# openpi refuses --resume together with --overwrite. RESUME=1 continues an
# interrupted run from the last checkpoint in the same EXP_NAME directory.
if [[ "${RESUME:-0}" == 1 && "${OVERWRITE:-0}" == 1 ]]; then
    echo "RESUME=1 and OVERWRITE=1 cannot be used together." >&2
    exit 2
fi
if [[ "${RESUME:-0}" == 1 ]]; then
    TRAIN_DATA_ARGS+=(--resume)
elif [[ "$SUBTASK_PREDICTION" == 0 || "${OVERWRITE:-0}" == 1 ]]; then
    TRAIN_DATA_ARGS+=(--overwrite)
fi

python scripts/b1k/train_b1k.py pi05_b1k \
    --exp-name "$EXP_NAME" \
    --assets-base-dir "$ASSETS_BASE_DIR" \
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-train-steps "$TRAIN_STEPS" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    --no-wandb-enabled \
    --data.repo-id "$REPO_ID" \
    --data.dataset-root "$DATASET_ROOT" \
    "${TRAIN_DATA_ARGS[@]}"

echo "Checkpoint root: $CHECKPOINT_BASE_DIR/pi05_b1k/$EXP_NAME"
