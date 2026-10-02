#!/usr/bin/env bash
set -euo pipefail
set +x

unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER_PY="$SCRIPT_DIR/train_goalkeeper_h20.py"
GOALKEEPER_DIR="$REPO_ROOT/goalkeeper"

: "${MODE:?MODE is required (preflight, smoke, or full)}"
: "${FEATURE_ROOT:?FEATURE_ROOT is required}"
: "${EXPECTED_TREE_SHA256:?EXPECTED_TREE_SHA256 is required}"
: "${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 is required}"
: "${EXPECTED_CONTRACT_SHA256:?EXPECTED_CONTRACT_SHA256 is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${ATTEMPT_ID:?ATTEMPT_ID is required}"
: "${JOB_NAME:?JOB_NAME is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
case "$MODE" in
    preflight|smoke|full) ;;
    *) echo "MODE must be preflight, smoke, or full" >&2; exit 2 ;;
esac
for value_name in RUN_ID ATTEMPT_ID JOB_NAME; do
    value="${!value_name}"
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
        echo "$value_name must be a safe, nonempty identifier" >&2
        exit 2
    }
done
for hash_name in EXPECTED_TREE_SHA256 EXPECTED_MANIFEST_SHA256 EXPECTED_CONTRACT_SHA256; do
    value="${!hash_name}"
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo "$hash_name must be a lowercase SHA-256 digest" >&2; exit 2; }
done

PYTHON="${GK_PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON" >&2; exit 1; }
[[ -f "$WORKER_PY" && -f "$GOALKEEPER_DIR/pyproject.toml" ]] || {
    echo "Incomplete GoalKeeper worker bundle" >&2
    exit 1
}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Fresh OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT"
ATTEMPT_ROOT="$OUTPUT_ROOT/attempts/$ATTEMPT_ID"
LOG_DIR="$ATTEMPT_ROOT/logs"
CONTRACT_DIR="$ATTEMPT_ROOT/contracts"
ARTIFACT_DIR="$ATTEMPT_ROOT/artifacts"
mkdir -p "$LOG_DIR" "$CONTRACT_DIR"
LOG_FILE="$LOG_DIR/worker.log"
exec > >(tee -a "$LOG_FILE") 2>&1

finished=0
cleanup() {
    status=$?
    if [[ -n "${STAGE_DIR:-}" && -d "$STAGE_DIR" && "${KEEP_LOCAL_STAGE:-0}" != 1 ]]; then
        rm -rf -- "$STAGE_DIR"
    fi
    if (( status != 0 && finished == 0 )); then
        "$PYTHON" "$WORKER_PY" write-json "$ATTEMPT_ROOT/_FAILED" \
            "{\"format\":\"goalkeeper-h20-failure-v1\",\"mode\":\"$MODE\",\"run_id\":\"$RUN_ID\",\"attempt_id\":\"$ATTEMPT_ID\",\"exit_status\":$status}" \
            >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup EXIT

printf '[GoalKeeper] run=%s attempt=%s mode=%s job=%s\n' "$RUN_ID" "$ATTEMPT_ID" "$MODE" "$JOB_NAME"
printf '[GoalKeeper] fixed topology: 1 worker, 8 H20 GPUs, global batch 64, per-rank batch 8\n'

"$PYTHON" "$WORKER_PY" validate-runtime --output "$CONTRACT_DIR/runtime_contract.json"
"$PYTHON" "$WORKER_PY" validate-feature "$FEATURE_ROOT" \
    --expected-tree "$EXPECTED_TREE_SHA256" \
    --expected-manifest "$EXPECTED_MANIFEST_SHA256" \
    --expected-contract "$EXPECTED_CONTRACT_SHA256" \
    > "$CONTRACT_DIR/source_feature_validation.json"

LOCAL_STAGE_PARENT="${LOCAL_STAGE_PARENT:-/dev/shm}"
[[ -d "$LOCAL_STAGE_PARENT" && -w "$LOCAL_STAGE_PARENT" ]] || {
    echo "LOCAL_STAGE_PARENT must be an existing writable directory: $LOCAL_STAGE_PARENT" >&2
    exit 1
}
STAGE_DIR="$(mktemp -d "$LOCAL_STAGE_PARENT/goalkeeper-h20.${RUN_ID}.${ATTEMPT_ID}.XXXXXX")"
LOCAL_FEATURE_ROOT="$STAGE_DIR/features"
mkdir "$LOCAL_FEATURE_ROOT"
printf '[GoalKeeper] staging immutable feature tree to %s\n' "$LOCAL_FEATURE_ROOT"
cp -a --reflink=auto "$FEATURE_ROOT/." "$LOCAL_FEATURE_ROOT/"
"$PYTHON" "$WORKER_PY" validate-feature "$LOCAL_FEATURE_ROOT" \
    --expected-tree "$EXPECTED_TREE_SHA256" \
    --expected-manifest "$EXPECTED_MANIFEST_SHA256" \
    --expected-contract "$EXPECTED_CONTRACT_SHA256" \
    > "$CONTRACT_DIR/local_feature_validation.json"
find "$LOCAL_FEATURE_ROOT" -type f -exec chmod a-w {} +

export PYTHONPATH="$GOALKEEPER_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - "$LOCAL_FEATURE_ROOT/manifest.jsonl" "$CONTRACT_DIR/dataset_preflight.json" <<'PY'
import json
import sys
from pathlib import Path
from goalkeeper.data import check_dataset, load_manifest

manifest, output = sys.argv[1:]
checked = check_dataset(Path(manifest).parent)
rows = load_manifest(manifest)
splits = {name: sum(row["split"] == name for row in rows) for name in ("train", "dev", "calibration")}
if any(count == 0 for count in splits.values()):
    raise ValueError(f"GoalKeeper requires nonempty train/dev/calibration splits; found {splits}")
if any(row["source"] == "synthetic_test" for row in rows):
    raise ValueError("Nebula GoalKeeper training accepts only prepared real features")
result = {
    "format": "goalkeeper-dataset-preflight-v1",
    "records": len(rows),
    "splits": splits,
    "dataset_contract_sha256": checked["dataset_contract_sha256"],
    "feature_id": rows[0]["feature_id"],
    "action_spec": rows[0]["action_spec"],
}
Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
PY

EPOCHS=20
[[ "$MODE" == smoke ]] && EPOCHS=1
"$PYTHON" "$WORKER_PY" write-json "$CONTRACT_DIR/launch_contract.json" \
    "{\"format\":\"goalkeeper-h20-launch-v1\",\"mode\":\"$MODE\",\"workers\":1,\"world_size\":8,\"gpus_per_worker\":8,\"global_batch_size\":64,\"per_rank_batch_size\":8,\"epochs\":$EPOCHS,\"run_id\":\"$RUN_ID\",\"attempt_id\":\"$ATTEMPT_ID\",\"job_name\":\"$JOB_NAME\",\"tree_sha256\":\"$EXPECTED_TREE_SHA256\",\"manifest_sha256\":\"$EXPECTED_MANIFEST_SHA256\",\"contract_sha256\":\"$EXPECTED_CONTRACT_SHA256\"}" \
    >/dev/null

if [[ "$MODE" == preflight ]]; then
    "$PYTHON" "$WORKER_PY" write-json "$ATTEMPT_ROOT/_PREFLIGHT_OK.json" \
        "{\"format\":\"goalkeeper-h20-preflight-v1\",\"run_id\":\"$RUN_ID\",\"attempt_id\":\"$ATTEMPT_ID\",\"world_size\":8,\"tree_sha256\":\"$EXPECTED_TREE_SHA256\"}" \
        >/dev/null
    finished=1
    printf '[GoalKeeper] preflight complete: %s\n' "$ATTEMPT_ROOT/_PREFLIGHT_OK.json"
    exit 0
fi

mkdir "$ARTIFACT_DIR"
export GK_MASTER_ADDR=127.0.0.1
export GK_MASTER_PORT="${GK_MASTER_PORT:-29531}"
export GK_NODE_RANK=0
export GK_NNODES=1
export GK_GPUS_PER_NODE=8
export GK_GLOBAL_BATCH_SIZE=64
export GK_WORKERS="${GK_WORKERS:-4}"
export GK_EPOCHS="$EPOCHS"
export GK_PYTHON="$PYTHON"
export GK_MANIFEST="$LOCAL_FEATURE_ROOT/manifest.jsonl"
export GK_DATASET_CONTRACT="$LOCAL_FEATURE_ROOT/dataset_contract.json"
export GK_OUTPUT="$ARTIFACT_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

printf '[GoalKeeper] starting 8-rank DDP training\n'
bash "$GOALKEEPER_DIR/scripts/train_head_ddp.sh"
printf '[GoalKeeper] calibrating rank-0 best checkpoint\n'
"$PYTHON" -m goalkeeper.cli calibrate \
    "$LOCAL_FEATURE_ROOT/manifest.jsonl" \
    "$ARTIFACT_DIR/best.pt" \
    "$ARTIFACT_DIR/calibration.json" \
    --device cuda:0
"$PYTHON" "$WORKER_PY" verify-artifacts "$ARTIFACT_DIR" \
    --expected-manifest "$EXPECTED_MANIFEST_SHA256" \
    --expected-contract "$EXPECTED_CONTRACT_SHA256" \
    --result "$CONTRACT_DIR/artifact_contract.json"
"$PYTHON" "$WORKER_PY" write-json "$ATTEMPT_ROOT/_SUCCESS.rank0.json" \
    "{\"format\":\"goalkeeper-h20-success-v1\",\"run_id\":\"$RUN_ID\",\"attempt_id\":\"$ATTEMPT_ID\",\"mode\":\"$MODE\",\"world_size\":8,\"global_batch_size\":64,\"manifest_sha256\":\"$EXPECTED_MANIFEST_SHA256\"}" \
    >/dev/null
finished=1
printf '[GoalKeeper] training and calibration complete: %s\n' "$ATTEMPT_ROOT/_SUCCESS.rank0.json"
