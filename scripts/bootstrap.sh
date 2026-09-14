#!/usr/bin/env bash
# One-shot bootstrap for this workspace on a fresh machine.
#
# Runs the stages in dependency order and skips whatever is already done, so it
# is safe to re-run after a Pod restart (the tmpfs envs and the staged dataset do
# not survive one).
#
#   ./scripts/bootstrap.sh                # all stages
#   ./scripts/bootstrap.sh envs dataset   # only the named stages
#   ./scripts/bootstrap.sh models
#
# Stages
#   tools     uv tool install conda-pack (needed by pack_envs.sh)
#   envs      build the b1k-pi05 conda environment (scripts/setup_envs.sh)
#   dataset   stage the 3 RGB cameras + parquet to local disk, strip depth keys,
#             and regenerate openpi's task prompt registry
#   models    download the π0.5 base checkpoint
#
# Prepare the simulator separately with scripts/setup_eval_env.sh on the eval host.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_DATASET="${SOURCE_DATASET:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
LOCAL_DATASET="${LOCAL_DATASET:-/var/tmp/behavior-1k/2026-challenge-demos}"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
CACHE_ROOT="${PRETRAINED_CACHE_ROOT:-/mnt/workspace/wangfaye/cache}"
OPENPI_CACHE_ROOT="${OPENPI_CACHE_ROOT:-${OPENPI_DATA_HOME:-$CACHE_ROOT/openpi}}"

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(tools envs dataset models)

for stage in "${STAGES[@]}"; do
    case "$stage" in
        tools|envs|dataset|models) ;;
        *) echo "Unknown stage: $stage (expected tools, envs, dataset, models)" >&2; exit 2 ;;
    esac
done

has_stage() { local s; for s in "${STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
log() { echo "[$(date '+%F %T')] $*"; }

if has_stage tools; then
    log "stage: tools"
    if command -v conda-pack >/dev/null 2>&1; then
        log "  conda-pack present"
    else
        UV_LINK_MODE=copy uv tool install conda-pack
    fi
fi

if has_stage envs; then
    log "stage: envs"
    ./scripts/setup_envs.sh
fi

if has_stage dataset; then
    log "stage: dataset"
    if [[ -f "$LOCAL_DATASET/meta/info.json" ]]; then
        log "  local dataset already staged at $LOCAL_DATASET"
    else
        # Random-access video decoding over the OSS/FUSE mount runs at ~1.8
        # samples/s; from local disk the same pipeline is GPU-bound instead.
        SOURCE_DATASET="$SOURCE_DATASET" DEST_ROOT="$LOCAL_DATASET" \
            ./scripts/stage_dataset_local.sh
    fi
    # lerobot existence-checks every dtype=="video" feature, so leaving the depth
    # keys in place makes it fall back to the (unreachable) Hub; the b1k configs
    # only consume the 3 RGB cameras anyway.
    "$ENV_ROOT/b1k-pi05/bin/python" ./scripts/strip_depth_from_info.py "$LOCAL_DATASET"
    "$ENV_ROOT/b1k-pi05/bin/python" ./scripts/gen_task_registry.py --dataset "$LOCAL_DATASET"
fi

if has_stage models; then
    log "stage: models"
    OPENPI_CACHE_ROOT="$OPENPI_CACHE_ROOT" ./scripts/download/download_pi05_base.sh
fi

log "bootstrap done"
echo
echo "next:"
echo "  train π0.5 : NUM_GPUS=8 CUDA_DEVICES=0,1,2,3,4,5,6,7 BATCH_SIZE=64 FSDP_DEVICES=8 \\"
echo "               OPENPI_DATA_HOME=$OPENPI_CACHE_ROOT \\"
echo "               DATASET_ROOT=$LOCAL_DATASET ./scripts/train_pi05_b1k.sh full"
echo "  pack envs  : ./scripts/pack_envs.sh"
