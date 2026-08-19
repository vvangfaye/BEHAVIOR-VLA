#!/usr/bin/env bash
# One-shot bootstrap for this workspace on a fresh machine.
#
# Runs the stages in dependency order and skips whatever is already done, so it
# is safe to re-run after a Pod restart (the tmpfs envs and the staged dataset do
# not survive one).
#
#   ./scripts/bootstrap.sh                # all stages
#   ./scripts/bootstrap.sh envs dataset   # only the named stages
#   HF_TOKEN=hf_xxx ./scripts/bootstrap.sh models
#
# Stages
#   tools     uv tool install conda-pack (needed by pack_envs.sh)
#   envs      build the b1k-pi05 / b1k-groot conda envs  (scripts/setup_envs.sh)
#   dataset   stage the 3 RGB cameras + parquet to local disk, strip depth keys,
#             and regenerate openpi's task prompt registry
#   models    download GR00T-N1.7-3B + Cosmos-Reason2-2B through the hf mirror
#   relstats  precompute GR00T's relative-action statistics (parallel)
#
# Not covered on purpose: the OmniGibson/Isaac Sim evaluation environment.
# Isaac Sim needs a GPU with RT cores and glibc >= 2.33; this host has neither
# (H20 is Hopper, glibc 2.32), so that stage belongs on the eval machine.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_DATASET="${SOURCE_DATASET:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
LOCAL_DATASET="${LOCAL_DATASET:-/var/tmp/behavior-1k/2026-challenge-demos}"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
CACHE_ROOT="${PRETRAINED_CACHE_ROOT:-/mnt/workspace/wangfaye/cache}"

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(tools envs dataset models relstats)

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
    if [[ -d "$CACHE_ROOT/huggingface/hub/models--nvidia--Cosmos-Reason2-2B" ]]; then
        log "  GR00T + Cosmos already cached"
    else
        PRETRAINED_CACHE_ROOT="$CACHE_ROOT" ./scripts/download/download_groot_n17.sh
    fi
fi

if has_stage relstats; then
    log "stage: relstats"
    if [[ -f "$LOCAL_DATASET/meta/relative_stats.json" ]]; then
        log "  relative_stats.json present"
    else
        # r1pro's torso/left_arm/right_arm actions are relative to state, so GR00T
        # needs these stats. Upstream computes them serially (~13 h) and only
        # writes at the very end; this version fans out and saves per key.
        ( cd Isaac-GR00T && HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
            "$ENV_ROOT/b1k-groot/bin/python" ../scripts/gen_groot_rel_stats.py \
                --dataset-path "$LOCAL_DATASET" \
                --modality-config examples/b1k/r1pro.py \
                --jobs "${JOBS:-64}" )
    fi
fi

log "bootstrap done"
echo
echo "next:"
echo "  train π0.5 : NUM_GPUS=8 CUDA_DEVICES=0,1,2,3,4,5,6,7 BATCH_SIZE=64 FSDP_DEVICES=8 \\"
echo "               DATASET_ROOT=$LOCAL_DATASET ./scripts/train_pi05_b1k.sh full"
echo "  train GR00T: NUM_GPUS=8 CUDA_DEVICES=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=128 \\"
echo "               DATASET_PATH=$LOCAL_DATASET ./scripts/train_gr00t_b1k.sh full"
echo "  pack envs  : ./scripts/pack_envs.sh"
