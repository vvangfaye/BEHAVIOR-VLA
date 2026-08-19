#!/usr/bin/env bash
# Stage the parts of the challenge dataset the π0.5 pipeline actually reads from
# the OSS/FUSE mount onto fast local disk. Random-access training over FUSE is
# I/O-bound (~1.8 samples/s); a local copy removes that bottleneck.
#
# The pipeline uses 3 RGB cameras (openpi/src/openpi/configs/robots/b1k.py) plus
# the parquet state/action data. Depth videos are skipped, cutting the copy from
# ~3 TB to ~425 GB.
#
# Usage:
#   ./scripts/stage_dataset_local.sh                 # all 100 chunks
#   NUM_CHUNKS=20 ./scripts/stage_dataset_local.sh   # first 20 chunks
set -euo pipefail

SOURCE_DATASET="${SOURCE_DATASET:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
DEST_ROOT="${DEST_ROOT:-/root/datasets/behavior-1k/2026-challenge-demos}"
JOBS="${JOBS:-32}"
NUM_CHUNKS="${NUM_CHUNKS:-}"

RGB_KEYS=(
    observation.rgb.zed_link_camera_0
    observation.rgb.left_realsense_link_camera_0
    observation.rgb.right_realsense_link_camera_0
)

if [[ ! -f "$SOURCE_DATASET/meta/info.json" ]]; then
    echo "Source dataset not found: $SOURCE_DATASET/meta/info.json" >&2
    exit 1
fi

mkdir -p "$DEST_ROOT"

# Copy metadata wholesale; it is small and always needed.
echo "Copying meta/"
mkdir -p "$DEST_ROOT/meta"
cp -r --no-preserve=ownership "$SOURCE_DATASET/meta/." "$DEST_ROOT/meta/"

# Determine the chunk list.
mapfile -t all_chunks < <(cd "$SOURCE_DATASET/data" && ls -d chunk-* | sort)
if [[ -n "$NUM_CHUNKS" ]]; then
    all_chunks=("${all_chunks[@]:0:$NUM_CHUNKS}")
fi
echo "Staging ${#all_chunks[@]} chunks with $JOBS parallel streams"

# Build the copy job list: parquet data + RGB videos, preserving layout.
job_list="$(mktemp)"
trap 'rm -f "$job_list"' EXIT
for chunk in "${all_chunks[@]}"; do
    find "$SOURCE_DATASET/data/$chunk" -name '*.parquet' -print0 |
        while IFS= read -r -d '' f; do
            rel="${f#"$SOURCE_DATASET"/}"
            printf '%s\t%s\n' "$f" "$DEST_ROOT/$rel" >> "$job_list"
        done
    for key in "${RGB_KEYS[@]}"; do
        src_dir="$SOURCE_DATASET/videos/$key/$chunk"
        [[ -d "$src_dir" ]] || continue
        find "$src_dir" -name '*.mp4' -print0 |
            while IFS= read -r -d '' f; do
                rel="${f#"$SOURCE_DATASET"/}"
                printf '%s\t%s\n' "$f" "$DEST_ROOT/$rel" >> "$job_list"
            done
    done
done

total="$(wc -l < "$job_list")"
echo "Copying $total files"

copy_one() {
    local line="$1" src dst
    src="${line%%$'\t'*}"
    dst="${line#*$'\t'}"
    mkdir -p "$(dirname "$dst")"
    cp -n "$src" "$dst"
}
export -f copy_one

xargs -a "$job_list" -P "$JOBS" -I {} bash -c 'copy_one "$@"' _ {}

echo "Done. Local dataset at: $DEST_ROOT"
du -sh "$DEST_ROOT"
