#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/mnt/afs/visitor34/code}"
GROOT_DIR="${GROOT_DIR:-$CODE_ROOT/Isaac-GR00T}"
SOURCE_DATASET="${SOURCE_DATASET:-/data/2026-challenge-demos}"
VIEW_ROOT="${VIEW_ROOT:-$CODE_ROOT/dataset_views/gr00t/2026-challenge-demos}"

if [[ ! -f "$SOURCE_DATASET/meta/info.json" ]]; then
    echo "Dataset not found or incomplete: $SOURCE_DATASET/meta/info.json" >&2
    exit 1
fi

mkdir -p "$VIEW_ROOT"
ln -sfn "$SOURCE_DATASET/data" "$VIEW_ROOT/data"
ln -sfn "$SOURCE_DATASET/videos" "$VIEW_ROOT/videos"
if [[ -d "$SOURCE_DATASET/annotations" ]]; then
    ln -sfn "$SOURCE_DATASET/annotations" "$VIEW_ROOT/annotations"
fi
mkdir -p "$VIEW_ROOT/meta"
cp -a "$SOURCE_DATASET/meta/." "$VIEW_ROOT/meta/"
cp "$GROOT_DIR/examples/b1k/r1pro.json" "$VIEW_ROOT/meta/modality.json"

echo "$VIEW_ROOT"
