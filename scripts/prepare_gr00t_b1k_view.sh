#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GROOT_DIR="${GROOT_DIR:-$CODE_ROOT/Isaac-GR00T}"
SOURCE_DATASET="${SOURCE_DATASET:-/var/tmp/behavior-1k/2026-challenge-demos}"
MODE="${1:-full}"

case "$MODE" in
    smoke)
        VIEW_ROOT="${VIEW_ROOT:-$CODE_ROOT/dataset_views/gr00t/2026-challenge-demos-smoke}"
        ;;
    full)
        VIEW_ROOT="${VIEW_ROOT:-$CODE_ROOT/dataset_views/gr00t/2026-challenge-demos}"
        ;;
    *)
        echo "Usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac

if [[ ! -f "$SOURCE_DATASET/meta/info.json" ]]; then
    echo "Dataset not found or incomplete: $SOURCE_DATASET/meta/info.json" >&2
    exit 1
fi

mkdir -p "$VIEW_ROOT"
mkdir -p "$VIEW_ROOT/meta"

if [[ "$MODE" == "smoke" ]]; then
    mkdir -p "$VIEW_ROOT/data" "$VIEW_ROOT/videos"
    ln -sfn "$SOURCE_DATASET/data/chunk-000" "$VIEW_ROOT/data/chunk-000"
    for video_key_dir in "$SOURCE_DATASET"/videos/*; do
        [[ -d "$video_key_dir/chunk-000" ]] || continue
        video_key="$(basename "$video_key_dir")"
        mkdir -p "$VIEW_ROOT/videos/$video_key"
        ln -sfn "$video_key_dir/chunk-000" "$VIEW_ROOT/videos/$video_key/chunk-000"
    done

    cp --no-preserve=ownership "$SOURCE_DATASET"/meta/*.json \
        "$SOURCE_DATASET"/meta/*.jsonl \
        "$SOURCE_DATASET"/meta/*.parquet \
        "$VIEW_ROOT/meta/" 2>/dev/null || true
    if [[ -d "$SOURCE_DATASET/meta/episodes/chunk-000" ]]; then
        mkdir -p "$VIEW_ROOT/meta/episodes"
        cp -r --no-preserve=ownership "$SOURCE_DATASET/meta/episodes/chunk-000" \
            "$VIEW_ROOT/meta/episodes/"
    fi
else
    ln -sfn "$SOURCE_DATASET/data" "$VIEW_ROOT/data"
    ln -sfn "$SOURCE_DATASET/videos" "$VIEW_ROOT/videos"
    if [[ -d "$SOURCE_DATASET/annotations" ]]; then
        ln -sfn "$SOURCE_DATASET/annotations" "$VIEW_ROOT/annotations"
    fi
    cp -r --no-preserve=ownership "$SOURCE_DATASET/meta/." "$VIEW_ROOT/meta/"
fi

cp "$GROOT_DIR/examples/b1k/r1pro.json" "$VIEW_ROOT/meta/modality.json"

echo "$VIEW_ROOT"
