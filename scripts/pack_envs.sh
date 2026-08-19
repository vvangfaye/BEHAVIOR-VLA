#!/usr/bin/env bash
# Pack the training conda environments into relocatable tarballs via conda-pack.
#
# Usage:
#   ./scripts/pack_envs.sh                  # pack b1k-pi05 and b1k-groot
#   ./scripts/pack_envs.sh b1k-groot        # pack a single env
#   OUT_DIR=/mnt/workspace/wangfaye/envs ./scripts/pack_envs.sh
#
# On the target machine:
#   mkdir -p /dev/shm/wangfaye/conda_envs/b1k-groot
#   tar xzf b1k-groot.tar.gz -C /dev/shm/wangfaye/conda_envs/b1k-groot
#   source /dev/shm/wangfaye/conda_envs/b1k-groot/bin/activate
#   conda-unpack
#
# openpi and gr00t are installed in editable mode, so the unpacked env still
# reads their source from this repository path.
set -euo pipefail

ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
OUT_DIR="${OUT_DIR:-$ENV_ROOT}"
THREADS="${THREADS:-16}"

ENVS=("$@")
if [[ ${#ENVS[@]} -eq 0 ]]; then
    ENVS=(b1k-pi05 b1k-groot)
fi

if ! command -v conda-pack >/dev/null 2>&1; then
    echo "conda-pack not found. Install it with: uv tool install conda-pack" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

for name in "${ENVS[@]}"; do
    prefix="$ENV_ROOT/$name"
    output="$OUT_DIR/$name.tar.gz"

    if [[ ! -x "$prefix/bin/python" ]]; then
        echo "Environment not found: $prefix" >&2
        exit 1
    fi

    echo "Packing $prefix -> $output"
    conda-pack \
        --prefix "$prefix" \
        --output "$output" \
        --n-threads "$THREADS" \
        --ignore-editable-packages \
        --ignore-missing-files \
        --force
    du -sh "$output"
done
