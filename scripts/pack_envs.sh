#!/usr/bin/env bash
# Pack the conda environments into relocatable tarballs via conda-pack.
#
# Usage:
#   ./scripts/pack_envs.sh                   # pack b1k-pi05, b1k-eval
#   ./scripts/pack_envs.sh b1k-eval          # pack a single env
#   FORMAT=tar ./scripts/pack_envs.sh        # skip gzip when the CPU is busy
#   OUT_DIR=/somewhere ./scripts/pack_envs.sh
#
# Output defaults to /data/oss_bucket_0/wangfaye/conda_env, which survives pod
# restarts. The environments themselves live on /dev/shm and do not, so restoring
# from a tarball is the fast path — rebuilding from scratch takes over 40 minutes
# and depends on GitHub, which is flaky from this network.
#
# On the target machine:
#   mkdir -p /dev/shm/wangfaye/conda_envs/b1k-eval
#   tar xzf b1k-eval.tar.gz -C /dev/shm/wangfaye/conda_envs/b1k-eval
#   source /dev/shm/wangfaye/conda_envs/b1k-eval/bin/activate
#   conda-unpack
#
# Then re-run the matching setup script (setup_envs.sh / setup_eval_env.sh). Both
# are idempotent, and a tarball can predate later dependency fixes; for the eval
# env it also rewrites the Vulkan/EGL manifests for the current GPU driver.
#
# OpenPI is installed in editable mode; the unpacked environment still reads
# its source from this repository path.
set -euo pipefail

ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
OUT_DIR="${OUT_DIR:-/data/oss_bucket_0/wangfaye/conda_env}"
THREADS="${THREADS:-16}"
# tar.gz by default. Pass FORMAT=tar to skip compression: gzip is CPU-bound and
# the simulator already oversubscribes this host's cores during an eval sweep.
FORMAT="${FORMAT:-tar.gz}"

ENVS=("$@")
if [[ ${#ENVS[@]} -eq 0 ]]; then
    ENVS=(b1k-pi05 b1k-eval)
fi

if ! command -v conda-pack >/dev/null 2>&1; then
    echo "conda-pack not found. Install it with: uv tool install conda-pack" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

for name in "${ENVS[@]}"; do
    prefix="$ENV_ROOT/$name"
    output="$OUT_DIR/$name.$FORMAT"

    if [[ ! -x "$prefix/bin/python" ]]; then
        # Not every environment is built on every host, so skip rather than abort
        # and lose the ones that are.
        echo "Skipping $name: not built at $prefix" >&2
        continue
    fi

    echo "Packing $prefix -> $output"
    conda-pack \
        --prefix "$prefix" \
        --output "$output" \
        --format "$FORMAT" \
        --n-threads "$THREADS" \
        --ignore-editable-packages \
        --ignore-missing-files \
        --force
    du -sh "$output"
done
