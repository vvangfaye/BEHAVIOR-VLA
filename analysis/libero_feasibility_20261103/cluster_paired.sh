#!/bin/bash
# Paired LIBERO comparison for an eval stage of a cluster multi-stage job (2026-11-05).
# Usage: cluster_paired.sh TAG STAGE REF_LOG [MORE_REF_LOGS...]
#   fetches the AOSS mirror of TAG, extracts <STAGE>/console.log from its tensorboard.tgz into
#   results/cluster_<TAG>_<STAGE>/console.log and runs paired.py with each REF_LOG as the reference.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG=$1; STAGE=$2; shift 2
CP="$HERE/../cluster_probe_20261104"
(cd "$CP" && timeout 150 uv run --offline --with boto3 python fetch_run.py "$TAG" >/dev/null 2>&1) || true
OUT="$HERE/results/cluster_${TAG}_${STAGE}"
mkdir -p "$OUT"
tar -xzOf "$CP/cluster_runs/$TAG/tensorboard.tgz" "$STAGE/console.log" > "$OUT/console.log"
n=$(grep -a -c "\[libero eval\] task_id=" "$OUT/console.log" || true)
echo "$TAG/$STAGE: $n episodes"
for ref in "$@"; do
    python3 "$HERE/paired.py" "$ref" "$OUT/console.log" 2>&1 | tail -3
done
