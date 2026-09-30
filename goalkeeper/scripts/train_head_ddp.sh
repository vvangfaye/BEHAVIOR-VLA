#!/usr/bin/env bash
# Run on each node. Example 4 nodes x 8 H20; head training also works on 1 GPU.
# The network rendezvous address, node rank and dataset must be supplied by you.
set -euo pipefail
: "${GK_MASTER_ADDR:?Set the reachable rank-0 training address}"
: "${GK_NODE_RANK:?Set the unique node rank (0 to GK_NNODES-1)}"
: "${GK_MANIFEST:?Set the shared training manifest path}"
: "${GK_OUTPUT:?Set the shared output directory}"
GK_NNODES=${GK_NNODES:-4}
GK_GPUS_PER_NODE=${GK_GPUS_PER_NODE:-8}
GK_MASTER_PORT=${GK_MASTER_PORT:-29531}
GK_PYTHON=${GK_PYTHON:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
"$GK_PYTHON" -m torch.distributed.run \
  --nnodes "$GK_NNODES" --nproc-per-node "$GK_GPUS_PER_NODE" \
  --node-rank "$GK_NODE_RANK" --master-addr "$GK_MASTER_ADDR" --master-port "$GK_MASTER_PORT" \
  -m goalkeeper.cli train "$GK_MANIFEST" "$GK_OUTPUT" \
  --batch-size "${GK_BATCH_PER_GPU:-16}" --workers "${GK_WORKERS:-4}" \
  --epochs "${GK_EPOCHS:-20}" --hidden-dim "${GK_HIDDEN_DIM:-128}" \
  --learning-rate "${GK_LEARNING_RATE:-0.0003}"
