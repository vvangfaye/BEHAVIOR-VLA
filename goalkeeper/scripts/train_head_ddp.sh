#!/usr/bin/env bash
# Run on each node. Example 4 nodes x 8 H20; head training also works on 1 GPU.
# GK_GLOBAL_BATCH_SIZE is divided across all ranks (128 / 32 = 4 per rank).
# The network rendezvous address, node rank and sealed dataset must be supplied by you.
set -euo pipefail
: "${GK_MASTER_ADDR:?Set the reachable rank-0 training address}"
: "${GK_NODE_RANK:?Set the unique node rank (0 to GK_NNODES-1)}"
: "${GK_MANIFEST:?Set the shared training manifest path}"
: "${GK_DATASET_CONTRACT:?Set the immutable dataset_contract.json path}"
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
  --dataset-contract "$GK_DATASET_CONTRACT" \
  --global-batch-size "${GK_GLOBAL_BATCH_SIZE:-128}" --workers "${GK_WORKERS:-4}" \
  --epochs "${GK_EPOCHS:-20}" --hidden-dim "${GK_HIDDEN_DIM:-128}" \
  --learning-rate "${GK_LEARNING_RATE:-0.0003}"
