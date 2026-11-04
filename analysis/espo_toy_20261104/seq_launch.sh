#!/bin/bash
# Sequential long-horizon toy, tuned comparison at two interaction budgets (2026-11-04).
cd "$(dirname "$0")"
source /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/activate
for E in 64 32; do
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 setsid nohup python espo_seq_toy.py --episodes $E --radius 0.25 --iters 25 \
        --json seq_ep${E}.json > seq_ep${E}.log 2>&1 < /dev/null &
done
