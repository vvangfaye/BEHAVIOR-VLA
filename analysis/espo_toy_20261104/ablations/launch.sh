#!/bin/bash
# ESPO component ablations on the binary 8-D toy (fixed eta=3, 5 test seeds, 30 iterations). Launched 2026-11-04.
cd "$(dirname "$0")/.."
source /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/activate
base=(--methods espo --task binary --du 8 --success-radius 0.8 --demo-std 0.35 --reward-noise 0.0 --iters 30 --fixed-lr 3.0)
for cfg in "qfeat 10 128 0" "moments 10 128 0" "mean 10 128 0" "qfeat 1 128 0" "qfeat 100 128 0" "qfeat 10 16 0" "qfeat 10 128 1"; do
    set -- $cfg
    tag="kind-$1_lam-$2_M-$3_wh-$4"
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 setsid nohup python espo_toy_train.py "${base[@]}" --espo-kind "$1" --lam "$2" \
        --model-samples "$3" --espo-whiten "$4" --json "ablations/$tag.json" > "ablations/$tag.log" 2>&1 < /dev/null &
done
