# BEHAVIOR 2026 Challenge workspace

This workspace is prepared for reproducing the two official 2026 BEHAVIOR
Challenge baselines:

- **π0.5**, using the challenge `behavior` branch of OpenPI.
- **GR00T N1.7**, using the challenge Isaac-GR00T fork.
- **OmniGibson evaluation**, using BEHAVIOR-1K v3.9.0 and Isaac Sim 5.1.

Official documentation: [challenge](https://behavior.stanford.edu/challenge/index.html)
and [baseline guide](https://behavior.stanford.edu/challenge/baselines.html).

For joint **subtask language prediction + action training** on BEHAVIOR 2026, see
[the subtask training guide](docs/subtask_training.md) and
`scripts/train_pi05_subtask_b1k.sh`. This is separate from replacing the task prompt with a subtask.

## Current status

| Component | Path | Status |
| --- | --- | --- |
| OpenPI source | `openpi/` | Installed, `behavior` branch (`0cc8e355`) |
| π0.5 environment | `/dev/shm/wangfaye/conda_envs/b1k-pi05` | Ready; linked as `openpi/.venv` |
| Isaac-GR00T source | `Isaac-GR00T/` | Installed (`ace36d93`) |
| GR00T environment | `/dev/shm/wangfaye/conda_envs/b1k-groot` | Ready; linked as `Isaac-GR00T/.venv` |
| BEHAVIOR-1K source | `BEHAVIOR-1K/` | v3.9.0 |
| OmniGibson evaluator | Miniforge `behavior` environment | Installed |
| Isaac Sim | Evaluator environment | 5.1, installed |
| Robot assets | `BEHAVIOR-1K/omnigibson-data/omnigibson-robot-assets` | 3.8.2, ready |
| 2026 task instances | `BEHAVIOR-1K/omnigibson-data/2026-challenge-task-instances` | Ready |
| BEHAVIOR encryption key | `BEHAVIOR-1K/omnigibson-data/omnigibson.key` | Ready |
| Core BEHAVIOR assets | `BEHAVIOR-1K/omnigibson-data/behavior-1k-assets` | Archive downloaded; extraction in progress |
| π0.5 provided checkpoint | `BEHAVIOR-1K/baseline-checkpoints/extracted/pi05_turn_on_the_radio` | Ready |
| GR00T provided checkpoint | `BEHAVIOR-1K/baseline-checkpoints/extracted/turning_on_radio_GR00T-checkpoint-150000` | Ready |

Both training environments were verified on the current host (2x NVIDIA H20,
driver 580.82.07): `jax 0.5.3` reports a CUDA device, `torch 2.7.1+cu128`
detects the H20, and the `flash-attn 2.7.4.post1` kernel runs.

The environments live in tmpfs and do not survive a Pod restart. Rebuild them
with `./scripts/setup_envs.sh`, or restore from the packed tarballs produced by
`./scripts/pack_envs.sh`.

## Repository layout

```text
/mnt/afs/visitor34/code/
├── README.md
├── openpi/                       # π0.5 training and serving
├── Isaac-GR00T/                  # GR00T N1.7 training and serving
└── BEHAVIOR-1K/
    ├── OmniGibson/               # simulator and evaluator source
    ├── omnigibson-data/
    │   ├── behavior-1k-assets/
    │   ├── omnigibson-robot-assets/
    │   ├── 2026-challenge-task-instances/
    │   └── omnigibson.key
    ├── baseline-checkpoints/
    │   ├── *.zip                 # original provided archives
    │   └── extracted/            # directly servable checkpoints
    └── .downloads/hf-assets/     # downloaded core-assets archive
```

The training demonstrations are separate:

```text
/data/2026-challenge-demos
```

The full demonstration release is about 3.27 TB. It is not needed to evaluate
the provided checkpoints. For training a single task, download only that task's
`chunk-xxx` plus the shared metadata.

## Environments

Both training environments are prefix-based conda environments outside the
repository, symlinked in as `.venv` so the training scripts find them:

| Model | Environment | Python |
| --- | --- | --- |
| π0.5 | `/dev/shm/wangfaye/conda_envs/b1k-pi05` | 3.11.13 |
| GR00T N1.7 | `/dev/shm/wangfaye/conda_envs/b1k-groot` | 3.10.13 |

Build or rebuild them:

```bash
cd /mnt/workspace/wangfaye/code/BEHAVIOR-Challenge
./scripts/setup_envs.sh              # both, or pass pi05 / groot
```

Pack them into relocatable tarballs (`conda-pack`, needs
`uv tool install conda-pack`):

```bash
./scripts/pack_envs.sh
```

Restore a tarball on another host:

```bash
mkdir -p /dev/shm/wangfaye/conda_envs/b1k-groot
tar xzf b1k-groot.tar.gz -C /dev/shm/wangfaye/conda_envs/b1k-groot
source /dev/shm/wangfaye/conda_envs/b1k-groot/bin/activate
conda-unpack
```

`openpi` and `gr00t` are installed in editable mode, so a restored environment
still imports their source from this repository path.

### π0.5

```bash
cd /mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/openpi
source .venv/bin/activate
```

### GR00T N1.7

```bash
cd /mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/Isaac-GR00T
source .venv/bin/activate
```

### OmniGibson evaluator

```bash
source /mnt/afs/visitor34/venvs/miniforge3/bin/activate
conda activate behavior

export OMNIGIBSON_DATA_PATH=/mnt/afs/visitor34/code/BEHAVIOR-1K/omnigibson-data
```

`OMNIGIBSON_DATA_PATH` is required. Without it, OmniGibson searches the
repository's default `BEHAVIOR-1K/datasets/` directory and cannot find the
challenge task metadata.

## Verify installed data

```bash
cd /mnt/afs/visitor34/code/BEHAVIOR-1K

cat omnigibson-data/behavior-1k-assets/VERSION
cat omnigibson-data/omnigibson-robot-assets/VERSION
test -f omnigibson-data/2026-challenge-task-instances/metadata/task.jsonl
test -f omnigibson-data/omnigibson.key
```

Expected asset versions are `3.9.0` and `3.8.2`.

If the core-assets archive ever needs to be extracted again:

```bash
cd /mnt/afs/visitor34/code/BEHAVIOR-1K
mkdir -p omnigibson-data/behavior-1k-assets
unzip -q .downloads/hf-assets/behavior-1k-assets-3.9.0.zip \
  -d omnigibson-data/behavior-1k-assets
```

## Evaluate the provided π0.5 checkpoint

Start the policy server in terminal 1:

```bash
cd /mnt/afs/visitor34/code/openpi
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro \
  --task b1k/turning_on_radio \
  --repo-id turning_on_radio \
  --policy.config pi05_b1k \
  --policy.dir /mnt/afs/visitor34/code/BEHAVIOR-1K/baseline-checkpoints/extracted/pi05_turn_on_the_radio \
  --control_mode receding_horizon \
  --action_horizon 16 \
  --port 8000
```

The provided checkpoint stores normalization statistics under
`assets/turning_on_radio`. Pass `--repo-id turning_on_radio` explicitly;
otherwise the server derives `b1k/turning_on_radio` from `--task` and looks in
the wrong checkpoint subdirectory.

Then run the common evaluator command below in terminal 2.

## Evaluate the provided GR00T N1.7 checkpoint

Start the policy server in terminal 1:

```bash
cd /mnt/afs/visitor34/code/Isaac-GR00T
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python scripts/b1k/serve_b1k.py \
  --model-path /mnt/afs/visitor34/code/BEHAVIOR-1K/baseline-checkpoints/extracted/turning_on_radio_GR00T-checkpoint-150000 \
  --modality-config-path examples/b1k/r1pro.py \
  --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 \
  --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/healthz
```

Then run the common evaluator command below in terminal 2.

## Run the OmniGibson evaluator

Run exactly one policy server on port 8000. In a second terminal:

```bash
source /mnt/afs/visitor34/venvs/miniforge3/bin/activate
conda activate behavior

cd /mnt/afs/visitor34/code/BEHAVIOR-1K
export OMNIGIBSON_DATA_PATH=/mnt/afs/visitor34/code/BEHAVIOR-1K/omnigibson-data

/mnt/afs/visitor34/venvs/miniforge3/envs/behavior/bin/python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 \
  --port 8000 \
  --output-dir ./eval_logs/turning_on_radio \
  --write-video
```

Metrics JSON and rollout videos are written to:

```text
/mnt/afs/visitor34/code/BEHAVIOR-1K/eval_logs/turning_on_radio
```

## Train the baselines

Run the commands below from the workspace root:

```bash
cd /mnt/afs/visitor34/code
```

### Persistent pretrained checkpoints

Pretrained weights are stored on AFS so they survive Pod restarts:

| Model | Persistent location |
| --- | --- |
| π0.5 base | `/mnt/afs/visitor34/devhome/cache/openpi/openpi-assets/checkpoints/pi05_base/params` |
| GR00T N1.7 and Cosmos Reason 2 | `/mnt/afs/visitor34/devhome/cache/huggingface/hub` |

The download scripts are idempotent and reuse completed files:

```bash
./scripts/download/download_pi05_base.sh
./scripts/download/download_groot_n17.sh
```

The GR00T download script persists both required Hugging Face repositories:
`nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B`. It and the GR00T
training script use `http://127.0.0.1:7890` by default. Override or disable it
with `PROXY_URL`, for example:

```bash
PROXY_URL=http://127.0.0.1:7890 ./scripts/download/download_groot_n17.sh
PROXY_URL= ./scripts/train_gr00t_b1k.sh smoke
```

`train_gr00t_b1k.sh` exports the persistent `HF_HOME` and `HF_HUB_CACHE`
automatically. `train_pi05_b1k.sh` uses OpenPI's persistent cache under
`/mnt/afs/visitor34/devhome/cache/openpi`.

### System dependency after a new Pod starts

The virtual environments are persistent, but apt packages in the container are
not. Install FFmpeg before training; GR00T's `torchcodec` requires its shared
libraries, and OpenPI also uses FFmpeg for video decoding.

```bash
apt-get update
apt-get install -y ffmpeg
```

Verify GR00T's decoder:

```bash
/mnt/afs/visitor34/venvs/isaac-groot/bin/python -c \
  "import torchcodec; print(torchcodec.__version__)"
```

The tested version is `torchcodec 0.4.0` with FFmpeg 4 and PyTorch
`2.7.1+cu128`.

### π0.5

Environment:

```text
/mnt/afs/visitor34/code/openpi/.venv
```

Run the two-step smoke test:

```bash
NUM_WORKERS=0 TRAIN_STEPS=2 BATCH_SIZE=1 NUM_GPUS=1 CUDA_DEVICES=0 \
  ./scripts/train_pi05_b1k.sh smoke
```

The smoke path uses episodes 0 and 1, computes normalization statistics from a
small sample, loads the three RGB cameras, and restores the 12 GB base
checkpoint.

On the current H100 MIG 3g.40GB instance, checkpoint restoration succeeds but
training-state initialization fails with an out-of-memory error. XLA reports a
minimum requirement of about 49.97 GiB even with batch size 1. Use a full 80 GB
H100 for this full-parameter configuration.

Run full training on a sufficiently large GPU:

```bash
NUM_GPUS=1 CUDA_DEVICES=0 ./scripts/train_pi05_b1k.sh full
```

When full normalization statistics do not yet exist, this command samples at
most 500,000 random frames by default instead of traversing all 211 million
frames. Override the limit with `NORM_MAX_FRAMES`, for example
`NORM_MAX_FRAMES=100000`.

### GR00T N1.7

Environment:

```text
/mnt/afs/visitor34/venvs/isaac-groot
```

Before the first download, accept the gated `nvidia/Cosmos-Reason2-2B` model on
Hugging Face and provide a valid `HF_TOKEN`.

Run the two-step smoke test:

```bash
NUM_GPUS=1 CUDA_DEVICES=0 NUM_WORKERS=0 \
  ./scripts/train_gr00t_b1k.sh smoke
```

Smoke mode creates a non-destructive dataset view at:

```text
/mnt/afs/visitor34/code/dataset_views/gr00t/2026-challenge-demos-smoke
```

It exposes only `chunk-000`, so initialization scans 2 parquet files instead of
all 955 files. It does not copy the large videos or modify the source dataset.

This exact command was verified on the 40 GB H100 MIG instance:

```text
steps:          2/2
train_loss:     1.2644
train_runtime:  113.8 seconds
result:         Training completed
```

The verified output is:

```text
/mnt/afs/visitor34/code/Isaac-GR00T/outputs/gr00t_b1k_smoke_20260729_021305
```

It is about 36 GB because the checkpoint includes model weights and a roughly
13 GB optimizer state.

Run full training over all dataset chunks:

```bash
NUM_GPUS=1 CUDA_DEVICES=0 ./scripts/train_gr00t_b1k.sh full
```

The released LeRobot v3 dataset is supported directly. Do not run the optional
in-place v2.1 conversion unless another tool specifically requires it.

### GR00T convergence validation

Use the bounded single-GPU experiment to verify that the VLA training loss
decreases before committing resources to a full run:

```bash
cd /mnt/afs/visitor34/code
MAX_STEPS=1000 NUM_WORKERS=0 ./scripts/run_groot_convergence.sh
```

This uses batch size 1, one cached shard from the `chunk-000` smoke view,
BF16, a `1e-4` learning rate, and saves model weights without optimizer state.
It tests whether the training pipeline can learn from a small data slice; it
does not measure closed-loop simulator success.

Verified on the current H100 MIG 3g.40GB instance:

```text
steps:                 1000/1000
train runtime:         298.0 seconds
first loss mean:       1.133160
last loss mean:        1.031260
relative improvement: 8.99%
minimum loss:          0.979700
result:                PASS (threshold: 5%)
```

The reproducible artifacts are stored in:

```text
experiments/groot_convergence/gr00t_b1k_convergence_20260730_044942/
```

They include `config.txt`, `train.log`, `loss.csv`, and `summary.txt`. The
corresponding model checkpoint is under
`Isaac-GR00T/outputs/gr00t_b1k_convergence_20260730_044942/`.

## Proxy

Use port 7890 only when external downloads need it:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export all_proxy="$ALL_PROXY"
```

For large Hugging Face files, Xet may behave poorly through this proxy. Force
the ordinary HTTP downloader when needed:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 hf download ...
```
