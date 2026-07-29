# BEHAVIOR 2026 Challenge workspace

This workspace is prepared for reproducing the two official 2026 BEHAVIOR
Challenge baselines:

- **π0.5**, using the challenge `behavior` branch of OpenPI.
- **GR00T N1.7**, using the challenge Isaac-GR00T fork.
- **OmniGibson evaluation**, using BEHAVIOR-1K v3.9.0 and Isaac Sim 5.1.

Official documentation: [challenge](https://behavior.stanford.edu/challenge/index.html)
and [baseline guide](https://behavior.stanford.edu/challenge/baselines.html).

## Current status

| Component | Path | Status |
| --- | --- | --- |
| OpenPI source | `openpi/` | Installed, `behavior` branch (`0cc8e355`) |
| π0.5 environment | `/mnt/afs/visitor34/venvs/openpi-behavior` | Ready; linked as `openpi/.venv` |
| Isaac-GR00T source | `Isaac-GR00T/` | Installed (`ace36d93`) |
| GR00T environment | `/mnt/afs/visitor34/venvs/isaac-groot` | Ready; linked as `Isaac-GR00T/.venv` |
| BEHAVIOR-1K source | `BEHAVIOR-1K/` | v3.9.0 |
| OmniGibson evaluator | Miniforge `behavior` environment | Installed |
| Isaac Sim | Evaluator environment | 5.1, installed |
| Robot assets | `BEHAVIOR-1K/omnigibson-data/omnigibson-robot-assets` | 3.8.2, ready |
| 2026 task instances | `BEHAVIOR-1K/omnigibson-data/2026-challenge-task-instances` | Ready |
| BEHAVIOR encryption key | `BEHAVIOR-1K/omnigibson-data/omnigibson.key` | Ready |
| Core BEHAVIOR assets | `BEHAVIOR-1K/omnigibson-data/behavior-1k-assets` | Archive downloaded; extraction in progress |
| π0.5 provided checkpoint | `BEHAVIOR-1K/baseline-checkpoints/extracted/pi05_turn_on_the_radio` | Ready |
| GR00T provided checkpoint | `BEHAVIOR-1K/baseline-checkpoints/extracted/turning_on_radio_GR00T-checkpoint-150000` | Ready |

The environments have already passed CUDA/import checks from a host context with
access to the H100 MIG GPU. A sandbox without GPU device passthrough may report
CUDA error 304 even though the host environment is valid.

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

### π0.5

```bash
cd /mnt/afs/visitor34/code/openpi
source .venv/bin/activate
```

### GR00T N1.7

```bash
cd /mnt/afs/visitor34/code/Isaac-GR00T
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
