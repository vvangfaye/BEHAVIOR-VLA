# BEHAVIOR-VLA：π 模型训练与 BEHAVIOR 评测

本工作区保留 OpenPI 的 π 模型代码，以及 BEHAVIOR-1K / OmniGibson 仿真评测工程。工作区级脚本主要对接 π0.5。

## 目录

| 路径 | 用途 |
| --- | --- |
| `openpi/` | π0、π0-FAST、π0.5 模型代码，训练、数据处理和推理服务 |
| `BEHAVIOR-1K/` | 仿真、任务定义、评测及其依赖；完整保留上游工程 |
| `scripts/` | π0.5 环境准备、训练、评测、数据处理和分析入口 |
| `scripts/download/` | 数据与 π0.5 基础权重下载工具 |
| `experiments/pi05_long300k/` | π0.5 长训练 loss 图 |
| `experiments/viz/` | π0.5 子任务训练收敛图 |

`DATASET_STATS.md` 是数据统计，`SUBTASK_ANNOTATIONS.md` 是子任务标注与 OpenPI 接入说明，`EVAL_SUMMARY.md` 和 `EVAL_FINDINGS.md` 是历史 π0.5 评测记录。

## 环境与数据

源码目录不包含已安装环境、数据集、仿真资产或模型权重。脚本仍有历史机器的默认路径，运行前按本机情况设置 `ENV_ROOT`、`CONDA`、`DATASET_ROOT`、缓存路径和 `EVAL_ENV`。

- `scripts/setup_envs.sh`：创建 π0.5 环境并链接到 `openpi/.venv`，仅支持 `pi05`。
- `scripts/setup_eval_env.sh`：BEHAVIOR / Isaac Sim 评测环境准备脚本。
- `scripts/setup_eval_assets.sh`：下载并解压仿真资产，链接到 `BEHAVIOR-1K/omnigibson-data`；密钥需要单独准备。
- `scripts/pack_envs.sh`：默认打包 `b1k-pi05` 与 `b1k-eval`。
- `scripts/bootstrap.sh`：串联训练工具、π 环境、数据暂存、π 权重下载；不包含仿真评测环境。
- `scripts/download/download_pi05_base.sh`：下载 π0.5 基础权重，可用 `OPENPI_CACHE_ROOT` 指定缓存根目录。

BEHAVIOR 的安装与运行要求见 `BEHAVIOR-1K/README.md`，OpenPI 的要求见 `openpi/README.md`。已配置的评测环境可以通过 `EVAL_ENV` 直接复用。

## 训练

从仓库根目录运行，先以 smoke 模式验证数据和环境：

```bash
DATASET_ROOT=/path/to/2026-challenge-demos ./scripts/train_pi05_b1k.sh smoke
DATASET_ROOT=/path/to/2026-challenge-demos ./scripts/train_pi05_b1k.sh full
```

训练入口支持 `NUM_GPUS`、`CUDA_DEVICES`、`BATCH_SIZE`、`TRAIN_STEPS`、`EXP_NAME`。子任务训练还支持 `SUBTASK_PROMPTS`、`WHOLE_TASK_PROB` 和 `FRAME_STRIDE`，见 `SUBTASK_ANNOTATIONS.md`。

`run_pi05_90min.sh` 和 `run_pi05_long300k.sh` 保留了历史实验配方；其中恢复训练路径需要按实际 checkpoint 调整。

## BEHAVIOR 闭环评测

```bash
PI05_CKPT=/path/to/checkpoint \
PI05_REPO_ID=behavior-1k/2026-challenge-demos \
EVAL_ENV=/path/to/behavior-env \
OMNIGIBSON_DATA_PATH=/path/to/omnigibson-data \
INSTANCE_INDICES="0" \
./scripts/eval_b1k.sh turning_on_radio
```

脚本默认 `MODEL=pi05`，启动 OpenPI 策略服务并运行 OmniGibson。`PI05_REPO_ID` 必须匹配 checkpoint 的 `assets/<repo-id>` 目录；官方 radio checkpoint 通常需要 `turning_on_radio`。默认策略服务 GPU 为 0，仿真 GPU 为 1，可通过 `SERVER_GPU`、`EVAL_GPU` 调整。

结果写入 `eval_results/pi05/<task>/`，包括 JSON、视频和日志。批量评测使用 `scripts/eval_sweep.sh`，可设置 `WORKERS` 和 `TASK_FILE`；同样需要配置 checkpoint、环境与资产路径。

```bash
python scripts/summarize_closed_loop.py eval_results/pi05
```

开环误差评估需在 OpenPI 环境运行：

```bash
openpi/.venv/bin/python scripts/eval_openloop.py \
  --pi05-ckpt /path/to/checkpoint \
  --dataset /path/to/2026-challenge-demos \
  --samples-per-task 2 --horizon 32 \
  --out eval_results/openloop_pi05.json
```

开环动作误差不能代替闭环任务成功率。绘图入口为 `plot_pi05_long300k_loss.py` 和 `plot_pi05_subtask_convergence.py`。
