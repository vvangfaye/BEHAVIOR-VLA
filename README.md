# BEHAVIOR 2026：π0.5 / subtask

本仓库用于 BEHAVIOR 2026 数据上的 π0.5 训练、subtask 联合训练、在线推理和
OmniGibson 闭环评测。训练与模型服务使用 OpenPI，星云提交脚本位于 `nebula/`。

完整的联合训练机制、数据预检和推理参数见
[Subtask 训练与推理指南](docs/subtask_training.md)。

PyTorch 版普通训练、subtask 联合训练、权重转换和梯度累积见
[Torch 训练指南](docs/pytorch_training.md)。

## 入口

以下命令在仓库根目录、已配置环境的 GPU 机器上执行。

| 用途 | 入口 | 参数 |
| --- | --- | --- |
| 普通 π0.5 训练 | [train_pi05_b1k.sh](scripts/train_pi05_b1k.sh) | `smoke` / `full` |
| Subtask 联合训练 | [train_pi05_subtask_b1k.sh](scripts/train_pi05_subtask_b1k.sh) | `check` / `smoke` / `full` |
| Torch 普通训练 | [train_pi05_pytorch_b1k.sh](scripts/train_pi05_pytorch_b1k.sh) | `smoke` / `full` |
| Torch Subtask 联合训练 | [train_pi05_subtask_pytorch_b1k.sh](scripts/train_pi05_subtask_pytorch_b1k.sh) | `check` / `smoke` / `full` |
| 在线策略服务 | [serve_b1k.py](openpi/scripts/b1k/serve_b1k.py) | checkpoint、任务及 subtask 生成参数 |
| 闭环评测 | [eval_b1k.sh](scripts/eval_b1k.sh) | 任务名；自动启动策略服务和仿真 |
| 多任务评测 | [eval_sweep.sh](scripts/eval_sweep.sh) | `TASK_FILE`、`WORKERS` |
| 星云单机 8 卡双任务联合训练 | [submit.sh](nebula/submit.sh) | 队列、数据和环境包配置 |
| 星云多机联合训练 | [submit_pi05_multihost.sh](nebula/submit_pi05_multihost.sh) | `MODE=discover/preflight/smoke/full` |
| 星云闭环评测 | [submit_pi05_eval.sh](nebula/submit_pi05_eval.sh) | `EVAL_PROFILE=smoke/reduced/full` |

## 训练

准备 LeRobot 格式的数据集，设置 `DATASET_ROOT`。
Subtask 段表默认读取 `$DATASET_ROOT/meta/subtask_segments.parquet`。
可以用 `EPISODES_FILE` 指定训练 episode 名单；未指定时 `full` 检查整个数据集。

```bash
# 只预检数据
DATASET_ROOT=/path/to/dataset \
  bash scripts/train_pi05_subtask_b1k.sh check

# 单卡联合训练：batch 从 1 起步，按显存和吞吐调整
DATASET_ROOT=/path/to/dataset \
NUM_GPUS=1 CUDA_DEVICES=0 BATCH_SIZE=1 \
TRAIN_STEPS=50000 EXP_NAME=pi05_subtask_run \
  bash scripts/train_pi05_subtask_b1k.sh full
```

普通 π0.5 训练使用 `scripts/train_pi05_b1k.sh full`，不设置
`SUBTASK_PREDICTION=1`。训练默认输出至 `openpi/outputs/`，可用
`ASSETS_BASE_DIR`、`CHECKPOINT_BASE_DIR` 覆盖。

联合训练始终输入完整任务，以 GT subtask 计算文本损失，并训练动作预测。
当前空标注策略默认 `drop_episode`，过滤含空标注的整个 episode；预检报告保存在
`openpi/outputs/subtask_preflight/`。原始数据不被删除。

## 推理与闭环评测

```bash
PI05_CKPT=/path/to/checkpoint/step \
PI05_REPO_ID=训练时使用的REPO_ID \
SUBTASK_PREDICTION=1 INSTANCE_INDICES=0 NUM_ROLLOUTS=1 \
  bash scripts/eval_b1k.sh turning_on_radio
```

`PI05_REPO_ID` 必须匹配 checkpoint 的 `assets/` 子目录。
`SUBTASK_PREDICTION=1` 使用联合训练权重，先生成 subtask，再预测动作。
普通 π0.5 评测省略此开关。`MODEL` 默认 `pi05`，也兼容显式 `MODEL=pi05`。

默认 GPU 0 跑策略、GPU 1 跑仿真，可通过 `SERVER_GPU`、`EVAL_GPU` 配置。
评测需要单独准备 OmniGibson / Isaac Sim 环境、支持的 GPU 和资产；
`EVAL_ENV` 指向评测环境，`OMNIGIBSON_DATA_PATH` 指向资产目录。
`OUTPUT_ROOT` 默认 `eval_results/`，结果位于 `pi05/<task>/` 下。

[eval_openloop.py](scripts/eval_openloop.py) 保留用于普通 π0.5 或 GT subtask
条件化的动作误差诊断，不用于验证联合模型的自主 subtask 生成；后者使用闭环入口。
[summarize_closed_loop.py](scripts/summarize_closed_loop.py) 可汇总并比较不同 checkpoint 的结果。

## 星云

先配置本机的 `nebula/config.sh`、队列、OSS 挂载、数据路径和环境包。
该配置文件不提交到 Git。

```bash
# 单机 8 卡、两个任务的数据混合训练
bash nebula/submit.sh

# 多机正式训练：默认只训练 3 步，必须显式设置 TRAIN_STEPS
MODE=full TRAIN_STEPS=50000 SAVE_INTERVAL=5000 \
  bash nebula/submit_pi05_multihost.sh

# 小规模闭环评测；默认 repo_id 对应双任务联合训练
EVAL_PROFILE=smoke CHECKPOINT_SOURCE=/path/to/checkpoint/step \
  bash nebula/submit_pi05_eval.sh
```

多机默认 2 台 × 8 卡、全局 batch 64；当前要求新实验目录，暂不支持断点续训。
星云评测 worker 当前要求两张 L20；`EVAL_PROFILE=full` 需显式设置 `CONFIRM_FULL=1`。
如果权重使用了不同的训练 `REPO_ID`，提交评测时同步设置 `PI05_REPO_ID`。

## 环境和数据工具

- [setup_envs.sh](scripts/setup_envs.sh)：构建 π0.5 环境，默认只安装 π0.5，保留 `pi05` 参数兼容。
- [setup_eval_env.sh](scripts/setup_eval_env.sh)、[setup_eval_assets.sh](scripts/setup_eval_assets.sh)：准备仿真环境与资产。
- [pack_envs.sh](scripts/pack_envs.sh)：默认打包 `b1k-pi05` 和 `b1k-eval`。
- [bootstrap.sh](scripts/bootstrap.sh)：按 `tools/envs/dataset/models` 阶段准备训练机器。
- [stage_dataset_local.sh](scripts/stage_dataset_local.sh)：暂存数据到本地磁盘。
- [download_pi05_base.sh](scripts/download/download_pi05_base.sh)：下载 π0.5 基座权重。
- [gen_subtask_prompts.py](scripts/gen_subtask_prompts.py)：从标注生成 subtask 段表。

数据和标注说明：[数据统计](DATASET_STATS.md)、[Subtask 标注](SUBTASK_ANNOTATIONS.md)、
[任务内 Subtask 变化](SUBTASK_VARIABILITY_BY_TASK.md)、[完整任务指令](FULL_TASK_INSTRUCTIONS.md)。

## 目录与维护

```text
openpi/       π0.5 模型、训练、推理及共享依赖
scripts/      BEHAVIOR 数据处理和训练评测入口
nebula/       星云提交、worker 和评测汇总
docs/         Subtask 训练与推理指南
BEHAVIOR-1K/  外部仿真器源码与资产（不提交到本仓库）
```

OpenPI 内部共享模型、配置、客户端和测试保留上游兼容性；项目入口仅维护 π0.5。
ALOHA、DROID、LIBERO、UR5 的独立示例已移除。
GR00T / G0.5 专用入口、旧实验启动脚本、历史对比报告和图表已移除；需要追溯时
可查看 Git 提交 `c0f6485`。实验产物写入已忽略的 `experiments/` 或 `outputs/`。
