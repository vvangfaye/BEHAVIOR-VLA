# BEHAVIOR 的 PyTorch π0.5 训练

> 本文保留已有 Torch 实现的使用说明。当前运行与代码阅读以 JAX 为主，入口见
> [仓库 README](../README.md)和 [JAX subtask 指南](subtask_training.md)。
> 不应将本文的精度、batch、优化器或分布式设置直接当作 JAX 的默认值。

支持普通动作训练和 subtask 文本/动作联合训练，复用 BEHAVIOR 数据配置、
归一化和标注预检。训练器支持单卡、torchrun DDP、梯度累积及断点续训。
本文命令从仓库根目录执行；数据、权重和环境路径按实际机器设置。

## 环境

```bash
bash scripts/setup_pytorch_env.sh
```

默认创建独立的 `b1k-pi05-torch` conda 环境，链接到 `openpi/.venv-torch`。
可通过 `ENV_ROOT` 或 `PYTORCH_ENV` 指定位置。原 JAX 环境的 `openpi/.venv` 链接保留。
安装器复用现有环境构建流程，并安装仓库中与 Transformers 5.5.4 配套的扩展。
扩展包括 π0.5 的条件归一化；只安装 PyPI Transformers 还不足以运行此模型。

若已手动准备 Torch 环境，先以该环境的 Python 安装 OpenPI 和客户端，再运行：

```bash
/path/to/torch-env/bin/python openpi/scripts/install_transformers_patch.py
```

训练入口会校验扩展文件与仓库一致，防止使用旧补丁。启动时将 `JAX_PLATFORMS=cpu`，
共享配置和 tokenizer 仍依赖 JAX/Flax，但训练张量和 GPU 计算使用 Torch。

## 基座权重

已有 `model.safetensors` 的 Torch π0.5 基座可直接使用。JAX 基座需要先转换一次：

```bash
bash scripts/convert_pi05_to_pytorch.sh /path/to/pi05_base /path/to/pi05_base_torch
```

输入目录应含 `params/`；输出目录含 `model.safetensors` 和配置记录。
转换默认保存 FP32 权重，可用 `CONVERT_PRECISION=bfloat16` 减小文件体积。
训练加载后参数仍转为 FP32。转换在 CPU 上执行，需要足够主机内存；不会加载到 GPU。
转换器按模型配置选择 π0.5 结构，严格核对参数，不依赖 checkpoint 目录名称。

## 普通训练

先运行两步 smoke：

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
PYTORCH_WEIGHT_PATH=/path/to/pi05_base_torch \
NUM_GPUS=1 CUDA_DEVICES=0 BATCH_SIZE=1 \
bash scripts/train_pi05_pytorch_b1k.sh smoke
```

默认 smoke 使用 episode 0、1；如果只下载了其他任务，设置 `EPISODES_FILE`。
正式训练将最后一个参数改为 `full`，并设置 `TRAIN_STEPS`、`EXP_NAME` 和合适的 batch。
`PYTORCH_WEIGHT_PATH` 接受权重目录或 safetensors 文件；首次训练必须提供基座权重。

## Subtask 联合训练

```bash
# 预检、输出有效 episode 名单；不加载基座和训练
DATASET_ROOT=/path/to/2026-challenge-demos \
EPISODES_FILE=/path/to/episodes.txt \
bash scripts/train_pi05_subtask_pytorch_b1k.sh check

# 正式训练示例；batch 需按实际显存调整
DATASET_ROOT=/path/to/2026-challenge-demos \
EPISODES_FILE=/path/to/episodes.txt \
PYTORCH_WEIGHT_PATH=/path/to/pi05_base_torch \
NUM_GPUS=1 CUDA_DEVICES=0 BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 \
TRAIN_STEPS=50000 EXP_NAME=pi05_subtask_torch_run \
bash scripts/train_pi05_subtask_pytorch_b1k.sh full
```

默认标注为 `$DATASET_ROOT/meta/subtask_segments.parquet`，可用 `SUBTASK_PROMPTS` 覆盖。
空标注默认 `drop_episode`；预检输出的有效 episode 名单同时用于归一化统计和训练。
标注规则和动作块边界限制仍见 [subtask 指南](subtask_training.md)。

joint 脚本默认 `SUBTASK_ACTION_CONDITIONING=subtask_only`。高层读取完整任务和观测，
subtask 标签以 next-token CE 训练；低层独立读取观测与 GT subtask，不读取完整任务或
高层 KV cache。两个前向共享模型参数与 SigLIP 输出，高层文本位置使用因果注意力。
默认总损失是
动作 flow-matching MSE 加 subtask CE，`SUBTASK_LOSS_WEIGHT` 可调文本项权重。
语言 logits 只计算 label 对应的位置，并按 `SUBTASK_LOSS_CHUNK_SIZE=8` 分块重算。
旧 joint checkpoint 继续训练时设置 `SUBTASK_ACTION_CONDITIONING=task_and_subtask`；
切换到新目标请开新实验做 fine-tuning。推理从 checkpoint 的 assets 自动识别模式。
两种模式权重结构相同，新增的高层前向仍会增加计算与激活开销，不能沿用旧训练的最大 batch。

## 显存、batch 和多卡

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `BATCH_SIZE` | 所有 GPU 一次前向的总样本数 | `NUM_GPUS` |
| `GRAD_ACCUM_STEPS` | 每次参数更新累计的前向/反向次数 | `1` |
| `TRAIN_PRECISION` | 计算精度：`bfloat16` / `float32` | `bfloat16` |
| `GRAD_CHECKPOINTING` | 激活重算以降低显存 | `1` |
| `MAX_TOKEN_LEN` | 联合训练文本长度 | `512` |
| `MAX_SUBTASK_LEN` | 最大 subtask token 数，含 EOS | `64` |

单卡 `BATCH_SIZE=8 GRAD_ACCUM_STEPS=8` 的有效 batch 是 64，每次前向只放 8 个样本。
这不表示单卡能一次放入 64 个样本；H20 上的最大 batch 需要真实测量。

默认采用 BF16 autocast，训练参数、梯度和 AdamW 状态保留 FP32；当前 Torch 训练器
不维护 EMA，`pi05_b1k_pytorch` 配置已关闭 EMA。CPU 测试使用 FP32。
显存日志区分 allocated、reserved 和峰值。

单机两卡示例：`NUM_GPUS=2 CUDA_DEVICES=0,1 BATCH_SIZE=8`，每卡前向 4 个样本。
脚本自动通过 torchrun 启动 DDP。全局 batch 必须能被 GPU 数整除。
DDP 复制模型和优化器，不像 JAX FSDP 那样分片参数。多机可通过 torchrun 直接调用
`openpi/scripts/b1k/train_b1k_pytorch.py`；现有星云多机提交脚本仍使用 JAX。

## Checkpoint 和续训

默认 checkpoint 路径：

```text
openpi/outputs/checkpoints/pi05_b1k_pytorch/<EXP_NAME>/<completed_steps>/
```

数字表示已完成的 optimizer 更新次数。例如两步 smoke 最终保存到 `2/`。
每个完整 checkpoint 包含模型、AdamW 状态、随机数状态、元数据和归一化 assets。
最后一步总会保存，即使 `SAVE_INTERVAL` 大于训练总步数。

续训使用相同 `EXP_NAME` 和数据配置，设置 `RESUME=1`；不再需要基座权重路径。
续训从 checkpoint 的 `assets/` 读取原训练归一化统计，不会因为外部统计缓存缺失而重新计算。
当前要求 GPU 数、batch、累积次数以及数据/模型/优化器配置保持一致，可提高目标
`TRAIN_STEPS`。不接受只有推理权重的目录作为完整训练恢复点。
恢复时会跳转到对应的数据 batch；多进程数据加载中的随机图像增强会重新初始化，
因此不承诺中断前后逐位一致。
JAX 和 Torch 的优化器状态不同，无法跨框架直接续训；可以转换模型权重后重新微调。

## 使用 Torch 权重评测

```bash
PI05_CKPT=/path/to/pi05_b1k_pytorch/pi05_subtask_torch_run/50000 \
PI05_REPO_ID=behavior-1k/2026-challenge-demos \
SUBTASK_PREDICTION=1 INSTANCE_INDICES=0 NUM_ROLLOUTS=1 \
bash scripts/eval_b1k.sh turning_on_radio
```

闭环入口检测到 `model.safetensors` 后默认使用 `openpi/.venv-torch`。
可通过 `PI05_ENV` 覆盖策略环境。普通模型省略 `SUBTASK_PREDICTION`；联合模型会先
生成 subtask，再预测动作。`PI05_REPO_ID` 必须与训练时一致。仿真环境和资产单独准备。

## 本地验证范围

2026-09-18 的条件分解改动通过 113 项 CPU 回归，覆盖 JAX/Gemma 和小型 HF 模型前反向、
Torch BF16 计算与推理、完整任务对低层动作的独立性、生成/缓存/外部 subtask、
JAX/Torch checkpoint 保存恢复，以及 BEHAVIOR 数据变换和启动脚本参数解析。
1 项双进程 DDP 测试因本地执行沙箱禁止 loopback socket 而跳过。
这些检查使用小模型和测试数据，未运行完整 BEHAVIOR 视频数据训练、H20 显存测量或仿真闭环。
