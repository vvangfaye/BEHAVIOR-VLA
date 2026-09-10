# π0.5 subtask 监督训练（BEHAVIOR 2026 / JAX）

本实现参考 [OpenPI PR #942](https://github.com/Physical-Intelligence/openpi/pull/942)
的联合监督方式，接入本仓库已有的 BEHAVIOR 2026 段表。

训练目标是：从图像、机器人状态和整任务指令预测当前 subtask 文本；动作专家在真实
subtask 条件下学习连续动作。不是把整任务 prompt 替换成 subtask。

```text
dataset frame
  ├─ task_index → 整任务 prompt
  ├─ episode_index + frame_index → 当前 subtask 标签
  └─ images / state / action chunk
          ↓ Repack → B1KInputs → normalization
Task: ... State: ...;\nSubtask: <真实标签><EOS>\nAction:
          ↓ PaliGemma + action expert
loss = flow-matching MSE + subtask_loss_weight × subtask next-token CE
```

图像与任务前缀使用双向注意力，subtask 和动作提示文本使用因果注意力，动作块能看到完整
真实 subtask。交叉熵监督第一个词直到 EOS，padding、任务前缀和 `Action:` 不计入文本损失。
词表投影复用原有 embedding 权重，没有增加 checkpoint 参数。

## 1. 数据准备

需要现有训练环境 `openpi/.venv`、BEHAVIOR 2026 LeRobot v3 数据、原始 annotations
或已生成的段表，以及 π0.5 基础权重。使用既有 OpenPI 环境；无需升级生产环境依赖。

从 BEHAVIOR-VLAs 根目录生成段表（已有有效段表可跳过）：

```bash
openpi/.venv/bin/python scripts/gen_subtask_prompts.py \
  --dataset /path/to/2026-challenge-demos
```

段表格式：`episode_index | start | end | subtask`。区间为 **[start, end)**，帧号是
episode 内的局部帧号，episode id 必须是原始 LeRobot id；选子集或设置 FRAME_STRIDE
时不能重编号。使用生成脚本中 metadata 的 `annotation_path` 映射，不能按文件名猜配对。

训练标签严格要求每个被选 episode 的 `[0, length)` 都被覆盖。未知 episode、空文本、
重复/重叠区间、间隙和越界帧会报错，不会回退为整任务标签，也不会自动延长最后一段。
尾部 settle 帧应由段表生成步骤明确覆盖。预检失败时修正段表或用 EPISODES_FILE 选择
已验证的完整 episodes，不要把缺标注帧伪装成有效监督。

## 2. 先预检，再 smoke test

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
  bash scripts/train_pi05_subtask_b1k.sh check
```

默认检查全数据的标注覆盖和 token 预算，不加载视频、基础模型或优化器。报告保存至
`openpi/outputs/subtask_preflight/<EXP_NAME>.json`。预算检查结合最长任务、最长 subtask
和机器人状态数字的保守长度上界，并预留完整生成预算；数据 tokenizer 也对每条实际样本再次严格检查。

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
INIT_PARAMS=/path/to/pi05_base/params \
NUM_GPUS=1 CUDA_DEVICES=0 NUM_WORKERS=0 BATCH_SIZE=1 TRAIN_STEPS=2 \
  bash scripts/train_pi05_subtask_b1k.sh smoke
```

smoke 默认只用原始 episodes 0 和 1。可通过 EPISODES_FILE 指定每行一个原始 episode id。
需要完整的对应视频和动作数据。若未设置 INIT_PARAMS，则沿用配置中的 π0.5 基础权重路径，
可以通过 OPENPI_DATA_HOME 复用已有缓存。

不要把 INIT_PARAMS（加载模型权重开始新实验）和 RESUME（恢复整个优化器/训练状态）混为一谈。
subtask 训练默认不会覆盖已有同名实验；使用新的 EXP_NAME，或者明确选择 RESUME=1。
只有明确要替换旧实验时才设置 OVERWRITE=1，不能与 RESUME=1 同用。

## 3. 完整训练

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
INIT_PARAMS=/path/to/pi05_base/params \
EXP_NAME=pi05_b1k_subtask_v1 \
NUM_GPUS=8 CUDA_DEVICES=0,1,2,3,4,5,6,7 FSDP_DEVICES=8 \
BATCH_SIZE=64 NUM_WORKERS=8 TRAIN_STEPS=50000 LR_DECAY_STEPS=50000 \
  bash scripts/train_pi05_subtask_b1k.sh full
```

资源参数需要按实际硬件调整；本次本地实现没有测量完整模型的 GPU 显存需求。
脚本会先预检，再复用/计算与原动作训练一致的 normalization stats，然后启动 JAX trainer。
默认关闭 W&B。checkpoint 保存到 `openpi/outputs/checkpoints/pi05_b1k/<EXP_NAME>/`。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| SUBTASK_PROMPTS | `<DATASET_ROOT>/meta/subtask_segments.parquet` | 帧级标签段表 |
| MAX_TOKEN_LEN | 512 | 完整语言序列上限，包含离散 state 和动作提示 |
| MAX_SUBTASK_LEN | 64 | 标签 token 上限，包含 EOS |
| SUBTASK_LOSS_WEIGHT | 1.0 | 文本 CE 权重 |
| SUBTASK_LOSS_CHUNK_SIZE | 8 | 每次词表投影的 token 数量 |
| WHOLE_TASK_PROB | 0 | 预测训练必须为 0，任务 prompt 始终保留 |
| FRAME_STRIDE | 1 | 在原始帧索引上做采样 |
| EPISODES_FILE | 未设置 | 训练和预检使用的原始 episode id 子集 |
| INIT_PARAMS | 原配置路径 | 新实验初始化的参数目录 |

如果超出 token 预算，应增加对应上限后重新预检。本实现不会静默截掉 subtask、EOS、state
或 Action 提示。文本 CE 仅投影监督位置，并按块 rematerialize，避免完整 `[batch, prompt_len,
vocabulary]` logits；减小 chunk size 可降低这一部分的临时显存，代价是更多计算。

原脚本的 `SUBTASK_PROMPTS=...` 且 **SUBTASK_PREDICTION=0** 仍是旧的“subtask 替换 prompt”
模式，不会训练语言生成；新增 wrapper 强制启用预测监督。也可以直接使用底层 CLI：

```bash
cd openpi
.venv/bin/python scripts/b1k/train_b1k.py pi05_b1k \
  --exp-name pi05_subtask_custom \
  --model.train-subtask-prediction \
  --model.max-token-len 512 \
  --model.max-subtask-len 64 \
  --data.dataset-root /path/to/2026-challenge-demos \
  --data.repo-id behavior-1k/2026-challenge-demos \
  --data.subtask-prompts-path /path/to/meta/subtask_segments.parquet \
  --data.subtask-whole-task-prob 0 \
  --no-wandb-enabled
```

直接运行 trainer 不会自动运行全表预检，请先执行 check 命令。逐样本校验始终启用。

## 4. 观察训练是否有效

日志除总 loss 外还包含：

- `action_loss`：连续动作 flow-matching MSE。
- `subtask_loss`：逐样本平均的 subtask CE，包含 EOS。
- `subtask_token_accuracy`：按有效 token 数加权的 teacher-forced token 准确率。
- `subtask_tokens`：每条样本的平均监督 token 数。

先检查小数据切片上 subtask CE 下降、token 准确率上升，并同时观察动作损失。token 准确率
是使用真实前文时的指标，不是自由生成文本正确率，也不等于机器人任务成功率。

## 5. 自动生成 subtask 的在线推理

已补齐 `整任务 + 当前图像/状态 → 自回归生成 subtask → 动作块`。
使用联合监督训练得到的 checkpoint，启动 B1K 服务：

```bash
cd openpi
.venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio \
  --repo-id behavior-1k/2026-challenge-demos \
  --policy.config pi05_b1k --policy.dir /path/to/subtask-checkpoint/49999 \
  --subtask-prediction --max-token-len 512 --max-subtask-len 64 \
  --subtask-interval 1 --action-horizon 16 --port 8000
```

`repo-id` 必须对应 checkpoint 的 assets 目录；token 长度配置与训练保持一致。
普通 action-only checkpoint 没有经过新增的文本监督，不能仅靠开启服务参数就获得可靠的
subtask 预测。JAX checkpoint 由原有加载流程恢复；PyTorch 暂不支持此功能。

- 默认不需要输入真值 subtask。`create_trained_policy` 检测到模型配置中的
  `train_subtask_prediction=True` 时会创建 `SubtaskPolicy`。
- 语言前缀与训练完全一致。逐 token 解码使用固定容量 KV cache，保留每条样本独立的
  位置、有效掩码和 EOS 停止状态。动作端直接使用生成的 token，包含 EOS 和 Action 提示，
  不经过文字解码再编码。语言生成和动作采样各做一次图像编码，未跨阶段复用视觉特征。
- `subtask_interval=1` 表示每次请求新动作块时重新生成；结合 `action_horizon=16`，通常
  执行 16 个控制步后根据新观测重新预测。设置 N>1 可在 N 次动作块请求之间复用 subtask，
  但动作预测始终读取当前图像和状态。多环境批处理必须使用 interval=1 或独立 policy 实例。
- 这是按动作块重新预测的调度，并未新增独立的 subtask 完成判别器。重新生成相同文本
  表示继续当前 subtask；切换质量仍需真实闭环评测。
- 显式传入 `subtask` 时执行给定文本的动作推理，输出标记 `provided`。下一次不提供
  subtask 时恢复自主生成，不会缓存真值供后续使用。任务或手动 subtask 改变、episode reset
  时清空相关状态；B1K 动作缓存和录制 wrapper 的 reset 也已接通。
- 没有生成 EOS、标签为空或 token 预算不足会报错，不会截断后继续动作预测。

`policy.infer` 输出 `actions`、`subtask`、`subtask_source`（generated / cached / provided）
和耗时。B1K websocket 保留原有 `action` 字段，并附带 `subtasks` 列表；服务日志也记录
新生成的文本。每个机器人环境应有自己的有状态 policy/server 实例。

原有闭环评测脚本也已接入（从 BEHAVIOR-VLAs 根目录运行）：

```bash
MODEL=pi05 SUBTASK_PREDICTION=1 \
PI05_CKPT=/path/to/subtask-checkpoint/49999 \
OUTPUT_ROOT=/path/to/subtask-evaluation \
  bash scripts/eval_b1k.sh turning_on_radio
```

可用 MAX_TOKEN_LEN、MAX_SUBTASK_LEN、SUBTASK_INTERVAL、SUBTASK_TEMPERATURE 调整服务参数。
默认 greedy 解码（temperature=0）；请给预测 subtask 与 oracle 评测使用不同输出目录。

## 实现范围与回归测试

当前包含联合训练、自动 subtask 生成及动作推理、给定 subtask 的对比推理和 B1K 服务接入。
本次仅实现和验证本地代码，没有启动 GPU 训练或 OmniGibson 闭环任务。

在安装了仓库训练依赖的环境中，从 `openpi/` 运行：

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  src/openpi/training/subtask_data_test.py \
  src/openpi/training/subtask_pipeline_test.py \
  src/openpi/models/subtask_test.py \
  src/openpi/models/subtask_inference_test.py \
  src/openpi/policies/subtask_policy_test.py \
  scripts/b1k/check_subtask_data_test.py
```

测试覆盖段表边界/缺标注、输入透传、EOS/首词监督、因果注意力、分块 CE 与完整 CE 的数值
和梯度一致性、真实 Pi0/Gemma 接口的联合反向传播、小样本过拟合和原动作 API 回归。
模型测试使用小词表和轻量图像编码器，避免下载数 GB 权重；不代表已完成 BEHAVIOR 全量训练。

本次本地验证（2026-09-10）：新增测试与相关旧路径回归共 **58 项通过**，使用 CPU、
Python 3.12、JAX 0.5.3、Flax 0.10.2、Torch 2.7.1。另用官方 257,152 词表 tokenizer
验证了标签解码、EOS 和离散状态的长度预算，并检查了启动脚本的预检、episode 子集、
恢复训练及参数传递。新增 Python 文件通过 Ruff；启动脚本通过语法检查。

推理验证包括：KV cache 解码与 teacher forcing logits 一致、不同长度前缀与 EOS 处理、
小模型训练后保存/恢复参数再自由生成正确标签、bfloat16 参数与缓存解码、自动与给定
subtask 路由、B1K 输入/动作适配、录制 wrapper reset、websocket 文本输出和服务 CLI。

这里的数据链路测试使用合成的 BEHAVIOR 格式帧和段表，不读取真实演示视频。未执行真实
数据上的完整 π0.5 GPU 训练，也未验证完整 checkpoint 的恢复、收敛或在线机器人成功率。
