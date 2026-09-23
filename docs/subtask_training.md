# π0.5 subtask 监督训练（BEHAVIOR 2026 / JAX）

本实现参考 [OpenPI PR #942](https://github.com/Physical-Intelligence/openpi/pull/942)
的联合监督方式，接入本仓库已有的 BEHAVIOR 2026 段表。

本文以 JAX 为主，普通 π0.5 的环境与启动方式见[仓库 README](../README.md)。
Subtask 监督使用 joint 联合训练：

| 模式 | 监督目标 | 动作块边界处理 | 启动开关 |
| --- | --- | --- | --- |
| Joint predictor | subtask 文本 CE + 动作 flow matching | 不自动过滤跨段动作块 | `SUBTASK_PREDICTION=1` |

Joint 的 `SUBTASK_ACTION_CONDITIONING=subtask_only` 只控制其动作分支的语言输入，
高层仍然学习根据完整任务和观测预测 subtask。

```text
joint 高层: Task + State + image → Subtask tokens (CE)
joint 低层: Subtask + State + image → action chunk (flow matching)
```

joint 训练脚本现在默认 `SUBTASK_ACTION_CONDITIONING=subtask_only`，对应
`p(s | o, T) p(a | o, s)`。高层保留完整任务，用真值 subtask teacher forcing，交叉熵监督
第一个词直到 EOS；padding、任务前缀和 `Action:` 不计入文本损失。低层使用独立的
`State: ...; Subtask: <GT tokens + EOS> Action:` 前缀，全部有效上下文双向可见，
不包含完整任务。两个语言前向共享权重，只复用不含语言信息的 SigLIP 输出，不复用高层
隐藏状态或 KV cache，避免完整任务通过图像/文本隐藏状态间接进入动作分支。

没有新增模型参数，但相比旧 joint 增加了一次高层语言前向，
训练速度和峰值显存需要在实际 GPU 上测量。此改动对齐层级条件关系，不代表复现论文的
完整预训练、多数据源混合配方，也不包含历史记忆或 subtask 完成判别器。

可选的视觉历史记忆已经接入 JAX action expert，默认关闭。启用方法、缓存生成和
训练/推理成本说明见 [FrameSamp + Modul](frame_memory_training.md)；它保留本文的 subtask 联合监督。

### 与旧 checkpoint 的兼容

- 旧 joint 模式为 `SUBTASK_ACTION_CONDITIONING=task_and_subtask`，动作仍可见完整任务。
  继续旧实验的 `RESUME=1` 时必须显式设置这个变量。
- 新训练会在每个 checkpoint 的 `assets/subtask_conditioning.json` 写入模式。
  推理服务自动读取，显式指定冲突模式会报错。
- 无该文件的旧 checkpoint 默认使用旧模式，不会因为升级代码就更换动作输入。
  `Pi0Config` 本身保留旧默认；直接调用 Python trainer 时，新模式需设置
  `--model.subtask-action-conditioning subtask_only`。
- 从旧权重切换到新模式，请开新实验做 fine-tuning；不要恢复旧优化器状态，也不要只在
  eval 时删掉完整任务就当作已训练的新模型。

## 1. 数据准备

需要现有训练环境 `openpi/.venv`、BEHAVIOR 2026 LeRobot v3 数据、原始 annotations
或已生成的段表，以及 π0.5 基础权重。使用既有 OpenPI 环境；无需升级生产环境依赖。

从 BEHAVIOR-VLAs 根目录生成段表（已有有效段表可跳过）。默认 `legacy` 保持旧 checkpoint 的标签语义；也可生成包含角色和来源信息的 role-aware 格式并写到新文件：

```bash
openpi/.venv/bin/python scripts/gen_subtask_prompts.py \
  --dataset /path/to/2026-challenge-demos

openpi/.venv/bin/python scripts/gen_subtask_prompts.py \
  --dataset /path/to/2026-challenge-demos \
  --label-format role_aware_v1 \
  --out /path/to/role_aware_v1.parquet
```

核心列仍为 `episode_index | start | end | subtask`。区间为 **[start, end)**，帧号是
episode 内的局部帧号，episode id 必须是原始 LeRobot id；选子集或设置 FRAME_STRIDE
时不能重编号。role-aware 表还保存对象角色、原始 annotation 路径、source segment/skill id、
`is_annotated` 和 `provenance`。role-aware 格式保留相邻同文本的不同原始标注段；
legacy 格式仍会合并相邻同文本区间。两者均不会从 segment end 虚构 completion、success
或 recovery 标签。

所有保留 episode 的 segment 必须精确覆盖 `[0, length)`，并具有非空文本标签。
空标签（包括 imputed 区间）按 `SUBTASK_EMPTY_LABEL_POLICY` 报错或剔除整条 episode。
不会用整任务、相邻 subtask 或 EOS-only 目标填补缺失监督；原段表和传入的 EPISODES_FILE 均不修改。

如需遇到空标签立即停止，设置 `SUBTASK_EMPTY_LABEL_POLICY=error`。直接调用预检 CLI
或 trainer 仍默认严格报错；自动筛选由启动脚本组织。未知 episode、缺少全部标注、
保留 episode 的重复/重叠区间、间隙、越界和 token 超限仍会报错；剔除后没有 episode
时也会停止。尾部 settle 帧应由段表生成步骤明确覆盖。

## 2. 先预检，再 smoke test

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
  bash scripts/train_pi05_subtask_b1k.sh check
```

默认检查全数据；设置 EPISODES_FILE 则只检查指定子集。筛选后检查标注覆盖和 token
预算，不加载视频、基础模型或优化器。报告保存至
`openpi/outputs/subtask_preflight/<EXP_NAME>.json`，最终原始 episode ID 名单保存至同目录
`<EXP_NAME>.episodes.txt`。报告包含 requested/selected/dropped 数量、丢弃的 episode
及空标签区间。预算检查结合最长任务、最长 subtask
和机器人状态数字的保守长度上界，并预留完整生成预算；数据 tokenizer 也对每条实际样本再次严格检查。

例如只训练当前两任务时，保留已有的 400 个 episode 名单作为输入：

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
EPISODES_FILE=/path/to/two_tasks_episodes.txt \
EXP_NAME=pi05_subtask_two_tasks_check \
  bash scripts/train_pi05_subtask_b1k.sh check
```

若这份子集恰好有 78 个 episode 含空标签，报告应显示 requested=400、dropped=78、
episodes=322；这是按所述数据情况推算，需以实际生成的报告为准。`full` 和 `smoke`
也会执行同一预检，并将最终名单自动用于归一化统计（需要新计算时）和训练器。
joint 保留已有 normalization stats 的复用行为；切换数据子集时使用独立 assets 目录或 REPO_ID，
自行核对统计来源，避免复用包含验证集的统计。

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

## 3. Joint 训练

### Joint 联合训练

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
INIT_PARAMS=/path/to/pi05_base/params \
REPO_ID=behavior-1k/joint_v1 \
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
| SUBTASK_EMPTY_LABEL_POLICY | drop_episode | 剔除含空标签的整条 episode；error 表示严格报错 |
| MAX_TOKEN_LEN | 512 | 完整语言序列上限，包含离散 state 和动作提示 |
| MAX_SUBTASK_LEN | 64 | 标签 token 上限，包含 EOS |
| SUBTASK_LOSS_WEIGHT | 1.0 | 文本 CE 权重 |
| SUBTASK_LOSS_CHUNK_SIZE | 8 | 每次词表投影的 token 数量 |
| SUBTASK_ACTION_CONDITIONING | subtask_only | 启动脚本默认；旧 joint 恢复时显式选择 task_and_subtask |
| WHOLE_TASK_PROB | 0 | 预测训练必须为 0，任务 prompt 始终保留 |
| FRAME_STRIDE | 1 | 在原始帧索引上做采样 |
| EPISODES_FILE | 未设置 | 训练和预检使用的原始 episode id 子集 |
| INIT_PARAMS | 原配置路径 | 新实验初始化的参数目录 |

如果超出 token 预算，应增加对应上限后重新预检。本实现不会静默截掉 subtask、EOS、state
或 Action 提示。文本 CE 仅投影监督位置，并按块 rematerialize，避免完整 `[batch, prompt_len,
vocabulary]` logits；减小 chunk size 可降低这一部分的临时显存，代价是更多计算。

直接运行 trainer 不会自动运行全表预检，请先执行 check 命令。逐样本校验始终启用。
若直接运行 trainer，必须把预检报告中的最终名单作为 `--data.episodes-index` 传入；
仅生成名单文件而不传给 trainer，不会改变它的采样范围。

## 4. 训练与验证集划分

先按 `(task_index, task_instance_id)` 分组切分，保证同一初始状态的全部轨迹只出现在一侧：

```bash
openpi/.venv/bin/python openpi/scripts/b1k/split_subtask_episodes.py \
  --dataset-root /path/to/2026-challenge-demos \
  --retained-episode-file /path/to/retained_episodes.txt \
  --task-indices 14 56 \
  --output-dir /path/to/split --seed 1 --val-fraction 0.2
```

输出包含 `train_episodes.txt`、`val_episodes.txt` 和带输入/输出 SHA-256、逐任务统计及 group
分配的 `split_manifest.json`。训练入口会再次拒绝 train/val episode 交集。

```bash
DATASET_ROOT=/path/to/2026-challenge-demos \
SUBTASK_PROMPTS=/path/to/subtask_segments.parquet \
EPISODES_FILE=/path/to/split/train_episodes.txt \
VAL_EPISODES_FILE=/path/to/split/val_episodes.txt \
INIT_PARAMS=/path/to/pi05_base/params \
REPO_ID=behavior-1k/joint_two_tasks_s1 \
EXP_NAME=pi05_b1k_joint_two_tasks_s1 \
NUM_GPUS=4 CUDA_DEVICES=0,1,2,3 BATCH_SIZE=64 \
TRAIN_STEPS=50000 LR_DECAY_STEPS=50000 \
  bash scripts/train_pi05_subtask_b1k.sh full
```

首次运行先将 `full` 改为 `check` 检查数据，再用小 batch 和 `TRAIN_STEPS=2` 跑 `smoke`。
上例按步数训练；若设置 `TRAIN_EPOCHS`，脚本会校验 `TRAIN_STEPS` 是否等于预检报告
`training_contract.train_steps`，不会自动代填。计算时必须使用同一段表、episode 子集、
frame stride 和全局 batch，并将 `LR_DECAY_STEPS` 设为相同总步数。

验证 loader 使用训练统计、独立 episode 列表和确定性 RNG；显式启用 validation 后的错误不会被静默忽略。

## 5. 离线全技能评测

```bash
openpi/.venv/bin/python scripts/eval_openloop.py \
  --dataset /path/to/2026-challenge-demos \
  --pi05-ckpt /path/to/checkpoint \
  --repo-id behavior-1k/joint_two_tasks_s1 \
  --episodes-file /path/to/split/val_episodes.txt \
  --subtask-prompts /path/to/role_aware_v1.parquet \
  --subtask-conditioning provided \
  --samples-per-subtask 32 \
  --out /path/to/result.json
```

`--repo-id` 必须与 checkpoint `assets/` 下保存 norm stats 的目录名一致。joint predictor 的外部标签审计
使用 `provided` 并逐样本断言来源为 `provided`。
后者使用 GT 标签，不测量模型自主生成技能的准确率。报告只在
segment 内且非 episode padding 的动作位置计算 L1/RMSE，并按 task、canonical subtask、action
group 和 chunk position 汇总，同时记录 checkpoint、段表和 split 哈希。

比较两种标签格式时，先生成参考结果，再在另一段表上增加
`--sample-keys-file /path/to/reference-result.json`。这会复用完全相同的 episode/frame 和有效 action
horizon，只从当前段表重新解析 subtask；若当前 segment 无法覆盖参考 horizon 会直接报错。该指标是
held-out demonstration action error，不是闭环成功率或 q-score。

## 6. 观察训练是否有效

Joint 日志除总 loss 外还包含：

- `action_loss`：连续动作 flow-matching MSE。
- `subtask_loss`：逐样本平均的 subtask CE，包含 EOS。
- `subtask_token_accuracy`：按有效 token 数加权的 teacher-forced token 准确率。
- `subtask_tokens`：每条样本的平均监督 token 数。

先检查小数据切片上 subtask CE 下降、token 准确率上升，并同时观察动作损失。token 准确率
是使用真实前文时的指标，不是自由生成文本正确率，也不等于机器人任务成功率。

## 7. 自动生成 subtask 的在线推理

已补齐 `整任务 + 当前图像/状态 → 自回归生成 subtask → 动作块`。
使用联合监督训练得到的 checkpoint，启动 B1K 服务：

```bash
cd openpi
.venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio \
  --repo-id behavior-1k/2026-challenge-demos \
  --policy.config pi05_b1k --policy.dir /path/to/subtask-checkpoint/49999 \
  --subtask-prediction --max-token-len 512 --max-subtask-len 64 \
  --action-horizon 16 --port 8000
```

`repo-id` 必须对应 checkpoint 的 assets 目录；token 长度配置与训练保持一致。
普通 action-only checkpoint 没有经过新增的文本监督，不能仅靠开启服务参数就获得可靠的
subtask 预测。本节使用 JAX checkpoint，已有 Torch 实现另见 [PyTorch 指南](pytorch_training.md)。

- 默认不需要输入真值 subtask。`create_trained_policy` 检测到模型配置中的
  `train_subtask_prediction=True` 时会创建 `SubtaskPolicy`。
- 语言前缀与训练完全一致。逐 token 解码使用 KV cache，保留每条样本独立的
  位置、有效掩码和 EOS 停止状态。动作端直接使用生成的 token，包含 EOS 和 Action 提示，
  不经过文字解码再编码。语言生成和动作采样各做一次图像编码，未跨阶段复用视觉特征。
- 新模式默认 `subtask_interval=2`，旧模式默认 1。2 是工程初始值，不是论文指定频率。
  `subtask_interval=1` 表示每次请求新动作块时重新生成；结合 `action_horizon=16`，通常
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
新生成的文本。websocket 每次返回一个控制步，由服务端缓存动作块；每个机器人环境应有自己的
有状态 policy/server 实例。观测字段、`subtask_sources` 和 reset 协议见
[B1K 远程推理](../openpi/docs/remote_inference.md)。

原有闭环评测脚本也已接入（从 BEHAVIOR-VLAs 根目录运行）：

```bash
MODEL=pi05 SUBTASK_PREDICTION=1 \
PI05_CKPT=/path/to/subtask-checkpoint/49999 \
PI05_REPO_ID=behavior-1k/joint_v1 \
INSTANCE_INDICES=0 NUM_ROLLOUTS=1 \
OUTPUT_ROOT=/path/to/subtask-evaluation \
  bash scripts/eval_b1k.sh turning_on_radio
```

可用 MAX_TOKEN_LEN、MAX_SUBTASK_LEN、SUBTASK_INTERVAL、SUBTASK_TEMPERATURE 调整服务参数。
`EXECUTE_HORIZON` 单独控制每次执行多少步，默认 16；例如
`SUBTASK_INTERVAL=2 EXECUTE_HORIZON=16` 通常每 16 步重新预测动作、每 32 步重新预测 subtask。
期间始终用最新图像/状态和缓存的 subtask 重建低层输入。可用 interval=1/2/4 做闭环消融，
较慢高层刷新可能延迟技能切换，不能保证提高成功率。
默认 greedy 解码（temperature=0）；请给预测 subtask 与 oracle 评测使用不同输出目录。
星云 eval worker 根据 checkpoint 条件模式选择刷新间隔；以运行参数和 rollout metadata 为准。

### 给定 subtask 的对比推理

固定外部 subtask 使用 `EXTERNAL_SUBTASK`，需设置 `PROMPT_MODE=whole_task`，
并使用 joint checkpoint 和 `SUBTASK_PREDICTION=1`。
例如 joint 的单实例 smoke：

```bash
MODEL=pi05 SUBTASK_PREDICTION=1 PROMPT_MODE=whole_task \
EXTERNAL_SUBTASK="move to paper bag" \
PI05_CKPT=/path/to/joint-checkpoint \
PI05_REPO_ID=behavior-1k/joint_v1 \
INSTANCE_INDICES=0 NUM_ROLLOUTS=1 MAX_STEPS=100 \
OUTPUT_ROOT=/path/to/external-subtask-smoke \
  bash scripts/eval_b1k.sh carrying_in_groceries
```

Evaluator 会在每次新 observation 上重新注入该文本，并要求服务端来源始终为 `provided`；
rollout metadata 记录 `external_fixed`、`provided`、`oracle=false`。这只评测给定初始技能的
执行能力，不代表完整任务调度能力。

### Joint checkpoint + 外部技能 selector

```bash
MODEL=pi05 SUBTASK_PREDICTION=1 \
PROMPT_MODE=whole_task MAX_TOKEN_LEN=512 \
EXTERNAL_SUBTASK_SELECTOR=state_machine_v1 \
PI05_CKPT=/path/to/joint-checkpoint/step \
PI05_REPO_ID=behavior-1k/joint_v1 \
INSTANCE_INDICES=0 NUM_ROLLOUTS=1 \
OUTPUT_ROOT=/path/to/joint-selector-evaluation \
  bash scripts/eval_b1k.sh carrying_in_groceries make_rose_centerpieces
```

此模式向 `SubtaskPolicy` 提供外部 subtask，跳过本次技能生成，来源记录为 `provided`。
低层输入继续遵循 checkpoint 的 joint 动作条件配置。

`state_machine_v1` 仅支持上面两个任务，外部 BEHAVIOR evaluator 必须实现
`--external-subtask-selector`。它使用仿真特权状态，结果标记
`oracle=true`、`privileged=true`、`official_benchmark=false`，用于诊断执行器能力。
`EXTERNAL_SUBTASK` 与 `EXTERNAL_SUBTASK_SELECTOR` 互斥；固定技能 smoke 可用前者替换 selector。

## 实现范围与回归测试

当前包含联合训练、自动 subtask 生成及动作推理、给定 subtask 的对比推理和 B1K 服务接入。
训练是否完成、闭环是否成功，需要查看对应训练日志、checkpoint 和 rollout 结果；平台 worker
还会写入 `_RUNNING`、`_SUCCESS`、`_FAILED` 或 manifest。状态文件不是所有本地入口的统一协议，
worker 成功退出也不等同于机器人完成任务。

在安装了仓库训练依赖的环境中，从 `openpi/` 运行：

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  src/openpi/training/subtask_data_test.py \
  src/openpi/training/subtask_pipeline_test.py \
  src/openpi/models/subtask_test.py \
  src/openpi/models/subtask_inference_test.py \
  src/openpi/policies/subtask_policy_test.py \
  src/openpi/shared/subtask_contract_test.py \
  scripts/b1k/check_subtask_data_test.py
```

回归测试覆盖段表边界/缺标注、role-aware provenance、输入透传、EOS/首词监督、
因果注意力、真实 Pi0/Gemma 接口的
联合反向传播、验证集隔离、外部 subtask 来源传输及原动作 API。模型测试使用小词表和轻量
图像编码器；真实数据 preflight 不解码视频，也不代表完整 checkpoint 已完成收敛或闭环验证。

条件分解回归还检查固定观测、subtask 和噪声时，改变完整任务不会改变低层动作结果；
训练一步打开初始化为零的 adaRMS gates 后，
改变低层 subtask 会改变采样动作，避免把“模型忽略所有上下文”误当作通过独立性检查。
模式随 checkpoint 保存，恢复训练时不能无意切换条件关系。
