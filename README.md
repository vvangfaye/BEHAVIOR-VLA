# BEHAVIOR 2026：π0.5 / subtask（JAX）

本仓库使用 OpenPI 在 BEHAVIOR 2026 数据上训练 π0.5，并通过 OmniGibson 进行闭环评测。
当前使用与代码阅读以 **JAX 训练和推理**为主。双任务实验对应
`carrying_in_groceries` 和 `make_rose_centerpieces`。

本文介绍模型模式、实际入口、运行示例和代码阅读顺序。标注格式、预检规则与联合训练细节见
[Subtask 训练与推理指南](docs/subtask_training.md)。已有 Torch 实现说明保留在
[PyTorch 指南](docs/pytorch_training.md)，不作为本文的运行主线。

## 1. 先区分训练模式

| 模式 | 模型输入与监督 | 动作块边界处理 | 训练开关 |
| --- | --- | --- | --- |
| 普通 π0.5 | 观测 + 完整任务 → 动作；仅动作 loss | 原始动作块 | `SUBTASK_PREDICTION=0`（不设置 `SUBTASK_PROMPTS`） |
| Joint predictor | 高层预测 subtask；动作层使用 GT subtask；文本 CE + 动作 loss | 不自动过滤跨 subtask 边界的动作块 | `SUBTASK_PREDICTION=1` |

只设置 `SUBTASK_PROMPTS` 不等于开启 joint 模式；
普通入口还保留按概率混合完整任务与 subtask prompt 的旧用法。

新版 joint 启动脚本默认 `SUBTASK_ACTION_CONDITIONING=subtask_only`：

```text
训练高层：当前图像/状态 + 完整任务 → GT subtask 的 next-token CE
训练低层：当前图像/状态 + GT subtask → 动作 flow-matching loss

自主推理：当前图像/状态 + 完整任务 → 生成 subtask
          当前图像/状态 + 生成的 subtask → 动作块
```

高低层共享模型参数；训练时复用不含语言信息的 SigLIP 输出，动作分支不复用高层的
语言隐藏状态或 KV cache。**joint 中的 `subtask_only` 控制动作分支的语言输入，高层仍训练 subtask 预测。**
`Pi0Config` 本身为兼容旧权重保留 `task_and_subtask` 默认值；绕过 shell 直接运行 Python
trainer 时，新 joint 模式需要显式传入 `--model.subtask-action-conditioning subtask_only`。
当前未实现历史记忆或独立的 subtask 完成判别器。

## 2. 环境、数据和入口

以下命令从仓库根目录，在已准备数据和权重的 Linux GPU 机器上执行。

- 训练与策略服务：`openpi/.venv`，由 [setup_envs.sh](scripts/setup_envs.sh) 创建并链接。
  可配置 `CONDA`、`ENV_ROOT`、`PI05_ENV`；镜像源与路径按运行机器调整。
- 闭环仿真：由 [setup_eval_env.sh](scripts/setup_eval_env.sh) 和
  [setup_eval_assets.sh](scripts/setup_eval_assets.sh) 准备独立环境及资产。
  外部 `BEHAVIOR-1K/` 源码和资产不随本仓库提交；selector 评测要求环境中的 evaluator
  支持 `--external-subtask-selector`。
- 数据：BEHAVIOR 2026 LeRobot 数据，包含 `meta/info.json`、动作数据和所选 episode 的 RGB 视频。
- 权重：π0.5 JAX 基座的 `params/` 目录，可用
  [download_pi05_base.sh](scripts/download/download_pi05_base.sh) 下载；该脚本的代理和缓存路径需要按机器配置。

JAX 的数据读取仍使用 PyTorch `DataLoader`，共享环境中的 Torch/视频解码依赖需要保留。

| 用途 | 入口 | 参数或开关 |
| --- | --- | --- |
| 普通动作训练 | [train_pi05_b1k.sh](scripts/train_pi05_b1k.sh) | `smoke` / `full` |
| Joint 联合训练 | [train_pi05_subtask_b1k.sh](scripts/train_pi05_subtask_b1k.sh) | `check` / `smoke` / `full` |
| 数据预检 | [check_subtask_data.py](openpi/scripts/b1k/check_subtask_data.py) | 段表、episode 子集、token 预算 |
| 在线策略服务 | [serve_b1k.py](openpi/scripts/b1k/serve_b1k.py) | checkpoint、任务和条件模式 |
| 闭环评测 | [eval_b1k.sh](scripts/eval_b1k.sh) | 任务名；自动启动策略服务和仿真 |
| 多任务评测 | [eval_sweep.sh](scripts/eval_sweep.sh) | `TASK_FILE`、`WORKERS` |
| 离线动作误差诊断 | [eval_openloop.py](scripts/eval_openloop.py) | `--subtask-conditioning none/provided` |
| 闭环结果汇总 | [summarize_closed_loop.py](scripts/summarize_closed_loop.py) | 不同 checkpoint 的结果目录 |
| GoalKeeper 目标筛选研究原型 | [goalkeeper/README.md](goalkeeper/README.md) | 独立小头训练、候选动作筛选、可选闭环服务 |

GoalKeeper 代码已放入 [goalkeeper/](goalkeeper/)；它使用本仓库的 OpenPI B1K 输入字段和
`scripts/eval_b1k.sh` 评测入口。GoalKeeper 服务可在 joint checkpoint 上让同一轮
候选动作共享一个预测 subtask，避免候选采样触发额外 subtask 刷新。配置方式、
小头训练要求和验证边界见其 README。

## 3. JAX 训练

### 公共设置

```bash
export DATASET_ROOT=/path/to/2026-challenge-demos
export INIT_PARAMS=/path/to/pi05_base/params
export REPO_ID=behavior-1k/2026-challenge-demos

# 可选：只训练指定 episode；文件中每行一个原始 episode_index
# export EPISODES_FILE=/path/to/train_episodes.txt
```

以下示例分别运行，假定没有遗留的其他模式开关。`smoke` 未指定 episode 名单时使用
原始 episode 0、1；`full` 未指定名单时使用全数据。只下载双任务数据时应提供对应名单。

### 普通 π0.5

```bash
SUBTASK_PREDICTION=0 \
NUM_GPUS=1 CUDA_DEVICES=0 BATCH_SIZE=1 NUM_WORKERS=0 \
EXP_NAME=pi05_ordinary_smoke \
  bash scripts/train_pi05_b1k.sh smoke
```

普通整任务训练不设置 `SUBTASK_PROMPTS`。正式训练将 `smoke` 改为 `full`，并显式设置
`BATCH_SIZE`、`TRAIN_STEPS` 和新的 `EXP_NAME`。

### Joint：预检 → 两步 smoke → 正式训练

```bash
EXP_NAME=pi05_joint_check \
  bash scripts/train_pi05_subtask_b1k.sh check

NUM_GPUS=1 CUDA_DEVICES=0 BATCH_SIZE=1 NUM_WORKERS=0 \
TRAIN_STEPS=2 EXP_NAME=pi05_joint_smoke \
  bash scripts/train_pi05_subtask_b1k.sh smoke

NUM_GPUS=1 CUDA_DEVICES=0 BATCH_SIZE=1 \
SUBTASK_ACTION_CONDITIONING=subtask_only \
TRAIN_STEPS=50000 LR_DECAY_STEPS=50000 EXP_NAME=pi05_joint_full \
  bash scripts/train_pi05_subtask_b1k.sh full
```

示例 batch 用于起步，不代表硬件可用上限。JAX 的 `BATCH_SIZE` 是所有设备合计的全局 batch，
需要能被全局设备数整除。单机多卡设置 `NUM_GPUS`、`CUDA_DEVICES` 和 `FSDP_DEVICES`。
joint 新模式增加高层语言前向，显存和吞吐应在实际训练机器上测量。
`TRAIN_STEPS` 不会自动改变学习率衰减长度；默认 `LR_DECAY_STEPS` 对应配置中的 30000 步，
需要按整个训练周期衰减时，应像示例一样显式设置。

### 预检、保存与恢复

- 段表默认是 `$DATASET_ROOT/meta/subtask_segments.parquet`，可用 `SUBTASK_PROMPTS` 覆盖。
  `[start, end)` 使用 episode 内局部帧号，episode ID 保留原始 LeRobot 编号。
- Joint 默认 `SUBTASK_EMPTY_LABEL_POLICY=drop_episode`，含空监督标签的整个 episode
  从训练名单中剔除；设置 `error` 则停止。报告和最终名单写入
  `openpi/outputs/subtask_preflight/`，原数据与输入名单不修改。
- 归一化统计通常复用已有文件；需要新计算时使用预检后的 episode 子集。
  切换数据子集时核对已有统计，使用独立 `REPO_ID` 或 `ASSETS_BASE_DIR` 管理不同实验的数据资产。
- checkpoint 默认写入 `openpi/outputs/checkpoints/pi05_b1k/<EXP_NAME>/<step>/`，评测要传具体 step 目录。
  `ASSETS_BASE_DIR`、`CHECKPOINT_BASE_DIR` 可覆盖默认路径。
- 当前 JAX 配置默认使用 BF16 模型计算、AdamW 和 `ema_decay=0.99`；并非全 BF16 参数/优化器状态。
  开启 EMA 时，checkpoint 中供推理使用的 `params/` 保存 EMA 权重。
- `INIT_PARAMS` 用于新实验的权重初始化；`RESUME=1` 恢复相同实验的训练状态。
  `RESUME` 与 `OVERWRITE` 互斥。普通入口默认允许覆盖同名实验，建议每次使用新的 `EXP_NAME`。

新 joint checkpoint 保存 `assets/subtask_conditioning.json`，记录动作条件模式。
服务端读取该文件；缺少此文件的旧 joint checkpoint 默认 `task_and_subtask`，动作仍可见完整任务。
旧 joint 继续训练需设置 `SUBTASK_ACTION_CONDITIONING=task_and_subtask`。
切换到新模式应使用旧权重启动新实验，不能用恢复优化器状态或切换评测参数代替重新训练。

## 4. 推理与闭环评测

### Joint 自主生成 subtask

```bash
PI05_CKPT=/path/to/checkpoint/step \
PI05_REPO_ID=behavior-1k/2026-challenge-demos \
SUBTASK_PREDICTION=1 \
INSTANCE_INDICES=0 NUM_ROLLOUTS=1 \
OUTPUT_ROOT=/path/to/results/joint_generated \
  bash scripts/eval_b1k.sh carrying_in_groceries
```

`PI05_REPO_ID` 必须匹配 checkpoint 的 `assets/<REPO_ID>/norm_stats.json`。
自主生成时不设置 `EXTERNAL_SUBTASK`、`EXTERNAL_SUBTASK_SELECTOR`。
普通 π0.5 使用同一入口，设置 `SUBTASK_PREDICTION=0`。

| 参数 | 含义与默认值 |
| --- | --- |
| 模型 `action_horizon` | BEHAVIOR 配置默认预测 32 步动作 |
| `EXECUTE_HORIZON` | 闭环每次执行多少步后请求新动作，默认 16 |
| `SUBTASK_INTERVAL` | 每几次动作块请求重新生成技能；直接评测入口未设置时，新 joint 默认 2，旧 joint 默认 1 |
| `SUBTASK_TEMPERATURE` | subtask 生成温度，默认 0（greedy） |
| `MAX_TOKEN_LEN` / `MAX_SUBTASK_LEN` | joint 服务默认 512 / 64，应与训练保持一致 |

例如 interval=2、execute horizon=16 时，通常每 16 步更新动作、每 32 步更新技能。
复用技能期间仍读取最新图像和状态；这不是历史记忆，也不是完成检测。
指令变化和 episode reset 会清理相关缓存。

### Joint + 外部 selector 诊断

设置 `SUBTASK_PREDICTION=1 EXTERNAL_SUBTASK_SELECTOR=state_machine_v1`，运行同一评测入口。
当前 selector 仅支持 `carrying_in_groceries` 和 `make_rose_centerpieces`，向 joint 提供外部技能。
结果标记为 `oracle=true`、`privileged=true`、`official_benchmark=false`，用于诊断动作执行能力。
固定技能调试可改用 `EXTERNAL_SUBTASK="move to paper bag"`，不能与 selector 同时设置。

### 仿真与结果

直接评测默认 GPU 0 跑策略、GPU 1 跑仿真，可通过 `SERVER_GPU`、`EVAL_GPU` 配置。
`PI05_ENV` 默认 `openpi/.venv`；`EVAL_ENV` 指向仿真环境，
`OMNIGIBSON_DATA_PATH` 指向资产目录。结果默认位于 `eval_results/pi05/<task>/`。
`INSTANCE_INDICES` 是评测 split 内的实例索引，不是训练 episode ID；不设置时 evaluator 默认运行全部实例。

离线 [eval_openloop.py](scripts/eval_openloop.py) 的 `none` 模式使用完整任务，
`provided` 用于给 joint 提供 GT subtask。
它报告 demonstration action error，不验证自主技能生成或闭环成功率。

## 5. 星云入口

提交前准备本机 `nebula/config.sh`、队列、OSS 挂载、环境包、数据和 checkpoint 路径。
`config.sh` 不提交到 Git；脚本中的既有实验路径和校验值需要与所用数据匹配。

| 用途 | 入口 | 配置 |
| --- | --- | --- |
| 单机 8 卡双任务 joint | [submit.sh](nebula/submit.sh) | 单 worker；由对应 two-tasks worker 准备数据 |
| 多机 / 指定硬件训练 | [submit_pi05_multihost.sh](nebula/submit_pi05_multihost.sh) | `MODE=discover/preflight/smoke/full`、`DATASET_PROFILE` |
| 闭环评测 | [submit_pi05_eval.sh](nebula/submit_pi05_eval.sh) | `EVAL_PROFILE=smoke/speed/two_tasks/reduced/full` |

训练 profiles 包括 `two_tasks`、`two_tasks_joint_h200` 和 `full100`，均使用 joint。
H200 profiles 有固定设备数、batch 和 epoch / 数据校验，
这些是提交配置，不代表新版 joint 已测得的显存上限。

```bash
# 默认 two_tasks 配置正式训练；该 profile 的 TRAIN_STEPS 默认仍是 3，需显式设置
DATASET_PROFILE=two_tasks MODE=full TRAIN_STEPS=50000 SAVE_INTERVAL=5000 \
  bash nebula/submit_pi05_multihost.sh

# 单实例 joint 闭环 smoke；默认限制 100 步，用于检查链路
EVAL_PROFILE=smoke PI05_MODEL_FORMAT=jax SUBTASK_PREDICTION=1 \
CHECKPOINT_SOURCE=/path/to/checkpoint/step PI05_REPO_ID=训练时使用的REPO_ID \
  bash nebula/submit_pi05_eval.sh
```

多机入口已支持 `MODE=full RESUME=1`，需要复用已有 checkpoint 对应的 `OUTPUT_ROOT` 和
`EXP_NAME`，并保持训练模式与相关配置一致；新实验要求输出目录不存在。

评测 worker 对 `smoke/reduced/full` 要求至少 2 张 L20，`speed/two_tasks` 使用 4 个 worker
及至少 4 张 L20；`full` 还要求 `CONFIRM_FULL=1`。`two_tasks` 默认运行
2 个任务 × 20 个实例 × 3 次 rollout = 120 条结果；启用 selector 时它们是诊断结果。
星云 joint 评测 worker 同样根据 checkpoint 条件模式选择刷新间隔，并记录到运行 metadata。

## 6. 理解代码的阅读路径

按“一条训练样本如何变成 loss，再如何变成机器人动作”的顺序阅读：

| 顺序 | 文件与函数 | 需要理解的内容 |
| --- | --- | --- |
| 1. 启动 | [train_pi05_b1k.sh](scripts/train_pi05_b1k.sh) | 环境变量如何选择模式，组织预检、norm stats 和 Python 参数 |
| 2. 配置 | [config.py](openpi/src/openpi/training/config.py)：`pi05_b1k`、`LeRobotB1KDataConfig`；[Pi0Config](openpi/src/openpi/models/pi0_config.py) | 数据配置、动作块长度、文本预算、EMA、模型开关 |
| 3. 样本 | [data_loader.py](openpi/src/openpi/training/data_loader.py)：`create_b1k_dataset()`、`transform_dataset()`；[subtask_data.py](openpi/src/openpi/training/subtask_data.py) | 当前帧如何匹配标签，episode 子集与 frame stride 如何确定采样起点 |
| 4. 输入 | [b1k_policy.py](openpi/src/openpi/policies/b1k_policy.py)；[transforms.py](openpi/src/openpi/transforms.py)；[tokenizer.py](openpi/src/openpi/models/tokenizer.py) | 图像/状态/动作字段、归一化、GT 标签、EOS 和高低层 token 序列 |
| 5. 模型 | [pi0.py](openpi/src/openpi/models/pi0.py)：`embed_prefix()`、`embed_suffix()`、`compute_loss_with_metrics()` | π0.5 也在此文件；先看动作 flow matching，再看 `factorized` 分支和文本 CE |
| 6. 更新与保存 | [train_b1k.py](openpi/scripts/b1k/train_b1k.py)：`init_train_state()`、`train_step()`、`main()`；[checkpoints.py](openpi/src/openpi/training/checkpoints.py) | 梯度、优化器、EMA、训练状态与推理权重的区别 |
| 7. 策略 | [serve_b1k.py](openpi/scripts/b1k/serve_b1k.py)；[policy_config.py](openpi/src/openpi/policies/policy_config.py)；[subtask_policy.py](openpi/src/openpi/policies/subtask_policy.py)：`infer()` | checkpoint 模式识别，生成 / 缓存 / 外部 subtask，动作采样 |
| 8. 闭环 | [eval_b1k_wrapper.py](openpi/src/openpi/shared/eval_b1k_wrapper.py)：`act_receding_horizon()`、`act()`、`reset()` | 执行动作缓存、重新观察、技能切换和 episode 重置 |

先跟踪一个具体例子：当前帧对应 `move to radio`，看它如何成为模型输入与动作监督，
以及闭环如何更新到 `press radio`。跑通概念后，再读 [gemma.py](openpi/src/openpi/models/gemma.py)、
[siglip.py](openpi/src/openpi/models/siglip.py) 和 [sharding.py](openpi/src/openpi/training/sharding.py)。

小型可执行例子见 [subtask_data_test.py](openpi/src/openpi/training/subtask_data_test.py)、
[subtask_test.py](openpi/src/openpi/models/subtask_test.py) 和
[subtask_policy_test.py](openpi/src/openpi/policies/subtask_policy_test.py)。
这些测试验证段表边界、损失与输入隔离、生成和缓存行为；不等同于完整数据训练或闭环成功率验证。

## 7. 目录与辅助工具

底层细节见 [B1K 数据流与配置](openpi/docs/b1k.md)、
[归一化统计](openpi/docs/norm_stats.md)和 [B1K 远程推理协议](openpi/docs/remote_inference.md)。

```text
openpi/       JAX 模型、训练器、策略服务及共享数据依赖
scripts/      BEHAVIOR 数据处理和训练评测入口
nebula/       星云提交、worker 和评测汇总
docs/         训练与推理细节
BEHAVIOR-1K/  外部仿真器源码与资产（不提交到本仓库）
```

- [bootstrap.sh](scripts/bootstrap.sh)：按 `tools/envs/dataset/models` 阶段准备训练机器。
- [stage_dataset_local.sh](scripts/stage_dataset_local.sh)：将数据暂存至本地磁盘。
- [gen_subtask_prompts.py](scripts/gen_subtask_prompts.py)：从标注生成段表。
- [split_subtask_episodes.py](openpi/scripts/b1k/split_subtask_episodes.py)：按任务实例分组划分 train/val，防止同一初始场景泄漏。
- [pack_envs.sh](scripts/pack_envs.sh)：打包训练和评测环境。

数据说明：[数据统计](DATASET_STATS.md)、[Subtask 标注](SUBTASK_ANNOTATIONS.md)、
[任务内 Subtask 变化](SUBTASK_VARIABILITY_BY_TASK.md)、[完整任务指令](FULL_TASK_INSTRUCTIONS.md)。
实验产物写入已忽略的输出目录；共享模型、配置和客户端仍保留 OpenPI 兼容性。
