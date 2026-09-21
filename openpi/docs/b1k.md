# BEHAVIOR 2026 的 π0.5 / JAX 集成

本文说明本仓库的数据流和底层代码入口。环境安装、普通训练、联合训练与闭环命令见
[仓库 README](../../README.md)，标签和边界处理见
[subtask 指南](../../docs/subtask_training.md)。以下文件路径均相对 `openpi/`。

## 从演示数据到动作

```text
BEHAVIOR LeRobot 数据 + 原始 episode 子集 + 可选 subtask 段表
  → 预检 / 筛选 episode
  → create_b1k_dataset：按帧取样，匹配 prompt / subtask / action mask
  → RepackTransform → B1KInputs → MappedDeltaActions → Normalize
  → tokenizer / padding → Observation + action targets
  → Pi0.compute_loss_with_metrics → train_step → checkpoint

checkpoint params + assets
  → create_trained_policy → Policy / SubtaskPolicy
  → B1KPolicyWrapper → 每个控制步的 23 维动作
```

`MappedDeltaActions` 由 `LeRobotB1KDataConfig.extra_delta_transform` 控制，
当前 B1K 默认开启。推理输出按反向流程反归一化、恢复绝对关节目标。

## 数据和机器人映射

数据根目录需要 `meta/info.json`、任务和 episode 元数据、动作 parquet，以及所选 episode
的三路 RGB 视频。`DATASET_ROOT` 是本地数据位置；`REPO_ID` 是保存和查找 norm stats
使用的标识，不必等于任务名。选择子集时始终保留原始 `episode_index`。

[R1Pro 配置](../src/openpi/configs/robots/b1k.py)注册为 `b1k/R1Pro`：

| 数据字段 | 用途 |
| --- | --- |
| `observation.rgb.zed_link_camera_0` | 头部相机 |
| `observation.rgb.left_realsense_link_camera_0` | 左腕相机 |
| `observation.rgb.right_realsense_link_camera_0` | 右腕相机 |
| `observation.state` | 原始 proprio 向量，按机器人配置提取 |
| `action` | 23 维控制动作 |

动作顺序为 base 3 维、torso 4 维、左臂 7 维、左夹爪 1 维、右臂 7 维、右夹爪 1 维。
[B1KInputs](../src/openpi/policies/b1k_policy.py)从原始 proprio 提取对应状态，并将每个夹爪
两根手指的位置求和为一个宽度值。原始 proprio 不能直接替换成已经提取的 23 维状态，
否则会再次按原始索引取值。

躯干和双臂默认使用映射后的 delta action；base 和夹爪保持配置定义的控制量。
不要套用 ALOHA / DROID 的动作索引或 normalization stats。

训练任务文本来自数据集任务元数据；服务默认任务文本来自
[任务注册表](../src/openpi/configs/tasks/__init__.py)，也可由观测中的 `prompt` 覆盖。
Joint 高层始终读取完整任务，strict / executor 以当前 subtask 替换语言 prompt。

## 配置和训练入口

[训练配置](../src/openpi/training/config.py)中的 `pi05_b1k` 使用
`Pi0Config(pi05=True, action_horizon=32)`、R1Pro 和 `tolerance_s=5e-3` 的视频时间戳容差。
配置文件里的默认数据路径属于原示例，运行时应通过启动脚本覆盖。

优先使用仓库根目录的 `scripts/train_pi05_b1k.sh` 及 joint / executor 包装脚本。
它们依次组织预检、最终 episode 名单、归一化统计和 Python trainer。
本目录旧 `scripts/b1k/train_b1k.sh` / `.sbatch.sh` 保留，但不是这套 subtask 工作流的入口。

如需调试底层 trainer，可从仓库根目录执行以下普通训练示例。先按
[归一化指南](norm_stats.md)为同一配置与子集准备 assets：

```bash
cd openpi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/b1k/train_b1k.py pi05_b1k \
  --exp-name pi05_manual_smoke \
  --data.dataset-root /path/to/2026-challenge-demos \
  --data.repo-id behavior-1k/manual_smoke \
  --data.episodes-index 0 1 \
  --weight-loader.params-path /path/to/pi05_base/params \
  --batch-size 1 --num-workers 0 --num-train-steps 2 \
  --assets-base-dir ./outputs/assets --no-wandb-enabled
```

原始 episode 0、1 只用于示例；应选择机器上已下载的 episode。
直接调用 trainer 不执行启动脚本的全表预检或自动过滤，也不会自动计算 norm stats。
Subtask 模式应先通过脚本预检，并把最终名单传为 `--data.episodes-index`。

| 模式 | 模型与数据配置的关键差别 |
| --- | --- |
| 普通 | `train_subtask_prediction=False`、`train_subtask_executor=False`，完整任务 prompt |
| Joint | `train_subtask_prediction=True`；新模式显式设置 `subtask_action_conditioning=subtask_only` |
| Strict | 两个模型训练开关均为 False；数据 `strict_subtask_action_chunks=True`，subtask prompt |
| Pure executor | `train_subtask_executor=True`，segment 采样与 `action_loss_mask` |

三种 subtask 模式互斥，要求 `subtask_whole_task_prob=0`。Joint 的低层 subtask-only
使用独立语言序列，不能用 ordinary / strict 的 tokenizer 配置代替。

[train_b1k.py](../scripts/b1k/train_b1k.py)通过
[data_loader.py](../src/openpi/training/data_loader.py)的 `create_b1k_data_loader()` 构造训练数据，
使用 Torch DataLoader 取样，再转成 JAX 数组。`init_train_state()` 初始化模型和优化器，
`train_step()` 计算 loss、梯度并更新 EMA，`main()` 管理验证与 checkpoint。

`BATCH_SIZE` 是全局 batch，需要能被全局设备数整除。BF16 是模型计算配置，
不意味着 trainable params、AdamW 两份动量和 EMA 副本都变成 BF16。
训练步数与 LR 衰减长度分别配置；增加步数时核对 `LR_DECAY_STEPS`。

## 保存、恢复和服务

checkpoint 默认位于 `outputs/checkpoints/pi05_b1k/<EXP_NAME>/<step>/`。
`params/` 用于推理，启用 EMA 时保存平滑后的参数；训练状态用于 `RESUME=1` 续训。
`INIT_PARAMS` 只初始化新实验的权重。两者不能互换。

Joint 另存 `assets/subtask_conditioning.json`，服务读取其中的动作条件模式；旧 checkpoint
缺少该文件时默认旧 `task_and_subtask`。切换条件模式需要新实验训练，不能仅切换服务参数。

服务入口是 [serve_b1k.py](../scripts/b1k/serve_b1k.py)，使用 `--policy.config` 和
`--policy.dir`，没有 `policy:checkpoint` 子命令。B1K 的 websocket 返回单步 `action`；
内部 policy 返回动作块 `actions`。服务命令和协议见[远程推理](remote_inference.md)。

正式比较模型时，训练与验证按任务实例分组切分，复用训练集统计；先做数据预检和小规模 smoke，
再评估闭环成功率。预检通过、训练 loss 下降、离线动作误差变小分别验证不同环节，均不能代替闭环结果。
