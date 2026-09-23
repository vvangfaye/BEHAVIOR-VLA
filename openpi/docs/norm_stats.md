# BEHAVIOR π0.5 的归一化统计

训练和推理必须使用相同的状态 / 动作定义与 normalization stats。
当前 B1K 的 π0.5 配置使用 mean/std 归一化（`use_quantile_norm=False`），
统计由本仓库的数据变换计算，不直接使用数据集 `meta/stats.json` 代替。

## 统计的是什么

数据经 [LeRobotB1KDataConfig](../src/openpi/training/config.py) 重排，再由
[B1KInputs](../src/openpi/policies/b1k_policy.py)提取 R1Pro 状态和动作。
默认开启 `extra_delta_transform`，把躯干和双臂目标转成相对当前状态的 delta。
[compute_norm_stats.py](../scripts/compute_norm_stats.py)在这些变换之后统计 `state`、`actions`。

训练随后执行 Normalize 和模型输入变换；推理输出执行 Unnormalize 和对应的绝对动作恢复。
因此更换机器人映射、delta 规则或数据分布后，不能仅凭维度相同就继续复用旧统计。
具体索引见 [B1K 集成](b1k.md)。客户端传原始观测，不应提前重复做这套归一化。

## 优先让训练入口管理

[公共训练脚本](../../scripts/train_pi05_b1k.sh)查找：

```text
<ASSETS_BASE_DIR>/pi05_b1k/<REPO_ID>/norm_stats.json
```

默认 `ASSETS_BASE_DIR` 是 `openpi/outputs/assets`。文件缺失时脚本计算统计，存在时通常复用。
不同实验应按数据来源设置独立的 `REPO_ID` 或 `ASSETS_BASE_DIR`。

| 训练模式 | 新计算统计时的样本范围 | 已有文件的处理 |
| --- | --- | --- |
| 普通 π0.5 | 指定训练 episode；未指定则全数据 | 直接复用 |
| Joint | 空标签预检后的训练 episode | 直接复用，不自动核对段表哈希 |

`NORM_MAX_FRAMES` 可限制统计样本量，属于近似统计；使用时记录该配置。
验证 loader 使用训练集统计，不应为验证集重新拟合。传入预检前的名单或复用包含验证集的
旧 assets，会破坏这项隔离；本地 joint 的文件存在检查不会自动发现这类问题。

## 手动计算普通训练的统计

以下命令从仓库根目录运行，仅演示原始 episodes 0、1；应与实际训练子集一致：

```bash
cd openpi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_b1k \
  --dataset-root /path/to/2026-challenge-demos \
  --repo-id behavior-1k/manual_smoke \
  --episodes-index 0 1 \
  --assets-base-dir ./outputs/assets \
  --batch-size 1 --num-workers 0
```

## checkpoint 与推理

[checkpoint 保存逻辑](../src/openpi/training/checkpoints.py)将使用的统计带入模型资产：

```text
<checkpoint-step>/assets/<REPO_ID>/norm_stats.json
```

闭环设置 `PI05_REPO_ID=<REPO_ID>`；直接启动服务传 `--repo-id`。这两个标识应匹配保存目录，
不根据本次评测任务名猜测。例如双任务共享一份训练统计时，两项任务使用同一个 repo ID。

`assets/subtask_conditioning.json` 记录的是 joint 的语言条件模式，
与 `assets/<REPO_ID>/norm_stats.json` 用途不同，不能相互代替。
完整启动示例见[仓库 README](../../README.md)和 [subtask 指南](../../docs/subtask_training.md)。
