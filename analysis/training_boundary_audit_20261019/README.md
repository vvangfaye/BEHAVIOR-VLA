# 训练窗口边界审计

本目录最初只分析已有训练索引；后续增加了固定原始观测的离线诊断。索引计数只说明合法训练支持集的性质，不是模型误差、实际训练曝光频率或闭环因果结论。

截至 2026-10-20，已按用户批准上传 54,651,349 字节的观测缓存与代码，并完成唯一作业 `b1k-boundary-probe-20261019-pilot01`（UID `7f68ca4e90e047adbe7975dd03049fac`）：1 张 Spot 5090，实际运行 1,542 秒（约 26 分钟），4 个既有 checkpoint 各评估 36 个相同观测，无训练或仿真。四份结果已下载，SHA-256 与 worker 回执一致，逐样本输入、目标、动作噪声及归一化配对校验通过。回执见 `pilot_results_receipt.json`，分析规则在读取结果前保存于 `pilot_analysis_plan.json`。

**本次 pilot 未支持一致的首动作边界退化。** 前 16 步的近边界全维 MAE 增加约 2–3%，但动作组方向不一致，正的边界交互集中在同两个样本。事后排除这两个样本后，三个模型的交互方向均反转；这只是集中度分析，不能据此删除正式结果。三任务、九边界、单次噪声及真实子任务条件不足以归因闭环收益变化，不建议据此扩大边界假设。数值与限制见 `pilot_conclusion.json`，动作组完整结果见 `pilot_interactions_v1/summary.json`。全维 MAE 混合了不同物理单位，仅作辅助概览。

当前判断见 `research_status.json`：边界过滤尚未被证实是闭环问题的原因，也不是推荐的论文方法。已核验的 FineART 公开实现默认允许动作窗口跨子任务标签，不能把本项目的过滤行为泛化为其缺陷。

## 复现

在项目根目录运行（仅 Python 标准库）：

```bash
python3 -m unittest discover -s analysis/training_boundary_audit_20261019 -p 'test_*.py' -v
python3 analysis/training_boundary_audit_20261019/audit.py \
  --cohort quality_20261016=analysis/retraining_20261016_quality_taskbalanced_v1/cpu/bundle/inputs/compiled/indexes \
  --cohort recovered_20261013=analysis/retraining_20261013/macpro_leading_fill_v1/recovered_index.json \
  --output analysis/training_boundary_audit_20261019/report.json
```

`report.json` 保留每个输入文件的 SHA-256、验证后的索引语义 fingerprint，按 cohort、source、task 统计。两个 cohort 表示不同训练准备方案，不能相加为一个训练集。

## 本次真实索引结果

| cohort / source | 回合 | 保留区间 | 保留区间帧数 | 无法作为观测起点的帧数 | 占比 | 从未进入前 16 target 槽位的动作帧数 |
|---|---:|---:|---:|---:|---:|---:|
| quality_20261016 / replay | 15,949 | 282,142 | 161,068,145 | 8,746,402 | 5.43% | 4,514,272 |
| quality_20261016 / full_episode | 270 | 5,154 | 2,506,369 | 159,774 | 6.37% | 82,464 |
| recovered_20261013 / recovered | 3,949 | 98,841 | 48,757,662 | 3,064,071 | 6.28% | 1,581,456 |

在 replay 中，265,494 个区间末端可由索引确认紧邻另一个不同标签；699 个末端后存在索引缺口，15,948 个是回合结束，另有 1 个未索引尾段。不能把上表所有末端都表述为已标注的语义交接。

## 定义与实现依据

- 冻结索引中的区间是 `[start,end)`，模型预测 H=32 个动作；当前闭环配置执行其前 K=16 个动作。
- `openpi/training/recovery_index.py:59` 只允许 `start <= anchor <= end-H`，短于 H 的区间在构建索引时被删除（同文件第 176 行）。
- `openpi/training/subtask_mixture.py:348` 在这个合法起点集合内采样；后续质量采样器 `analysis/retraining_20261016_quality_taskbalanced_v1/task_balanced_windows.py` 仍使用同一限制。
- `openpi/training/subtask_mixture_data.py:167` 要求完整 H 步真实动作，不接受 padded target。
- `openpi/models/pi0.py:290` 对全部 H 步做动作目标损失；这是源码默认路径，具体运行若用了 frozen patch 则必须另核对应 patch。
- `analysis/repeatability_1688977_seed0_30_20261019/protocol.json` 的 `executed_action_horizon` 为 16。

相对于每个保留区间：

1. 最后 H−1=31 帧从未作为训练观测起点。
2. 最后 H−K=16 帧的动作只可能出现在远期 target 槽位，未出现在任何合法窗口的前 K 槽位。
3. 这些动作并未全部丢失；每个保留区间的所有动作都至少出现在一个 H 步目标里。
4. 区间末尾不等于物理事件或任务完成；报告分别统计相邻异标签、相邻同标签、索引缺口、未索引尾段、回合结束。

## 已有数据的限制

冻结索引已经剔除了短段。replay 源统计保留了被剔除的 774 段、18,626 帧；recovered 源统计保留了 733 段、17,148 帧，但没有其逐段位置、长度和任务。因此本报告无法计算这些短段中原本能提供多少 K 步窗口，也不能按任务恢复它们。报告中放宽 H→K 得到的可新增起点数，只针对已保留区间，是原始完整数据上的下界。

同时，原始非 mixture loader 会读取每帧并以该帧子任务作为 conditioning，不强制目标 H 步都属于同一标签（`openpi/training/data_loader.py:214–240`）。这与后续 mixture 管线有结构差别，但必须绑定具体 checkpoint 的训练包，不能据当前源码倒推所有历史 checkpoint。

## 审稿人视角

恢复窗口末端、mask 不可信 target、只训练真正执行的 prefix，本身属于常见 padding / masked behavior cloning / 可变窗口设计，不能据此宣称算法创新。审计中的计数也不证明这些边界比普通状态更重要。

这里可检验的实证问题是：**为了得到语义一致、完整长度的动作窗口而过滤数据，是否系统性删去了长任务中控制切换所需的观测条件，导致局部动作目标改善而完整任务收益下降？**

要回答它，应在相同原始数据、初始化、优化器、任务曝光和训练预算下至少比较：

1. 现有完整窗口过滤。
2. 原始跨标签窗口训练（起点标签保持原样）。
3. 恢复末段观测，按有效动作 mask 训练。
4. 保持同样新增样本数但在区间内部增采的控制，以及只改 H 或 K 的控制。

先按离边界距离测旧/新模型的真实动作误差与条件动作敏感性，再做统一自主闭环评测。增加样本数、边界重加权、改变执行频率和删掉远期损失必须分别控制。只有当现象跨任务、模型及独立数据成立，且能排除这些简单解释，才值得讨论更一般的训练原则。

即使所有实验都成立，最先成立的贡献也是一个可复现实证失效机制；如果标准 masked BC 即可修复，就应如实陈述，不能把修复包装成新的算法范式。

## 已冻结的离线边界探针

`probe_manifest.py` 只读当前 quality 的 `0000_replay.json`，从同一回合相邻且标签不同的区间选择边界。左段至少 64 帧，右段至少 32 帧；每任务最多 3 个边界，来自不同的 `(task_instance_id,episode_index)`。seed 0 加语义身份的 SHA-256 决定顺序。它不读取视频、不跑推理、不增加训练监督。

```bash
python3 analysis/training_boundary_audit_20261019/probe_manifest.py \
  --index analysis/retraining_20261016_quality_taskbalanced_v1/cpu/bundle/inputs/compiled/indexes/0000_replay.json \
  --output-dir /tmp/b1k-boundary-probe-new \
  --seed 0 --max-per-task 3 --action-noise-seed 0
```

CLI 拒绝覆盖已有产物。正式结果位于本目录：`probe_manifest.json`、`selected_episodes.json`、兼容现有 `--episodes-file` 的 `selected_episodes.txt`，以及 `probe_sampling_statistics.json`。实际选出 100 个任务、300 个边界、300 个回合、1,200 个观测；四个距离桶 `64/32/16/1` 各 300 条，`valid_action_steps` 分别是 `32/32/16/1`。

runner 可用 `scripts/eval_openloop.py --sample-keys-file .../probe_manifest.json --episodes-file .../selected_episodes.txt --subtask-conditioning provided --action-noise-seed 0`。其余 dataset / checkpoint / label 路径必须按实际部署提供。新 runner 使用 `b1k-openloop-noise-v1:{seed}:{episode}:{frame}:{H}:{D}` 派生相同初始 action noise，并输出逐样本 `action_noise_sha256`。正式比较时必须检查两个 checkpoint 的样本身份、观测预处理和这些哈希完全一致；共享初始 noise 不等于控制了所有 runtime 随机性。完整 100 任务探针尚未运行；上述获批 pilot 已对其中选定的 3 个任务执行真实模型推理。

这些探针来自训练索引，因此是 training-seen 的离线诊断，不能作为独立泛化集或闭环成功证据。选取了 32 步真实未来不表示跨标签动作可作为当前标签的有效监督；现有 evaluator 仍应按 `valid_action_steps` 屏蔽跨边界目标。
