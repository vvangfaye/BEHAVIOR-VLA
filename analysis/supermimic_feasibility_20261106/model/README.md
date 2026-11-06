# B1K 小型动作条件世界模型探针

本目录只进行离线拟合和预测评估，不启动仿真、不更新 VLA 策略、不报告真实成功率。

```sh
analysis/rlinf_adaptation_20261020/venv/bin/python \
  analysis/supermimic_feasibility_20261106/model/train_world_model.py \
  --data /absolute/path/transitions.npz --output /absolute/path/results \
  --epochs 30 --threads 8 --seed 7
```

输入直接兼容相邻 `data/export_transitions.py` 的 `b1k_action_conditioned_transition_v1`：23 维当前/后续 proprio，16×23 实际执行命令及 mask，576 维 RGB 均值池化描述，目标布尔值及 masks，reward，split，episode_index，chunk_index，instance。已有 split 为训练/验证/测试；不随机重分轨迹或相邻片段，每 epoch 仅在隔离后的训练集合内确定性 shuffle。所有标准化及可选 PCA 仅拟合训练集当前观测。

主实验只用实际执行完整 16 个动作的 chunk。提前终止的部分 chunk 保留在输入表，但从训练、标准化、验证与测试指标中排除，单独报告数量。原因是提前终止后的实际长度带有未来结果信息，不能当作可部署的当前动作条件；full 模型的 mask 因此恒为全 1。完整长度的终态 chunk 仍能训练 reward/goal，缺失的 next observation 仅影响 dynamics/visual loss。

默认比较两个训练模型及两个廉价基线：

- `full`：当前 proprio、visual、goals 及有效性，实际执行动作和 action mask。
- `no_action`：完全相同的当前观察输入，将动作和 action mask 输入置零；与 full 保持同架构、同参数数量及同 seed 的初始权重，检验动作信息是否带来超越观测自相关的预测改善。
- `persistence`：保持当前 proprio、raw visual、goals，reward=0。
- `train_mean_reward`：同样保持当前观察，reward 恒为完整 chunk 训练集均值；不使用验证/测试集均值。另报告三个 split 的零奖励比例、正负奖励数量及 reward 非零子集 MSE，避免稀疏奖励被低总 MSE 掩盖。

模型始终预测总 reward（含 shaping）。另按记录中的 `task_reward != 0` 选出实际任务事件，报告这些行上的总 reward 预测 MSE及三个 split 的任务奖励零/正/负计数和比例。总 reward 的非零值可能只来自势函数变化，不能代替 task event；没有 `task_reward` 字段的合成旧数据明确标为 unavailable。

`--include-no-visual` 可加上 state/goals/action 模型，`--visual-dim 0` 关闭 RGB 投影，`--ensemble 3` 可拟合三个独立初始化。默认两个小型 192 隐层 MLP，CPU 训练，终态没有 next observation 时仅屏蔽 dynamics/visual 损失，仍使用目标与 reward 标签。下一时刻 proprio/visual 以残差预测。目标 BCE 对发生变化的 entries 加 5 倍权重，两种模型完全一致；奖励使用训练集均值方差归一化。

输出 `results.json` 及 `full.pt`、`no_action.pt`。checkpoint 包含所有训练集变换和网络配置，可通过 `torch.load(..., weights_only=True)` 加载。按最低验证集组合损失选择 epoch，测试集不参与选择。

报告包含三个 split 的 proprio RMSE、标准化 RMSE、raw 576 维及投影 visual RMSE、逐 goal 的变化事件 precision/recall/AP/AUROC、变化目标的正确率和 reward MSE。Goal 名字读自 NPZ 同名 JSON 的 `goal_names`，没有时只使用明确的序号名。1/2/4 chunk open-loop 从真实起点出发，后续只反馈预测状态，重放日志中的动作，不跨 episode/split，不在每步重置真实未来观察。

Open-loop 必须有下一时刻观察，因此排除缺失 terminal next observation 的窗口，并逐 horizon 报告排除计数；完整终态 chunk 的 reward 另包含在 one-step reward 指标中。视觉误差的有效性要求起点及目标描述都存在，与 loss mask 一致。加权 BCE 后的 sigmoid 只作未校准的 goal/event 分数，不能解释为真实任务成功概率。

CPU 吞吐只计小模型预测，不包含 VLA/渲染/物理/IPC，不能与 GPU 仿真速度直接相除。可选 ensemble 的梯度分歧只针对拟合 reward head 的归一化动作导数，不是已验证的 Q 梯度、环境导数或策略改进方向。全体评估仍是观察性 factual action replay；是否可用于真实动作优化需后续独立仿真干预验证。负结果仅约束当前数据、表示、模型和预算。

契约测试：

```sh
analysis/rlinf_adaptation_20261020/venv/bin/python -m pytest \
  analysis/supermimic_feasibility_20261106/model/test_world_model.py -q
```

测试覆盖 train-only PCA/normalization、NaN terminal mask、masked action invariance、终态 reward、跨轨迹窗口边界、split 泄漏拒绝、PR/AUC ties，以及完整 CPU 训练/保存/评估流程。合成数据结果不代表 B1K 实测。
