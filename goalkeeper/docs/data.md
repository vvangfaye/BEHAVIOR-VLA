# 离线数据与推理输入契约

## 训练 episode 输入：`export`

NPZ 使用 `allow_pickle=False`。`T` 个动作连接 `T+1` 个观测，`actions[t]` 把 `obs[t]` 推进到 `obs[t+1]`。

| 字段 | 形状 / 含义 |
|---|---|
| rgb | uint8 `[T+1,3,height,width,3]`；头、左腕、右腕顺序 |
| proprio | float `[T+1,S]`；与真实机器人和主干一致 |
| actions | float `[T,A]`；执行给环境的动作，不能是未经过后处理的 latent |
| truth | 0/1 `[T+1,N]`；离线训练 oracle 真值，不是输入 |
| visibility | -1/0/1 `[T+1,N]`；未知/不可验证/可验证，不是输入 |
| atom_keys | Unicode `[N]`；必须等于 `compile_bddl(text).atoms` 的 `.key` 顺序 |

也支持分别提供 `rgb_head/rgb_left/rgb_right` 三个 `[T+1,H,W,3]` 数组，以保留不同相机分辨率。公开 LeRobot schema 的头相机为 720×720、腕相机 480×480，state 为 61 维、action 为 23 维；不要强行将原始三路图像堆成同尺寸数组。参见 [公开 info.json](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/blob/main/meta/info.json)。实际运行可使用官方低分辨率 wrapper，训练时需匹配其图像处理。

可见性是谓词的可验证性，不等于“看见一个像素”。`inside(cup,cabinet)` 是否可验证取决于对象、容器、遮挡和关系，直接把分割像素计数当作完整关系可见性会产生错误监督。若暂时不能可信标注，填 -1；相应观测损失被屏蔽，不能假装观测头已学好。

训练可用 privileged state/segmentation 生成标签，但采集脚本必须和推理服务分进程运行，不能给服务塞一个可访问 env 的对象。当前服务模块不导入训练记录读取器。输入白名单是防止误接数据的工程边界，不是抵抗恶意代码的沙箱。

## 训练记录输入：`train`

每条 manifest JSONL 含 `id, group_id, source, record, bddl, feature_id, action_spec`。路径相对 manifest 所在目录。`source` 只允许 `training_demo`、`training_rollout`、`synthetic_test`。来源字段需要用户从真实采集日志审核；一个字符串不能自动证明实例合法。

`group_id` 表示可共享物理初始状态的最大泄漏单元：至少包括 scene、task、training instance。来自同一初始状态的全部 demos、裁剪窗口、重试与 counterfactual branches 必须使用同组；如相邻/重复初始状态仍强相关，应进一步合并。绝不能给每帧创建独立组来获得窄置信区间。

| 字段 | 形状 |
|---|---|
| visual | float `[history,cameras,patches,visual_dim]` |
| proprio | float `[history,S]` |
| queries | float `[N,text_dim]` |
| actions | float `[K,H,A]` |
| truth / visibility | `[history,N]` |
| future_truth | 0/1 `[K,N]` |
| atom_keys | Unicode `[N]` |

history 的采样间隔是 H 个动作，与推理时决策间隔相同。当前 mini-batch 支持 N、K、DNF 分支数不同，不支持混用不同 history、H、相机数、视觉维数或机器人维数。模型按短窗口训练、按整 episode 递归推理；长时记忆分布偏移需要实测，不能只看离线窗口损失。

K>1 时所有 action candidates 必须从**同一个决策状态**实际执行，并分别记录 t+H 的后果。分支需要恢复物理状态、机器人控制器、抓取约束和随机源，验证恢复的一致性；把 simulator 的 `dump_state/load_state` 名字写进代码不等于这些条件自动满足。因此当前未提供未经环境验证的官方分支采集器。

`feature_id` 包含冻结模型 ID、40 位 commit 和特征处理版本；当前 SigLIP adapter 需在真实权重上联调。SigLIP 文本长度有限，长 init 会被截断；初版把目标查询放在前面，仍需检查对象消歧上下文是否丢失。`action_spec` 应包含机器人配置、控制顺序、单位、规范化与后处理的版本/hash。小头自行拟合的均值方差不能替换 Comet 的 normalization。

每组按稳定 hash 分配 train/dev/calibration，大致 80/10/10。校准器每个 bin 内先计算各初始状态组的平均真值，再对组等权求平均，并按组数计算区间。其对象是组平衡的 bin 可靠性；不是单状态真值概率的统计保证。只在 calibration 组生成阈值/区间，不能在 dev 反复选定同一个校准集合来获得声称的有限样本保证。

## 当前实现边界

BDDL 的基数及配对量词 `forn/forpairs/fornpairs`、类型继承等未实现。编译器拒绝这些任务，不静默替换为较简单目标；`audit` 给出覆盖率。需要在正式 100 任务定义上确认它们是否出现，并补官方语义的测试。目标为空、DNF 组合爆炸也不能按“可预测目标任务”训练。

官方 2026 robot 配置可能与 2025 Comet 不一致。先跑官方 baseline、记录实际字段/shape，并核验 action_spec。不要为了“接口跑通”截断状态或填零扩展动作后就宣称迁移成功。任务无有效筛选器时，应使用经过验证的主干原始服务，并在全量结果中明确记录，不能从总分中删除。
