# LIBERO-Long 可行性检查（libero_feasibility_20261103）

**授权**：用户 2026-11-03："先从LIBERO-Long开始做可行性检查吧"。这次检查只用本机的空闲 GPU（2 × RTX 6000 Ada，48 GB），不提交集群作业。

## 要回答的问题

1. **环境：** LIBERO（robosuite/MuJoCo）能否在本机和集群节点上用 EGL 渲染。本机已确认有 libEGL_nvidia 和 glvnd。
2. **基线：** π0.5 少量示范起点（`RLinf/RLinf-Pi05-LIBERO-SFT`，7.5 GB）在 LIBERO-10 上的成功率，按任务列出。
3. **吞吐：** 每局耗时、每小时能跑多少局、显存占用。用这些数据推算一次 PPO 迭代的成本。参照 RLinf 配置 `libero_10_ppo_openpi_pi05.yaml`：64 个环境、每次迭代采 8 遍、学习率 5e-6、每批训 4 遍。
4. **迁移到集群：** 5090 节点和 H100 节点能否渲染。H100 节点上缺 libEGL_nvidia，可能要退回 CPU 渲染。

## 代码与环境

- `RLinf/`：从 GitHub 克隆，固定在 commit `91a302ec`，与 `rlinf_adaptation_20261020/upstream` 一致，LIBERO 环境代码逐字相同。BEHAVIOR 实验冻结的那份 upstream 不做任何改动。
- 环境由 RLinf 自带的 `requirements/install.sh embodied --model openpi --env libero --no-root` 创建，全部装在 `RLinf/.venv` 里，不用 sudo，不改系统。它会做这些事：
  1. `uv venv`；
  2. `uv sync --extra embodied`；
  3. 安装 `rlinf-libero` 并下载 LIBERO 资源；
  4. 安装 `rlinf-openpi==0.1.1`；
  5. 安装预编译的 flash-attn。
  这一步执行外部代码，自动权限策略拦下了，**需要用户确认**。
- `models/RLinf-Pi05-LIBERO-SFT/`：直接用 curl 从 HuggingFace 下载，safetensors 格式，只是数据。

## 状态

- 2026-11-03：RLinf 克隆完成，权重在下载中。安装脚本等待用户确认。
- 2026-11-03 21:3x：用户确认（"允许运行这个安装脚本"）。
- 第一次安装失败（exit 2）：安装脚本里有一处未加引号的 `$SCRIPT_DIR`，路径中的空格（"Behavior-1k Challenge"）把它截断了。
- 处理方式：删除装了一半的 `.venv`，把克隆移到无空格路径 `/mnt/data1/faye/code/libero_rlinf_20261103/RLinf`，本目录下的 `RLinf` 改为指向它的软链接，然后在那里重跑同一条安装命令。
- 权重已下载并校验：`model.safetensors` 7.47 GB，812 个张量，3.6B 参数。

## 结果（2026-11-03）

| 检查项 | LIBERO-10 | CALVIN ABC→D |
|---|---|---|
| 环境 | `RLinf/.venv`（rlinf-libero，MuJoCo 3.8.1），EGL 渲染正常 | `RLinf/.venv-calvin`（mees/calvin，PyBullet），不需要下载 CALVIN 数据集 |
| 起点权重 | RLinf-Pi05-LIBERO-SFT（少量示范） | RLinf-Pi05-CALVIN-ABC-D-SFT |
| 冒烟测试 | 10 局，成功 3 局（success_once） | 10 个环境 × 2400 步，avg_len 3.5，rollout 349 s |
| 完整基线 | **500 局 27.8%**（success_at_end 23.8%）。各任务成功数（每任务 50 局）：0:10, 1:25, 2:24, 3:44, 4:4, 5:0, 6:11, 7:14, 8:1, 9:6 | 待做。RLinf 文档：SFT 3.838 → Flow-SDE RL 4.717 |
| 速度（GPU 0 单卡） | 50 个环境 × 10 轮 = 500 局，59.5 分钟，约 500 局/小时；显存 46.5/48 GB | 10 个环境约 69 环境步/秒，单环境约 7 步/秒 |

- **饱和情况：** LIBERO-10 公开最好成绩约 97–98%（X-VLA 97.6% 等）；CALVIN 上 π0.5 + RL 已到 4.72/5。两者都只适合当试验台，比较方法和样本效率，不适合拿来报绝对提升。
- **运维注意：**
  - RLinf 安装脚本不支持带空格的路径，所以 RLinf 放在 `/mnt/data1/faye/code/libero_rlinf_20261103`。
  - 默认配置 `component_placement: all` 会占用所有 GPU。
  - 本机 /tmp 已用 95%，Ray 临时目录要改到 `/mnt/data1/faye/ray_tmp_libero`。

## RL 策略验证（2026-11-03 晚，目标：CVPR）

### 关键发现：RLinf 的具身 PPO 默认就是“轨迹等权”

- 只要训练环境是 `auto_reset: False` 且 `ignore_terminations: False`（LIBERO-10 和 CALVIN 的训练配置都是），RLinf 就会：
  1. 在 actor 端用 `compute_loss_mask(dones)` 算出每条轨迹的有效步数 `loss_mask_sum`；
  2. 在 PPO 的 actor 和 critic 损失里改用 `masked_mean_ratio`，把每一步除以 `loss_mask_sum / max_episode_steps`。
- 结果是每条轨迹的总权重相同，和 BEHAVIOR 上的 balanced_ppo 是同一个机制。差别只有两点：critic 也归一化了；整体尺度不同。
- 合成数据核验（`patches/` 和本节的测试）：两条轨迹分别有 2 个和 6 个有效 chunk。默认设置下两条轨迹总权重都是 12；关掉归一化后每步权重相同，总权重为 0.25 对 0.75。
- 影响：
  1. 正在跑的 `results/ppo_baseline`（RLinf 默认设置）其实就是**轨迹等权**这一臂；
  2. 对照组要显式关掉归一化；
  3. RLinf 公开的 LIBERO / CALVIN RL 结果本身就是用轨迹等权训练出来的，写论文时不能把“轨迹等权”当作新方法，只能当作分析或消融。
- 代码：
  - 之前在优势上乘权重的补丁（`patches/superseded_trajectory_balance_advweights.diff`）会和 RLinf 默认的归一化叠加，变成按 1/L² 加权，已撤回，从未用于任何运行。
  - 新开关 `+algorithm.length_normalization=False`（`patches/length_normalization_switch.diff`）恢复普通的逐步平均（token-mean），actor 和 critic 两个损失都恢复。默认值 True，与 RLinf 原行为完全一致。

### 各臂（同一预算：每次迭代 128 局，PPO 超参数与 RLinf 相同）

| 臂 | 覆盖参数 | 结果目录 |
|---|---|---|
| 轨迹等权（RLinf 默认） | 无 | `results/ppo_baseline` |
| 逐步平均对照 | `+algorithm.length_normalization=False` | `results/ppo_tokenmean` |
| 轨迹等权 + γ=1 | `algorithm.gamma=1.0` | `results/ppo_balanced_gamma1` |

- 64 个环境会让 GPU 0 显存溢出：EGL 每个环境约 0.54 GB，rollout 模型约 12 GB。所以改为 32 个环境 × 4 轮 rollout，每次迭代仍是 128 局（`results/ppo_local_oom_64envs` 是失败的那次）。
- 启动：`bash run_ppo_local_libero.sh <tag> [hydra overrides]`；CALVIN 用 `bash run_ppo_local_calvin.sh <tag> [overrides]`。

### 开放问题：SFT 起点成功率比 RLinf 报告的低

- 我们测的 `RLinf-Pi05-LIBERO-SFT`：LIBERO-10 全部 500 个初始状态，success_once 27.8%。
- RLinf 和 πRL 论文给同一个少量示范 π0.5 报的是 LIBERO-Long 43.9%，PPO 后 93.0%，GRPO 后 77.6%。
- 我们的评测配置和 RLinf 官方的 `evaluations/libero/libero_10_openpi_pi05_eval.yaml` 逐行一致，只改了 GPU 位置、视频开关和权重路径。设置为：noise_level 0.3，task eval，520 步，auto_reset，ignore_terminations。
- 差距的来源还没查清，可能是采样随机性或评测种子，也可能论文用的是别的协议。待 GPU 空出来后，用确定性采样（noise_level 0）再测一次。
- 论文中各臂之间的比较都用同一套协议，这一点不受影响。

### 运行记录

| 时间 | 运行 | 布局 | 结果 |
|---|---|---|---|
| 11-03 23:21 | `ppo_local_hung_nccl_p2p` | 流水线 | 两张 GPU 跨 NUMA，NCCL P2P 卡住 22 分钟。之后加了 `NCCL_P2P_DISABLE=1` |
| 11-03 23:31 | `ppo_local_oom_64envs` | 流水线，64 个环境 | GPU 0 显存溢出 |
| 11-03 23:39 | `ppo_pipeline_balanced_uncached_iter1` | 流水线，32 个环境 × 4 轮，不缓存 | 第 1 次迭代：2248 s；rollout 701 s（其中 predict 507 s），actor 训练 1545 s。训练局 success_once 39/128 = 30.5%，approx_kl 0.0056，clip_fraction 0.045，grad_norm 14.8（被裁剪到 1），critic explained_variance -1.18，loss_mask_fraction 0.866。按这个速度 10 次迭代要 6.2 小时，所以在第 2 次迭代中途停掉 |
| 11-04 00:32 | `ppo_balanced` | 共置，2 × 16 个环境，前缀缓存（pageable，每 64 次命中核对一次） | 运行中，20 次迭代。第 1 次迭代：810 s，是流水线布局的 2.8 倍速度（rollout 390 s，其中 predict 278 s；actor 训练 418 s）。缓存：每个 rank 62 GiB，命中 576 次、未命中 192 次，9 次核对的最大差值为 0.0。训练局 success_once 41/128 = 32.0%，approx_kl 0.0075，clip_fraction 0.047。轨迹长度：成功 278.5 步，失败 468.9 步；按逐步平均，失败轨迹的权重占比为 81.7%，按轨迹等权为 72.7% |

- 采样剖析：actor 训练时 89% 的时间花在冻结的 VLM 前缀上，后缀 7%，观测变换 2%。
- 为什么不用 pinned 内存：PyTorch 的 pinned 分配器会把每块大小向上取整到 2 的幂，前缀缓存会从约 197 GB 涨到约 270 GB。所以改用 pageable 内存，每个 micro-batch 拷贝约 40 ms，相比约 0.4 s 的前缀计算可以忽略。

- 注意 `num_action_chunks` 是 10：训练配置沿用 RLinf 的 pi0_5 默认值，所以每个 chunk 是 10 个环境步，每局 48 个 chunk，每次迭代 6144 个样本，也就是 3 个 global batch × 4 个 epoch = 12 次优化步。评测配置是 5。
- `traj/success_frac` 用“奖励之和 > 0”来判成功，比环境统计的 success_once 少几局（35 对 41），可能有一部分成功局的奖励没有落进 rollout 窗口。从逐步平均那一臂开始，另外记录 `traj_term/*`，按“提前终止 = 成功”来判断。

### 文献核查：轨迹长度归一化已经发表（2026-11-04 01:40）

- **RLinf-VLA**（arXiv 2510.06710）：
  - **§IV-C3 / §VIII-C2 “Loss Normalization by Trajectory Length”：** 有效步数为 T_i^succ 的轨迹，每一步按 1/T_i^succ 缩放，目的是“prevents longer trajectories from dominating the gradient”。
  - **Fig. 12 消融：** OpenVLA-OFT + GRPO，在 LIBERO-Goal 上加归一化后明显更好；在 ManiSkill 上看不出好处（Fig. 13）。
  - **附录 X-C “Gamma with action chunk”：** 在 chunk 级别，γ_chunk 应取 γ_action^c；直接用 γ_chunk = 0.99 明显更差（ManiSkill，c = 8）。
- 对 CVPR 的影响：
  - balanced_ppo 本身不能算新方法。
  - 现在 LIBERO-Long、CALVIN 上的对照实验（π0.5 + PPO，长程任务）可以看作对 RLinf 消融的扩展：他们原来只在 GRPO + OpenVLA-OFT + LIBERO-Goal/ManiSkill 上做过。另外我们补了机制分析，即失败轨迹权重占比和成功/失败轨迹长度。
  - 论文的切入点需要用户决定。
- γ 的换算：RLinf 的 π0.5 LIBERO 配置在 chunk 级别用 γ = 0.99，c = 10，相当于动作级 γ ≈ 0.999。所以这里的 γ=1 臂和默认值只差一点（48 个 chunk 上是 0.99^48 = 0.62 和 1 的区别）。

### checkpoint 评测的加载问题（2026-11-04 01:50，已修复）

- 第 5 步的 checkpoint（`ppo_balanced/.../global_step_5/actor/model_state_dict/full_weights.pt`）有 6.5 GB，DCP 训练状态另有 8.6 GB。和 SFT 起点相比，有 190 个张量变了，最大 |Δ| 4.4e-4，说明训练的更新确实写进了 checkpoint。
- **问题：** 评测模式的 π0.5 不建价值头，即使设 `add_value_head=True` 也一样；而 RLinf 评测是严格加载 `load_state_dict(strict=True)`，checkpoint 里多出 8 个 `value_head.*` 键就会报错。
- **解决：**
  - `strip_value_head.py` 先去掉价值头，另存一份临时副本给评测用，评测完删除。
  - `check_ckpt_load.py` 在 CPU 上验证过：missing 0，unexpected 0，STRICT LOAD OK。
  - `run_eval_ckpt.sh` 和 `run_calvin_eval_ckpt.sh` 已经改好；队列执行到评测那一步时会读到新版本。

### 评测用的是确定性 ODE 采样器（2026-11-04 02:4x，核对代码 `openpi/env_io.py::_predict_eval`）

- RLinf 在评测模式下用的是 “Deterministic Euler ODE sampler”：固定初始噪声，跑 5 步 Euler，不加 SDE 噪声。
- 所以评测配置里的 `noise_level: 0.3` 对评测没有作用。SFT 起点 27.8% 和 RLinf 报告 43.9% 之间的差距，不是采样噪声造成的，原来计划的“确定性评测”核查也就不需要了。
- 训练 rollout 用的是 Flow-SDE：在随机选中的某一步加 noise_level 0.5 的噪声，chunk 长度 10。我们训练局的成功率约 30%，和评测的 27.8% 接近，所以 chunk 长度 5 和 10 的差别看来也不是原因。
- **和 ESPO 的关系：** ESPO 优化的正是部署时这个 ODE 采样器“实际执行的动作”的分布；Flow-SDE PPO 优化的是训练用 SDE 链的似然。这一点和登记稿的出发点一致。

### 预算对比：πRL 在 LIBERO-Long 上的配方（arXiv 2510.25889，附录 J）

- **πRL 的配方**（π0.5 + PPO）：64 个环境 × 8 轮 rollout，每次迭代 512 局。预测 10 步、执行 10 步，update_epoch 4，lr 5e-6，用 cosine 学习率调度，γ 0.99，λ 0.95。
  - 一共训练 **500 次迭代，约 25.6 万局**，成功率从 43.9% 提到 93–94%。
  - 硬件是 8 × H100。
- **我们的每个臂：** 20 次迭代 × 128 局 = 2560 局，约为它的 **1%**。每次迭代只有 12 次优化步，πRL 是 48 次。
- **解读：** `ppo_balanced` 前 13 次迭代的训练成功率大约停在 33%（24–44% 之间来回），这和预算这么小是吻合的。所以本地的 LIBERO 实验比较的是**低预算下的样本效率**，这正是 ESPO 声称的优势所在。
- 如果 ESPO 和 PPO 都没有明显变化，需要考虑：
  - 公平地调高两者的学习率，比如 2e-5；
  - 或者用更长的预算。

### `ppo_balanced` 跑完（2026-11-04 04:47，20 次迭代，共 4 小时 15 分钟）

- **训练局成功率**（每次迭代 128 局）：前 5 次迭代平均 **31.9%**，后 5 次平均 **37.5%**，全部 20 次平均 33.8%。逐次数值在 `results/ppo_balanced/curve.json`。
  - 后 5 次比前 5 次高 5.6 个百分点。每 5 次迭代是 640 局，标准误约 1.9pp，所以这个差距大约是 2 个标准误，只能说明有温和的学习。
- **KL 和裁剪：** 每次迭代的 approx_kl 在 0.003–0.0075 之间，clip_fraction 约 0.035。critic 的 explained_variance 从 −2.45 升到 0.5–0.64。
- **轨迹长度：** 成功轨迹平均 250–308 步，失败轨迹约 470 步。按逐步平均算，失败轨迹的权重占比为 0.75–0.88。
- **checkpoint：** global_step_5/10/15/20，每个 15 GB。第 20 步的 500 局评测排在 ESPO 之后。

### ESPO v1 冒烟测试（`results/espo_smoke`，2026-11-04 04:49 开始）

**第 1 次迭代**（rollout 来自 SFT 策略）：

| 项目 | 数值 |
|---|---|
| 单次迭代 | 1210 s：rollout 389 s，actor 819 s（PPO 是 418 s） |
| 训练局 success_once | 33.6% |
| 回报预测器 Q(z, u) | 训练 R² 0.833，留出 R² **0.739** |
| ESPO 损失 | 归一化后起点约 1，4 个 epoch 平均 0.883 |
| 特征目标位移 | 0.243 |
| 前缀缓存 | 命中 960 次 = 192 × 5（准备阶段第二遍 1 次 + 4 个 epoch），未命中 192 次，15 次核对最大差值 0.0 |
| 成功判定核对 | 按“提前终止”算的成功比例 0.336，和环境给的 success_once 完全一致 |

- 整条流程在 GPU 上跑通，没有报错。
- 按这个速度，`espo_v1` 的 20 次迭代约需 6.7 小时。

### ESPO v1 失败，改为 ESPO v2（2026-11-04 06:35）

- **`espo_v1`** 使用 Flow-SDE rollout（noise 0.5），lr 5e-6，没有 critic 预热，也没有信赖域。
  - 训练成功率：第 1 次迭代（SFT 策略）37.5%，第 2 次 12.5%，第 3 次 **3.1%**，策略在崩溃，所以停掉了。
  - Q 的拟合一直正常（留出 R² 0.66–0.80），grad_norm 从 15.6 降到 7.0。
- **判断的原因**（三条一起起作用）：
  1. **采样器不一致**：rollout 用的是 Flow-SDE，在某一步加 noise 0.5 的噪声并做漂移修正；而 ESPO 优化的是 ODE 采样器，也就是部署用的那个。玩具实验里 rollout 本来就是 ODE 采样，登记稿的设计也是如此。
  2. **没有步长控制**：ESPO 向固定目标做回归，Adam 在 12 步里一直朝同一方向走，每个参数每步移动约一个 lr。PPO 有裁剪，梯度也带噪声互相抵消，每次迭代的 KL 只有约 0.005。
  3. **critic 刚初始化**：explained_variance 是 −2，所以优势估计几乎是噪声。PPO 的裁剪能扛住，ESPO 的回归扛不住。
- **`espo_v2`** 的改动：
  - `actor.model.openpi.noise_level=0.0`：rollout、模型采样和评测用的都是同一个 ODE 采样器。RLinf 在 σ=0 时 log-prob 记为 0，这对 ESPO 没有影响。
  - `actor.optim.lr=1e-6`；
  - `actor.optim.critic_warmup_steps=12`：第一次迭代只训练 critic；
  - **信赖域**（`espo.stop_frac=0.5`）：某个 epoch 的平均“特征到目标距离”降到初始目标位移的 50% 以下时，本次迭代剩下的 epoch 只训练 critic。
- **注意：** ESPO v2 的训练成功率是在 ODE 采样器下测的，PPO 是在 SDE 采样器下测的，两条训练曲线不能直接比。可比的是两臂的最终评测，都按 500 局、ODE 采样器的同一套协议。

### 基线差距的可能原因：重规划步长（2026-11-04 07:07）

- πRL 附录 J 给 LIBERO-Long 用的是 **H′ = 10**：每次预测 10 步，执行完 10 步再重新规划。Spatial/Object/Goal 三个套件是 5。RLinf 自带的评测配置 `libero_10_openpi_pi05_eval.yaml` 却写的是 `num_action_chunks: 5`。
- 对照数据：
  - `espo_v2` 第 1 次迭代是 critic 预热，策略还是 SFT，用 ODE 采样器、chunk 10，得到训练局 success_once **45.3%（58/128）**，几乎就是 πRL 报告的 43.9%。
  - 我们按 chunk 5 协议测的是 27.8%（500 局）。
  - 同样 chunk 10、但用 SDE 采样器（noise 0.5）时，PPO 第 1 次迭代是 32%。
- 结论：所有 LIBERO-Long 的最终评测改用 **chunk 10（`EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"`）**，这和 πRL 的协议、也和训练时的设置一致。另外补测一个 chunk 10 的 SFT 基线（`eval_sft_c10`）。这些都在队列 v5（`run_queue5.sh`）里。

### 已确认：按 chunk 10 协议，SFT 起点和 πRL 吻合（2026-11-04 08:57）

| 协议 | `RLinf-Pi05-LIBERO-SFT` 在 500 个初始状态上的成功率 | 各任务成功数（每任务 50 局） |
|---|---|---|
| chunk 5（RLinf 自带评测配置） | 27.8%（139/500） | 0:10, 1:25, 2:24, 3:44, 4:4, 5:0, 6:11, 7:14, 8:1, 9:6 |
| **chunk 10**（πRL 的 LIBERO-Long 协议 H′ = 10，与训练一致） | **47.6%（238/500）**，success_at_end 38.2% | 0:23, 1:40, 2:32, 3:49, 4:24, 5:0, 6:31, 7:15, 8:8, 9:16 |

- 两种协议的配对差为 +19.8pp：只在 chunk 5 下成功的有 29 局，只在 chunk 10 下成功的有 128 局，McNemar 精确检验 p < 1e-4。
- πRL 报告的是 43.9%。所以之前的差距来自评测协议，现在已经对上。
- 后面所有 LIBERO 评测都用 chunk 10，数据在 `results/eval_*_c10`。用两张卡跑 500 局约 21 分钟。
