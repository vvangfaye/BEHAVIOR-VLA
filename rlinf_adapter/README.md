# BEHAVIOR joint-subtask π₀.₅ → RLinf

适配现有 `1688977` 及同结构 checkpoint。保留“整任务 → 自生成子任务 → 动作”的输入格式，冻结 VLM 和子任务生成，只用在线 PPO 更新动作 expert、投影层和新增 value head。

基于 RLinf commit `91a302ec50ddacd1fdabc5859b85e3661d768236`。本目录是独立扩展，不修改原评测代码、checkpoint 或 RLinf 源码。`RLINF_EXT_MODULE=b1k_rlinf` 在 driver 和每个 worker 中注册；普通 OpenPI 配置仍走原始 builder。

## 已接入的接口

- 512-token 上限、64-token 子任务、greedy 生成、每次动作请求重新生成子任务。
- 与原 tokenizer 相同的 `Task / State / Subtask / Action` 格式及因果 attention mask。
- rollout 保存实际子任务 token、mask、图像和归一化状态；PPO 重算不重新生成子任务。
- 32 步预测、16 步执行、32 维模型输出、23 维机器人动作。
- 使用 checkpoint 自带均值/标准差；躯干和双臂增量恢复为绝对关节目标，夹爪位置按原定义处理。
- 原始 61 维观测可用于离线对齐；其中前 57 维包含策略所需状态。仿真训练显式请求 25 维命名本体量，合并两指后得到相同的 23 维策略状态，避免依赖另一版 OmniGibson 的默认排列。

## 夹爪执行接口

R1Pro 的辅助抓取控制器区分完全张开与略低于张开端点的目标。2026-10-25 增加可配置的执行映射：仅将动作通道 14、22 中 `[0.999, 1)` 的值对齐到 `1`，在反归一化后、送入环境前执行。训练采样与评测共用 `predict_action_batch` 的同一出口；不能只给 PPO 模型或只给评测启用。

配置项为 `actor.model.b1k_joint.execution_gripper_open_tolerance`。库与配置模板均默认 `0`，保留已有行为；显式设为 `0.001` 才启用上述候选映射。`rollout.model` 引用完整的 `actor.model`，使单独评测也使用相同接口。回放旧实验或检查 JAX/Torch 原始输出一致性时应设为 `0`。这是 R1Pro 辅助抓取配置下的边界处理，不应直接套用到其他机器人。

返回的动作和 `result["execution_actions"]` 是环境执行值；`result["policy_actions"]` 保留未修正的物理动作。`model_actions`、`forward_inputs["action"]`、采样链和 PPO log-prob 均保持原始策略样本，概率重算不能改用修正后的执行值。这是确定的环境动作映射，不修改策略采样分布。

切换接口后，后续在线 PPO 应重新收集对应接口下的轨迹；旧接口的 rollout 不能直接当作新接口的 on-policy 数据。现有 PPO 权重可以在新接口下评测，无需重新训练。

相关边界与 PPO 回放测试见 `tests/test_action_interface.py`。新接口下原 checkpoint 与 PPO checkpoint 的闭环验证单独记录于项目 `analysis/rlinf_adaptation_20261020/action_interface_fix_20261025/`；修复接口本身不等于证明 RL 提升。

该验证共完成四个回合，原模型与 PPO 均启用 `0.001`：实例 301 / seed 0 为原模型成功、PPO 失败，实例 302 / seed 0 为原模型失败、PPO 成功，双方成功率均为 1/2。两组初始位姿误差均为 0，四次原始策略概率重算误差均为 0。因此不能认为这个候选映射已消除 PPO 退化或带来稳定收益，默认不启用；已提交实验的配置与原始结果保留不变。

## 安装与转换

先按 RLinf 的 BEHAVIOR OpenPI 环境安装依赖，checkout 上述 commit，再安装本扩展：

```bash
pip install --no-deps -e /path/to/BEHAVIOR-VLAs/rlinf_adapter
export RLINF_ROOT=/path/to/RLinf
export PYTHONPATH="$RLINF_ROOT:$PYTHONPATH"
python -m b1k_rlinf.prepare \
  --checkpoint /path/to/original-jax-checkpoint \
  --norm-stats /path/to/original-jax-checkpoint/assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json \
  --output /path/to/new-joint-rl-checkpoint
```

转换读取 OCDBT 的完整数组，去除 NNX `value` 包装，再使用固定版本 RLinf 的 SigLIP、Gemma 和投影权重转换。无需原训练 GPU 拓扑，不加载旧 optimizer；输出 FP32 权重。norm stats 原字节复制。未完成的输出保留在 `.partial-*` 目录，成功后才重命名为目标目录。

## 检查与启动

配置模板使用两个可渲染的 GPU：GPU 0 仿真，GPU 1 推理，actor 使用两者。H200 可承担模型计算，但 OmniGibson 仍需要适合光追渲染的 GPU；多节点放置应另配。

```bash
export B1K_TRAIN_INSTANCES=/path/to/training-task-instances
export B1K_EVAL_INSTANCES=/path/to/held-out-task-instances
export ISAAC_PATH=/path/to/isaac-sim
export OMNIGIBSON_DATA_PATH=/path/to/omnigibson-data
python -m b1k_rlinf.launch \
  --rlinf-root "$RLINF_ROOT" \
  --checkpoint /path/to/new-joint-rl-checkpoint \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --output /path/to/fresh-results \
  --mode config
```

`config` 仅解析配置，不启动 Ray 或仿真。确认实例路径、任务、控制频率与原评测一致后，可将模式改为 `eval`、`smoke` 或 `train`。`smoke` 只跑 1 轮、每段 256 个控制步；`train` 默认上限 20 轮，每段 4096 步。新 optimizer 使用固定 `1e-6` 动作学习率和 `1e-4` value 学习率，不继承旧 SFT 衰减日程。

模板固定训练实例 1、评估实例 301，分别由 `B1K_TRAIN_INSTANCE` / `B1K_EVAL_INSTANCE` 覆盖；不要把评估实例用于训练。任务默认 `turning_on_radio`，由 `B1K_TASK` 覆盖；切换任务时还需要匹配 scene 和实例文件。控制频率默认 30 Hz，由 `B1K_ACTION_FREQUENCY` 显式覆盖。

## 验证

初始 CPU 验证中，真实 `1688977` checkpoint 可自生成子任务并输出 `16×23` 动作，Flow-SDE 采样与 PPO 重算的 log-prob 最大差为 `0`。固定同一观测、子任务 token 和初始噪声，与原始 JAX 的 10 步 FP32 ODE 输出相比，平均绝对误差 `9.02e-5`，最大绝对误差 `1.36e-3`。原始记录见项目 `analysis/rlinf_adaptation_20261020/validation.json`。

2026-10-23 的进一步验证：

- 修复仿真 step recorder 递归；已在 RTX 5090 上实际完成闭环回合。
- BF16 的 JAX / Torch 对照没有通过：实例 302 的成功结果不同，因此不能据此启动 PPO。
- FP32 CPU JAX / GPU Torch 的 12 个相同观测通过冻结阈值，归一化动作误差的最大均值为 `0.0001187`、最大值为 `0.0025588`。
- 在同一张 5090 上运行 FP32 JAX 和现有仿真，16 个观测全部通过 CPU / GPU 数值一致性检查；显存峰值 `30796 / 32607 MiB`，预热后每次动作 chunk 推理平均 `0.371 s`。关闭 JAX 预分配，使用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` 和 `JAX_DEFAULT_MATMUL_PRECISION=highest`。这验证了该组观测和场景下的兼容性，不保证所有任务的显存峰值。

GPU 实测记录见 `analysis/rlinf_adaptation_20261020/gpu_fp32_probe_20261023/summary.json`。

2026-10-23 晚间，完整 FP32 闭环对照通过：实例 301 的 JAX / Torch 均为 score 1、成功（分别 2673 / 1500 步）；实例 302 均为 score 0、失败（均为 3224 步）。JAX 301 复用已完成的 CPU FP32 回合，JAX 302 和 Torch 回合使用同一张 RTX 5090。16 个同观测比较的子任务均一致，归一化动作误差的最大均值为 `9.5306e-5`、最大值为 `0.0016709`；两个实例的初始状态、score 和成功结果均通过原定阈值。

随后完成一轮单进程 RLinf PPO：训练实例 1，256 个控制步、16 个 action chunks、4 次 optimizer 更新。expert 和 value head 均有非零梯度，冻结参数哈希保持一致、可训练参数哈希改变，checkpoint 保存及读回校验通过，更新前 log-prob 重算误差为 0。该批次 reward sum 为 0，因此它验证采样、优化和保存链路，不证明任务成功率提升。此验证不覆盖 Ray/FSDP。

运行期间的原始回执见 `analysis/rlinf_adaptation_20261020/cloud_smoke_gpu_20261023/completion_live_evidence.json`，完整归档核验见同目录 `acceptance_report.json`。平台作业的 FAILED 终态来自同卡接管完成后主动结束旧 supervisor；实验结果以最终 driver 状态、闭环 gate 和 PPO 产物为准。

```bash
RLINF_ROOT=/path/to/RLinf python -m pytest tests -q
```

测试包含实际小型 Gemma 的 causal attention、greedy 子任务生成、Flow-SDE 采样、PPO log-prob 重算与梯度更新；另与原 tokenizer、PIL resize 和机器人动作契约对齐。CPU 测试不是完整 3B checkpoint 的数值等价证明，也不代表 BEHAVIOR 闭环收益。

接入验收顺序：真实 checkpoint 权重结构检查 → 相同观测/初始噪声的 ODE 动作对齐 → 不更新权重的闭环对照 → 有限 PPO smoke。完整验证记录存于项目 `analysis/rlinf_adaptation_20261020/`。

已有专家观测缓存可先验证完整模型，不启动仿真：

```bash
python -m b1k_rlinf.verify \
  --checkpoint /path/to/new-joint-rl-checkpoint \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --sample /path/to/existing-observation.npz \
  --output /path/to/probe.json \
  --export-reference-inputs /path/to/reference-inputs.npz

python -m b1k_rlinf.reference_check \
  --checkpoint /path/to/original-jax-checkpoint \
  --reference-source /path/to/BEHAVIOR-VLAs/openpi/src \
  --inputs /path/to/reference-inputs.npz \
  --output /path/to/reference-parity.json
```

缓存 NPZ 使用既有诊断格式：原始 `observation.state`、三个 `observation.rgb.*` 图像和 `prompt`。第一步使用实际 checkpoint 自生成子任务并检验 PPO 重算；第二步让原始 JAX 实现读取相同的观测、子任务 token 和初始噪声，比较 10 步 FP32 ODE 输出。该比对不替代 GPU BF16 和闭环验收。

## 仿真缓存与存储预算

2026-10-24 盘点发现，逐回合把 Isaac / NVIDIA / Warp 缓存写入 AFS，32 个缓存目录累计占用约 270.7 GiB；完整回合的一份缓存约 9.2 GiB。

后续运行应将 `OMNIGIBSON_APPDATA_PATH`、`XDG_CACHE_HOME`、`XDG_CONFIG_HOME` 和 `WARP_CACHE_PATH` 显式指向节点 `/tmp` 下的独立临时目录，并在仿真关闭或初始化失败后清理；不要受继承的 `TMPDIR` 影响回到 AFS。禁用 core dump。AFS 保留权重、优化器恢复点、轨迹、视频、日志和实验配置。

当前两轮 PPO 恢复实现位于项目 `analysis/rlinf_adaptation_20261020/conservative_ppo_storage_20261024/`，对持久化输出执行 15 GiB 预算检查，并在保存 checkpoint 前预留写入空间。原始实验目录及其配置保持独立，清理清单见 `analysis/rlinf_storage_20261024/`。

### Conservative PPO two-worker continuation (2026-10-24)

The user authorized two RTX5090 workers for the same frozen experiment. Deployment and run receipts are in `analysis/rlinf_adaptation_20261020/conservative_ppo_dual_20261024` at the project root. Each worker owns disjoint trajectory and evaluation IDs. Both verify the same collection-policy hash; the leader gathers all four trajectories in original order and runs the unchanged single-process PPO update. Completed rollouts are replay-verified and referenced from the immutable prior run, without re-simulation or another raw-data copy. Shared outputs remain capped at 15 GiB; simulator caches remain node-local. The previous single-GPU job was suspended and its worker deleted before submitting this pair.
