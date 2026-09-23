# π0.5 JAX：FrameSamp + Modul 与 subtask 联合训练

本实现把历史视觉记忆接到现有 JAX π0.5 action expert，保留 subtask 文本 CE 和动作
flow matching 联合监督。默认关闭；设置 `FRAMESAMP_MODUL=1` 启用。
这是针对 BEHAVIOR 的实现，不能直接引用 RoboMME 的 44.51% 作为本模型的效果。

## 模型与数据路径

```text
同一 episode 已观测的 head camera 帧（包含当前已缓存帧）
  → 冻结的 SigLIP → 每帧 16×16 tokens 平均池化到 4×4
  → 从完整历史前缀均匀选最多 32 帧 → 512 个视觉 tokens（不足时 mask padding）
  → 拼接时间/行/列位置特征 → 投影到 action expert 的 1024 维
  → 每层 action features 作 Q，memory 作 K/V
  → 4 个 Q heads、1 个 KV head → 每个 action token 的 scale / shift
  → RMSNorm(action) × (1 + scale) + shift → 原有 time AdaRMSNorm → FFN

高层：Task + State + 当前 images → Subtask CE
低层：Subtask + State + 当前 images + 历史调制 → Action flow matching
```

调制只作用于 action expert 的 18 层，不向 VLM 拼接历史 tokens，也不向高层 subtask
预测器注入 memory。默认 `subtask_only` 时继续使用独立的低层语言前向；生成 subtask 的
KV cache 不传给动作分支。历史信息对动作损失可训练，但不会直接改变高层文本损失。

与论文配方的区别：使用本仓库 JAX 模型与 BEHAVIOR 三视角当前观测；历史默认仅取
`base_0_rgb`（R1Pro head camera）。历史帧候选默认每 **16 个原始控制步** 存一次，再从
候选前缀均匀取 32 帧。位置编码为本实现的 `thw_sincos_768`（时间、行、列各 256 维），
不是官方代码的逐位复现；小幅非零 affine 初始化用于向新增记忆模块传梯度。

`MEMORY_FRAME_STRIDE` 与训练样本的 `FRAME_STRIDE` 是两个不同参数。前者定义历史
缓存的时间间隔，后者定义从数据集中挑哪些帧作为动作训练样本。所有位置均保留原始
episode id 和帧号，采样严格限制在当前帧及以前，不读取未来帧。

## 1. 生成冻结视觉特征缓存

在具备现有 OpenPI 环境、数据和 GPU 的训练机器上，从仓库根目录运行：

```bash
cd openpi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/b1k/cache_frame_memory.py \
  --dataset-root /path/to/2026-challenge-demos \
  --repo-id behavior-1k/2026-challenge-demos \
  --init-params /path/to/current_subtask_checkpoint/params \
  --episodes-file /path/to/train_and_val_episodes.txt \
  --output-dir /path/to/frame_memory_cache \
  --frame-stride 16 --grid-size 4 --batch-size 32
```

`--episodes-file` 每行一个原始 episode id，需包含之后训练和验证使用的全部 episode。
省略则缓存整个数据集；全量数据占用可能很大。可以多次向同一目录补充不同 episode，
相同设置下已完成的 episode 会跳过。`--shard-index i --num-shards N` 可将同一 episode
名单分配给 N 个独立 GPU 进程；不需要启动 JAX distributed。每个进程必须使用同一
权重、相机、stride、grid 和数据。实际视频解码由 LeRobot 完成，可能涉及多个相机，
只有所选相机的特征会进入缓存。

缓存包括 `manifest.json`、逐 episode 的 FP16 `.npy` 和原子完成标记。manifest 记录
视觉权重和 episode metadata 的 SHA-256；启动前检查完成标记与数组形状。训练初始化
还会核对实际载入的 SigLIP 权重。哈希不覆盖全部原始视频内容；修改视频后需要重建缓存。

**缓存必须来自本次 `INIT_PARAMS` 的视觉权重。** 如果当前 subtask checkpoint 已经更新
过 SigLIP，不能用 π0.5 base 的缓存替代。启用记忆后 SigLIP 会在优化器中冻结，当前图像
分支和历史分支共用这套视觉权重。历史缓存不做随机图像增强。

## 2. 预检与新实验

首先生成缓存，再沿用现有 subtask 启动入口。第一次加记忆时使用新的 `EXP_NAME`，
通过 `INIT_PARAMS` 加载已有模型；新增参数单独初始化，不恢复旧优化器。

```bash
export DATASET_ROOT=/path/to/2026-challenge-demos
export INIT_PARAMS=/path/to/current_subtask_checkpoint/params
export EPISODES_FILE=/path/to/train_episodes.txt
export VAL_EPISODES_FILE=/path/to/val_episodes.txt
export REPO_ID=behavior-1k/joint_two_tasks_s1
export EXP_NAME=pi05_subtask_framesamp_v1
export FRAMESAMP_MODUL=1
export MEMORY_CACHE_DIR=/path/to/frame_memory_cache
export MEMORY_FRAME_STRIDE=16
export SUBTASK_ACTION_CONDITIONING=subtask_only

bash scripts/train_pi05_subtask_b1k.sh check
NUM_GPUS=1 CUDA_DEVICES=0 FSDP_DEVICES=1 NUM_WORKERS=0 \
  BATCH_SIZE=1 TRAIN_STEPS=2 \
  EXP_NAME=pi05_subtask_framesamp_smoke \
  bash scripts/train_pi05_subtask_b1k.sh smoke
```

`check` 同时验证 subtask 标签和 train/val 所需的历史缓存，不分配 GPU。`smoke` 使用
显式传入的 episode 名单；如果省略名单，沿用现有脚本的 episodes 0、1。
确认 smoke 成功后，用目标 GPU 数、全局 batch、训练步数执行 `full`。这里没有测量
目标硬件的 batch 上限；不能仅由新增参数比例推断 batch 64 一定能放下。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `FRAMESAMP_MODUL` | 0 | 启用 JAX 视觉记忆 |
| `MEMORY_CACHE_DIR` | 无 | 已完成的历史特征缓存 |
| `MEMORY_NUM_FRAMES` | 32 | 均匀采样的最大帧数 |
| `MEMORY_GRID_SIZE` | 4 | 每帧池化网格边长；每帧 16 tokens |
| `MEMORY_FRAME_STRIDE` | 16 | 原始控制帧上的历史候选间隔 |
| `MEMORY_CAMERA` | base_0_rgb | 历史相机 |

改变 stride/grid/camera 或视觉权重时，需要另建对应缓存；只改变帧预算可复用缓存，
但应开新实验。设 `MEMORY_FRAME_STRIDE=1` 可使用每个原始帧作为候选，需要按 stride 1
重新生成缓存，同时增加存储与在线编码成本。

直接调用 Python trainer 时，相应参数是：

```text
--model.framesamp-modul
--model.memory-num-frames 32
--model.memory-grid-size 4
--model.memory-frame-stride 16
--model.memory-camera base_0_rgb
--data.memory-cache-dir /path/to/frame_memory_cache
```

每个 checkpoint 保存 `assets/frame_memory.json`。记忆实验自身可用 `RESUME=1` 恢复，
但必须保持配置与缓存 provenance 相同；不能用旧无记忆 checkpoint 做完整状态 resume。
原无记忆模型和已有训练入口默认行为保持不变。PyTorch 路线没有实现此模块，显式启用会报错。

## 3. 多机训练

`nebula/submit_pi05_multihost.sh` 和 worker 启动脚本已经传递上述 memory 环境变量及
`INIT_PARAMS`。在原有提交命令上增加这些环境变量即可。`MEMORY_CACHE_DIR` 必须在
每台 worker 上以相同路径可读，通常使用现有共享挂载；提交脚本不会自动生成或上传缓存。
缓存检查会在每个 worker 的训练入口执行，避免某台机器缺 episode 文件时静默降级。

训练 batch 语义沿用既有 multihost 数据切分和 JAX mesh，memory 是同一 Observation
中的固定形状数组，跟随 batch 一起分片。这里没有提交或修改任何正在运行的训练任务。

## 4. 闭环与离线评测

`create_trained_policy` 自动读取 checkpoint 的记忆配置，仍按原 subtask 参数启动服务。
BEHAVIOR wrapper 每个控制步向记忆 buffer 提供观测，即使该步还在执行上一 action chunk；
只在 stride 边界编码新帧，其余历史特征复用。episode reset 清空历史，subtask 切换只重置
语言缓存与动作队列，保留当前 episode 的视觉历史。每个环境必须有独立 policy 实例。

直接调用 `Policy.infer` 而不经过 wrapper 时，每次请求被当作相隔一个 memory stride；
只有请求频率确实如此时才适用。不同频率的客户端应在控制步调用
`policy.observe_memory(obs, step=original_step)`，并在新 episode 调用 `policy.reset()`。
该接口目前是本地 Python 接口，未扩展通用 websocket 协议。

随机抽帧的 open-loop 评测必须传递同源缓存，不依赖 policy 内部在线历史：

```bash
openpi/.venv/bin/python scripts/eval_openloop.py \
  --dataset /path/to/2026-challenge-demos \
  --pi05-ckpt /path/to/memory_checkpoint \
  --repo-id behavior-1k/joint_two_tasks_s1 \
  --episodes-file /path/to/val_episodes.txt \
  --subtask-prompts /path/to/subtask_segments.parquet \
  --subtask-conditioning provided \
  --memory-cache-dir /path/to/frame_memory_cache \
  --out /path/to/memory_openloop.json
```

按 episode/frame 读取因果历史，并检查缓存与 checkpoint 的 provenance；缺少缓存会直接
报错。此处 `provided` 使用真值 subtask，测的是动作误差，不是自主 subtask 的闭环成功率。

## 成本与验证边界

当前生产配置（Gemma 2B + action Gemma 300M）通过 `nnx.eval_shape` 统计：

| 项目 | 数值 / 行为 |
| --- | --- |
| 原模型参数 | 3,353,433,872 |
| 新增参数 | **88,446,720（+2.64%）** |
| 新增权重体积 | BF16 约 168.7 MiB；FP32 约 337.4 MiB |
| 每个缓存帧 | 16 × 2048 × FP16 = 64 KiB，不计文件头 |
| 默认 stride 16 缓存 | 每百万原始帧约 4.10 GB（十进制），另有 episode 向上取整开销 |
| 训练 memory 输入 | 每样本 512 × 2048 × FP32 = 4 MiB；全局 batch 64 合计 256 MiB，之后按设备分片 |
| 推理历史编码 | 每 16 控制步额外编码一张所选相机图像，旧帧不重复编码 |
| 推理动作计算 | 每个 denoise step、每层新增 memory attention 和 modulation |

缓存生成是额外的一次性视觉前向与视频 I/O。训练时读取池化特征，无需逐样本重编码 32
张历史图，但仍有投影、跨注意力、梯度、优化器状态和缓存 I/O 成本。在线 buffer 保留
episode 的全部候选特征，**512 是每次模型读取的 token 预算，不是 buffer 的存储上限**。

上述权重和输入体积不等于总显存增量；激活、rematerialization、FSDP、优化器和 EMA
都会影响结果。尤其本模式冻结 SigLIP：若 baseline 原来训练视觉编码器，公平的 memory
消融应使用同样冻结设置。不能把冻结带来的节省归因于 memory 模块。

本地 CPU 测试覆盖真实 JAX 小模型的训练梯度、动作条件变化、高层 subtask 隔离、
masked tokens、无历史退化、旧权重迁移、真实 B1K train_step、checkpoint round-trip、
因果缓存读取、在线 reset 与 subtask 切换。完整模型只做参数形状统计，尚未运行真实数据
GPU smoke、multihost 训练或 BEHAVIOR 闭环评测，不能声称已有成功率提升或耗时百分比。

测量时使用相同硬件、数据、batch、精度和视觉冻结设置，先排除 JIT 编译的 warm-up，
对比 steady-state step time、峰值显存与 checkpoint 体积。闭环日志中的 `memory_ms`
包含新增帧编码和历史打包耗时（包含 action 请求间累计的编码），`infer_ms` 包含动作
采样，subtask policy 另有 `subtask_ms`。`total_ms` 只覆盖当前 infer 调用；外部
`observe_memory` 发生在调用之前，比较端到端延迟应计时完整 `wrapper.act`。
