# GoalKeeper — BEHAVIOR 2026 研究原型

本目录已集成进 BEHAVIOR-VLAs，作为可单独安装的 Python 包。推理连接本仓库
`openpi/` 的 B1K policy，闭环评测复用上层 `scripts/eval_b1k.sh`。

基于公共 BDDL 的目标信念与动作后果模型，为 Comet / OpenPI 选择动作前缀。研究目标是减少长任务中的误判完成与目标回退。方案和证据边界见 [research.md](docs/research.md)。

**当前是核心实现，尚无 BEHAVIOR 跑分，尚未接入你的 32 张 H20。** 不包含 Comet 权重、官方数据或 OmniGibson。合成样本只验证软件；合成训练产物被运行时显式拒绝，不能用来假装比赛模型。

## 已实现

- 公共 BDDL 的有限语法编译、目标原子展开、否定与多分支处理。
- RGB patch / 目标文本 / proprio 交叉注意力、GRU 信念模型、动作条件目标后果头。
- 单机/DDP 训练、按初始状态分组划分、分组可靠性校准、NPZ 离线导出。
- 可撤销记忆、低置信度退回原候选、避免保守静止动作阻止必要开门的筛选规则。
- OpenPI 原始 `infer()` 候选采样适配、官方 v3.9.2 msgpack/WebSocket 协议和输入白名单。
- 读取官方原始 JSON 的评测汇总：缺失计零、拒绝重复 rollout、配对 bootstrap。

未完成：真实 100 个 BDDL 的覆盖审核；训练数据回放与分支采集；真实编码器/OpenPI 权重联调；GPU/仿真测试；全量评测；标量 critic 等论文对照。官方接口通过源码核对不等于模拟器联调通过。

## 本地安装与软件验证

建议 Linux 训练使用 Python 3.11/3.12；CUDA/PyTorch 按服务器驱动安装。桌面可用 CPU 跑测试。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m goalkeeper.cli compile examples/cup_in_cabinet.bddl
python -m goalkeeper.cli synthetic outputs/toy --groups 120
OMP_NUM_THREADS=1 python -m goalkeeper.cli train outputs/toy/manifest.jsonl outputs/toy-head --epochs 20 --hidden-dim 32 --device cpu
python -m goalkeeper.cli calibrate outputs/toy/manifest.jsonl outputs/toy-head/best.pt outputs/toy-head/calibration.json
```

示例 BDDL 是自编测试任务，不是官方任务。目标代理进度 `GoalProgram.score()` 也不是官方 Q。

## 真实数据

详见 [数据契约](docs/data.md)。先在训练场景重放或采集数据，得到 RGB、本体感知、实际动作和训练用真值；然后 `export` 提取冻结特征。H 步后果必须对应**实际执行的同一段动作**，包括任何后处理。不能先用长序列末端的成功标签，评测时只执行其前 16 步。

```bash
python -m goalkeeper.cli audit /absolute/path/to/public_bddl
python -m pip install -e '.[vision]'
python -m goalkeeper.cli export /path/episode.npz /path/public_task.bddl /path/features \
  --group-id TRAINING_SCENE_INITIAL_STATE_ID \
  --action-spec VERIFIED_ROBOT_AND_ACTION_SPEC_HASH \
  --encoder google/siglip-base-patch16-224 --revision VERIFIED_40_CHARACTER_COMMIT \
  --horizon 16 --history 4 --device cuda:0
```

`export` 产生 K=1 监督记录。K>1 排序数据需要从相同训练状态分支执行候选后再写入记录；本仓库没有虚构反事实标签的快捷路径。

## 训练和部署

```bash
python -m goalkeeper.cli train /path/manifest.jsonl /path/head --device cuda:0
python -m goalkeeper.cli calibrate /path/manifest.jsonl /path/head/best.pt /path/head/calibration.json --device cuda:0
```

多节点示例见 [train_head_ddp.sh](scripts/train_head_ddp.sh)，各节点填写相同数据/输出路径与 rendezvous 地址以及不同 node rank。先小规模验证，不必用全部 32 张 H20 训练这个小模型。

运行服务必须在选定的 OpenPI 环境安装本包，并填写 [配置模板](configs/serve.template.json)。模板采用已核对的公开数据 schema：proprio 为 61 维、action 为 23 维；仍需从实际机器人首帧确认顺序。task ID 和 commit 必须填写真实值。模型输入输出顺序和主干 normalization 必须匹配权重。

从 BEHAVIOR-VLAs 根目录，在已安装本仓库 `openpi/` 的策略环境中安装：

```bash
python -m pip install -e ./goalkeeper
# 训练/推理需要 SigLIP 编码器时再装 vision extra；版本与 openpi/pyproject.toml 对齐。
python -m pip install -e './goalkeeper[vision]'
```

服务配置中的 `task` 使用 `b1k/<task_name>`，`robot` 使用 OpenPI 机器人注册名，
`repo_id` 必须与 checkpoint 中的 normalization assets 一致。`instruction` 设为 `null`
时读取 OpenPI 任务注册表；`openpi_config`、`openpi_checkpoint`、`bddl`、`task_id` 和
`horizon` 必须对应同一实际任务与模型。支持普通或 joint subtask π0.5 的三路 RGB 推理；
joint checkpoint 在配置中设置 `"subtask_prediction": true`，并使 `max_token_len`、
`max_subtask_len` 与训练一致。`subtask_action_conditioning` 留为 `null` 时读取 checkpoint
元数据；`subtask_interval` 留为 `null` 时使用 OpenPI 默认值。每轮先生成或复用一个
subtask，再在相同 subtask 和观测下采样 K 个独立动作块供 GoalKeeper 筛选；刷新间隔
按轮计算。服务会将 `subtasks` 和 `subtask_sources` 写入响应。当前不支持 FrameSamp
checkpoint。新导出的 `feature_id` 包含
Transformers 版本；旧特征缓存与旧小头需要按迁入后的编码器重新导出、训练和校准。

```bash
python -m goalkeeper.cli serve /path/filled_config.json
# 在独立的官方 OmniGibson v3.9.2 环境运行：
python -m omnigibson.eval.eval --task-name OFFICIAL_TASK_NAME \
  --host 127.0.0.1 --port 8000 --output-dir /path/raw-eval --write-video
```

使用上层闭环脚本时，将每个任务配置保存为 `<task_name>.json`：

```bash
GOALKEEPER_CONFIG_DIR=/path/to/goalkeeper-configs \
INSTANCE_INDICES=0 NUM_ROLLOUTS=1 OUTPUT_ROOT=/path/to/results \
  bash scripts/eval_b1k.sh turning_on_radio
```

该入口将 GoalKeeper 服务接入原有 OmniGibson 评测流程，输出目录使用
`goalkeeper/<task_name>/`。`baseline`、`shadow`、`rerank` 和 joint 开关由各任务 JSON
配置决定；若同时设置 shell 的 `SUBTASK_PREDICTION=1`，脚本会检查 JSON 已开启 joint。
joint + GoalKeeper 的
`rerank`/`shadow` 需要按 joint 候选分布训练并校准 GoalKeeper 小头；普通策略小头的
效果不能直接外推。

每个服务进程绑定一个公共任务；reset 只清空该任务的 episode 状态。当前服务按序执行推理，尚非 50 端口提交平台。`baseline` 使用固定前缀控制，并非 Comet 官方默认 temporal ensemble 的原样复现；必须单独记录这项差别。`shadow` 执行所有预测但始终选择候选 0；`rerank` 才启用筛选。K 次采样会推进主干随机数状态，不能将 shadow 与 K=1 称为逐动作相同的配对复现。

协议参考 [v3.9.2 network_utils.py](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.2/OmniGibson/omnigibson/eval/utils/network_utils.py)。OpenPI 图像缩放使用上游原函数，保持其预处理语义。运行时仅将 RGB/proprio 和公共任务文本送入模型；不接受 instance segmentation、物体状态或世界坐标。

## 汇总正式评测

`expected.json` 是选定官方 split 的 `[task_name, instance_id, rollout_id]` 列表，instance ID 必须来自官方解析器，不能假定就是 0–9。输出写到单独文件，不改动官方 JSON/视频。

```bash
python -m goalkeeper.cli report /path/official-json expected.json \
  --baseline /path/matched-baseline-json --output /path/summary.json
```

完整提交需遵循 [官方要求](https://behavior.stanford.edu/challenge/submission.html)。当前核心代码没有证明完整成功率提升或单张 24GB 可容纳组合模型；这两项都是进入正式打榜前的必测项。
