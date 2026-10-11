# Nebula 16 H200 subtask 主线训练

这是独立于旧 full100 / two_tasks 模板的入口：
`nebula/submit_subtask_mainline.sh`、`nebula/subtask_mainline.py` 和
`nebula/train_subtask_mainline.py`。默认拓扑固定为两节点、每节点八张 H200，
全局 16 卡、FSDP=8；global batch 是全局 microbatch，必须能被 16 整除。
它使用明确的 optimizer update 步数，不继承旧脚本的两个 epoch 或原始大数据步数。

本地 `prepare` 和 `submit --dry-run` 不需要 AI-Hub、OSS SDK、认证或者集群挂载，
不会提交云作业或下载数据。它们只能确认代码、索引和配置合约。
实际两节点通信、数据解码、前后向更新与 checkpoint 恢复，必须在已配置的 GPU 环境验收。

## 恢复旧数据和冻结三源

使用 Python 3.10+；索引准备需要 `pyarrow`，不会解码或重编码视频。
恢复指从之前整条丢弃的 episode 中重新使用已有可靠标签的区间，不给空白帧伪造标签。
对 `[start, end)` 区间，只采样 `start <= frame <= end - 32`，整个 action chunk
留在同一 subtask 内。原 episode ID、原 frame ID、30 Hz 时基均保留。

此前审计的 3,949 条被丢弃轨迹包含 48,483,795 个有标签帧；按 32 帧窗口保守计算，
候选 anchor 为 45,402,692 个。这是排除 held-out 实例之前的候选量，实际可训练数量
以索引生成回执为准。本次只准备代码，没有在本机恢复或转存这批完整数据。

以下命令在数据已经就绪的机器上执行，`/ready`、`/plans` 都须替换成实际路径。
原始 expert metadata 可能声明三路深度视频；先建立只复制 metadata 的 RGB 视图，
`data/` 和 `videos/` 引用原文件，保留 source 数据不变。必须先生成视图，再冻结索引；
冻结后不可再运行原地修改 metadata 的 `strip_depth_from_info.py`。

```bash
python scripts/prepare_subtask_rgb_view.py \
  --dataset-root /ready/datasets/expert-original \
  --output /ready/datasets/expert-rgb

python scripts/prepare_subtask_recovery.py \
  --dataset-root /ready/datasets/expert-rgb \
  --labels /ready/datasets/expert-rgb/meta/subtask_segments.parquet \
  --episode-audit-report /plans/original-subtask-audit.json \
  --held-out-instances /plans/full299/campaign.json \
  --source-kind recovered --root-id expert --repo-id behavior-1k/2026-challenge-demos \
  --output /plans/recovered-v1.json

python scripts/prepare_subtask_recovery.py \
  --dataset-root /ready/datasets/expert-rgb \
  --labels /ready/datasets/expert-rgb/meta/subtask_segments.parquet \
  --episodes-file /plans/replay-episodes.txt \
  --held-out-instances /plans/full299/campaign.json \
  --source-kind replay --root-id expert --repo-id behavior-1k/2026-challenge-demos \
  --output /plans/replay-v1.json
```

`original-subtask-audit.json` 是原 `check_subtask_data` 回执，恢复选择其中
`dropped_episode_indices`。若 annotations 存在外部，`--labels` 指向原标签 parquet。
`replay-episodes.txt` 每行一个原 episode ID，使用有完整标签、覆盖要保持能力的固定小池，
避免把全部原始数据搬入新训练工作集。recovered 和 replay 可以共享同一个物理 root，
必须使用同一 `root-id`；两者 episode 集合不能重叠。

每批新成功轨迹使用已有导出器验收通过的 LeRobot 根目录，不经 expert RGB 视图工具：

```bash
python scripts/prepare_subtask_recovery.py \
  --dataset-root /ready/datasets/collected-v1 \
  --labels /ready/datasets/collected-v1/meta/subtask_segments.parquet \
  --episodes-file /plans/collected-v1-episodes.txt \
  --held-out-instances /plans/full299/campaign.json \
  --source-kind collected --root-id collected_v1 --repo-id collected_v1 \
  --output /plans/collected-v1.json

python scripts/prepare_subtask_mixture.py \
  --recovered-index /plans/recovered-v1.json \
  --collected-index /plans/collected-v1.json \
  --replay-index /plans/replay-v1.json \
  --weak-groups /plans/full299/campaign.json --weak-group-multiplier 3 \
  --recovered-weight 0.8 --collected-weight 0.1 --replay-weight 0.1 \
  --batch-size 128 --seed 42 --end-step 5000 \
  --output /plans/mixture-v1.json
```

每个 collected 导出目录须保留 `meta/collection_provenance.json`、`_EXPORTED`、
`_VERIFIED_EXPORT`、`format_verification.json` 和 `export_report.json`；索引会验证真实 reader
验收、成功轨迹来源及帧数对齐。不同导出目录使用不同 root ID，允许各自从 episode 0 开始；
内容 hash 重复的轨迹会被拒绝，不能靠换目录重复计入。多目录可重复传入
`--collected-index`，不需要合并或重编号视频。

采样依次选择 **source → task/subtask 组 → instance → 区间 → 合法 anchor**。
默认三源比例 80/10/10 是每次抽样的概率，和三源帧数、目录数无关；短 batch 不保证精确配额。
299 个弱组在各源已有的数据范围内权重为 3，其余组为 1，不虚构不存在的成功轨迹。
因此大旧池不会按总帧数淹没新采集数据，长轨迹也不会自动占据大部分样本。
三源必须使用相同 held-out task/instance 列表，并排除对应实例。

后续累计足够的新成功轨迹后，新增冻结 phase 可以调整为 45/45/10；以下 `5000`
必须来自已完成 checkpoint 的 `assets/subtask_mixture.json` 中的 `step`，不能猜测：

```bash
python scripts/prepare_subtask_mixture.py \
  --previous-mixture /plans/mixture-v1.json --start-step 5000 --end-step 10000 \
  --recovered-index /plans/recovered-v1.json \
  --collected-index /plans/collected-v1.json --collected-index /plans/collected-v2.json \
  --replay-index /plans/replay-v1.json \
  --weak-groups /plans/full299/campaign.json --weak-group-multiplier 3 \
  --recovered-weight 0.45 --collected-weight 0.45 --replay-weight 0.1 \
  --batch-size 128 --seed 42 --output /plans/mixture-v2.json
```

保留所有旧索引；新 manifest 保存已消耗前缀，再追加未来数据。例子中的 5,000/10,000
是展示更新步数语义的占位预算，最终按 H200 实测吞吐和评测预留时间设置，不等同于完整 epoch。
阶段切换需要停稳旧作业再 resume；当前入口不会在训练中悄悄替换正在读取的数据池。

## 输入与配置

先通过有效区间工具生成 recovered、collected、replay 索引及三源 mixture manifest。
所有索引必须是带 fingerprint 的 `b1k_valid_intervals`，三源使用
`b1k_three_source_mixture`。prepare 调用真实 `MixturePlan.load`，核验文件 SHA256、
索引 schema、留出实例、重复数据及采样权重。每个正权重源必须有可用数据，
不要用空 collected 源占据正采样比例。

作业配置模板见 `nebula/subtask_mainline.example.json`（队列、卡型名、runtime 名和大文件路径须替换成实际配置）：

```json
{
  "schema_version": 1,
  "run_id": "subtask_mainline_v1",
  "mode": "train",
  "resume": false,
  "runtime_python": "/ready/environment/bin/python",
  "openpi_data_home": "/ready/openpi-cache",
  "tokenizer_sha256": "REPLACE_WITH_VERIFIED_TOKENIZER_SHA256",
  "output_root": "/shared/mainline/subtask_mainline_v1",
  "data_roots": {
    "expert": "/ready/datasets/expert-rgb",
    "collected_v1": "/ready/datasets/collected_v1"
  },
  "resources": {
    "queue": "REPLACE_WITH_ACTUAL_H200_QUEUE",
    "gpu_card_type": "REPLACE_WITH_ACTUAL_SCHEDULER_CARD_TYPE",
    "algo_name": "REPLACE_WITH_ACTUAL_RUNTIME",
    "cpu": 3200,
    "memory": 256000
  },
  "checkpoint": {
    "step": 1375000,
    "root": "/ready/checkpoints/1375000",
    "manifest_sha256": "REPLACE_WITH_VERIFIED_SHA256",
    "norm_relative_path": "assets/ORIGINAL_REPO_ID/norm_stats.json",
    "norm_sha256": "REPLACE_WITH_VERIFIED_SHA256"
  },
  "seed": 42,
  "global_batch_size": 128,
  "train_steps": 5000,
  "save_interval": 100,
  "keep_period": 500,
  "num_workers": 4,
  "learning_rate": {
    "peak": 0.00001,
    "floor": 0.000001,
    "warmup_steps": 100,
    "decay_steps": 5000
  },
  "rendezvous_timeout": 600,
  "coordinator_port": 12355
}
```

`data_roots` 必须恰好覆盖全部索引的 root_id；路径是 worker 已就绪的本地或共享挂载路径。
索引保留原 dataset_root 和原始 bytes，worker 通过 roots 映射定位现有文件。
prepare 只把 manifest 的 `indexes.path` 改成包内相对路径，保留索引原文、SHA256、
group_weights 与分阶段配比，不改变 episode/frame 身份。缺少映射或证据会停止。

本入口不提供 AOSS→Nebula 传输、1TB 数据同步或 conda 环境恢复。`runtime_python`
必须已经包含实际 OpenPI / LeRobot / JAX / CUDA 依赖；数据、1375000 checkpoint
及原 normalization 必须已挂载。`openpi_data_home` 指向已就绪的 OpenPI 缓存，
其中 `big_vision/paligemma_tokenizer.model` 必须通过 `tokenizer_sha256` 校验；worker
显式设置 `OPENPI_DATA_HOME`，避免 tokenizer 在离线环境尝试下载。tokenizer hash
进入稳定续训合约，缓存物理路径可以重新定位。输出目录须两节点共享、可写，支持独占创建和原子替换。
CPU/memory 为平台 cluster JSON 单位，每 worker 的 GPU 请求是 800（八张卡）。
真实硬件预检要求八张 GPU 均报告 H200 且至少 130000 MiB，不接受 L20/L20Z 名称当作 H200 证据。

## 本地准备与检查

```bash
bash nebula/submit_subtask_mainline.sh prepare \
  --config /path/job.json --mixture /path/mixture.json \
  --output /path/prepared-mainline-v1

bash nebula/submit_subtask_mainline.sh submit \
  --prepared /path/prepared-mainline-v1 --dry-run
```

`review.json` 列出资源、步数、依赖路径和 `cloud_submitted: false`；
`bundle/` 包含训练源码、全部小索引、roots、manifest 和作业配置。
`bundle_manifest.json` 为每个文件记录 SHA256。prepare 不复制视频、参数或密码，
也不读取旧 `nebula/config.sh` 的默认 L20 队列。使用新的本地 prepared 目录，避免覆盖提交证据。

用户核对资源配置和路径后，显式提供 `AI_HUB_CLI`、`AI_HUB_TOKEN`、
`AI_HUB_NAMESPACE`、`AIHUB_MODEL_NAME`、`OSS_ACCESS_ID`、`OSS_ACCESS_KEY`、
`OSS_BUCKET`、`OSS_ENDPOINT`。这些认证值只用于真正的 submit，不写入 bundle。
删去 `--dry-run` 才调用 `AI_HUB_CLI train mdl`。入口使用配置中的队列、卡型和 runtime，
固定请求两 worker、每 worker 八卡，关闭平台自动 failover。

提交前独占创建本地 `submission_intent.json`，CLI 结果写入经过密钥脱敏的日志和 receipt。
即使 CLI 超时或进程中断，已有 intent 也会阻止盲目重提；先按原作业身份核对平台状态。
这不是跨机器的云端全局锁，不要复制一个 prepared bundle 到别处后重复 submit。

## Worker 验收、保存和续训

每个 rank 验证 checkpoint `_READY`、完整参数清单、23 维**转换后的 normalization**、
`task_and_subtask` 条件和 frozen bundle。原始 observation.state 保留数据声明的 57/61 等维数；
真正的 B1K transform 才把它提取成模型使用的 23 维。

两 rank 通过新 attempt 的共享 rendezvous 发现彼此，核验两台不同主机、相同代码/索引/norm。
实际 runtime 调用 `create_dataset`，从每份索引所在 root 解码真实样本，验证三路 RGB、
完整 raw state、32×23 action chunk、episode/frame/task 身份。`mode: preflight` 在这一步结束，
只发布 `gpu_training_verified: false`；它不是 GPU 前后向或网络 collective 的验收。
普通 `train` 也必须先通过该读取检查，然后直接调用 `train_b1k.py pi05_b1k`。
JAX 分布式入口再确认实际 2×8 拓扑和 mesh，trainer 保存后执行 collective checkpoint restore。

新实验用 1375000/params 初始化并创建新 optimizer；保留原 norm、joint
`task_and_subtask`、max token 512、action horizon 32。周期性保存完整 optimizer
和 sampler 消耗合约；`keep_period` 默认是五倍 save_interval，避免每个短间隔都永久保留。
每次成功记录在 `attempts/<attempt_id>/success.json`，`latest_success.json` 是可原子更新的指针。

续训必须确认旧云作业和 worker 已终止，准备新 bundle/attempt，保留同一 run_id、output_root，
设 `resume: true`。launcher 不再传初始 weight-loader，trainer 恢复完整 optimizer、step 和
确定性采样位置。baseline/code/模型/batch/seed/optimizer 的稳定合约必须相同。
这里保证 optimizer、EMA、step 和抽样索引的连续恢复；原 loader 的 worker 数据增强 RNG
不在 checkpoint 中，不承诺训练结果逐位复现。
**`learning_rate.decay_steps` 独立固定，不随新 train_steps 自动增长**，避免悄悄改掉历史学习率。

允许增加总步数或修改尚未消耗的未来 phase，也允许为未来 phase 加入新冻结采集批次。
worker 会读取最新完整 checkpoint 的 `assets/subtask_mixture.json`，用真实
`compare_consumed_contract` 规则拒绝改动已经消耗的样本前缀。旧 attempt 的合约与成功记录保持可追溯。
恢复到只含初始权重、缺 optimizer、缺消耗合约或只有半写入文件的目录会失败。

```bash
PYTHONPATH=openpi/src python -m pytest --confcutdir=openpi/src/openpi/training \
  openpi/src/openpi/training/recovery_index_test.py \
  openpi/src/openpi/training/subtask_mixture_test.py \
  openpi/src/openpi/training/subtask_mixture_data_test.py \
  openpi/src/openpi/training/subtask_data_test.py \
  scripts/test_prepare_subtask_mixture.py scripts/test_prepare_subtask_rgb_view.py \
  nebula/test_subtask_mainline.py -q
bash -n nebula/submit_subtask_mainline.sh
```

这些 CPU 测试覆盖本地防重、dry-run、密钥脱敏、16 卡 batch、真实索引校验、硬件识别、
输入元数据变动、参数校验、INIT/RESUME 区分、未来 phase 续训和固定 LR。
真实 H200 训练速度、显存、两节点通信及训练效果仍由集群实际运行验收。

2026-10-11 本地验收：上述 **128 项 CPU 测试通过**（包括既有 subtask loader 回归测试），
Python 3.12 编译、shell 语法和 `git diff --check` 通过。CPU reader 集成测试使用 fake
LeRobot 与真实 parquet metadata；另用已有 native export 的 metadata 验证了 collected
索引兼容性。本机没有 JAX/LeRobot/GPU runtime，未执行完整 tokenizer/模型前后向，
没有提交训练或转移完整数据；实际环境仍须完成 worker preflight 和 16 卡短训练/恢复验收。
