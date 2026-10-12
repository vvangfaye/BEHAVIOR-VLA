# SenseCore：2000 个完整 episode 的原生策略采集

本流程单独运行 100 个 task × public instance 301–320 × rollout 0，共 2000 个完整 episode。用户已确认这些 public 实例全部用于微调；每个实际入训练的实例写入 `training_seen=true`、`independent_evaluation_eligible=false`。本轮数据与原来的 benchmark 输出、低成功率 subtask 采集分别保存，`official_benchmark=false`。

**当前阶段（2026-10-12）：v2 pilot 已通过完整归档 SHA 与真实 LeRobot / PyAV 回读验收。两个启动前 DNS 失败的生产分片 1、4 已按用户指令补交 g1，旧 UID 与未产生轨迹的证据保存在 submission_history。其余分片已恢复按最新闲时容量提交；本线程每 5 分钟刷新容量并续交，原 admission deadline 后停止。** 实际进度以平台 UID、对象存储回执及完成报告为准。

验证沿用原 subtask 计划冻结的 **102 组训练实例**，这些实例 ID 均不超过 300；它们必须同时从原始 replay、旧数据恢复以及新采集训练索引中排除。已经用于微调的 301–320 实例，后续评测只能报告训练见过实例上的结果，不能作为独立验证集。`held_out_instances.json` 是显式的 `(task, instance_id)` 列表；三源训练中的每个索引必须绑定同一份划分。

## 1. 录制内容与训练选择

录制从环境 reset 后的初始观测开始，直到环境实际 `terminated` 或 `truncated`。每个成功执行的控制步保存其前置观测、实际执行的动作以及该步返回的终止标志，不能把尚未执行的预测动作当作训练目标。

原始 HDF5 包含完整 R1Pro proprioception、23 维动作、三路 onboard RGB、逐帧 subtask/source、连续 frame index、每步 `next.terminated` / `next.truncated`，以及终止后的最后观测。checkpoint、source manifest、run / attempt / task / instance / rollout 身份与各类 SHA 一起保存。仿真 reset 必须创建新的 attempt 文件，不拼接成同一个 episode。

| 原始结果 | 保存完整运行记录 | 默认进入行为克隆微调 |
| --- | --- | --- |
| 环境完整终止，task success 与 final goal success 一致且为 true，未截断 | 是 | 是，随后还需真实读取验收 |
| 环境终止但任务失败 | 是 | 否 |
| 达到环境限制而 truncated | 是 | 否 |
| 进程中断、文件未完成或 hash 不一致 | 保留已有诊断 / attempt | 否，不能计为完整 episode |

最终 q-score 只作为原始结果保留。不会因为整条任务的最终分数，给失败 episode 的前缀补造“成功 subtask”标签。

subtask 来自实际执行动作对应的策略响应，来源仅为 `generated`、`cached` 或缺失。它们是 **self-generated weak labels**，不是人工标注，也不是逐个 subtask 成功的证明。段表显式写入 `is_human_annotated=false`、`subtask_success_verified=false`，保留原始文本和 source 变化。

## 2. 冻结计划与容量提交

入口为 [sco/full_episode_collection.py](../sco/full_episode_collection.py)。它复用已验证的 checkpoint **1688977**、基础 source manifest 和 runtime，并冻结本轮 overlay 的逐文件 SHA。默认每个生产分片最多运行 5 个完整 episode；计划含 1 个单 episode pilot 和 400 个生产分片，共 401 个分片、2000 次完整运行机会。pilot 固定为 `turning_on_radio / 301 / rollout 0`，本身属于这 2000 个实例之一。

以下路径均应指向本轮实际冻结文件；`collection_root` 必须是新目录。SDK / SCO 环境、私有对象存储客户端及认证使用现有配置，不把凭据写入命令或日志。

```bash
python sco/full_episode_collection.py freeze \
  --root "$collection_root" \
  --ready "$verified_checkpoint_ready" \
  --held-out-instances "$validation_102_json" \
  --codec-wheels "$pinned_linux_h5py_wheels" \
  --run-id "$run_id"

python sco/full_episode_collection.py submit \
  --root "$collection_root" \
  --index 0 \
  --capacity "$fresh_spot_capacity_json"
```

固定 pilot 完成后，使用同一私有 AOSS 客户端环境验收；该命令会绑定精确 UID，并下载整份归档核对 SHA。验收通过后，每次用新容量观察提交有限批次，容量过期即停止本次提交，不自动扩大排队规模。

```bash
python sco/full_episode_collection.py accept-pilot --root "$collection_root"
python sco/full_episode_collection.py submit-batch \
  --root "$collection_root" --capacity "$fresh_spot_capacity_json"

python sco/restore_full_episode_collection.py inventory --root "$collection_root" \
  --output "$native_inventory_json"
python sco/restore_full_episode_collection.py restore --root "$collection_root" \
  --output "$successful_raw_root"
```

恢复器默认只取成功 episode，下载归档和 HDF5 全文件 hash、结果、身份及已验收 source 必须一致；它不会生成训练 ready 标记。`--include-failures` 仅用于另行恢复失败原始记录。

[sco/watch_full_episode_collection.py](../sco/watch_full_episode_collection.py) 可在提交机持续等待固定 pilot、自动执行上述验收，然后在容量记录仍新鲜时逐个提交剩余分片。容量过期时等待刷新，不以历史空卡数继续提交；pilot 失败、UID 变化或未知提交意图会停止展开。控制器有明确的 `--deadline`，不恢复失败实例，不占 GPU 等待。当前阶段见 `controller_status.json`，`all_shards_submitted` 只表示全部已提交，不代表采集完成。

本轮使用 `computing-cluster-5090-01g` 的闲时资源，单作业为 `n12lp.nn.i10a.1`：1 GPU、14 vCPU、60 GiB 内存；单分片的最长执行时间为 96 小时。已取消固定 64 张卡上限，提交额度由**当前闲时 GPU、CPU、内存三者的最小完整 worker 容量**决定，并扣除同池 owner 的待调度、启动中和未知提交意图。

容量观察最多有效 120 秒。创建前会再次检查余量；没有可用完整 worker、容量记录过期或身份不匹配时不提交。不要用一份旧容量记录循环创建全部作业。每个 `submissions/NNNN.json` 在云端创建前保存提交意图，创建成功后记录精确 UID；遇到不确定返回必须按该意图和 UID 核对，禁止直接覆盖回执重交。

生产分片要求 `pilot_accepted.json` 绑定同一 plan fingerprint，并有原始完整轨迹、真实 codec / reader、观测动作对齐的验收证据。**格式验收不以机器人任务是否成功为门槛**，不能挑选模型成功结果来替代固定 pilot。计划默认 `automatic_failure_recovery=false`，每个实例只有一次完整运行预算；失败或抢占不会自动扩展采集范围。

## 3. 原始轨迹对象存储与 AFS 用量

[sco/run_full_episode_collection.py](../sco/run_full_episode_collection.py) 把 RGB 历史写入 worker 的独立临时盘，启动时检查至少 128 GiB 可用空间。AFS 保存 compact queue、JSON、manifest、回执和日志，不累计全部原始 RGB。

每条完整 episode 在发布结果 JSON 前，先流式上传 tar 归档，内容为 `episode.hdf5` 与 `result.json`。AOSS 前缀为：

```text
behavior1k/full-episode-collection/<run_id>/<plan_fingerprint前12位>/
  episodes/<task>/<instance>/attempt-1.tar
  episodes/<task>/<instance>/attempt-1.json
  shards/<shard_index>.json
```

multipart 使用每片 Content-MD5，完成后核对对象字节数与 multipart ETag，并记录归档 SHA-256、HDF5 文件 SHA、实际数组 content SHA 和 subtask SHA。driver 还会回读 episode receipt 并核对身份、结果及 archive HEAD。上传成功、回执匹配后，生产 worker 才释放对应临时 HDF5；pilot 保留临时源以执行真实格式回读。上传失败或回执不一致时不释放未确认副本。

multipart / HEAD 证据不等于已经下载整个归档并完成 SHA 回读。后续恢复训练候选时，需要重新校验下载 tar 的 SHA、解包 HDF5 的 SHA 及其内部 content / label / outcome 哈希；只有验证后的成功原始文件才交给 exporter。失败轨迹保留在对象存储，默认不恢复到行为克隆训练目录。

三路 224×224 RGB 每帧约 0.43 MiB 未压缩；全量可能达到 TB 级。实际对象存储预算应按 pilot 的归档字节 / 帧数推算，训练导出仅暂存成功候选，并按分片增量处理。96 小时作业超时不是主训练的数据冻结时刻；训练可以先使用截止时已验收的数据。

## 4. 真实 codec pilot gate

[scripts/verify_full_episode_codec.py](../scripts/verify_full_episode_codec.py) 从固定 pilot 的真实 HDF5 中读取最多前 64 帧，使用真实 LeRobot writer 编码，再用真实 LeRobot / PyAV 解码；检查实际 action / state 对齐及三路 RGB 与源图像的误差。该探针支持完整但失败的 episode，因此不会把“格式可用”混同为“任务成功”。

```bash
python scripts/verify_full_episode_codec.py \
  --trajectory "$pilot_hdf5" \
  --output "$fresh_codec_probe_root"
```

pilot driver 已接入该检查，完成报告内嵌 `codec_verification`。probe 始终标记 `format_probe_only=true`、`ready_for_training=false`，不能把这个最多 64 帧的小切片当作完整成功训练轨迹。生产提交前必须核实报告与同一 pilot 的文件 SHA、content SHA、平台完成状态和 plan 身份一致，再生成验收回执。

原推理环境没有 `h5py`。codec pilot 使用冻结 overlay 中来自 PyPI 的 h5py 3.14.0 Linux wheels，按实际 Python ABI 选择并解包到该 worker 独立 scratch 目录，通过局部 `PYTHONPATH` 加载，不改共享推理环境。依赖文件也进入 source SHA 清单。后续运行完整 exporter 的环境需同时具备 h5py、pyarrow 和真实 LeRobot writer/reader。

提交器对启动前的私有配置 GET 使用最多 6 次有限重试，仅对网络故障及 429 / 指定 5xx 响应重试；403、SHA 不匹配或超出 32 MiB 上限立即失败。配置校验完成后 driver 只执行一次，不借此增加 episode 运行预算。

进入 driver 后的启动失败会向同一私有前缀的 `diagnostics/NNNN.json` 写入去除 URL 的错误记录，方便 worker 退出后排障。失败记录不会替代完成回执或 pilot 验收。

## 5. 完整成功 episode 导出与验收

[scripts/export_full_episode_collection.py](../scripts/export_full_episode_collection.py) 提供四个入口。`inspect` 可统计成功、失败和截断的完整原始记录；`export` 输入必须是显式选出的、唯一且 hash 验证通过的完整成功轨迹。不要将全部原始文件不加筛选地传给 `export`。

```bash
python scripts/export_full_episode_collection.py inspect \
  --trajectories "$raw_episode_1" "$raw_episode_2" \
  --held-out-instances "$validation_102_json"

python scripts/export_full_episode_collection.py export \
  --trajectories "$successful_episode_1" "$successful_episode_2" \
  --output "$fresh_native_dataset_root" \
  --repo-id "$native_repo_id" \
  --held-out-instances "$validation_102_json"

python scripts/export_full_episode_collection.py verify \
  --dataset-root "$fresh_native_dataset_root"

python scripts/export_full_episode_collection.py index \
  --dataset-root "$fresh_native_dataset_root" \
  --root-id "$unique_native_root_id" \
  --held-out-instances "$validation_102_json" \
  --output "$fresh_native_index_json"
```

默认 `--training-instance-min 301`，覆盖用户已批准的全部 public 实例。输出为独立 LeRobot v3 root，保存全部执行帧、`meta/collection_provenance.json`、`meta/subtask_segments.parquet`、canonical task prompt、`retained_episodes.txt` 及导出报告。

仅完成文件导出时 `ready_for_training=false`。`verify` 用真实读取器核对完整 state / action 数组、episode / frame / task 身份、source 与 outcome 哈希，并读取代表性 RGB 和 action horizon。通过后写入 `native_format_verification.json` 与 `_VERIFIED_FULL_EPISODE_EXPORT`；之后 `index` 才可生成训练索引。mock reader 的合同测试不会生成这个标记。

## 6. 32 步训练窗口与三源接入

原始保存和 LeRobot 导出均保留完整 episode，包括缺失 subtask 的帧。训练索引保留非空弱标签连续区间中的完整 32 步窗口。对于 `[start, end)` 区间，合法 anchor 为 `start ≤ frame ≤ end−32`，不跨 subtask 切换、空标签区间或 episode 终止，也不使用 episode 尾部 padding。

相邻同文本 subtask 的 `generated → cached` 来源变化不会切断训练窗口；source 的原始变化仍保留在段表与 HDF5。长度不足 32 的非空区间以及完全没有非空标签的 episode 仍在完整归档 / 导出中，但没有 joint 训练 anchor。不会用整任务、相邻标签或 EOS 目标补造缺失 subtask。

新 native 索引使用现有三源采样器的 `collected` 类别，同时在 provenance 中明确 `native_collection_kind` 与 `self_generated_weak_subtask`。旧数据恢复、原始 replay、新采集仍为三类；native episode 是 `collected` 中新增的独立 root。所有索引保持原始 root / episode / instance 身份，不能把 native 轨迹伪装成旧固定 subtask 采集 proof。

```bash
python scripts/prepare_subtask_mixture.py \
  --recovered-index "$recovered_index_json" \
  --replay-index "$replay_index_json" \
  --collected-index "$subtask_collection_index_json" \
  --collected-index "$fresh_native_index_json" \
  --recovered-weight "$recovered_weight" \
  --collected-weight "$collected_weight" \
  --replay-weight "$replay_weight" \
  --batch-size 128 \
  --end-step "$absolute_training_end_step" \
  --output "$fresh_mixture_json"
```

三个权重应沿用本次主训练冻结配方；重复 `--collected-index` 不会额外增加一份 `collected` 总权重。若需要继续原先的弱组加权，可额外传 `--weak-groups "$full299_campaign_json"` 与已确定的 multiplier。Native 标签可能与人工技能名称不同，未匹配的组保留默认组权重，不作未经核验的文本映射。

模型保持 joint subtask CE + action flow-matching，动作条件模式、action horizon 与归一化统计沿用初始化 checkpoint / 主训练合同。Nebula 16 卡入口已调用同一个 interval reader；新 root 在训练机器上必须可见，迁移路径通过 roots mapping 指定。启动前仍需真实 reader / token 预检及 GPU smoke，文件格式通过不等于模型已训练或成功率已提高。

如果训练已开始，增加数据要使用新 mixture phase，并通过 `--previous-mixture`、`--start-step` 保留已经消费的采样合同；不得直接覆盖旧 mixture 或索引文件。验证列表也必须保持一致，旧数据恢复与 replay 不得把 102 个验证实例重新放回训练。
