# BEHAVIOR-1K sub-task（技能）标注说明与实例

日期：2026-08-12
数据来源：数据集自带的 `annotations/` 目录（OSS 源）
`.../behavior-1k_2026-challenge-demos_20260728235542/annotations/`

---

## 概览

数据集在 demo 之外，**自带逐帧的 sub-task（技能）分段标注**：

- 覆盖 **100 个任务、20,000 个 episode**（每任务 200 个 json：`annotations/task-XXXX/episode_XXXXXXXX.json`）。
- 另有两个汇总表：`skill_summary.csv`（技能词表 + 统计）、`skill_type_summary.csv`（技能类型统计）。
- 共约 **406,341 个技能片段**，**35 种技能**，3 大类型。

> 本文保留标注结构和历史数据统计，计数不是当前过滤后训练集的大小。
> 当前 π0.5 支持普通动作训练和 joint 联合训练，
> 请以 [subtask 训练与推理指南](docs/subtask_training.md) 为准。

---

## 实例：`task-0000/episode_00000010.json`

任务 **turning on radio**，全长 1776 帧（≈ 59s，30 fps），切成 **4 个子任务**：

| # | 帧区间 [start, end) | 时长 | skill_id | 描述 | 类型 | 目标物体 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | [0, 265) | 8.8s | 1 | move to | navigation | radio_89 |
| 1 | [265, 1162) | 29.9s | 2 | pick up from | uncoordinated | 从 coffee_table 拿起 radio_89 |
| 2 | [1162, 1434) | 9.1s | 67 | press | coordinated | 按 radio_89（开机键） |
| 3 | [1434, 1776) | 11.4s | 3 | place on | uncoordinated | 把 radio_89 放回 coffee_table |

语义即：**走到收音机 → 拿起来 → 按开机 → 放回桌上**。

### 单个片段的字段 schema（以 segment 1 为例）

```json
{
  "skill_idx": 1,
  "skill_id": [2],
  "skill_description": ["pick up from"],
  "object_id": [["radio_89", "coffee_table_koagbh_0"]],
  "manipulating_object_id": ["radio_89"],
  "memory_prefix": [],
  "spatial_prefix": [],
  "frame_duration": [265, 1162],
  "mp_ef": [],
  "skill_type": ["uncoordinated"]
}
```

| 字段 | 含义 |
| --- | --- |
| `skill_idx` | 该 episode 内子任务序号（从 0 起） |
| `skill_id` / `skill_description` | 技能编号 / 文本描述（与 `skill_summary.csv` 对应） |
| `object_id` | 涉及的对象（含被操作物 + 参照物，如目标家具） |
| `manipulating_object_id` | 实际被机器人操作的物体 |
| `frame_duration` | **[起始帧, 结束帧)，结束帧不包含在当前段内** |
| `skill_type` | `navigation` / `uncoordinated` / `coordinated` |
| `memory_prefix`/`spatial_prefix`/`mp_ef` | 记忆前缀 / 空间关系前缀 / 运动基元·末端信息（多为空） |

顶层还有 `task_name`、`meta_data.task_duration`、`meta_data.valid_duration`。

---

## 技能词表统计（`skill_summary.csv`，前几名）

| skill_id | 描述 | 片段数 n | 出现任务数 | 均时长 |
| --- | --- | --- | --- | --- |
| 1 | move to | 143,566 | 100 | 18.64s |
| 2 | pick up from | 93,092 | 99 | 17.47s |
| 4 | place in | 43,743 | 61 | 13.31s |
| 3 | place on | 34,580 | 71 | 13.01s |
| 90 | push to | 13,456 | 49 | 25.48s |
| 10 | open door | 12,810 | 41 | 29.48s |
| 91 | place on next to | 10,482 | 24 | 12.91s |

共 **35 种技能**。

## 技能类型统计（`skill_type_summary.csv`）

| 类型 | 片段数 n | 均时长 | min | max |
| --- | --- | --- | --- | --- |
| navigation | 139,556 | 19.01s | 0.1s | 298.2s |
| uncoordinated | 230,674 | 15.86s | 0.03s | 225.0s |
| coordinated | 36,111 | 18.0s | 0.1s | 198.87s |

合计约 **406,341** 个片段。

---

## 已实现的管线（2026-08-20）

### 对齐规则（已用真实数据验证）

标注文件与数据集 episode **没有显式 id 关联**，需按"任务内排序配对"建立：

- 同一任务下，标注文件按文件名数字升序（编号是原始采样 id，如 10..3000），
  数据集 episode 按 `episode_index` 升序，**按序一一对应**。
- 证据：`corr(task_duration, length) = 0.98`（随机配对仅 0.09），多任务复核 corr 0.98–1.00。
- 数据集 episode 与标注**同起点**（frame 0 == frame 0），且只会更长：
  `length - task_duration >= 0` 在 20000 集中成立 19999 例，多出的是尾部 settle 帧。
- 因此标注的 `frame_duration [start,end)` 可**直接当作数据集局部帧区间**；
  `task_duration` 之后的尾帧继承最后一个技能。

### 生成脚本与产物

```bash
openpi/.venv/bin/python scripts/gen_subtask_prompts.py \
    --dataset /var/tmp/behavior-1k/2026-challenge-demos
```

产物 `meta/subtask_segments.parquet`，**行程编码（RLE）的段表**：
`episode_index | start | end(不含) | subtask`

| 指标 | 值 |
| --- | --- |
| 覆盖 | **20000/20000 episode**，100/100 任务 |
| 段数 | 388,409（平均 19.4 段/episode） |
| 文件大小 | **1.95 MB** |
| 不同子任务串 | 1615 |
| 尾部填充帧占比 | 1.1%（2,413,661 / 210,916,774） |

**为什么不用逐帧表**：逐帧版是 2.1 亿行、**402 MB**，`from_parquet` 建表要 **355 秒**且吃数十 GB
内存——多 dataloader worker 会各建一份直接 OOM。段表版 **1.95 MB / 1.04 秒**（约 340 倍提速），
每个 worker 都能轻松持有。

### 当前训练与推理入口

上面的 RLE 统计对应 legacy 标签格式。生成脚本另支持 `--label-format role_aware_v1`，
记录对象角色、原始标注来源及 `is_annotated`，保留相邻同文本的不同源片段边界。
它会产生不同的段表，不能直接套用上面的历史计数。

Joint 高层保留完整任务，以 GT subtask 计算文本 CE；当前启动脚本默认
`SUBTASK_ACTION_CONDITIONING=subtask_only`，低层只用观测和 GT subtask 学习动作。
推理先生成 subtask，再生成动作。旧 `task_and_subtask` 模式仍保留用于旧 checkpoint。

- 数据预检：`bash scripts/train_pi05_subtask_b1k.sh check`。
- 联合训练：`bash scripts/train_pi05_subtask_b1k.sh full`。
- 闭环评测：设置 `SUBTASK_PREDICTION=1`，运行 `scripts/eval_b1k.sh`。
- 联合训练默认以 `drop_episode` 过滤包含空标注的 episode；这与历史上仅替换 prompt、
  缺标注时回退整任务的条件化训练不同。

完整参数、checkpoint 配置和标注预检说明见
[当前指南](docs/subtask_training.md)。
