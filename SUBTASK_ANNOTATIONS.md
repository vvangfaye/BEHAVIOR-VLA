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

> 重要：当前 π0.5 训练与推理**都没用**这些 sub-task，只喂了整任务级 prompt
> （`prompt_from_task=True`，`tasks.jsonl` 每任务一句话）。见文末"如何接入"。

---

## 实例：`task-0000/episode_00000010.json`

任务 **turning on radio**，全长 1776 帧（≈ 59s，30 fps），切成 **4 个子任务**：

| # | 帧区间 [start, end] | 时长 | skill_id | 描述 | 类型 | 目标物体 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | [0, 265] | 8.8s | 1 | move to | navigation | radio_89 |
| 1 | [265, 1162] | 29.9s | 2 | pick up from | uncoordinated | 从 coffee_table 拿起 radio_89 |
| 2 | [1162, 1434] | 9.1s | 67 | press | coordinated | 按 radio_89（开机键） |
| 3 | [1434, 1776] | 11.4s | 3 | place on | uncoordinated | 把 radio_89 放回 coffee_table |

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
| `frame_duration` | **[起始帧, 结束帧]——把子任务精确对齐到帧** |
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

## 如何接入（sub-task 条件化）

1. 把 `annotations/` 暂存本地。
2. 遍历 20,000 个 json，按每段 `frame_duration` 把 `skill_description`（+ `manipulating_object_id` 物体名）
   展开成一列 **per-frame 语言**（每帧对应"当前正在做的子任务"）。
3. 训练/推理改喂当前 sub-task 而非整任务：
   - π0.5：openpi 数据管线支持逐帧 `prompt`，改造主要在数据侧（生成 per-frame 语言列）；
     这与 π0.5 本身的分层设计（高层预测 sub-task 语言、低层动作专家执行）一致。
4. 推理时若无高层规划器，可先用固定/启发式子任务序列，或再训一个高层 sub-task 预测头。

> 动机：当前把 ~5 分钟、约 10 个子步骤的长程任务当成"整任务→扁平动作"，模型分不清当前处于哪一步，
> 与评估中观察到的"乱走 + 手臂几乎不动"相互印证（见 `EVAL_FINDINGS.md` / `EVAL_SUMMARY.md`）。

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

### openpi 侧接入

- 新增 `openpi/src/openpi/transforms.py::PromptFromSubtask`：持有
  `episode_index -> (starts 数组, texts 列表)`，用 `np.searchsorted` 做区间查找；
  未命中时**回退整任务 prompt**，训练不会因个别未标注帧崩掉。
- `create_b1k_dataset` 在 `subtask_prompts_path` 有值时用它替换 `PromptFromLeRobotTask`。
- 配置与 CLI 开关（已验证存在）：
  ```
  --data.subtask-prompts-path {None}|STR     # 段表路径，默认 None = 保持原整任务 prompt
  --data.frame-stride INT                    # 帧步长采样，默认 1
  ```

### 边界验证（episode 0，对应标注 `task-0000/episode_00000010.json`）

| 帧 | 查表结果 |
| --- | --- |
| 0, 264 | `move to radio` |
| 265, 1161 | `pick up from radio to coffee table koagbh` |
| 1162 | `press radio` |
| 1434, 1955 | `place on radio to coffee table koagbh` |

与标注段 `[0,265) [265,1162) [1162,1434) [1434,1776)` 逐帧吻合；1776→1955 为尾部继承。
未知 episode 正确回退到整任务 prompt。

### 待办

1. **闭环推理仍缺高层**：`openpi` 中搜不到任何 sub-task/high-level 机制，所以闭环时没有东西
   产出"当前该做哪个子任务"。需要二选一：训练一个高层子任务预测头，或先用规则/脚本化子任务序列。
2. 建议先做**开环验证**：用 GT 子任务喂入，量 `left_arm/right_arm` 的开环 L1 能否跑赢 `hold`，
   再决定是否投入高层头的训练。
