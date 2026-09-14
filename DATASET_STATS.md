# BEHAVIOR-1K 2026-challenge-demos 数据集统计

日期：2026-08-12
数据来源：本机暂存副本 `/var/tmp/behavior-1k/2026-challenge-demos` 的 meta 元数据实测
（`info.json`、`meta/episodes/chunk-*/*.parquet`、`tasks.jsonl`）。

---

## 总体规模

| 项 | 值 |
| --- | --- |
| 机器人 | R1Pro |
| 数据格式 | LeRobot v3.0 |
| 帧率 | 30 fps |
| 任务数 | 100 |
| 集数（episodes） | 20,000（**每任务恰好 200 集，完全均衡**） |
| 总帧数 | **210,916,774**（≈ 1,953 小时 ≈ 81 天演示） |
| chunk 大小 | 1000 |

## 每集长度分布（单位：帧）

| min | p25 | 中位 | 均值 | p75 | max |
| --- | --- | --- | --- | --- | --- |
| 148 | 6,604 | 10,051 | 10,546 | 13,507 | 45,761 |

换算时长（30 fps）：中位 **5.6 分钟**、均值 5.9 分钟、最长 25.4 分钟/集。

## 每任务集数分布

100 个任务，每个任务 min = median = mean = max = **200 集**，合计 20,000——完全均衡。

## 观测与动作维度

- `action` float32 **[23]**
  - base `[0:3]`、torso `[3:7]`、left_arm `[7:14]`、left_gripper `[14:15]`、
    right_arm `[15:22]`、right_gripper `[22:23]`
- `observation.state` float32 **[61]**
- RGB 相机 3 路：
  - `zed_link_camera_0`（头部）**720×720×3**
  - `left_realsense_link_camera_0`（左腕）**480×480×3**
  - `right_realsense_link_camera_0`（右腕）**480×480×3**
- 其他特征：`next.reward[1]`、`next.terminated[1]`、`next.truncated[1]`、
  相机外参 `observation.robot2cam_pose.*[7]`、`timestamp`、`frame_index`、
  `episode_index`、`index`、`task_index`

## 本机磁盘占用（约 1003G）

| 内容 | 大小 | mp4 数 |
| --- | --- | --- |
| zed 头部 RGB | 531G | 3,050 |
| left realsense 腕部 RGB | 202G | 1,117 |
| right realsense 腕部 RGB | 202G | 1,119 |
| data（parquet：state / action） | 70G | 955 |
| meta | 6.5M | — |

## 任务示例（100 个中前 8）

`turning_on_radio`、`picking_up_trash`、`putting_away_Halloween_decorations`、
`cleaning_up_plates_and_food`、`can_meat`、`setting_mousetraps`、`hiding_Easter_eggs`、
`picking_up_toys`

## 测试（评估）情况

**关键点：released demo 数据集只有 `train` 一个 split（全部 20000 集，`splits: {train: 0:20000}）`，
不含任何测试轨迹。** 因此测试不是"在留出数据上算误差"，而是把训练好的策略放进
OmniGibson / Isaac Sim 里做**闭环 rollout**，在与训练不同的**场景实例**上跑。

- **测试实例来自独立资产包**，不在 demo 数据集内：
  `<DATA_PATH>/2026-challenge-task-instances/`（场景 + 物体摆放配置）。
- **三个评估 mode**（`omnigibson/eval/eval.py`）：`train` / `public_test` / `hidden_test`。
- **实例编号**（`eval_utils.py`）：`TEST_INSTANCE_IDS = 301..340`，每任务共 40 个测试实例：
  - `public_test` = **301–320**（20 个，公开）
  - `hidden_test` = **321–340**（20 个，隐藏，用于最终排名）
- **q_score 指标**（`score_utils.py:compute_final_q_score`）：
  - 每个 rollout 产出 `q_score.final ∈ [0,1]`，即满足的 BDDL 目标条件比例（部分完成也给分）。
  - **成功率 task_sr** = 该任务下 `q_score==1`（全部目标达成）的实例占比。
  - **每任务 q_score** = 该任务所有实例的均值；**总体 q_score** = 100 个任务的均值。

## 注意事项

- 这是本机暂存副本，只放了 **3 路 RGB**；depth 相机（`observation.depth_linear.*`）
  故意未拷贝——π0.5 只用 RGB，省去约一半空间。
- 本机 `meta/info.json` 已剔除 depth 视频键（原件备份为 `info.json.orig`）；但
  `meta/episodes/*.parquet` 元数据里仍保留 depth 列引用，本机并无对应 depth 视频文件。
  不影响 RGB 训练。
- `/var/tmp` 为根 overlay 分区，**非持久化**；数据源头在 OSS，可用
  `scripts/stage_dataset_local.sh` 重新暂存。
