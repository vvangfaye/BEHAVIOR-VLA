# π0.5 BEHAVIOR 评测摘要

历史实验：50k 与 270k checkpoint。详见 [EVAL_FINDINGS.md](EVAL_FINDINGS.md)。这些是已有实验记录，当前工作区未重新运行评测，视频和原始结果也未随源码提供。

| checkpoint | rollout 数 | 任务数 | 完整成功数 | 平均 q_score | 非零分 rollout |
| --- | --- | --- | --- | --- | --- |
| 50k | 475 | 99 | 0 | 0.0045 | 14 |
| 270k | 488 | 99 | 0 | 0.0145 | 31 |

在共同的 471 个任务/实例对上，平均 q_score 从 0.0046 提升到 0.0150，约 3.3 倍；仍未完整完成任务。

- q_score 是目标条件完成比例，非零分不代表完整成功。
- 总体开环 L1 容易被夹爪维度主导，应同时看左右臂分组误差、hold 基线和闭环结果。
- `agent_distance` 是累计路径长度，不能直接解读为净位移或有效操作。历史“手臂几乎不动”的说法已在后续调查中纠正。

评测入口：`scripts/eval_b1k.sh`（单任务）、`scripts/eval_sweep.sh`（批量）、`scripts/eval_openloop.py`（开环）、`scripts/summarize_closed_loop.py`（结果汇总）。
