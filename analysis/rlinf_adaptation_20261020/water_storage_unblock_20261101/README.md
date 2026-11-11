# 搬运饮水：归档与续跑

用户授权：2026-11-01，将不用的旧实验数据归档到现有私有 AOSS，释放空间后自动继续搬运饮水采样和 PPO。

## 固定范围

- 精确源清单：`plan.json`，29 个文件，83.514 GiB 分配空间。
- 保留当前 1688977 原模型、转换模型、仿真环境、代码、已完成基线及其他运行任务的依赖。
- 原实验不变：32 条训练轨迹，17 次 PPO 优化，24 个固定评测回合。
- 仍使用原容量门槛：64 GiB 本实验预算，加 20 GiB 余量；不得用降低门槛代替释放空间。

## 归档位置与证据

桶：`260010165`。

- 首次归档：`behavior1k/storage-reclaim-20261101/water-unblock-a1/`。
- 单文件重试：`behavior1k/storage-reclaim-20261101/water-unblock-a2/`；仅处理首次因 DNS 错误保留的索引 0，不与首次作业重复写入源文件。
- 大分段重试：`behavior1k/storage-reclaim-20261101/water-unblock-a3/`；先停止 a2 并核实 worker 退出，再对相同文件使用 128 MiB 有序分段，减少请求开销。a2 未释放源文件。
- 顺序回读重试：`behavior1k/storage-reclaim-20261101/water-unblock-a4/`；大文件分段回读实测仍慢，因此停止 a3 并核实退出，恢复顺序完整回读并提高临时 GET 错误重试次数。a3 未释放源文件。
- 分块新建归档：`behavior1k/storage-reclaim-20261101/water-unblock-a5/`。已有大对象读取持续慢，因此停止 a4 后，将同一源文件保存为 150 个不超过 256 MiB 的新对象；每块完整回读并按顺序计算整文件 SHA-256，再独立复核源文件。AOSS 原备份不改动。
- 索引 0 已有的对象只读复用：`models/solarwm/2026-10-23/sgf-stage2-81f/model.pt`。
- 每个归档版本包含 `plan.json`、`metadata.tar.gz`、`metadata.COMPLETE.json`；每个成功文件包含 `receipts/NNNN/manifest.json`、`complete.json`。
- AFS 原路径旁保留 `文件名.aoss.json` 恢复指针。未归档的小文件和目录保留。

每个源文件仅在完整 AOSS 回读 SHA-256 与源文件一致、源身份及内容复核一致、manifest 和 COMPLETE 回读成功、实时依赖审核允许后释放。重试的分段回读覆盖整个对象，按原顺序计算完整哈希，不是抽样验证。

## 恢复方法

先确认目标空间，再恢复到一个新的空目录，避免覆盖当前训练数据。

本次是按清单归档文件，不是整棵目录迁移。未入选的大文件仍在 AFS；要恢复完整目录时，先复制这些保留文件，再补回归档清单中的文件。

1. 读取该版本 `metadata.COMPLETE.json`，下载 `metadata.tar.gz` 并验证其中的长度和 SHA-256。
2. 检查 tar 条目均为安全相对路径，恢复小文件；逐项核对 `ARCHIVE_METADATA_MANIFEST.json` 中的哈希。
3. 从每个已发布 COMPLETE 对应的 manifest 读取实际对象 key、源相对路径、大小和 SHA-256。索引 0 使用上述已有对象，其余使用 manifest 指向的新对象。
4. 将大文件下载到目标中的源相对路径，完整核对大小与 SHA-256。若 `storage_kind` 为 `ordered_parts`，逐块验证 `parts` 中的大小与 SHA-256，严格按 `number` 拼接，再验证整文件大小与 SHA-256；`key` 是分块前缀，不能当作一个完整文件下载。
5. 按 metadata manifest 恢复符号链接，并检查链接指向的模型文件已恢复。

各次尝试的 `archive_status.json` 区分已释放、处理中和保留源文件；`continuation_status.json` 中的真实个人配额判断决定何时允许原搬运饮水控制器继续。最终 `restore_index.json` 汇总各次成功文件和对应完整性回执。作业成功状态不能替代逐文件验证或真实配额检查。

## 验证

`test_worker.py` 覆盖：正常归档、已有对象只读复用、读回内容损坏、归档期间源变化、保护目录。`test_parts.py` 另覆盖分块拼接哈希、分块损坏、源变化和保护目录。共九项通过。分段回读另验证了全字节覆盖与完整哈希不匹配时拒绝通过。

## 防止采样后再次卡在容量检查

原容量检查每个阶段都要求额外空闲 64 + 20 GiB，会把本任务已完成轨迹的占用再次预留。`capacity_accounting_fix.json` 记录了修复：64 GiB 总任务预算与 20 GiB 余量不变，只从剩余预留中扣除已确认写入的本任务数据。

一个无 GPU 的有界 CPU 作业仅统计有完成回执的轨迹，以及有 `ready.json` 的 checkpoint；不计未完成文件、软链接或可能共享到外部的硬链接。计量比配额快照更新时不抵扣，避免多算可用空间。排队期间仍适用原来的六小时准入有效期。八项容量测试通过。

水任务执行文件仅 `capacity_admission.py` 改动，更新已记录在执行 manifest；模型、奖励、优化器、随机种子、轨迹数和评测实例未变。原始 manifest 保存在本目录，基线结果继续复用。
