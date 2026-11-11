# 前缀缓存实测（prefix_cache_probe_20261103）

**授权**：用户 2026-11-03：“前缀缓存的实测还在等你确认。我确认”。方案为 1 张闲时卡、约 20 分钟、上限 1 GPU·小时。

## 目的

多轮 learner（`rl_multiround_20261102`）有 80% 的时间花在 KL 检查上。粗算每个 chunk 前向约 99% 的计算量在冻结的 VLM 前缀上：3×256 个图像 token 加补齐到 512 的文本 token，经过 SigLIP + Gemma-2B。一轮之内这部分是常量，但现在每个训练步、每次 KL 检查都要重算。本实测用真实模型和真实 rollout 回答下面几个问题：

1. 前缀和后缀（action expert，1 个去噪步）各占多少时间？
2. 用缓存的前缀以后，KL 前向和梯度步能快多少？
3. 缓存路径算出的 logprob 和梯度，与现在逐次重算是否一致？
   - 批组成相同时应逐位相同；
   - 不同时，差异与现有的 batch 4 / batch 8 差异作比较。
4. 每个 chunk 缓存多少字节？有效 token 有多少（去掉 padding 能省多少）？
5. 训练时的显存峰值（之前没有记录），以及 host pinned 缓存的拷贝开销。

## 范围与资源

- **资源**：1 × H100（share-cluster，`n6ls.iu.i40.1`：1 GPU / 22 vCPU / 240 GiB），闲时（Spot）。
  - 单次作业上限 2400 s；全部尝试合计 ≤ 3600 GPU·s。`probe.py submit` 会强制检查这个上限。
  - 只有在闲时卡被回收时才重新提交。
- **模型、rollout、运行时**：直接用 `rl_multiround_20261102` 冻结版本（哈希校验）。
  - 只把 `driver.py` 换成 `prefix_probe_driver.py`；
  - bootstrap 只改两处，让它接受 `world_size=1`。改后的副本保存在 `jobs/<id>/bootstrap.py`。
- **数据**：32 条恢复的 modem rollout 中的前 3 条，均匀取 384 个 chunk；梯度测量用 84 个 chunk（相当于 4 卡 learner 里一张卡一个优化步的量）。
- **不训练、不保存模型**：唯一产出是 JSON（`results/<id>/prefix_probe.json`，通过 AOSS 取回）。最后一步 AdamW 只用来测优化器状态占多少显存。

## 文件

- `prefix_probe_driver.py`：测量驱动，含 `PrefixCache`。它包装 `kl_batch.collate` 来记录当前批次的行，并替换模型的前缀构建函数；用 prompt token 校验缓存行与当前批次对齐。
- `tests/test_prefix_cache.py`：toy 模型上的 CPU 测试，覆盖等价性、不重算、错位检测、拆分与拼接。
  运行：`PYTHONPATH=../rl_multiround_20261102/runtime:../rlinf_adaptation_20261020/upstream ../rlinf_adaptation_20261020/venv/bin/python tests/test_prefix_cache.py`
- `probe.py submit|status|collect`：提交、状态查询、取回结果。
- `jobs/<id>/`：提交回执 `submission.json`、`latest.json`、`worker_plan.json`、`bootstrap.py`，以及取回的 `retrieved/`。
