#!/usr/bin/env python3
"""Read archived B1K receipts only; never start policies, environments or cloud jobs.

Run from any directory. Only results.json and report.md next to this script are
written. Platform allocation duration is not pure compute duration. Missing
per-component timers are explicitly retained as missing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def summary(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"n": 0}
    def quantile(q):
        k = (len(values) - 1) * q
        return values[math.floor(k)] * (math.ceil(k) - k) + values[math.ceil(k)] * (k - math.floor(k)) if k % 1 else values[int(k)]
    return {"n": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p10": quantile(.1),
            "p90": quantile(.9), "min": values[0], "max": values[-1]}


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def analyze(campaign):
    inputs = {}
    def read(path):
        body = path.read_bytes()
        inputs[str(path.relative_to(PROJECT))] = {
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        return json.loads(body)

    jobs = [(p.parent, read(p)) for p in sorted((campaign / "jobs").glob("*/latest.json"))]
    receipts = [read(p) for p in sorted((campaign / "receipts").glob("*/*.json"))]
    assert len({(r["round"], r["episode"]) for r in receipts}) == len(receipts)
    groups = defaultdict(list)
    for r in receipts:
        match = re.fullmatch(r"mr_sample_r(\d+)_(s\d+)_a(\d+)", r["source_task"])
        assert match, r["source_task"]
        key = (int(match[1]), match[2], int(match[3]))
        assert r["chunks"] == math.ceil(r["status"]["steps"] / 16)
        groups[key].append(r)

    role_cost = defaultdict(float)
    state_counts = defaultdict(Counter)
    sample_rows, eval_rows, learner_rows = [], [], []
    for directory, row in jobs:
        role = row["role"]
        state_counts[role][row["state"]] += 1
        allocated = None
        if row.get("start_time") and row.get("complete_time"):
            allocated = max(0., timestamp(row["complete_time"]) - timestamp(row["start_time"]))
            role_cost[role] += allocated * row["gpu_count"] / 3600
        progress = row.get("progress", {})
        worker = None
        if "started_at" in progress and "updated_at" in progress:
            worker = progress["updated_at"] - progress["started_at"]
        if role == "sample" and row["state"] == "SUCCEEDED":
            key = (row["round"], row["payload"]["sampler_id"], row["attempt"])
            owned = groups.get(key, [])
            assert len(owned) == progress["collected"], (row["id"], len(owned))
            chunks = sum(r["chunks"] for r in owned)
            sample_rows.append({"job": row["id"], "round": row["round"],
                "episodes": len(owned), "chunks": chunks,
                "bottom_level_actions": sum(r["status"]["steps"] for r in owned),
                "worker_seconds": worker, "allocated_seconds": allocated,
                "worker_seconds_per_chunk_inclusive": worker / chunks,
                "allocation_seconds_per_chunk_inclusive": allocated / chunks})
        elif role == "eval" and row["state"] == "SUCCEEDED":
            case = row["payload"]["case"]
            base = directory / "retrieved" / "final" / f"i{case[0]}_s{case[1]}"
            init = read(base / "initialization.json")
            result = read(base / "result.json")
            assert init["stage"] == "ready"
            chunks = math.ceil(result["steps"] / 16)
            remaining = worker - init["elapsed_seconds"]
            assert remaining > 0
            eval_rows.append({"job": row["id"], "instance": case[0], "seed": case[1],
                "worker_seconds": worker, "allocated_seconds": allocated,
                "simulator_initialization_seconds": init["elapsed_seconds"],
                "remaining_worker_seconds_inclusive": remaining,
                "chunks": chunks, "bottom_level_actions": result["steps"],
                "remaining_worker_seconds_per_chunk_inclusive": remaining / chunks})
        elif role == "learn" and row["state"] == "SUCCEEDED":
            timing = read(directory / "retrieved" / "timing.json")
            learner_rows.append({"job": row["id"], "allocated_seconds": allocated,
                "gpu_count": row["gpu_count"], "phase_seconds": timing["phase_seconds"],
                "driver_model_load_seconds": timing["marks"]["model_loaded"] - timing["driver_started_at"]})

    total = sum(role_cost.values())
    status = read(campaign / "status.json")
    assert abs(total - status["gpu_hours_bound"]) < 1e-8
    assert sum(r["episodes"] for r in sample_rows) == len(receipts)
    sample_worker = sum(r["worker_seconds"] for r in sample_rows)
    sample_chunks = sum(r["chunks"] for r in sample_rows)
    eval_worker = sum(r["worker_seconds"] for r in eval_rows)
    init_seconds = sum(r["simulator_initialization_seconds"] for r in eval_rows)
    scenarios = []
    for speed in (2, 5, 10, None):
        replacement = 0 if speed is None else role_cost["sample"] / speed
        remaining = total - role_cost["sample"] + replacement
        scenarios.append({"all_sampling_cost_speedup_assumption": "infinite" if speed is None else speed,
            "campaign_gpu_hours_if_other_costs_fixed": remaining,
            "campaign_gpu_hour_speedup": total / remaining,
            "note": "Optimistic scenario, not prediction: replaces all sampling cost, retains evaluation and training; excludes new world-model training/validation cost."})
    training_total=role_cost["sample"]+role_cost["learn"]
    training_scenarios=[]
    for speed in (2,5,10,None):
        replacement=0 if speed is None else role_cost["sample"]/speed
        remaining=role_cost["learn"]+replacement
        training_scenarios.append({"all_sampling_cost_speedup_assumption":"infinite" if speed is None else speed,
            "training_loop_gpu_hours_if_learner_cost_fixed":remaining,
            "training_loop_gpu_hour_speedup":training_total/remaining,
            "note":"Training-loop-only optimistic scenario: final evaluation excluded; all sampling costs assumed replaceable; extra world-model fitting and model-error validation costs omitted."})
    result = {
        "scope": "Archived B1K installing_a_modem multiround run, 2026-11-03. No new simulator runs.",
        "campaign": str(campaign.relative_to(PROJECT)),
        "cost_definition": "Sum platform complete_time-start_time times gpu_count; includes startup, failures and teardown; excludes queue waiting. Same calculation as campaign controller.",
        "job_state_counts": {k: dict(v) for k, v in state_counts.items()},
        "gpu_hours_by_role": dict(role_cost), "gpu_hours_total": total,
        "sample": {"episodes": len(receipts), "chunks": sample_chunks,
            "steps": sum(r["status"]["steps"] for r in receipts),
            "worker_seconds": sample_worker,
            "weighted_worker_seconds_per_chunk_inclusive": sample_worker / sample_chunks,
            "per_job_seconds_per_chunk_inclusive": summary(r["worker_seconds_per_chunk_inclusive"] for r in sample_rows),
            "per_job_worker_seconds": summary(r["worker_seconds"] for r in sample_rows),
            "per_episode_worker_seconds_amortized": sample_worker / len(receipts),
            "note": "Includes policy/model loading, each episode simulator initialization, rollouts, writes and shutdown; not isolated physics or inference latency.",
            "by_round": {str(k): {"episodes": sum(x["episodes"] for x in sample_rows if x["round"] == k),
                "chunks": sum(x["chunks"] for x in sample_rows if x["round"] == k),
                "worker_seconds": sum(x["worker_seconds"] for x in sample_rows if x["round"] == k)} for k in (1, 2, 3)},
            "jobs": sample_rows},
        "evaluation": {"episodes": len(eval_rows), "worker_seconds": eval_worker,
            "simulator_initialization_seconds": summary(r["simulator_initialization_seconds"] for r in eval_rows),
            "simulator_initialization_fraction_of_worker": init_seconds / eval_worker,
            "remaining_worker_seconds_per_chunk_inclusive": summary(r["remaining_worker_seconds_per_chunk_inclusive"] for r in eval_rows),
            "weighted_remaining_worker_seconds_per_chunk_inclusive": (eval_worker-init_seconds)/sum(r["chunks"] for r in eval_rows),
            "note": "Remaining includes VLA model loading, rollout inference, environment/render/IPC, I/O and teardown. Subtraction does not isolate rollout. Initialization is measured on evaluation, not training sampling.",
            "jobs": eval_rows},
        "learner": learner_rows,
        "unavailable_component_timers": {
            "sampling_environment_initialization": "Not exported in sampler receipts; cannot substitute eval timings as training measurements.",
            "vlm_prefix_and_action_inference": "No component timer in archived sampler/eval outputs.",
            "subtask_generation": "Included in predict_action_batch; no independent timer.",
            "physics_versus_rendering_versus_ipc": "No component timer.",
            "rollout_tensor_and_video_writes": "No isolated timer; included in inclusive worker duration."},
        "amdahl_sampling_only_scenarios": scenarios,
        "training_loop_excluding_final_evaluation":{
            "gpu_hours":training_total,
            "sampling_gpu_hour_fraction":role_cost["sample"]/training_total,
            "scenarios":training_scenarios},
        "measured_eval_initialization_removal_upper_bound": {
            "gpu_hours": init_seconds/3600,
            "campaign_gpu_hour_speedup_if_removed_for_successful_eval_only": total/(total-init_seconds/3600),
            "note": "Ideal upper bound for removing all measured successful-eval initialization; persistent environment reloading is not zero cost. Does not estimate sampling initialization."},
        "limitations": ["GPU-hours across heterogeneous GPUs are resource accounting, not equivalent FLOPs.",
            "GPU-hour Amdahl bounds are not wall-clock bounds for this parallel scheduled workflow.",
            "All-sampling replacement bounds include non-replaceable work and are intentionally optimistic.",
            "Simulator initialization is repeatedly executed for each episode by current code; can benchmark persistent-process/cache reuse separately.",
            "A state-only world model cannot automatically replace image generation needed by the existing image-conditioned VLA."],
        "inputs": inputs}
    return result


def render(result):
    sample, ev = result["sample"], result["evaluation"]
    costs = result["gpu_hours_by_role"]
    rows = "\n".join(f"| {label} | {costs[role]:.2f} | {100*costs[role]/result['gpu_hours_total']:.1f}% |" for role,label in [("sample","训练轨迹采样"),("eval","正式评测"),("learn","PPO训练")])
    scenarios = "\n".join(f"| {x['all_sampling_cost_speedup_assumption']} | {x['campaign_gpu_hours_if_other_costs_fixed']:.2f} | {x['campaign_gpu_hour_speedup']:.2f}× |" for x in result['amdahl_sampling_only_scenarios'])
    loop=result['training_loop_excluding_final_evaluation']
    training_scenarios="\n".join(f"| {x['all_sampling_cost_speedup_assumption']} | {x['training_loop_gpu_hours_if_learner_cost_fixed']:.2f} | {x['training_loop_gpu_hour_speedup']:.2f}× |" for x in loop['scenarios'])
    return f'''# B1K 现有轨迹采样耗时审计

只读分析 2026-11-03 modem 三轮在线 PPO 的已归档记录；没有启动仿真、策略或云作业。
复现：`python3 analysis/supermimic_feasibility_20261106/profiling/analyze_existing.py`。

## 实测成本

| 阶段 | GPU 小时 | 占总量 |
|---|---:|---:|
{rows}

总计 {result['gpu_hours_total']:.2f} GPU 小时，逐作业平台开始/结束时刻与控制器总量完全一致；包含失败尝试、启动和退出，不含排队。不同 GPU 的 GPU 小时不等于等效算力。

- 成功采样 {sample['episodes']} 条、{sample['chunks']:,} 个 chunk、{sample['steps']:,} 个底层动作；每 chunk 最多 16 动作。
- 96 个成功采样 worker 的时间合计除以 chunk 为 **{sample['weighted_worker_seconds_per_chunk_inclusive']:.3f} 秒/chunk**；均摊 **{sample['per_episode_worker_seconds_amortized']/60:.2f} 分钟/条**。这是包含模型加载、每回合仿真初始化、推理、环境、写盘和退出的均摊值，不是纯 rollout 延迟。
- {ev['episodes']} 个成功评测任务保留了 `initialization.json`：仿真初始化中位数 **{ev['simulator_initialization_seconds']['median']:.1f} 秒**，p10–p90 **{ev['simulator_initialization_seconds']['p10']:.1f}–{ev['simulator_initialization_seconds']['p90']:.1f} 秒**；占这些 worker 总时长 **{100*ev['simulator_initialization_fraction_of_worker']:.1f}%**。
- 扣除已测初始化后，评测剩余时间均摊 **{ev['weighted_remaining_worker_seconds_per_chunk_inclusive']:.3f} 秒/chunk**。其中仍含模型加载、推理、环境步进/渲染、I/O、退出，不能标成 physics 或 inference。

## 哪些分解没有数据

采样收据没有导出初始化计时；策略前缀/动作推理、子任务生成、物理/渲染/IPC、tensor/视频写盘均没有独立计时。详细 `results.json` 将这些值留空并说明原因。评测初始化结果只能提示优化方向，不能冒充训练采样的实测初始化占比。

代码已经明确每回合都新建 `Simulator`（`runtime/experiment.py:105`），每次创建并删除独立的临时仿真缓存（`runtime/common.py:114`）。因此保持仿真进程、同进程切换实例和复用合法缓存值得单独 benchmark，但实际收益未测量。

## 训练循环的 World model 乐观 Amdahl 上限

针对“RL 采样慢”的目标，应先看采样＋PPO learner，**不含最后的正式评测**。该训练循环共 {loop['gpu_hours']:.2f} GPU 小时，采样占 **{100*loop['sampling_gpu_hour_fraction']:.1f}%**。假设全部采样成本可替换且 PPO learner 成本不变：

| 采样整体加速假设 | 新训练循环 GPU 小时 | 训练循环 GPU 小时加速 |
|---|---:|---:|
{training_scenarios}

其中采样整体 10 倍加速对应训练循环约 **{loop['scenarios'][2]['training_loop_gpu_hour_speedup']:.2f} 倍**；采样免费时的数学上限约 **{loop['scenarios'][-1]['training_loop_gpu_hour_speedup']:.2f} 倍**。这显示了减少真实环境采样的潜在价值，但实际 world model 增加拟合、数据更新和模型误差检查的成本；只替换物理步进也无法消除全部采样成本。

## 含最终评测的整批实验上限

假设所有训练采样成本都可替换、评测和 PPO 成本不变，且新增 world model 训练/验证免费：

| 采样整体加速假设 | 新总 GPU 小时 | 全流程 GPU 小时加速 |
|---|---:|---:|
{scenarios}

以上两个表衡量不同范围：前表是 RL 训练循环，后表是包括最终评测的整批实验。两者都是乐观上限情景，不是预测，也不是墙钟加速。正式评测仍需原仿真；状态模型也不能直接替代现有 VLA 所需 RGB 观察。

近端优先项是补齐 component timer 并测试仿真进程/缓存复用；world model 试验需同时测量真实采样减少量、模型训练成本、闭环成功率与跨实例误差。
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=PROJECT / "analysis/rl_multiround_20261102")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    result = analyze(args.campaign.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "report.md").write_text(render(result))
    print(json.dumps({"gpu_hours": result["gpu_hours_by_role"],
        "sample_seconds_per_chunk_inclusive": result["sample"]["weighted_worker_seconds_per_chunk_inclusive"],
        "eval_init_median_seconds": result["evaluation"]["simulator_initialization_seconds"]["median"],
        "training_loop_excluding_final_evaluation":result["training_loop_excluding_final_evaluation"],
        "sampling_free_upper_bound": result["amdahl_sampling_only_scenarios"][-1]}, indent=2))


if __name__ == "__main__":
    main()
