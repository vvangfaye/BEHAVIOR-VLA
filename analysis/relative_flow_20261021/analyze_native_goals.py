"""Descriptive goal-state comparison; never changes the frozen experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE / "closed_loop_exploratory_ada01"
BDDL = Path("/mnt/data1/faye/b1k-rehearsal-1688977/frozen_source/BEHAVIOR-1K/bddl3/bddl/activity_definitions")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clauses(path):
    tokens = iter(re.findall(r"\(|\)|[^\s()]+", path.read_text()))

    def parse(first=None):
        token = next(tokens) if first is None else first
        if token != "(":
            return token
        values = []
        for token in tokens:
            if token == ")":
                return values
            values.append(parse(token))
        raise ValueError("Unclosed BDDL expression")

    problem = parse()
    goal = next(x for x in problem if isinstance(x, list) and x and x[0] == ":goal")[1]
    if goal[0] != "and":
        raise ValueError("Expected top-level conjunction")
    return goal[1:]


def main():
    plan = read(ROOT / "plan.json")
    plan_hash = sha(ROOT / "plan.json")
    methods = plan["protocol"]["methods"]
    if methods != ["extra_fm", "relative_difference"]:
        raise ValueError("Unexpected comparator order")
    checksums = {"plan.json": plan_hash, str(Path(__file__)): sha(Path(__file__))}
    pairs, pending, accepted = [], [], 0
    for task, instance in plan["pairs"]:
        goal_path = BDDL / task / "problem0.bddl"
        goals = clauses(goal_path)
        checksums[str(goal_path)] = sha(goal_path)
        cases = {}
        for method in methods:
            folder = ROOT / "episodes" / method / f"{task}-{instance}"
            receipt_path = folder / "receipt.json"
            if not receipt_path.exists():
                continue
            receipt = read(receipt_path)
            if not receipt["accepted"]:
                raise ValueError(f"Unaccepted episode: {method}/{task}-{instance}")
            if receipt["plan_sha256"] != plan_hash:
                raise ValueError("Receipt belongs to a different plan")
            native_path = folder / "results/json" / f"{task}_{instance}_0.json"
            if sha(native_path) != receipt["result_sha256"]:
                raise ValueError("Native result hash mismatch")
            native = read(native_path)
            if (native["task"], native["instance_id"], native["rollout_id"]) != (task, instance, 0):
                raise ValueError("Episode identity mismatch")
            status = native["final_goal_state"]["top_level"]
            good, bad = set(status["satisfied"]), set(status["unsatisfied"])
            if good & bad or good | bad != set(range(len(goals))):
                raise ValueError("Goal status does not match the BDDL conjunction")
            initial = read(folder / "initial_state.json")
            cases[method] = dict(score=native["q_score"]["final"], success=native["success"],
                                 steps=native["steps"], satisfied=sorted(good),
                                 unsatisfied=sorted(bad), initial=initial)
            for source in [receipt_path, native_path, folder / "initial_state.json"]:
                checksums[str(source)] = sha(source)
            accepted += 1
        if len(cases) != 2:
            pending.append(dict(task=task, instance=instance, available_methods=list(cases)))
            continue
        control, candidate = (cases[method] for method in methods)
        if control["initial"]["objects"] != candidate["initial"]["objects"]:
            raise ValueError("Paired initial physical objects differ")
        gained = sorted(set(candidate["satisfied"]) - set(control["satisfied"]))
        lost = sorted(set(control["satisfied"]) - set(candidate["satisfied"]))
        pairs.append(dict(task=task, instance=instance,
                          control={k: v for k, v in control.items() if k != "initial"},
                          candidate={k: v for k, v in candidate.items() if k != "initial"},
                          score_delta=candidate["score"] - control["score"],
                          final_goal_gains=[dict(index=i, clause=goals[i]) for i in gained],
                          final_goal_losses=[dict(index=i, clause=goals[i]) for i in lost],
                          initial_physics_identical=True,
                          initial_observation_hashes_identical=(control["initial"]["observation"] ==
                                                               candidate["initial"]["observation"])))
    report = dict(observed_at=datetime.now(timezone.utc).isoformat(), accepted_episodes=accepted,
                  complete_pairs=len(pairs), total_pairs=len(plan["pairs"]), pairs=pairs, pending=pending,
                  source_sha256=checksums,
                  analysis_status="Supplementary descriptive analysis added after the first paired result; not a new promotion gate.",
                  interpretation="Goal gains/losses compare terminal states, not causal effects or within-episode progress. No partial-episode outcomes are used.")
    (ROOT / "goal_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 终局目标条件比较", "", f"已接受 {accepted}/12 回合，完成 {len(pairs)}/6 个配对。",
             "", "| 任务 / 实例 | 对照 score | 候选 score | 成功（对照/候选） | 候选多满足的终局条件 | 候选少满足的终局条件 |",
             "|---|---:|---:|---|---|---|"]
    for pair in pairs:
        a, b = pair["control"], pair["candidate"]
        gains = ", ".join(str(x["index"]) for x in pair["final_goal_gains"]) or "无"
        losses = ", ".join(str(x["index"]) for x in pair["final_goal_losses"]) or "无"
        lines.append(f"| {pair['task']} / {pair['instance']} | {a['score']:.4f} | {b['score']:.4f} | {a['success']}/{b['success']} | {gains} | {losses} |")
    lines += ["", "条件编号与表达式见 goal_comparison.json。只比较完成的配对，不把未完成回合当成失败。",
              "", "此表为看到首个配对结果后增加的描述性分析，不改变预先规定的指标或实验门槛。初始物理状态相同，RGB 并非逐像素相同；终局条件差异也不能单独证明因果机制。", ""]
    (ROOT / "goal_comparison.md").write_text("\n".join(lines))
    print(json.dumps(dict(accepted_episodes=accepted, complete_pairs=len(pairs),
                         gains_and_losses=[{k: p[k] for k in ["task", "instance", "final_goal_gains", "final_goal_losses"]} for p in pairs])))


if __name__ == "__main__":
    main()
