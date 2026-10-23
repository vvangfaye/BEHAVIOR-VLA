"""Audit real result artifacts; never infer completion from job state alone."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "retrieved"


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else None


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    episodes = []
    for instance in plan["eval_instances"]:
        for backend in ("jax", "torch"):
            result = read(f"{backend}/{instance}/result.json")
            if result is not None:
                assert result["backend"] == backend and result["instance"] == instance
                assert 0 < result["steps"] <= plan["eval_steps"]
                assert 0 <= result["score"] <= 1
                episodes.append(result)
    gate = read("gate.json")
    ppo = read("ppo/result.json")
    checks = {
        "all_four_episodes": len(episodes) == 4,
        "frozen_gate_passed": bool(gate and gate["passed"] and all(gate["checks"].values())),
        "ppo_completed": bool(ppo and ppo["passed"] and ppo["iterations"] == 1),
        "ppo_has_real_updates": bool(ppo and ppo["optimizer_steps"] > 0 and ppo["expert_gradient_nonzero"] and ppo["value_gradient_nonzero"]),
        "frozen_weights_unchanged": bool(ppo and ppo["frozen_sha_before"] == ppo["frozen_sha_after"]),
        "trained_weights_changed": bool(ppo and ppo["trainable_sha_before"] != ppo["trainable_sha_after"]),
        "checkpoint_round_trip": bool(ppo and ppo["checkpoint_round_trip"]),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "episodes": episodes,
        "gate": gate,
        "ppo": ppo,
        "final_driver_status": read("status.json"),
        "scope": "Single RTX 5090, unchanged-weight paired closed-loop acceptance, then one small PPO iteration. No success-rate improvement claim.",
    }
    (HERE / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": checks, "episodes": episodes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
