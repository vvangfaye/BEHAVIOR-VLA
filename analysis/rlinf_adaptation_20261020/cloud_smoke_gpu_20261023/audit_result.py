"""Audit real result artifacts; never infer completion from job state alone."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "retrieved"


def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else None


def main():
    if not ROOT.is_dir():
        raise RuntimeError("Result archive has not been retrieved; acceptance is unknown, not failed")
    plan = json.loads((HERE / "plan.json").read_text())
    gate = read("gate.json")
    episodes = []
    episode_sources = {}
    for instance in plan["eval_instances"]:
        for backend in ("jax", "torch"):
            key = f"{backend}/{instance}/result.json"
            result = read(key)
            source = key
            if result is None and gate:
                # The original gate embeds the exact completed episode records.
                # These are sufficient while the large tensor archive downloads.
                pair = next((p for p in gate.get("pairs", []) if p["instance"] == instance), None)
                result = pair.get(backend) if pair else None
                source = f"gate.json: pairs[{instance}].{backend}"
            if result is not None:
                assert result["backend"] == backend and result["instance"] == instance
                assert 0 < result["steps"] <= plan["eval_steps"]
                assert 0 <= result["score"] <= 1
                episodes.append(result)
                episode_sources[key] = source
    ppo = read("ppo/result.json")
    gpu_probe = read("gpu_fp32_probe.json")
    gpu_device = read("jax/device.json")
    first = ROOT / "jax/301/result.json"
    checks = {
        "same_card_fp32_gpu_probe_passed": bool(gpu_probe and gpu_probe["passed"]),
        "remaining_jax_episode_used_gpu": bool(gpu_device and gpu_device["backend"] == "gpu" and gpu_device["precision"] == "float32"),
        "completed_first_episode_preserved": first.exists() and hashlib.sha256(first.read_bytes()).hexdigest() == plan["reuse_files_sha256"]["jax/301/result.json"],
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
        "episode_sources": episode_sources,
        "gate": gate,
        "ppo": ppo,
        "final_driver_status": read("status.json"),
        "gpu_probe": gpu_probe,
        "gpu_device": gpu_device,
        "allocation_handoff": read("handoff_receipt.json"),
        "scope": "Single RTX 5090, unchanged-weight paired closed-loop acceptance, then one small PPO iteration. No success-rate improvement claim.",
    }
    (HERE / "acceptance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": checks, "episodes": episodes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
