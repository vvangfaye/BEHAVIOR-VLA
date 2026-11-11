"""Exact counterexample: replay advantage does not identify feedback change.

Analytic diagnostic, not a robot result. No random numbers or dependencies.
"""
from fractions import Fraction as Q
import json
from pathlib import Path


def main():
    # Plant x_next = x + a; goal |x_next| <= 0.65.
    # Both policies have exactly the same slope and nominal success.
    policies = {"parent": (Q(0), Q(-1, 2)), "adapted": (Q(1, 2), Q(-1, 2))}
    radius = Q(13, 20)
    disturbances = (Q(-1), Q(1))
    rows = {}
    for name, (bias, gain) in policies.items():
        branches = []
        for delta in disturbances:
            reactive = delta + bias + gain * delta
            replay = delta + bias
            branches.append({
                "delta": str(delta), "reactive_endpoint": str(reactive),
                "replay_endpoint": str(replay),
                "reactive_success": abs(reactive) <= radius,
                "replay_success": abs(replay) <= radius,
                "centered_physical_response": str(reactive - bias),
            })
        reactive_rate = Q(sum(r["reactive_success"] for r in branches), len(branches))
        replay_rate = Q(sum(r["replay_success"] for r in branches), len(branches))
        rows[name] = {
            "nominal_action": str(bias), "feedback_gain": str(gain),
            "nominal_success": abs(bias) <= radius,
            "reactive_success_rate": str(reactive_rate),
            "replay_success_rate": str(replay_rate),
            "replay_advantage": str(reactive_rate - replay_rate), "branches": branches,
        }
    assert all(row["nominal_success"] for row in rows.values())
    assert rows["parent"]["feedback_gain"] == rows["adapted"]["feedback_gain"]
    assert rows["parent"]["replay_advantage"] == "1"
    assert rows["adapted"]["replay_advantage"] == "0"
    assert [r["centered_physical_response"] for r in rows["parent"]["branches"]] == [
        r["centered_physical_response"] for r in rows["adapted"]["branches"]]

    # For this deterministic scalar example a four-way intervention separates
    # nominal action from feedback. This is not an unqualified prescription
    # for stochastic image-conditioned VLA grafting.
    hybrids = []
    for nominal_name, (bias, _) in policies.items():
        for response_name, (_, gain) in policies.items():
            rate = Q(sum(abs(delta + bias + gain * delta) <= radius
                         for delta in disturbances), len(disturbances))
            hybrids.append({"nominal_source": nominal_name,
                            "response_source": response_name,
                            "success_rate": str(rate)})
    for name in policies:
        rates = [h["success_rate"] for h in hybrids if h["nominal_source"] == name]
        assert rates[0] == rates[1]
    report = {
        "kind": "exact_analytic_counterexample", "rows": rows, "hybrids": hybrids,
        "finding": "Both policies succeed nominally and have identical feedback and centered physical responses, but the reactive-minus-replay success advantage drops from 1 to 0.",
        "implication": "Symmetric disturbances, nominal success matching, and replay controls alone do not identify feedback degradation; goal geometry and nominal operating point can change the score.",
        "limits": ["One-step deterministic scalar plant, not empirical VLA evidence.",
                   "The hybrid intervention is well-defined here; transferring it to stochastic VLA chunks needs separate validity and coupling checks.",
                   "Centered mean physical response is informative in this example, not a complete measure of stochastic multi-stage competence."],
    }
    target = Path(__file__).with_suffix(".json")
    with target.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
