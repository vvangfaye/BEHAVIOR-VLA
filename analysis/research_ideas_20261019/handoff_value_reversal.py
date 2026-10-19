"""Exact decision sandbox for downstream data value under changing handoffs.

This program evaluates assumed two-stage response tables, not trained policies.
It demonstrates discriminating and negative-control conditions; it provides no
evidence that a learned forecast will be accurate on a robot.
"""

import json
from pathlib import Path


DEFAULT_SUCCESS = {
    "none": (0.80, 0.10),
    "left_data": (0.95, 0.10),
    "right_data": (0.80, 0.90),
}


def downstream(data, right_mass, success_rates=None):
    """B succeeds in left/right handoff contexts with these stipulated rates."""
    success = (success_rates or DEFAULT_SUCCESS)[data]
    return (1.0 - right_mass) * success[0] + right_mass * success[1]


def choose_data(right_mass, success_rates=None):
    return max(("left_data", "right_data"),
               key=lambda name: downstream(name, right_mass, success_rates))


def evaluate_case(name, final_right_mass, teacher_right_mass, probe_right_mass,
                  success_rates=None):
    # Same current student in all cases. Training A is stipulated to raise its
    # success from .2 to .8. Every comparison pays for A plus one equal-cost
    # downstream data group.
    current_right_mass = 0.0
    final_a_success = 0.8
    choices = {
        "current_handoff": choose_data(current_right_mass, success_rates),
        "teacher_handoff": choose_data(teacher_right_mass, success_rates),
        "student_probe_forecast": choose_data(probe_right_mass, success_rates),
        "future_handoff_oracle": choose_data(final_right_mass, success_rates),
    }
    true_returns = {
        data: final_a_success * downstream(data, final_right_mass, success_rates)
        for data in ("left_data", "right_data")
    }
    best = max(true_returns.values())
    return {
        "name": name,
        "assumptions": {
            "current_A_success": 0.2,
            "current_right_handoff_mass": current_right_mass,
            "final_A_success_after_equal_training": final_a_success,
            "final_right_handoff_mass": final_right_mass,
            "teacher_right_handoff_mass": teacher_right_mass,
            "probe_forecast_right_handoff_mass": probe_right_mass,
            "same_cost_choices": ["A + left_data", "A + right_data"],
            "B_success_rates_left_right": success_rates or DEFAULT_SUCCESS,
        },
        "conditional_marginal_effect_of_left_data": {
            "current_handoff": (
                downstream("left_data", current_right_mass, success_rates)
                - downstream("none", current_right_mass, success_rates)
            ),
            "future_handoff": (
                downstream("left_data", final_right_mass, success_rates)
                - downstream("none", final_right_mass, success_rates)
            ),
        },
        "true_full_returns_after_training": true_returns,
        "decisions": {
            method: {
                "selected": selection,
                "true_return": true_returns[selection],
                "regret": best - true_returns[selection],
            }
            for method, selection in choices.items()
        },
    }


def main():
    cases = [
        evaluate_case("stationary_negative_control", 0.0, 0.0, 0.0),
        evaluate_case("student_shift_teacher_mismatch", 0.8, 0.0, 0.8),
        evaluate_case("teacher_already_matches_future", 0.8, 0.8, 0.8),
        evaluate_case("miscalibrated_probe_negative_control", 0.0, 0.0, 0.8),
        evaluate_case(
            "helpful_data_becomes_harmful_after_handoff_shift", 0.8, 0.0, 0.8,
            success_rates={
                "none": (0.80, 0.10),
                "left_data": (0.95, 0.05),
                "right_data": (0.70, 0.90),
            },
        ),
    ]
    stationary, shifted, aligned, wrong, sign_flip = cases
    assert stationary["decisions"]["current_handoff"]["regret"] == 0.0
    assert shifted["decisions"]["current_handoff"]["regret"] > 0.48
    assert shifted["decisions"]["teacher_handoff"]["regret"] > 0.48
    assert shifted["decisions"]["student_probe_forecast"]["regret"] == 0.0
    assert aligned["decisions"]["teacher_handoff"]["regret"] == 0.0
    assert wrong["decisions"]["student_probe_forecast"]["regret"] > 0.11
    marginal = sign_flip["conditional_marginal_effect_of_left_data"]
    assert marginal["current_handoff"] > 0.0
    assert marginal["future_handoff"] < 0.0
    # Exact conditional-success modeling with the true future distribution
    # always explains these cases. The example contains no extra learning
    # interaction beyond changing state occupancy.
    assert all(c["decisions"]["future_handoff_oracle"]["regret"] == 0.0
               for c in cases)
    result = {
        "evidence_type": "analytic decision sandbox with stipulated response rates",
        "purpose": "identify a necessary discrimination experiment, not establish a method win",
        "cases": cases,
        "required_real_experiment": [
            "Measure current, teacher-restored, micro-trained and final-student handoffs separately.",
            "Freeze predicted downstream-data ranking before joint training.",
            "Test ranking reversals on held-out scenes and independent training seeds.",
            "Compare to current-state and future-oracle conditional-success models.",
            "Charge micro-training and intervention cost to the method.",
            "Include stationary, teacher-aligned and inaccurate-forecast negative controls.",
        ],
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
