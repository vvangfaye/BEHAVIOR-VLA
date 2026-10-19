"""Analytic toy: equal behavior and repair diagnostics need not imply equal learning.

This is not a BEHAVIOR experiment. Both students use the identical bilinear
Bernoulli policy class, optimizer, labels and learning rate. They differ only
in behaviorally equivalent initial parameters. Skills are independent, so
the positive interaction is entirely explained by the success-rate product.
"""

import json
import math
from pathlib import Path


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def learn_positive(u, v, steps, lr):
    """SGD on -log sigmoid(u*v), using simultaneous parameter updates."""
    for _ in range(steps):
        residual = sigmoid(u * v) - 1.0
        du, dv = residual * v, residual * u
        u, v = u - lr * du, v - lr * dv
    return u, v


def evaluate_student(initial_v, steps=10, lr=0.02):
    initial_u = 0.0
    p0 = sigmoid(initial_u * initial_v)
    u1, v1 = learn_positive(initial_u, initial_v, steps, lr)
    p1 = sigmoid(u1 * v1)
    pu, pv = learn_positive(initial_u, initial_v, 1, lr)
    # There are two independent skills with identical initial parameterization.
    returns = {"00": p0 * p0, "10": p1 * p0,
               "01": p0 * p1, "11": p1 * p1}
    interaction = returns["11"] - returns["10"] - returns["01"] + returns["00"]
    product_prediction = (p1 - p0) ** 2
    assert math.isclose(interaction, product_prediction, abs_tol=1e-12)
    return {
        "initial_parameters_per_skill": [initial_u, initial_v],
        "initial_skill_success": p0,
        "oracle_repair_full_success": {"00": p0 * p0, "10": p0,
                                       "01": p0, "11": 1.0},
        "one_update_skill_success": sigmoid(pu * pv),
        "after_updates_skill_success": p1,
        "student_full_success_by_training_arm": returns,
        "joint_training_gain_from_00": returns["11"] - returns["00"],
        "raw_training_interaction": interaction,
        "interaction_predicted_by_independent_product": product_prediction,
        "residual_beyond_product": interaction - product_prediction,
    }


def main():
    students = {
        "small_local_jacobian": evaluate_student(1.0),
        "large_local_jacobian": evaluate_student(10.0),
    }
    small, large = students.values()
    assert small["oracle_repair_full_success"] == large["oracle_repair_full_success"]
    assert large["joint_training_gain_from_00"] > small["joint_training_gain_from_00"]
    result = {
        "evidence_type": "exact synthetic counterexample, not empirical robotics evidence",
        "policy_class": "independent skills, P(success)=sigmoid(u*v)",
        "optimizer": {"name": "SGD", "learning_rate": 0.02,
                      "positive_examples_per_trained_skill": 10},
        "students": students,
        "logical_conclusions": [
            "Behavior and oracle repair outcomes alone cannot identify training gains in general.",
            "A one-update probe distinguishes these two parameterizations.",
            "Positive joint-training interaction can be wholly due to independent success-rate multiplication.",
        ],
        "limits": [
            "No VLA, simulator, real data, handoff shift or shared-skill interference is modeled.",
            "The argument establishes insufficiency of behavior-only evidence, not novelty or efficacy of a method.",
            "The SGD counterexample does not quantify the effect under the project's AdamW configuration.",
        ],
    }
    output = Path(__file__).with_name("behavior_learning_counterexample.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
