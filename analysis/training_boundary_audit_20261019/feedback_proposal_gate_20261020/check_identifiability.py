"""Analytic limits of the proposed feedback-preserving VLA objective.

This is a mathematical diagnostic, not a VLA or simulator experiment.
Run with standard Python; writes its report beside this file.
"""
import json
from pathlib import Path


def main():
    # f(z,w)=a*z+b*w. Data cover w=0, with expert g(z)=2*z.
    # Exact interpolation on an open z interval fixes a=2. Parent=(1,2).
    penalties = []
    for b in [-3.0, 0.0, 1.0, 2.0, 5.0]:
        selected = (b - 2.0) ** 2
        full = (2.0 - 1.0) ** 2 + selected
        assert full - selected == 1.0
        penalties.append({"b": b, "selected": selected, "full": full})

    # The same policy change has opposite consequences in two plants.
    # x_next = plant*x + u, u=-k*x. Stability multiplier = plant-k.
    base_k, tuned_k = 0.8, 1.2
    plants = []
    for plant in [1.1, 0.3]:
        base = abs(plant - base_k)
        tuned = abs(plant - tuned_k)
        plants.append({
            "plant_gain": plant,
            "base_policy_jacobian": -base_k,
            "tuned_policy_jacobian": -tuned_k,
            "absolute_jacobian_change": abs(tuned_k-base_k),
            "base_error_contraction_factor": base,
            "tuned_error_contraction_factor": tuned,
            "feedback_change": "improves" if tuned < base else "worsens",
        })
    assert plants[0]["feedback_change"] == "improves"
    assert plants[1]["feedback_change"] == "worsens"
    assert plants[0]["absolute_jacobian_change"] == plants[1]["absolute_jacobian_change"]

    report = {
        "status": "complete",
        "evidence_type": "analytic counterexamples only; no VLA experiment",
        "exact_fit_equivalence": {
            "conditions": [
                "A smooth demonstration manifold parameterized by z, with normal coordinates w=0.",
                "A differentiable learned policy exactly agrees with the expert on an open part of the manifold.",
                "Full and selected Jacobian penalties use the same metric and sampling locations.",
                "Optimization is restricted to policies satisfying this exact-fit condition.",
            ],
            "reason": "Tangential derivatives are fixed by exact agreement on the manifold. The full penalty differs from the normal-direction penalty by a constant on the feasible set.",
            "affine_example": penalties,
            "implication": "Both penalties have the same minimizers on the exact-fit feasible set. Direction selection alone is not a generally distinct learning principle.",
            "does_not_prove": [
                "Equivalence for finite isolated demonstrations.",
                "Equivalence with imperfect fitting, finite penalty weights, or different optimization trajectories.",
                "That the demonstration manifold or its normals can be identified from our dataset.",
            ],
        },
        "policy_jacobian_cannot_sign_physical_effect": plants,
        "design_consequence": "Measure interventional closed-loop task outcomes and retain nominal outcomes; policy derivative magnitude is not a recovery-performance label.",
        "goal_complete": False,
    }
    target = Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({"status": report["status"], "report": str(target), "analytic_checks": 2}))


if __name__ == '__main__':
    main()
