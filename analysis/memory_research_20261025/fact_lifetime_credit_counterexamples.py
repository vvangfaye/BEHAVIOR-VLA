"""CPU algebra checks for the fact-lifetime credit proposal; no model training."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_adjoint_relay_algebra import dot, mm, mv, solve2, transpose


def redundant_probe():
    # lambda = (J J^T)^+ J g = J (J^T J)^-1 g for full-column-rank J.
    j = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    g = [1.0, 10.0]
    lam = mv(j, solve2(mm(transpose(j), j), g))
    masked = mv(transpose(j), [lam[0], 0.0, 0.0])
    transition = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    transition_relay = mv(transpose(j), mv(transpose(transition), lam))
    assert abs(masked[0] + 8.0 / 3.0) < 1e-12
    assert abs(transition_relay[0] - 1.0) < 1e-12
    return {
        "primitive_facts": ["A", "B"],
        "physical_transition": "(A,B) -> (A,c)",
        "endpoint_loss": "A + 10*B",
        "true_source_gradient": [1.0, 0.0],
        "primitive_identity_probe_relay": [1.0, 0.0],
        "redundant_probe": ["A", "B", "A+B"],
        "joint_semantic_covector": lam,
        "diagonal_persistence_mask_relay": masked,
        "dot_with_true_gradient": masked[0],
        "consistent_semantic_transition_matrix": transition,
        "transition_matrix_relay": transition_relay,
        "lesson": "Adding a redundant changing relation reverses the diagonal-mask relay. Primitive independent coordinates repair this example; a consistent full semantic transition Jacobian also repairs it.",
    }


def shared_parameters(coefficients):
    def states(theta):
        out = [theta]
        for coeff in coefficients[1:]:
            out.append(out[-1] + coeff * (theta - 1.0))
        return out
    def loss(theta):
        return 0.5 * states(theta)[-1] ** 2
    eps = 1e-6
    full = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
    endpoint = coefficients[-1]
    source_only = endpoint + coefficients[0]
    block_sum = sum(coefficients)
    sampled = [endpoint + 2.0 * c for c in coefficients[:-1]]
    assert abs(full - block_sum) < 1e-8
    assert abs(sum(sampled) / len(sampled) - block_sum) < 1e-12
    return {
        "theta": 1.0,
        "coefficients": coefficients,
        "forward": "m1=theta; mi=m(i-1)+ci*(theta-1)",
        "actual_states": states(1.0),
        "native_fact": 1.0,
        "probe": "f(m)=m",
        "all_input_state_jacobians": 1.0,
        "local_differential_fact_invariance_at_current_theta": True,
        "all_pointwise_fact_errors": 0.0,
        "true_full_parameter_gradient_finite_difference": full,
        "last_block_local_gradient": endpoint,
        "source_only_plus_last_block": source_only,
        "all_disjoint_blocks_semantic_relay": block_sum,
        "sample_one_prior_block_p_half_estimates": sampled,
        "sampled_estimator_expectation": sum(sampled) / len(sampled),
        "lesson": "Exact cut-state cotangents do not include every direct use of shared parameters. Injecting into disjoint blocks and accounting for each local parameter occurrence fixes the omission in this example.",
    }


def complete_fact_not_action_alignment():
    theta, truth, actor_weight, action_target = 1.0, 1.0, -1.0, 1.0
    action_error = actor_weight * theta - action_target
    return {
        "model": "f(m)=m=theta; action=w*f(m)",
        "fact_value": theta,
        "true_fact": truth,
        "fact_loss": 0.5 * (theta - truth) ** 2,
        "fact_gradient": theta - truth,
        "actor_weight": actor_weight,
        "action_target": action_target,
        "action_loss": 0.5 * action_error ** 2,
        "memory_action_gradient": actor_weight * action_error,
        "actor_gradient": theta * action_error,
        "lesson": "A perfect fact representation can receive an action gradient that corrupts its truth to compensate for a wrong reader. The relay adds optimization preference, not missing information. Same-label auxiliary supervision is an essential comparator.",
    }


if __name__ == "__main__":
    result = {
        "purpose": "Algebra counterexamples only; no trained model or robot performance.",
        "redundant_probe_sign_reversal": redundant_probe(),
        "shared_parameter_omission": shared_parameters([1.0, 2.0, 3.0]),
        "shared_parameter_direction_reversal": shared_parameters([1.0, -10.0, 3.0]),
        "perfect_fact_wrong_reader": complete_fact_not_action_alignment(),
    }
    dest = Path(__file__).with_name("fact_lifetime_credit_counterexamples.json")
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
