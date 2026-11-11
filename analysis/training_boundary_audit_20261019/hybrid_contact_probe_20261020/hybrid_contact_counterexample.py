#!/usr/bin/env python3
"""Exact scalar/rational audit of a known hybrid guard sensitivity effect.

Pure standard-library arithmetic. No robot policy, simulator, training, cloud,
or downloaded data is used. This is not evidence about any VLA checkpoint.
"""
from __future__ import annotations

from fractions import Fraction as Q
import json
import math
from pathlib import Path


def number(value: Q) -> dict:
    return {"fraction": str(value), "float": float(value)}


def pair(value: tuple[Q, Q]) -> list[dict]:
    return [number(v) for v in value]


def constant_endpoint(x: Q, y: Q, epsilon: Q, observation_time: Q):
    """x=0 guard; constant pre-field (epsilon,1), absorbing post-field (0,0)."""
    assert epsilon > 0 and x < 0
    hit = -x / epsilon
    if hit > observation_time:
        return (x + epsilon * observation_time, y + observation_time), hit, False
    return (Q(0), y + hit), hit, True


def profile_endpoint(x: Q, y: Q, early: Q, late: Q, observation_time: Q):
    """Clock changes normal speed at t=1/2; tangential speed remains one.

    Both velocities are state-independent. The x=0 guard, when enabled,
    switches the vector field to zero without a state reset.
    """
    assert x < 0 and early > 0 and late > 0
    switch = Q(1, 2)
    first_hit = -x / early
    if first_hit <= switch:
        hit = first_hit
    else:
        hit = switch + (-x - early * switch) / late
    if hit > observation_time:
        first_duration = min(switch, observation_time)
        return (x + early * first_duration + late * (observation_time - first_duration),
                y + observation_time), hit, False
    return (Q(0), y + hit), hit, True


def clock_stop_endpoint(x: Q, y: Q, early: Q, late: Q, stop_time=Q(1)):
    """Control: identical normal-speed profile, but stop at an external clock.

    No x-dependent event and no projection/reset of x. Observation is after stop.
    """
    switch = Q(1, 2)
    first_duration = min(switch, stop_time)
    return (x + early * first_duration + late * (stop_time - first_duration), y + stop_time)


def finite_difference(function, x: Q, y: Q, step: Q):
    columns = []
    for dx, dy in ((step, Q(0)), (Q(0), step)):
        positive = function(x + dx, y + dy)
        negative = function(x - dx, y - dy)
        columns.append(tuple((a - b) / (2 * step) for a, b in zip(positive, negative)))
    return tuple(tuple(columns[col][row] for col in range(2)) for row in range(2))


def matrix(matrix_value):
    return [[number(v) for v in row] for row in matrix_value]


def check_constant_family():
    x0, y0 = Q(-1, 100), Q(0)
    observation_time, difference_step = Q(3), Q(1, 1000000)
    cases = []
    for epsilon in (Q(1, 100), Q(1, 200)):
        end, hit, completed = constant_endpoint(x0, y0, epsilon, observation_time)
        assert completed
        finite = finite_difference(
            lambda x, y: constant_endpoint(x, y, epsilon, observation_time)[0],
            x0, y0, difference_step,
        )
        analytic = ((Q(0), Q(0)), (-1 / epsilon, Q(1)))
        assert finite == analytic
        assert end == (Q(0), y0 - x0 / epsilon)
        cases.append({
            "epsilon": number(epsilon), "nominal_hit_time": number(hit),
            "fixed_observation_time": number(observation_time), "nominal_endpoint": pair(end),
            "pre_mode_state_jacobian": [[0, 0], [0, 0]],
            "post_mode_state_jacobian": [[0, 0], [0, 0]],
            "endpoint_state_jacobian_analytic": matrix(analytic),
            "endpoint_state_jacobian_finite_difference": matrix(finite),
            "x_to_y_sensitivity_magnitude": number(1 / epsilon),
            "jacobian_spectral_norm": math.sqrt(1 + float(1 / epsilon) ** 2),
        })
    assert cases[0]["x_to_y_sensitivity_magnitude"]["fraction"] == "100"
    assert cases[1]["x_to_y_sensitivity_magnitude"]["fraction"] == "200"
    return {
        "initial_state": pair((x0, y0)), "finite_difference_step": number(difference_step),
        "normal_velocity_change": number(Q(1, 200)),
        "cases": cases,
        "interpretation": "An O(epsilon) constant-velocity change doubles event sensitivity. This first comparison also changes nominal duration, so it does not by itself exclude a duration explanation.",
    }


def check_equal_time_family():
    epsilon = Q(1, 100)
    x0, y0 = -epsilon, Q(-1)
    observation_time = Q(2)
    delta, difference_step = epsilon / 100, epsilon / 1000
    profiles = {
        "slow_at_guard": (3 * epsilon / 2, epsilon / 2),
        "fast_at_guard": (epsilon / 2, 3 * epsilon / 2),
    }
    cases = []
    for name, (early, late) in profiles.items():
        end, hit, completed = profile_endpoint(x0, y0, early, late, observation_time)
        assert completed and hit == 1 and end == (0, 0)
        # Both profiles have the same nominal normal displacement and exactly
        # the same action MSE relative to the constant reference (epsilon,1).
        normal_impulse = (early + late) / 2
        normal_action_mse = ((early - epsilon) ** 2 + (late - epsilon) ** 2) / 2
        vector_squared_error_integral = normal_action_mse  # y velocity error is zero; duration is one.
        vector_coordinate_mse = normal_action_mse / 2
        assert normal_impulse == epsilon
        assert normal_action_mse == epsilon ** 2 / 4
        assert vector_coordinate_mse == epsilon ** 2 / 8
        finite = finite_difference(
            lambda x, y: profile_endpoint(x, y, early, late, observation_time)[0],
            x0, y0, difference_step,
        )
        analytic = ((Q(0), Q(0)), (-1 / late, Q(1)))
        assert finite == analytic
        perturbed = []
        for offset in (-delta, delta):
            shifted, shifted_hit, hit_guard = profile_endpoint(x0 + offset, y0, early, late, observation_time)
            assert hit_guard and shifted_hit > Q(1, 2) and shifted_hit < observation_time
            assert shifted[1] == -offset / late
            perturbed.append({
                "initial_x_offset": number(offset), "hit_time": number(shifted_hit),
                "fixed_time_endpoint": pair(shifted), "endpoint_error_norm": number(abs(shifted[1])),
                "initial_error_norm": number(abs(offset)),
                "finite_perturbation_gain": number(abs(shifted[1] / offset)),
            })
        clock_jacobian = finite_difference(
            lambda x, y: clock_stop_endpoint(x, y, early, late), x0, y0, difference_step,
        )
        assert clock_jacobian == ((Q(1), Q(0)), (Q(0), Q(1)))
        assert clock_stop_endpoint(x0, y0, early, late) == (0, 0)
        clock_errors = []
        for offset in (-delta, delta):
            shifted = clock_stop_endpoint(x0 + offset, y0, early, late)
            assert shifted == (offset, Q(0))
            clock_errors.append({"initial_x_offset": number(offset), "endpoint": pair(shifted),
                                 "finite_perturbation_gain": number(Q(1))})
        cases.append({
            "name": name, "early_normal_velocity": number(early), "late_normal_velocity": number(late),
            "nominal_hit_time": number(hit), "fixed_observation_time": number(observation_time),
            "nominal_endpoint": pair(end), "pre_mode_state_jacobian": [[0, 0], [0, 0]],
            "post_mode_state_jacobian": [[0, 0], [0, 0]],
            "normal_displacement_before_nominal_hit": number(normal_impulse),
            "nominal_pre_contact_action_error_window": [0, 1],
            "normal_action_mse_vs_constant_reference": number(normal_action_mse),
            "vector_action_squared_error_integral_vs_reference": number(vector_squared_error_integral),
            "vector_action_coordinate_mse_vs_reference": number(vector_coordinate_mse),
            "full_observation_window_normal_action_mse_vs_reference": number(normal_action_mse / observation_time),
            "full_observation_window_vector_coordinate_mse_vs_reference": number(vector_coordinate_mse / observation_time),
            "endpoint_state_jacobian_analytic": matrix(analytic),
            "endpoint_state_jacobian_finite_difference": matrix(finite),
            "x_to_y_sensitivity_magnitude": number(1 / late), "perturbed_cases": perturbed,
            "clock_stop_control": {"stop_time": number(Q(1)), "state_jacobian": matrix(clock_jacobian),
                                   "perturbed_cases": clock_errors},
        })
    assert cases[0]["x_to_y_sensitivity_magnitude"]["fraction"] == "200"
    assert cases[1]["x_to_y_sensitivity_magnitude"]["fraction"] == "200/3"
    return {
        "epsilon": number(epsilon), "initial_state": pair((x0, y0)),
        "perturbation_magnitude": number(delta), "finite_difference_step": number(difference_step),
        "clock_profile_switch_time": number(Q(1, 2)), "cases": cases,
        "event_sensitivity_ratio_slow_to_fast": number(Q(3)),
        "squared_endpoint_error_ratio_slow_to_fast": number(Q(9)),
        "controlled_exactly": [
            "Initial physical state and its perturbation set",
            "State-triggered guard x=0, continuous identity reset, and absorbing post mode",
            "Nominal hit time=1 and final observation time=2",
            "Nominal endpoint=(0,0) and zero nominal endpoint error",
            "Mean nominal normal velocity and normal displacement",
            "Nominal action MSE against the same constant reference, on both normal-channel and vector-coordinate definitions",
            "Zero spatial Jacobians within each smooth mode",
        ],
        "time_interpretation": "The gain difference is not caused by longer nominal execution or a longer evaluation window: both are equal. Perturbed hitting times do differ; their state sensitivity is the event mechanism, not a nuisance we claim to remove. Replacing the state guard by an external clock stop removes the amplification for both profiles.",
        "coverage_interpretation": "Initial-state coverage is identical and there is no learning dataset in this toy. Intermediate paths differ as a consequence of the velocity profile; this is a mediator, not evidence that real VLA training-distribution coverage has been controlled.",
    }


def main():
    result = {
        "schema_version": 1, "kind": "exact_hybrid_guard_sensitivity_counterexample", "date": "2026-10-20",
        "computation": {"standard_library_only": True, "exact_rational_finite_differences": True,
                        "training": False, "robot_model": False, "simulation_job": False,
                        "network": False, "cloud_jobs": False},
        "hybrid_system": {
            "guard": "x=0, approached from x<0",
            "pre_vector_field": "(epsilon,1) in the constant example; (v_x(t),1) in the equal-time control",
            "post_vector_field": "(0,0), sticking; identity state reset",
            "event_endpoint": "(0, y0-x0/epsilon) when the guard is reached before the fixed observation time",
            "saltation_matrix_constant": "S=I+(f_plus-f_minus)*n^T/(n^T*f_minus)=[[0,0],[-1/epsilon,1]], n=(1,0)",
            "within_mode_caution": "Zero vector-field Jacobian means no local smooth expansion, not asymptotic contraction: the smooth flow state Jacobian is identity. This toy makes no claim that both modes are strictly contracting.",
        },
        "constant_velocity_comparison": check_constant_family(),
        "equal_time_equal_nominal_mse_comparison": check_equal_time_family(),
        "future_real_vla_test": {
            "hypothesis": "A real SFT checkpoint can preserve or improve nominal action error while increasing finite-amplitude state-error amplification specifically across physical contact-mode transitions.",
            "event_labels": "Use verified physical grasp/support/contact state changes from simulator state, not a generated subtask label or a gripper-command threshold.",
            "required_identity": "Real baseline and SFT checkpoints on the same verified initial snapshots, instruction, runtime, common inference noise, perturbation directions/magnitudes and execution protocol.",
            "duration_controls": [
                "Always observe at a fixed elapsed time; separately record nominal and perturbed event times.",
                "Match or stratify anchors on nominal time-to-event; compare event-crossing and no-event windows of the same duration.",
                "Use an isochronous toy/control, as above, before claiming sensitivity is more than longer nominal duration.",
            ],
            "ordinary_action_error_control": "Measure nominal action MSE separately and use matched-MSE checkpoints/windows or a controlled action-profile intervention. This toy demonstrates equality of nominal action MSE is insufficient; it does not establish equal MSE for real checkpoints.",
            "state_coverage_control": "Use identical evaluation snapshots and paired perturbations, hold training data/compute fixed in any training comparison, and report distance to available demonstrations separately. Do not infer coverage control from nominal rollout similarity.",
            "event_sensitivity_estimates": "For each paired perturbed/unperturbed rollout report entry separation, physical mode outcome, event-time difference, fixed-time exit separation, and after-event task predicates. Keep different physical modes as categorical outcomes instead of hiding them inside one Euclidean score.",
            "falsification": [
                "No checkpoint-related increase in transition-conditioned amplification on matched snapshots.",
                "The same amplification change appears equally in matched-duration no-event windows.",
                "The apparent difference disappears after matching nominal action error or nominal time-to-event.",
                "The only effect is different initial-state coverage or a failed snapshot restoration.",
            ],
        },
        "not_proven": [
            "This is a known saltation/event-time sensitivity phenomenon, not a new mathematical theory.",
            "No real VLA checkpoint has been compared and no empirical explanation of unstable fine-tuning is established.",
            "The toy has idealized absorbing contact, no observation noise, no occlusion and no stochastic contact solver.",
            "A pointwise stable continuous-mode Jacobian alone does not establish stability of a hybrid event map.",
            "Identifying a VLA failure mechanism, or an effective new training objective, requires additional evidence.",
        ],
    }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(destination), "exact_checks_passed": True,
                      "constant_sensitivities": [100, 200],
                      "isochronous_sensitivity_ratio": 3, "isochronous_squared_error_ratio": 9,
                      "clock_stop_gains": [1, 1]}))


if __name__ == "__main__":
    main()
