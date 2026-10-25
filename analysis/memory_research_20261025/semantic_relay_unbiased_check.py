"""Exact, seconds-scale CPU algebra check; no model or physical data is trained.

The recurrent toy uses the nonlinear fact coordinates (u - 0.7 v**2, v).
One shared parameter vector is used at every write.  An independently sampled
write index and Rademacher direction estimate its full terminal-loss gradient.
The correction's directional derivative is computed by forward tangent replay,
not by reading the reference reverse gradient used to check the answer.

The CVFG identity is established prior art, not a contribution of this script:
https://arxiv.org/html/2405.02952v1, Definition 3.3 / Theorem 3.4.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path


ALPHA = 0.7
RHO = 0.83
THETA = [0.23, -0.11]
INITIAL = [1.2, 0.8, -0.3]
WRITE_PROB = [0.2, 0.3, 0.5]
# c = x + b * theta[0] + d * theta[1]; all writes reuse theta.
INPUTS = [
    dict(x=0.2, b=0.3, d=-0.2, kappa=0.1, eta=0.2),
    dict(x=-0.7, b=-0.25, d=0.5, kappa=0.2, eta=-0.4),
    dict(x=0.5, b=0.4, d=0.1, kappa=-0.15, eta=0.6),
]
TARGET = [1.45, 0.2, -0.1]  # fact A, fact B, nuisance w
LOSS_WEIGHTS = [1.0, 0.07, 0.12]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def scale(c, a):
    return [c * x for x in a]


def norm2(a):
    return dot(a, a)


def transpose(a):
    return [list(row) for row in zip(*a)]


def mv(a, b):
    return [dot(row, b) for row in a]


def mm(a, b):
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def madd(a, b):
    return [add(ar, br) for ar, br in zip(a, b)]


def solve2(a, b):
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if abs(det) < 1e-12:
        raise ValueError("Singular joint fact basis in this toy")
    return [(a[1][1] * b[0] - a[0][1] * b[1]) / det,
            (a[0][0] * b[1] - a[1][0] * b[0]) / det]


def facts(m):
    return [m[0] - ALPHA * m[1] ** 2, m[1]]


def fact_jac(m):
    return [[1.0, -2.0 * ALPHA * m[1], 0.0], [0.0, 1.0, 0.0]]


def write(m, theta, step, leak):
    """Return next state and exact local partials A=dm'/dm, K=dm'/dtheta.

    The first write creates fact A using theta[0]. Subsequent writes preserve
    A if leak=0, while replacing B and updating a nuisance coordinate.  leak>0
    deliberately violates this model-level invariance.  No physical claim is
    made about either setting.
    """
    inp = INPUTS[step]
    first = 1.0 if step == 0 else 0.0
    drift = 0.0 if step == 0 else leak
    a = facts(m)[0] + first * theta[0] + drift * math.tanh(m[1])
    c = inp["x"] + inp["b"] * theta[0] + inp["d"] * theta[1]
    next_m = [a + ALPHA * c * c, c,
              RHO * m[2] + inp["kappa"] * a + inp["eta"] * theta[1]]
    da_dv = -2.0 * ALPHA * m[1] + drift * (1.0 - math.tanh(m[1]) ** 2)
    a_jac = [[1.0, da_dv, 0.0],
             [0.0, 0.0, 0.0],
             [inp["kappa"], inp["kappa"] * da_dv, RHO]]
    k_jac = [[first + 2.0 * ALPHA * c * inp["b"],
              2.0 * ALPHA * c * inp["d"]],
             [inp["b"], inp["d"]],
             [inp["kappa"] * first, inp["eta"]]]
    return next_m, a_jac, k_jac


def terminal_loss(m):
    coordinates = facts(m) + [m[2]]
    return 0.5 * sum(w * (x - y) ** 2
                     for w, x, y in zip(LOSS_WEIGHTS, coordinates, TARGET))


def terminal_gradient(m):
    err = sub(facts(m) + [m[2]], TARGET)
    gradient = mv(transpose(fact_jac(m)),
                  [LOSS_WEIGHTS[0] * err[0], LOSS_WEIGHTS[1] * err[1]])
    gradient[2] += LOSS_WEIGHTS[2] * err[2]
    return gradient


def trajectory(theta, leak):
    states = [INITIAL[:]]
    state_jacs, write_jacs = [], []
    for step in range(len(INPUTS)):
        next_m, a, k = write(states[-1], theta, step, leak)
        states.append(next_m)
        state_jacs.append(a)
        write_jacs.append(k)
    return states, state_jacs, write_jacs


def full_interval_jvp(start_state, after_write_index, theta, leak, direction):
    """Forward replay of only the suffix after a selected write; constant tape."""
    state, tangent = start_state[:], direction[:]
    for step in range(after_write_index + 1, len(INPUTS)):
        state, a, _ = write(state, theta, step, leak)
        tangent = mv(a, tangent)
    return state, tangent


def finite_difference_gradient(theta, leak, epsilon=1e-6):
    result = []
    for j in range(len(theta)):
        plus, minus = theta[:], theta[:]
        plus[j] += epsilon
        minus[j] -= epsilon
        result.append((terminal_loss(trajectory(plus, leak)[0][-1])
                       - terminal_loss(trajectory(minus, leak)[0][-1]))
                      / (2.0 * epsilon))
    return result


def reference_gradients(states, state_jacs, write_jacs):
    """Reverse reference and independent forward sensitivity check, toy only."""
    late = terminal_gradient(states[-1])
    state_grads = [None] * len(INPUTS)
    running = late
    for s in reversed(range(len(INPUTS))):
        state_grads[s] = running
        running = mv(transpose(state_jacs[s]), running)
    contributions = [mv(transpose(k), g)
                     for k, g in zip(write_jacs, state_grads)]
    shared = [sum(c[j] for c in contributions) for j in range(len(THETA))]
    sensitivity = [[0.0] * len(THETA) for _ in INITIAL]
    for a, k in zip(state_jacs, write_jacs):
        sensitivity = madd(mm(a, sensitivity), k)
    forward_shared = mv(transpose(sensitivity), late)
    return late, state_grads, contributions, shared, forward_shared


def make_anchors(states, late, mode):
    if mode == "zero":
        return [[0.0] * len(INITIAL) for _ in INPUTS]
    jt = fact_jac(states[-1])
    lam = solve2(mm(jt, transpose(jt)), mv(jt, late))
    result = []
    for s in range(len(INPUTS)):
        # B is overwritten in each suffix, so it is selected only at the end.
        selected = [lam[0], lam[1] if s == len(INPUTS) - 1 else 0.0]
        anchor = mv(transpose(fact_jac(states[s + 1])), selected)
        if mode == "deliberately_bad":
            # A wrong-sign/overscaled anchor is intentional, not a data label.
            anchor = scale(-5.0, anchor)
        result.append(anchor)
    return result


def enumerate_estimator(states, write_jacs, theta, leak, anchors, true_contributions,
                        true_shared, late):
    directions = [list(v) for v in itertools.product((-1.0, 1.0), repeat=3)]
    mean = [0.0] * len(theta)
    second_moment = 0.0
    position_variance = 0.0
    conditional_variance = 0.0
    per_write = []
    for s, (prob, k, h, true_c) in enumerate(
            zip(WRITE_PROB, write_jacs, anchors, true_contributions)):
        state_estimates, parameter_estimates = [], []
        for v in directions:
            end, tangent = full_interval_jvp(states[s + 1], s, theta, leak, v)
            assert max(abs(x - y) for x, y in zip(end, states[-1])) < 1e-12
            directional = dot(late, tangent)
            # No reference state_grads are used in the actual estimator.
            state_estimate = add(h, scale(directional - dot(h, v), v))
            parameter_estimate = scale(1.0 / prob, mv(transpose(k), state_estimate))
            state_estimates.append(state_estimate)
            parameter_estimates.append(parameter_estimate)
            weight = prob / len(directions)
            mean = add(mean, scale(weight, parameter_estimate))
            second_moment += weight * norm2(parameter_estimate)
        state_mean = [sum(v[j] for v in state_estimates) / len(directions)
                      for j in range(len(INITIAL))]
        parameter_mean = scale(1.0 / prob, true_c)
        cond = sum(norm2(sub(x, parameter_mean)) for x in parameter_estimates) / len(directions)
        conditional_variance += prob * cond
        position_variance += prob * norm2(sub(parameter_mean, true_shared))
        per_write.append(dict(
            write_index=s,
            sampling_probability=prob,
            anchor=h,
            enumerated_state_gradient_mean=state_mean,
            conditional_parameter_mean=parameter_mean,
            conditional_parameter_variance_trace=cond,
            state_gradient_variance_trace=sum(norm2(sub(x, state_mean))
                                              for x in state_estimates) / len(directions),
        ))
    total_variance = second_moment - norm2(mean)
    assert abs(total_variance - conditional_variance - position_variance) < 1e-10
    return dict(
        enumerated_shared_gradient_mean=mean,
        mean_error_vs_shared_reference_l2=math.sqrt(norm2(sub(mean, true_shared))),
        total_parameter_variance_trace=total_variance,
        direction_component_of_variance_trace=conditional_variance,
        write_index_component_of_variance_trace=position_variance,
        probability_sum=sum(WRITE_PROB),
        enumerated_outcomes=len(INPUTS) * len(directions),
        per_write=per_write,
    )


def run_case(name, leak, modes):
    states, a_jacs, k_jacs = trajectory(THETA, leak)
    late, state_grads, contributions, shared, forward_shared = reference_gradients(
        states, a_jacs, k_jacs)
    fd = finite_difference_gradient(THETA, leak)
    assert math.sqrt(norm2(sub(shared, fd))) < 1e-8
    assert math.sqrt(norm2(sub(shared, forward_shared))) < 1e-12
    estimates = {}
    for mode in modes:
        anchors = make_anchors(states, late, mode)
        estimate = enumerate_estimator(states, k_jacs, THETA, leak, anchors,
                                       contributions, shared, late)
        assert estimate["mean_error_vs_shared_reference_l2"] < 1e-12
        for s, item in enumerate(estimate["per_write"]):
            assert math.sqrt(norm2(sub(item["enumerated_state_gradient_mean"],
                                      state_grads[s]))) < 1e-12
            # Exact Rademacher state-level CVFG variance identity.
            expected = (len(INITIAL) - 1) * norm2(sub(state_grads[s], anchors[s]))
            assert abs(item["state_gradient_variance_trace"] - expected) < 1e-10
        estimate["mean_error_vs_finite_difference_l2"] = math.sqrt(
            norm2(sub(estimate["enumerated_shared_gradient_mean"], fd)))
        estimates[mode] = estimate
    # Forgetting 1/p_s averages contributions instead of summing them.
    wrong_without_importance = [sum(p * c[j] for p, c in zip(WRITE_PROB, contributions))
                               for j in range(len(THETA))]
    return dict(
        name=name,
        leak=leak,
        states=states,
        fact_A_by_state=[facts(m)[0] for m in states],
        shared_parameter_gradient_reference=shared,
        shared_parameter_gradient_forward_sensitivity=forward_shared,
        shared_parameter_gradient_finite_difference=fd,
        finite_difference_error_l2=math.sqrt(norm2(sub(shared, fd))),
        individual_write_contributions=contributions,
        omitted_importance_factor_negative_control=dict(
            expected_gradient=wrong_without_importance,
            error_l2=math.sqrt(norm2(sub(wrong_without_importance, shared)))),
        estimators=estimates,
    )


def run():
    exact = run_case("exact_fact_A_persistence", 0.0, ["zero", "semantic"])
    imperfect = run_case("imperfect_model_fact_A_persistence", 0.12,
                         ["zero", "semantic", "deliberately_bad"])
    for case in [exact, imperfect]:
        zero = case["estimators"]["zero"]["total_parameter_variance_trace"]
        semantic = case["estimators"]["semantic"]["total_parameter_variance_trace"]
        assert semantic < zero
    assert (imperfect["estimators"]["deliberately_bad"]["total_parameter_variance_trace"]
            > imperfect["estimators"]["zero"]["total_parameter_variance_trace"])
    return dict(
        purpose="Exact CPU mathematical verification, not model training or a physical experiment.",
        prior_formula="CVFG, https://arxiv.org/html/2405.02952v1, Definition 3.3 / Theorem 3.4",
        parameter=THETA,
        parameter_reuse="The same two theta entries occur in all three writes; every local partial K_s is counted once.",
        sampling="s~(0.2,0.3,0.5); conditional on s enumerate all eight independent Rademacher directions in R^3.",
        estimator="K_s^T [h_s + ((g_s-h_s)^T v) v] / p_s, with g_s^T v computed by suffix JVP.",
        verification="Sum K_s^T g_s = forward sensitivity gradient = exhaustive estimator expectation = central finite difference up to roundoff.",
        cases=[exact, imperfect],
        limits=[
            "Toy real-valued nonlinear recurrence only; no BEHAVIOR data or physical validity is asserted.",
            "Semantic anchor and native persistent-fact idea are evaluated algebraically, not demonstrated empirically.",
            "Unbiasedness requires the anchor to be fixed independently of the current Rademacher direction.",
            "The full suffix still costs linear time to replay; this script does not benchmark activation memory or throughput.",
            "Reference reverse gradients and saved states are used for verification only, not as inputs to the forward-direction correction.",
            "Position-sampling variance is separated because changing the anchor cannot remove it.",
            "An unbiased raw gradient does not imply unbiased clipping, Adam updates, better convergence, or better closed-loop behavior.",
            "The correction formula and variance identity are standard prior art; no new estimator claim is made.",
        ],
    )


if __name__ == "__main__":
    result = run()
    output = Path(__file__).with_name("semantic_relay_unbiased_check_result.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "cases": [{"name": c["name"],
                   "finite_difference_error_l2": c["finite_difference_error_l2"],
                   "gradient": c["shared_parameter_gradient_reference"],
                   "variance": {m: e["total_parameter_variance_trace"]
                                for m, e in c["estimators"].items()}}
                  for c in result["cases"]],
        "status": "all exact-enumeration assertions passed",
    }, indent=2))
