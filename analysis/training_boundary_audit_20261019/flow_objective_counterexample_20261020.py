"""Exact toy audit: conditional flow-matching loss is not terminal likelihood.

No policy weights, robot data, network calls, or simulator are used. The scalar
construction only disproves an unconditional identity; it does not identify a
cause of this project's fine-tuning results or establish a new method.
"""
from fractions import Fraction as Q
import json
from pathlib import Path


def text(q):
    return {"fraction": str(q), "float": float(q)}


def coefficients(piecewise_k):
    """Integrate E[(k(t) ((1-t)e+t*a) - (a-e))**2] over t.

    e ~ N(0,1), a is a fixed observed target. Result is A*a**2+B.
    All interval endpoints and slopes are rational, so integration is exact.
    """
    a_coefficient = Q(0)
    b_coefficient = Q(0)
    for start, end, k in piecewise_k:
        i0 = end - start
        i1 = (end**2 - start**2) / 2
        i2 = (end**3 - start**3) / 3
        a_coefficient += k**2 * i2 - 2 * k * i1 + i0
        b_coefficient += k**2 * (i0 - 2 * i1 + i2) + 2 * k * (i0 - i1) + i0
    return a_coefficient, b_coefficient


def discrete_example(c, steps=10):
    assert steps > 0 and steps % 2 == 0
    h = Q(1, steps)
    assert 1 + h*c > 0
    k_even, k_odd = c, -c / (1 + h*c)
    ks = [k_even if i % 2 == 0 else k_odd for i in range(steps)]
    intervals = [(i*h, (i+1)*h, k) for i, k in enumerate(ks)]
    a_coefficient, b_coefficient = coefficients(intervals)
    multiplier = Q(1)
    for k in ks:
        multiplier *= 1 + h*k
    assert multiplier == 1
    loss_good = b_coefficient  # a+ = 0
    loss_bad = 4*a_coefficient + b_coefficient  # a- = 2
    # Reference k=0 has losses 1 and 5. No beta or /2 rescaling.
    margin = (1-loss_good) - (5-loss_bad)
    return {
        "c": text(c), "euler_steps": steps,
        "even_interval_k": text(k_even), "odd_interval_k": text(k_odd),
        "deployed_euler_multiplier_exact": text(multiplier),
        "deployed_distribution": "N(0,1), identical to reference for every initial noise",
        "loss_a_squared_coefficient": text(a_coefficient),
        "loss_constant": text(b_coefficient),
        "loss_at_preferred_target_0": text(loss_good),
        "loss_at_rejected_target_2": text(loss_bad),
        "fm_reference_difference_preference_margin": text(margin),
        "true_log_density_ratio_preference_margin": 0,
        "exact_ode_and_euler_distinction": "Only the stated finite Euler sampler is identity here; the exact ODE is generally not."
    }


def exact_ode_example(c):
    factor = 1+c/3+2*c*c/15
    return {
        "c": text(c),
        "field": "v_c(t,x)=c*(1-2*t)*x",
        "solution": "X_t=exp(c*(t-t^2))*epsilon, hence X_1=epsilon",
        "terminal_distribution": "N(0,1), identical for every initial noise",
        "conditional_fm_loss": "factor*(a^2+1)",
        "factor": text(factor),
        "fm_reference_difference_preference_margin_a_plus_0_a_minus_2": text(4*(factor-1)),
        "true_log_density_ratio_preference_margin": 0,
        "scope": "Exact continuous ODE; not a claim about finite-step Euler evaluation of this field."
    }


def main():
    result = {
        "kind": "analytic_flow_objective_counterexample_v1",
        "date": "2026-10-20",
        "training_interpolant": "z_t=(1-t)*epsilon+t*a, epsilon~N(0,1), t~Uniform[0,1]",
        "training_target_velocity": "a-epsilon",
        "reference_field": "v_0=0",
        "continuous_ode_examples": [exact_ode_example(Q(c)) for c in [-1,0,1]],
        "finite_euler_examples": [discrete_example(Q(c)) for c in [-1,0,1]],
        "verified": [
            "For both separately defined constructions, the declared sampler outputs exactly the same sample for each initial noise.",
            "Conditional FM losses and a loss-difference preference margin can change while the deployed policy is identical.",
            "For the exact-ODE c=-1 example, every fixed-target FM loss is 20% lower than at c=0.",
            "An unconditional equality between reference-calibrated FM loss differences and terminal log-density ratios is false."
        ],
        "not_proven": [
            "That actual fine-tuning updates follow either toy family.",
            "That conditional flow matching has an incorrect population minimizer under its standard fixed-path assumptions.",
            "That standard FM training, or a validated preference surrogate, cannot improve policies.",
            "That a new policy-learning algorithm is required or has been identified."
        ],
        "finite_sampler_convention": "Intervals are left-closed and right-open; forward Euler samples each interval's left endpoint. Identity is in exact arithmetic, not a claim of bitwise equality in floating-point execution.",
        "prior_art": [
            {"title": "When Can Conditional Flow Matching Replace Pointwise Negative Log-Likelihood?", "url": "https://arxiv.org/html/2608.28010v1", "overlap": "Explicitly studies CFM loss differences versus endpoint likelihood ratios, including identical-endpoint counterexamples. This audit is not a novelty claim."},
            {"title": "Newton Matching for Generative Modeling", "url": "https://arxiv.org/pdf/2609.05727", "overlap": "Defines many-to-one terminal-density map and canonicalization preserving the terminal density. Main definitions independently reviewed, full proof corpus not audited."}
        ],
        "caution": "The paired field families use different sampler definitions; do not conflate exact ODE and finite Euler claims."
    }
    assert result["continuous_ode_examples"][0]["factor"]["fraction"] == "4/5"
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(out), "all_finite_euler_multipliers_exactly_one": True,
                      "continuous_c_minus_one_loss_factor": "4/5",
                      "finite_euler_c_one_surrogate_margin": result["finite_euler_examples"][2]["fm_reference_difference_preference_margin"]}))


if __name__ == "__main__":
    main()
