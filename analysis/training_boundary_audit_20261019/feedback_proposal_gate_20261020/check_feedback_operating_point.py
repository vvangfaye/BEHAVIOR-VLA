"""Exact scalar counterexample; not a VLA or robot experiment.

Good feedback at a teacher's nominal action need not remain good after an
adaptation changes that action, even in the same plant at the same state.
"""
import json
from pathlib import Path


def transition(x, action):
    return x + action + action * action


def trajectory(bias, gain, initial=0.01, horizon=10):
    states = [initial]
    for _ in range(horizon):
        x = states[-1]
        states.append(transition(x, bias + gain * x))
    return states


def main():
    policies = {
        "teacher": (0.0, -0.5),
        "adapted_with_matching_action_feedback": (-1.0, -0.5),
        "adapted_with_matching_physical_feedback": (-1.0, 0.5),
    }
    output = {}
    for name, (bias, gain) in policies.items():
        output[name] = {
            "nominal_action": bias,
            "action_response_derivative": gain,
            "nominal_next_state": transition(0.0, bias),
            "physical_closed_loop_derivative": 1.0 + (1.0 + 2.0 * bias) * gain,
            "states": trajectory(bias, gain),
        }
    # These are algebraic identities: both healthy maps equal .5*x+.25*x*x.
    for x in [-0.1, -0.01, 0.0, 0.01, 0.1]:
        expected = 0.5 * x + 0.25 * x * x
        assert abs(transition(x, -0.5 * x) - expected) < 1e-14
        assert abs(transition(x, -1.0 + 0.5 * x) - expected) < 1e-14
    assert all(v["nominal_next_state"] == 0 for v in output.values())
    assert output["teacher"]["physical_closed_loop_derivative"] == 0.5
    assert output["adapted_with_matching_action_feedback"]["physical_closed_loop_derivative"] == 1.5
    assert output["adapted_with_matching_physical_feedback"]["physical_closed_loop_derivative"] == 0.5
    result = {
        "status": "complete",
        "evidence_type": "analytic scalar nonlinear plant, not VLA evidence",
        "plant": "x_next = x + a + a^2",
        "policies": output,
        "finding": "Teacher feedback can be stabilizing and correctly measured, yet copying its action response after a nominal-action change can destabilize the same plant. Matching physical response needs the opposite action gain in this example.",
        "implication": "A teacher-usefulness gate alone does not justify derivative inheritance. The student's operating point and action-to-state response matter.",
        "not_established": [
            "That the robot's command map has this scalar nonmonotonic form.",
            "That this mechanism caused any observed finetuning result.",
            "That physical-response matching is a novel algorithm or improves VLA performance.",
            "That a local response target guarantees finite-horizon task success.",
        ],
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {"physical_gain": v["physical_closed_loop_derivative"], "final_error": v["states"][-1]} for k, v in output.items()}, indent=2))


if __name__ == "__main__":
    main()
