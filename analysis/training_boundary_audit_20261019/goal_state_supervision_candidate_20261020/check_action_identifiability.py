"""Exact finite counterexample, not a simulator or learned-policy experiment.

Both goals are independent and monotone. An action completes its named goal;
STOP terminates. The concept predictor is perfect even off the expert path.
"""
from itertools import product
from pathlib import Path
import json
import math

STATES = list(product((0, 1), repeat=2))
ACTIONS = ("A", "B", "STOP")
EXPERT = {(0, 0): "A", (1, 0): "B", (1, 1): "STOP"}

def run(policy, initial, limit=4):
    state = initial
    trace = []
    for _ in range(limit):
        action = policy[state]
        trace.append({"state": list(state), "action": action})
        if action == "STOP":
            return {"success": state == (1, 1), "trace": trace, "end": "stop"}
        state = (1, state[1]) if action == "A" else (state[0], 1)
    return {"success": False, "trace": trace, "end": "budget"}

def softmax_loss(logits, label, scale=16.0):
    scores = [scale * x for x in logits]
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(x - maximum) for x in scores)) - scores[ACTIONS.index(label)]

def good_logits(state):
    a, b = state
    return [2 - 4 * a, 1 - 4 * b, 0]

def count_logits(state):
    progress = sum(state)
    return [1 - progress, 0.5, progress - 1]

def main():
    surviving = []
    for choices in product(ACTIONS, repeat=len(STATES)):
        policy = dict(zip(STATES, choices))
        if all(policy[s] == a for s, a in EXPERT.items()):
            surviving.append(policy)
    rows = []
    for policy in surviving:
        rows.append({
            "action_on_unseen_combination": policy[(0, 1)],
            "expert_action_error": 0,
            "concept_error_on_all_combinations": 0,
            "rollout_from_unseen_combination": run(policy, (0, 1)),
        })
    assert len(surviving) == 3
    assert sum(row["rollout_from_unseen_combination"]["success"] for row in rows) == 1
    # The ambiguity also exists with simple affine action logits, not only lookup tables.
    affine = {}
    for name, logit_fn in [("factorwise", good_logits), ("count_based", count_logits)]:
        predictions = {s: ACTIONS[max(range(3), key=lambda i: logit_fn(s)[i])] for s in STATES}
        assert all(predictions[s] == a for s, a in EXPERT.items())
        affine[name] = {
            "mean_expert_cross_entropy_at_scale_16": sum(softmax_loss(logit_fn(s), a) for s, a in EXPERT.items()) / len(EXPERT),
            "action_on_unseen_combination": predictions[(0, 1)],
            "rollout": run(predictions, (0, 1)),
        }
    assert affine["factorwise"]["rollout"]["success"]
    assert not affine["count_based"]["rollout"]["success"]
    result = {
        "type": "finite_analytic_counterexample_not_robot_evidence",
        "expert_supervision": [{"state": list(s), "action": a} for s, a in EXPERT.items()],
        "concept_supervision": "Exact full goal vector available on all four combinations; identity predictor used by every policy.",
        "unseen_action_combination": [0, 1],
        "consistent_action_heads_before_extra_concept_labels": len(surviving),
        "consistent_action_heads_after_extra_concept_labels": len(surviving),
        "policies": rows,
        "affine_examples": affine,
        "implication": "Perfect factual-state supervision does not, by itself, identify the action on an unseen completion combination, even with a hard concept bottleneck and independent goals.",
        "limits": [
            "Does not show state supervision cannot improve real VLA learning through representation sharing or inductive bias.",
            "Does not predict the optimizer's choice among solutions.",
            "Does not reproduce actual robot failure rates.",
            "Does not establish that additional action supervision is necessary under every structural policy class."
        ],
        "required_followup": "Separate the effect of better state estimation from the action-composition inductive bias; report an oracle-concept action test before attributing closed-loop gains to state grounding.",
    }
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
