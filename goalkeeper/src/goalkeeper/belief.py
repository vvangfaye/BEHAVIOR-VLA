"""Uncertainty-aware memory and conservative action-chunk selection.

Intervals are empirical estimates, not certified per-state probabilities. See
docs/research.md for the assumptions needed by the conditional risk proposition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from .goals import Atom, GoalProgram, Literal


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.mean, self.lower, self.upper)):
            raise ValueError("Nonfinite belief")
        if not 0 <= self.lower <= self.mean <= self.upper <= 1:
            raise ValueError("Belief must satisfy 0 <= lower <= mean <= upper <= 1")

    def negate(self):
        return Estimate(1 - self.mean, 1 - self.upper, 1 - self.lower)


UNKNOWN = Estimate(0.5, 0.0, 1.0)


class BeliefMemory:
    def __init__(self, program: GoalProgram, decay=0.85, visibility_threshold=0.7):
        if not 0 < decay <= 1 or not 0 <= visibility_threshold <= 1:
            raise ValueError("Invalid memory parameters")
        self.program, self.decay = program, decay
        self.visibility_threshold = visibility_threshold
        self.reset()

    def reset(self):
        # PUBLIC init is a task specification, not a live sensor measurement.
        self.values = {a: UNKNOWN for a in self.program.atoms}
        self.ages = {a: 0 for a in self.program.atoms}

    def update(self, estimates: dict[Atom, Estimate], visibility: dict[Atom, float]):
        if set(estimates) != set(self.values) or set(visibility) != set(self.values):
            raise ValueError("Verifier atom set does not match static task")
        for atom, evidence in estimates.items():
            vis = visibility[atom]
            if not math.isfinite(vis) or not 0 <= vis <= 1:
                raise ValueError("Invalid visibility probability")
            if vis >= self.visibility_threshold:
                # Contradictory new evidence can undo a previous completed state.
                self.values[atom], self.ages[atom] = evidence, 0
            else:
                old = self.values[atom]
                self.values[atom] = Estimate(old.mean, old.lower * self.decay,
                                            1 - (1 - old.upper) * self.decay)
                self.ages[atom] += 1
        return dict(self.values)


def literal_estimate(literal: Literal, values: dict[Atom, Estimate]) -> Estimate:
    v = values[literal.atom]
    return v if literal.positive else v.negate()


def lower_progress(program: GoalProgram, values: dict[Atom, Estimate]) -> float:
    return max((sum(literal_estimate(l, values).lower for l in c) / len(c) if c else 1
                for c in program.clauses), default=0.0)


@dataclass(frozen=True)
class Selection:
    index: int
    reason: str
    scores: tuple[float, ...]
    regression_risks: tuple[float, ...]
    protected: tuple[str, ...]


class PreservationSelector:
    def __init__(self, confidence=0.9, risk_budget=0.2, penalty=0.5, minimum_gain=0.02,
                 minimum_progress=0.01):
        if not 0 <= confidence <= 1 or not 0 <= risk_budget <= 1 or penalty < 0 or minimum_gain < 0 or minimum_progress < 0:
            raise ValueError("Invalid selection parameters")
        self.confidence, self.risk_budget = confidence, risk_budget
        self.penalty, self.minimum_gain = penalty, minimum_gain
        self.minimum_progress = minimum_progress

    def choose(self, program: GoalProgram, current: dict[Atom, Estimate],
               futures: list[dict[Atom, Estimate]]) -> Selection:
        if not futures:
            raise ValueError("At least one candidate required; candidate 0 is the base policy")
        if any(set(f) != set(program.atoms) for f in futures) or set(current) != set(program.atoms):
            raise ValueError("Belief atom set mismatch")
        # Protect satisfied literals in ONE coherent alternative, not incompatible
        # alternatives at once. Recompute from evidence, so regression is reversible.
        best_clause = max(program.clauses, key=lambda c:
                          sum(literal_estimate(l, current).lower for l in c) / max(1, len(c)),
                          default=())
        protected = tuple(l for l in best_clause if literal_estimate(l, current).lower >= self.confidence)
        risks = tuple(min(1., sum(1 - literal_estimate(l, f).lower for l in protected))
                      for f in futures)
        scores = tuple(lower_progress(program, f) - self.penalty * risk for f, risk in zip(futures, risks))
        current_progress = lower_progress(program, current)
        # An idle action that merely keeps a door closed must not defeat opening
        # that door to put an object inside. Intervene only with positive predicted
        # progress over the CURRENT state; otherwise defer to the original policy.
        # This is abstention, not a certified constraint on the executed action.
        eligible = [i for i, risk in enumerate(risks)
                    if risk <= self.risk_budget and
                    (i == 0 or lower_progress(program, futures[i]) >= current_progress + self.minimum_progress)]
        if not eligible:
            return Selection(0, "abstain_no_candidate_within_estimated_budget", scores, risks,
                             tuple(l.key for l in protected))
        i = max(eligible, key=lambda j: scores[j])
        if i == 0 or scores[i] < scores[0] + self.minimum_gain:
            i, reason = 0, "base_policy_no_confident_improvement"
        else:
            reason = "estimated_progress_with_preservation"
        return Selection(i, reason, scores, risks, tuple(l.key for l in protected))
