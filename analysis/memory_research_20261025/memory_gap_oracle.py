#!/usr/bin/env python3
"""Exact toy oracle for memory gaps in finite deterministic option models.

This is not a simulator, a VLA benchmark, or a certificate for continuous
robot control.  It exhaustively enumerates deterministic reactive policies
over every reachable (visible observation, time, remaining budget).  Each
such observation key occurs at most once on a path in these acyclic toys.
Consequently a randomized reactive policy is a mixture of deterministic
tables and cannot improve the optimum here.  This does not justify reducing
arbitrary stationary randomized POMDP controllers to deterministic ones.

A separate exact belief-tree DP compares retaining the prefix bit with erasing
it once, while allowing unrestricted use of subsequent observed evidence.
That retention value generally differs from the reactive-policy optimum.

Both hidden worlds have the same goal: retrieve the target.  A visible prefix
reveals which sealed cabinet contains it.  The budget starts at the common
decision point, after that prefix.  A reactive policy sees the prefix too, but
may use it later only through a permitted observable environmental marker.
The history policy stores the actually observed cue in one internal bit.

Run with Python's standard library only:
    python memory_gap_oracle.py
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable


@dataclass(frozen=True)
class State:
    hidden_bit: int
    phase: str
    elapsed: int
    remaining: int
    marker: int | None = None


@dataclass(frozen=True)
class Terminal:
    success: bool
    reason: str


Observation = tuple[str, int | None, int | None, int, int]


class CabinetModel:
    """Two equally likely worlds, deterministic options, visible clocks.

    advance and prefix marker writing are pre-budget operations: they finish
    the prefix and initialize the common decision budget.  Post-prefix marker
    writing is a one-use zero-cost option; crucially it writes the chosen bit,
    not an oracle's hidden bit.  This makes the external-memory control strict:
    even free blind writing cannot create missing information.
    """

    def __init__(self, budget: int, marker_at_cue: bool = False,
                 marker_when_sealed: bool = False):
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        self.budget = budget
        self.marker_at_cue = marker_at_cue
        self.marker_when_sealed = marker_when_sealed

    def initial(self, bit: int) -> State:
        if bit not in (0, 1):
            raise ValueError("hidden bit must be 0 or 1")
        return State(bit, "cue", 0, self.budget)

    @staticmethod
    def observe(s: State) -> Observation:
        visible_bit = s.hidden_bit if s.phase in ("cue", "revealed") else None
        # The hidden bit is absent at sealed/marked decision points.  All
        # declared visible side channels, including time/budget, are retained.
        return (s.phase, visible_bit, s.marker, s.elapsed, s.remaining)

    def actions(self, s: State) -> tuple[str, ...]:
        if s.phase == "cue":
            return (("advance", "mark_0", "mark_1") if self.marker_at_cue
                    else ("advance",))
        base = ("commit_0", "commit_1", "stop")
        if s.phase in ("sealed", "marked"):
            base += ("look",)
        if s.phase == "sealed" and self.marker_when_sealed:
            base += ("mark_0", "mark_1")
        return base

    def step(self, s: State, action: str) -> State | Terminal:
        if action not in self.actions(s):
            raise ValueError(f"unavailable option {action!r}")
        if s.phase == "cue":
            marker = int(action[-1]) if action.startswith("mark_") else None
            phase = "marked" if marker is not None else "sealed"
            return State(s.hidden_bit, phase, 0, self.budget, marker)
        if action.startswith("mark_"):
            return State(s.hidden_bit, "marked", s.elapsed, s.remaining,
                         int(action[-1]))
        if action == "stop":
            return Terminal(False, "stopped")
        cost = 2 if action == "look" else 3
        if cost > s.remaining:
            return Terminal(False, "budget_exhausted")
        if action == "look":
            return State(s.hidden_bit, "revealed", s.elapsed + cost,
                         s.remaining - cost, s.marker)
        chosen = int(action[-1])
        return Terminal(chosen == s.hidden_bit,
                        "target_retrieved" if chosen == s.hidden_bit else "wrong_cabinet")

    def all_reachable(self) -> list[State]:
        pending = [self.initial(0), self.initial(1)]
        reached: set[State] = set()
        while pending:
            s = pending.pop()
            if s in reached:
                continue
            reached.add(s)
            for action in self.actions(s):
                result = self.step(s, action)
                if isinstance(result, State):
                    pending.append(result)
        return sorted(reached, key=repr)

    def observation_actions(self) -> list[tuple[Observation, tuple[str, ...]]]:
        by_observation: dict[Observation, tuple[str, ...]] = {}
        for s in self.all_reachable():
            obs, actions = self.observe(s), self.actions(s)
            if obs in by_observation and by_observation[obs] != actions:
                raise AssertionError("action availability leaks a hidden state")
            by_observation[obs] = actions
        return sorted(by_observation.items(), key=repr)


def rollout(model: CabinetModel, bit: int,
            decide: Callable[[Observation], str]) -> tuple[bool, list[dict]]:
    s = model.initial(bit)
    trace = []
    for _ in range(8):  # The declared transition graph is acyclic, depth <= 4.
        obs = model.observe(s)
        action = decide(obs)
        out = model.step(s, action)
        trace.append({"observation": list(obs), "action": action})
        if isinstance(out, Terminal):
            trace[-1]["terminal"] = {"success": out.success, "reason": out.reason}
            return out.success, trace
        s = out
    raise AssertionError("unexpected cycle in finite toy model")


def exact_reactive_optimum(model: CabinetModel) -> dict:
    """Enumerate every total deterministic observation-based decision rule.

    Policies must share an action wherever their entire visible observation
    keys agree.  Keys contain time and budget, so time-indexed policies and
    declared environmental external storage are included in the enumeration.
    Unreachable-under-a-policy entries are still enumerated, not omitted.
    """
    # The randomized-policy reduction needs a once-per-path observation key.
    # Acyclic hidden states alone would not establish that condition.
    def check_path(s: State, seen: frozenset[Observation]) -> None:
        obs = model.observe(s)
        if obs in seen:
            raise ValueError("repeated observation key invalidates the randomization reduction")
        for action in model.actions(s):
            successor = model.step(s, action)
            if isinstance(successor, State):
                check_path(successor, seen | {obs})

    for bit in (0, 1):
        check_path(model.initial(bit), frozenset())
    entries = model.observation_actions()
    keys, domains = zip(*entries)
    policy_count = 1
    for domain in domains:
        policy_count *= len(domain)
    index = {obs: i for i, obs in enumerate(keys)}
    best_successes = -1
    best_values = None
    examined = 0
    # Do not break at a perfect policy: enumerate the entire declared space.
    for values in itertools.product(*domains):
        examined += 1
        successes = 0
        for bit in (0, 1):
            state: State | Terminal = model.initial(bit)
            while isinstance(state, State):
                state = model.step(state, values[index[model.observe(state)]])
            successes += int(state.success)
        if successes > best_successes:
            best_successes, best_values = successes, values
    assert examined == policy_count and best_values is not None
    best = dict(zip(keys, best_values))
    return {"mean_success_upper_bound": best_successes / 2,
            "bound_kind": "exact reactive optimum in this acyclic, clock-observed toy",
            "randomization_argument": "Each observation key appears at most once per path; pre-sampling independent action choices at all keys gives a mixture of deterministic tables. An expected score cannot exceed the best table.",
            "total_policies_enumerated": examined,
            "observation_keys": len(entries),
            "best_policy": [{"observation": list(obs), "action": best[obs]} for obs in keys],
            "witness_rollouts": [rollout(model, bit, best.__getitem__)[1] for bit in (0, 1)]}


def history_lower_bound(model: CabinetModel) -> dict:
    traces = []
    success = 0
    for bit in (0, 1):
        memory = None

        def choose(obs: Observation) -> str:
            nonlocal memory
            if obs[0] == "cue":
                memory = obs[1]  # Only the visible prefix; no native-state read.
                return "advance"
            assert memory in (0, 1)
            return f"commit_{memory}"

        ok, trace = rollout(model, bit, choose)
        success += int(ok)
        traces.append(trace)
    return {"mean_success_lower_bound": success / 2,
            "policy_memory_bits": 1,
            "memory_source": "visible prefix cue, not hidden state at decision time",
            "witness_rollouts": traces}


def exact_future_history_value(model: CabinetModel,
                               prior: tuple[tuple[State, Fraction], ...]) -> dict:
    """Optimize *all future history* after a specified one-time information loss.

    This differs from a reactive-policy restriction.  Each new observation
    conditions the belief, and later actions may depend on that entire belief.
    All available future actions are considered; repeated look is additionally
    permitted after the bit has been revealed.  Such a redundant look costs 2
    and cannot improve the optimum, but is not silently forbidden here.

    The finite model is deterministic conditional on its hidden state.  The
    only uncertainty is the exact Fraction prior over the two possible worlds.
    """
    def canonical(entries: dict[State, Fraction]) -> tuple[tuple[State, Fraction], ...]:
        return tuple(sorted(((s, p) for s, p in entries.items() if p), key=repr))

    if not prior or sum((p for _, p in prior), Fraction(0)) != 1:
        raise ValueError("belief must be a normalized nonempty distribution")
    if any(p <= 0 for _, p in prior):
        raise ValueError("belief probabilities must be positive")
    combined: dict[State, Fraction] = {}
    for state, probability in prior:
        combined[state] = combined.get(state, Fraction(0)) + probability
    prior = canonical(combined)

    def future_actions(s: State) -> tuple[str, ...]:
        actions = model.actions(s)
        return actions + (("look",) if s.phase == "revealed" else ())

    def future_step(s: State, action: str) -> State | Terminal:
        if action == "look" and s.phase == "revealed":
            if s.remaining < 2:
                return Terminal(False, "budget_exhausted")
            return State(s.hidden_bit, "revealed", s.elapsed + 2,
                         s.remaining - 2, s.marker)
        return model.step(s, action)

    def successor_beliefs(belief, action):
        terminal_reward = Fraction(0)
        groups: dict[Observation, dict[State, Fraction]] = {}
        for state, probability in belief:
            result = future_step(state, action)
            if isinstance(result, Terminal):
                terminal_reward += probability * int(result.success)
            else:
                group = groups.setdefault(model.observe(result), {})
                group[result] = group.get(result, Fraction(0)) + probability
        successors = []
        for group in groups.values():
            mass = sum(group.values(), Fraction(0))
            posterior = canonical({s: p / mass for s, p in group.items()})
            successors.append((mass, posterior))
        return terminal_reward, successors

    @lru_cache(maxsize=None)
    def solve(belief):
        observations = {model.observe(s) for s, _ in belief}
        if len(observations) != 1:
            raise ValueError("one belief must condition on one current observation")
        domains = {future_actions(s) for s, _ in belief}
        if len(domains) != 1:
            raise ValueError("action availability exposes hidden information")
        best_value, best_action = Fraction(-1), None
        for action in next(iter(domains)):
            terminal_reward, successors = successor_beliefs(belief, action)
            value = terminal_reward + sum(
                (mass * solve(posterior)[0] for mass, posterior in successors),
                Fraction(0))
            if value > best_value:
                best_value, best_action = value, action
        return best_value, best_action

    value, action = solve(prior)
    return {"mean_success": float(value), "exact_fraction": str(value),
            "first_action": action, "belief_states_solved": solve.cache_info().currsize,
            "future_policy_class": "unrestricted future observation-history policies in this finite model",
            "future_look": "allowed, including redundant repeated looks while budget permits"}


def prefix_retention_oracle(budget: int) -> dict:
    """Compare retaining the prefix bit with erasing it once at handover.

    The same physical worlds and uniform prior are used in both cases.  The
    full-prefix value averages singleton beliefs because the prefix revealed
    the bit.  The erased-prefix value starts from their mixture.  Erasure
    removes no physical state and does not erase any future observation.
    """
    model = CabinetModel(budget)
    states = []
    for bit in (0, 1):
        state = model.step(model.initial(bit), "advance")
        assert isinstance(state, State)
        states.append(state)
    assert model.observe(states[0]) == model.observe(states[1])
    known = [exact_future_history_value(model, ((s, Fraction(1)),)) for s in states]
    unknown = exact_future_history_value(
        model, tuple((s, Fraction(1, 2)) for s in states))
    full = sum((Fraction(k["exact_fraction"]) for k in known), Fraction(0)) / 2
    erased = Fraction(unknown["exact_fraction"])
    expected_erased = Fraction(1, 2) if budget == 3 else Fraction(1)
    assert full == 1 and erased == expected_erased
    return {"budget": budget, "uniform_world_prior": ["1/2", "1/2"],
            "intervention": "erase only the prefix bit once after advance; retain all future evidence",
            "initial_observation": list(model.observe(states[0])),
            "full_prefix_mean_success": float(full),
            "erased_prefix_mean_success": float(erased),
            "retention_loss": float(full - erased),
            "retention_loss_exact": str(full - erased),
            "full_prefix_singleton_oracles": known,
            "erased_prefix_belief_oracle": unknown,
            "not_a_reactive_bound": True}


def run_case(name: str, model: CabinetModel, expected_reactive: float) -> dict:
    before = [model.observe(model.step(model.initial(bit), "advance")) for bit in (0, 1)]
    assert before[0] == before[1], "nominal sealed input must match, including clocks"
    reactive = exact_reactive_optimum(model)
    history = history_lower_bound(model)
    assert reactive["mean_success_upper_bound"] == expected_reactive
    assert history["mean_success_lower_bound"] == 1.0
    return {"name": name, "budget": model.budget,
            "marker_at_visible_cue": model.marker_at_cue,
            "marker_only_after_cue_disappears": model.marker_when_sealed,
            "nominal_sealed_observation": list(before[0]),
            "reactive": reactive, "history": history,
            "certified_gap_within_declared_class":
                history["mean_success_lower_bound"] - reactive["mean_success_upper_bound"]}


def main() -> None:
    cases = [
        run_case("tight_budget", CabinetModel(3), 0.5),
        run_case("look_is_affordable", CabinetModel(5), 1.0),
        run_case("observable_marker_can_carry_seen_bit", CabinetModel(3, marker_at_cue=True), 1.0),
        run_case("blind_marker_does_not_create_information", CabinetModel(3, marker_when_sealed=True), 0.5),
    ]
    output = {"status": "all four reactive checks and both prefix-retention checks passed",
              "goal": "retrieve the target; identical in both hidden worlds",
              "world_prior": [0.5, 0.5],
              "option_costs_after_prefix": {"commit": 3, "look": 2, "blind_mark_once": 0},
              "scope": [
                  "Finite deterministic two-cabinet model only; no simulator execution.",
                  "Every deterministic time/budget-indexed reactive policy is enumerated.",
                  "The acyclic once-per-key condition extends this toy optimum to behavioral randomized reactive policies; it is not a general stationary-POMDP result.",
                  "No certificate for continuous VLAs or an unmodeled action/observation interface.",
                  "No option has access to the hidden bit except the explicit look sensor.",
                  "A stored marker records the selected action bit, never an oracle bit.",
                  "Missing physical actions, uncontrolled option memory, or observation side channels invalidate transfer of this bound.",
                  "The budget starts after a fixed prefix. Prefix marker writing is explicitly permitted only in its positive control.",
              ], "cases": cases,
              "prefix_retention_oracles": [prefix_retention_oracle(3),
                                            prefix_retention_oracle(5)]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
