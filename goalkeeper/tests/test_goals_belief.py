from itertools import product
import pytest
from goalkeeper.goals import compile_bddl, UnsupportedGoal
from goalkeeper.belief import Estimate, UNKNOWN, BeliefMemory, PreservationSelector, lower_progress


def task(goal, objects="a b - cup box - cabinet", init=""):
    return compile_bddl(f"(define (problem test) (:objects {objects}) (:init {init}) (:goal {goal}))")


def test_quantification_and_negation():
    g = task("(forall (?x - cup) (not (or (open ?x) (inside ?x box))))")
    assert len(g.clauses) == 1 and len(g.atoms) == 4
    assert all(not lit.positive for lit in g.clauses[0])
    e = task("(exists (?x - cup) (inside ?x box))")
    assert len(e.clauses) == 2
    assert e.score({a: a.arguments[0] == "b" for a in e.atoms}) == 1


def test_equality_empty_domain_and_contradiction():
    assert task("(forall (?x - missing) (open ?x))").score({}) == 1
    assert task("(exists (?x - missing) (open ?x))").score({}) == 0
    assert task("(and (open a) (not (open a)))").score({}) == 0
    assert task("(= a a)").score({}) == 1


def test_reject_unsupported_and_explosion():
    with pytest.raises(UnsupportedGoal): task("(forpairs (?x - cup) (?y - cup) (touching ?x ?y))")
    with pytest.raises(ValueError): task("(open ?missing)")
    with pytest.raises(UnsupportedGoal): task("(open missing)")
    with pytest.raises(UnsupportedGoal):
        compile_bddl("(define (problem p) (:objects a b c - cup) (:goal (exists (?x - cup) (open ?x))))", max_clauses=2)


def test_init_is_not_live_truth_and_memory_can_revoke():
    g = task("(open a)", init="(open a)")
    a, = g.atoms
    m = BeliefMemory(g)
    assert m.values[a] == UNKNOWN and "INITIAL" in g.query(a)
    yes, no = Estimate(.99, .97, 1), Estimate(.01, 0, .03)
    m.update({a: yes}, {a: 1})
    hidden = m.update({a: no}, {a: 0})[a]
    assert hidden.lower < yes.lower and hidden.mean == yes.mean
    assert m.update({a: no}, {a: 1})[a] == no
    m.reset(); assert m.values[a] == UNKNOWN


def test_preservation_rejects_regression_and_abstains_when_unknown():
    g = task("(and (open a) (open b))")
    a, b = g.atoms
    yes, no = Estimate(1, .98, 1), Estimate(0, 0, .02)
    current = {a: yes, b: no}
    bad, good = {a: no, b: yes}, {a: yes, b: yes}
    selector = PreservationSelector()
    assert selector.choose(g, current, [bad, good]).index == 1
    unknown = {a: UNKNOWN, b: UNKNOWN}
    assert selector.choose(g, unknown, [unknown, unknown]).index == 0


def test_temporary_regression_must_not_be_blocked_by_idle_candidate():
    g = task("(and (inside a box) (not (open box)))")
    inside = next(a for a in g.atoms if a.predicate == "inside")
    opened = next(a for a in g.atoms if a.predicate == "open")
    yes, no = Estimate(1, 1, 1), Estimate(0, 0, 0)
    current = {inside: no, opened: no}
    base_open = {inside: no, opened: yes}
    idle = current.copy()
    result = PreservationSelector().choose(g, current, [base_open, idle])
    assert result.index == 0  # permits a necessary temporary decrease in goal score


def test_protect_only_one_consistent_goal_alternative():
    g = task("(or (open a) (not (open a)))")
    a, = g.atoms
    current = {a: Estimate(1, 1, 1)}
    result = PreservationSelector().choose(g, current, [current])
    assert result.protected == ("(open a)",)


def test_conditional_lower_progress_and_union_bound_without_independence():
    g = task("(or (and (open a) (not (open b))) (and (open b) (open box)))")
    atoms = g.atoms
    worlds = list(product([False, True], repeat=len(atoms)))
    weights = [1, 2, 1, 5, 2, 1, 7, 3]
    weights = [w / sum(weights) for w in weights]
    p = [sum(w * state[j] for w, state in zip(weights, worlds)) for j in range(len(atoms))]
    beliefs = {a: Estimate(v, max(0, v - .03), min(1, v + .03)) for a, v in zip(atoms, p)}
    expected = sum(w * g.score(dict(zip(atoms, state))) for w, state in zip(weights, worlds))
    assert lower_progress(g, beliefs) <= expected + 1e-12
    violation = sum(w for w, s in zip(weights, worlds) if not all(s))
    assert violation <= min(1, sum(1 - v.lower for v in beliefs.values())) + 1e-12


def test_duplicate_declarations_fail():
    with pytest.raises(ValueError): task("(open a)", objects="a - cup a - cup")


def test_official_bddl_instance_reference_syntax():
    # Verified against turning_on_radio/problem0.bddl in v3.9.2/bddl3.
    g = task("(and (toggled_on ?radio_receiver.n.01_1))",
             objects="radio_receiver.n.01_1 - radio_receiver.n.01")
    assert g.atoms[0].arguments == ("radio_receiver.n.01_1",)
    quantified = task("(exists (?a - cup) (open ?a))")
    assert {a.arguments[0] for a in quantified.atoms} == {"a", "b"}


def test_exact_count_is_not_at_least_count():
    g = task("(forn (2) (?x - cup) (open ?x))", objects="a b c - cup")
    neg = task("(not (forn (2) (?x - cup) (open ?x)))", objects="a b c - cup")
    for values in product([False, True], repeat=3):
        truth = dict(zip(g.atoms, values))
        assert (g.score(truth) == 1) == (sum(values) == 2)
        assert (neg.score(truth) == 1) == (sum(values) != 2)


def test_multiple_goal_roots_and_implication():
    g = task("(open a) (not (open b))")
    assert len(g.clauses) == 1 and len(g.clauses[0]) == 2
    implication = task("(imply (open a) (open b))")
    assert implication.score({a: False for a in implication.atoms}) == 1


def test_wildcard_not_treated_as_single_named_object():
    with pytest.raises(UnsupportedGoal, match="Wildcard"):
        task("(forall (?x - cup) (open ?x))", objects="a cup_* - cup")
