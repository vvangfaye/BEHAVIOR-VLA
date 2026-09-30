"""Compile PUBLIC BDDL text, never a simulator snapshot, to grounded goal clauses.

The score here is a research surrogate: maximum fraction of satisfied literals in
a DNF clause. It is NOT a replacement for the official evaluator's q_score.
Unsupported operators and combinatorial explosions raise, rather than silently
changing a task. A true static init relation is not evidence of its current truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product, combinations
import hashlib
import math
import re


class UnsupportedGoal(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Atom:
    predicate: str
    arguments: tuple[str, ...]

    @property
    def key(self) -> str:
        return "(" + " ".join((self.predicate, *self.arguments)) + ")"


@dataclass(frozen=True, order=True)
class Literal:
    atom: Atom
    positive: bool = True

    @property
    def key(self) -> str:
        return self.atom.key if self.positive else f"(not {self.atom.key})"


@dataclass(frozen=True)
class GoalProgram:
    name: str
    objects: dict[str, str]
    init: tuple[str, ...]
    clauses: tuple[tuple[Literal, ...], ...]
    source_sha256: str

    @property
    def atoms(self) -> tuple[Atom, ...]:
        return tuple(sorted({lit.atom for clause in self.clauses for lit in clause}))

    def query(self, atom: Atom) -> str:
        """A static query for a visual verifier, with explicitly initial context."""
        names = ", ".join(f"{a} ({self.objects[a]})" for a in atom.arguments)
        relevant = [s for s in self.init if set(atom.arguments) & set(re.findall(r"[^\s()]+", s))]
        initial = "; ".join(relevant)
        return (f"Task: {self.name}. Estimate whether {atom.key} is true NOW. "
                f"Objects: {names}. Public INITIAL constraints (not current evidence): {initial}")

    def score(self, truth: dict[Atom, bool]) -> float:
        if not self.clauses:
            return 0.0
        return max(sum(truth[l.atom] == l.positive for l in c) / len(c) if c else 1.0
                   for c in self.clauses)

    def as_dict(self) -> dict:
        return {"name": self.name, "source_sha256": self.source_sha256,
                "objects": self.objects, "public_init": list(self.init),
                "clauses": [[l.key for l in c] for c in self.clauses],
                "queries": {a.key: self.query(a) for a in self.atoms}}


def parse_sexpr(text: str) -> list:
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack: list[list] = []
    roots = []
    for token in tokens:
        if token == "(":
            node: list = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unexpected ')' in BDDL")
            stack.pop()
        else:
            if not stack:
                raise ValueError("Token outside expression")
            stack[-1].append(token)
    if stack or len(roots) != 1:
        raise ValueError("Expected one balanced BDDL expression")
    return roots[0]


def _render(node) -> str:
    return "(" + " ".join(map(_render, node)) + ")" if isinstance(node, list) else node


def _typed(tokens: list[str]) -> dict[str, str]:
    result, pending = {}, []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not isinstance(t, str):
            raise UnsupportedGoal("Nested type declarations are not supported")
        if t == "-":
            if not pending or i + 1 >= len(tokens):
                raise ValueError("Malformed typed object list")
            typ = tokens[i + 1]
            for name in pending:
                if name in result:
                    raise ValueError(f"Duplicate declaration: {name}")
                result[name] = typ
            pending = []
            i += 2
        else:
            pending.append(t)
            i += 1
    for name in pending:
        if name in result:
            raise ValueError(f"Duplicate declaration: {name}")
        result[name] = "object"
    return result


def compile_bddl(text: str, max_clauses: int = 1024, max_atoms: int = 512) -> GoalProgram:
    tree = parse_sexpr(text)
    if not tree or tree[0] != "define":
        raise ValueError("Expected a BDDL define expression")
    sections = {s[0]: s[1:] for s in tree[1:] if isinstance(s, list) and s}
    if ":goal" not in sections or ":objects" not in sections:
        raise ValueError("BDDL requires :objects and :goal")
    if not sections[":goal"]:
        raise ValueError("Expected at least one :goal expression")
    objects = _typed(sections[":objects"])
    name = sections.get("problem", ["unnamed"])[0]

    def clean(clauses):
        out = set()
        for clause in clauses:
            c = frozenset(clause)
            if any(Literal(l.atom, not l.positive) in c for l in c):
                continue
            out.add(tuple(sorted(c)))
            if len(out) > max_clauses:
                raise UnsupportedGoal("DNF expansion exceeds max_clauses; use base policy")
        return sorted(out)

    def combine(parts, conjunction):
        if not conjunction:
            return clean(c for part in parts for c in part)
        out = [()]
        for part in parts:
            if len(out) * len(part) > max_clauses:
                raise UnsupportedGoal("DNF expansion exceeds max_clauses; use base policy")
            out = clean(a + b for a in out for b in part)
        return out

    def visit(node, env, negative=False):
        if not isinstance(node, list) or not node:
            raise ValueError("Malformed goal expression")
        op, *args = node
        if op == "not":
            if len(args) != 1:
                raise ValueError("not requires one expression")
            return visit(args[0], env, not negative)
        if op in ("and", "or"):
            return combine([visit(a, env, negative) for a in args], (op == "and") != negative)
        if op in ("forall", "exists"):
            if len(args) != 2 or not isinstance(args[0], list):
                raise ValueError("Malformed quantifier")
            variables = _typed(args[0])
            if not all(v.startswith("?") for v in variables):
                raise ValueError("Quantified names must start with ?")
            pools = [[o for o, t in objects.items() if typ == "object" or t == typ]
                     for typ in variables.values()]
            if any("*" in o for pool in pools for o in pool):
                raise UnsupportedGoal("Wildcard object domain requires public resolved bindings; cannot ground as one object")
            if any(len(p) > max_atoms for p in pools):
                raise UnsupportedGoal("Quantifier domain too large")
            if math.prod(map(len, pools)) > max_clauses:
                raise UnsupportedGoal("Quantifier expansion too large")
            assignments = list(product(*pools))
            parts = [visit(args[1], env | dict(zip(variables, values)), negative)
                     for values in assignments]
            return combine(parts, (op == "forall") != negative)
        if op == "forn":
            if len(args) != 3 or not isinstance(args[0], list) or len(args[0]) != 1:
                raise ValueError("Malformed exact-count quantifier")
            count = int(args[0][0]); variables = _typed(args[1])
            if count < 0 or len(variables) != 1: raise UnsupportedGoal("forn requires one typed variable and a nonnegative count")
            variable, typ = next(iter(variables.items()))
            if not variable.startswith("?"): raise ValueError("Quantified names must start with ?")
            pool = [o for o, t in objects.items() if typ == "object" or t == typ]
            if any("*" in o for o in pool): raise UnsupportedGoal("Wildcard object domain requires public resolved bindings")
            if len(pool) > max_atoms: raise UnsupportedGoal("Quantifier domain too large")
            if count > len(pool): return [()] if negative else []
            if math.comb(len(pool), count) > max_clauses: raise UnsupportedGoal("Exact-count expansion exceeds max_clauses")
            yes = [visit(args[2], env | {variable: o}) for o in pool]
            no = [visit(args[2], env | {variable: o}, True) for o in pool]
            # Official NQuantifier.evaluate requires EXACTLY N, not >= N.
            # Explicitly negate nonselected children to preserve that semantics.
            exact = combine([combine([yes[i] if i in chosen else no[i] for i in range(len(pool))], True)
                             for chosen in map(set, combinations(range(len(pool)), count))], False)
            if not negative: return exact
            return combine([[tuple([Literal(l.atom, not l.positive)]) for l in clause] for clause in exact], True)
        if op == "imply":
            if len(args) != 2: raise ValueError("imply requires two expressions")
            return visit(["or", ["not", args[0]], args[1]], env, negative)
        if op in ("when", "forpairs", "fornpairs"):
            raise UnsupportedGoal(f"Unsupported logical operator: {op}")
        if any(isinstance(a, list) for a in args):
            raise UnsupportedGoal(f"Unsupported nested expression: {op}")
        # BDDL goal leaves also spell explicitly declared instances with a '?'
        # prefix, e.g. ?radio_receiver.n.01_1, without an enclosing quantifier.
        # Resolve only exact declared instances; do not treat arbitrary free
        # variables as object references. Bound quantifier variables take priority.
        grounded = tuple(env[a] if a in env else a[1:] if a.startswith("?") and a[1:] in objects else a
                         for a in args)
        if any(a.startswith("?") for a in grounded):
            raise ValueError("Unbound goal variable")
        if any("*" in a for a in grounded):
            raise UnsupportedGoal("Wildcard object reference needs public resolved bindings")
        if op == "=":
            if len(grounded) != 2:
                raise ValueError("Equality requires two arguments")
            return [()] if (grounded[0] == grounded[1]) != negative else []
        if any(a not in objects for a in grounded):
            raise UnsupportedGoal(f"Undeclared goal object in {op}{grounded}")
        return [(Literal(Atom(op, grounded), not negative),)]

    # Official evaluate_state treats multiple top-level goal conditions as AND.
    clauses = tuple(clean(visit(["and", *sections[":goal"]], {})))
    program = GoalProgram(name, objects, tuple(_render(i) for i in sections.get(":init", [])),
                          clauses, hashlib.sha256(text.encode()).hexdigest())
    if len(program.atoms) > max_atoms:
        raise UnsupportedGoal("Too many grounded atoms")
    return program
