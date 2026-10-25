#!/usr/bin/env python3
"""Compile reviewed evidence certificates into legal history selections (CPU only).

This is a certificate checker, NOT a visual annotator, tracker, fact extractor,
model trainer, or proof that an unobserved world state stayed unchanged.
The caller supplies a reviewed AND/OR proof graph: all nodes/edges of a proof
are required (AND); alternative proof_ids for a requirement are alternatives
(OR). Edges point from evidence toward the proof's claim root. A proof records
required identity edges explicitly, so a detached witness is not sufficient.

Input schema (see synthetic_fixture() for a complete runnable example):
  frames: [{id, step, view, ...source metadata...}]
  next_step, fixed_next_frame_ids, history_frame_ids (chronological candidates)
  graph: {nodes: [{id, kind, reviewed, frame_ids, ...claim fields...}],
          edges: [{id, kind, source, target, reviewed, frame_ids}]}
  proofs: [{id, root, node_ids, edge_ids, identity_edge_ids}]
  requirements: {guards: [...], keep: [...], write: [...]}
    Each requirement is {fact_key, value, proof_ids, confirmed_step}.
  revisions: [{fact_key, value, at_step, proof_ids}]
    These are DECLARED reviewed revisions. Latest means latest in this supplied
    ledger, not latest in every frame of an uninspected physical recording.
  selection_groups (optional): [{frame_ids, min_count, max_count}]

Historical proofs must use frames strictly before next_step. Write proofs may
also use the fixed next observation, never other future frames. Each selected
history must retain a complete proof for every guard/K/W requirement and the
latest declared historical revision of every guard/K fact. The returned
loader_manifest uses frame IDs to select source images; IDs must NOT be tokenized
as model features. Proof IDs, native IDs and labels are supervision metadata.
No counterfactual motor target is constructed. Explicit unknown_fact_keys are
never permitted as known guard/K/W labels.

Usage:
  python semantic_edit_compiler.py --self-test
  python semantic_edit_compiler.py --demo-episode10710 --max-subsets 16
  python semantic_edit_compiler.py --input reviewed_graph.json --output out.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


class CertificateError(ValueError):
    """A supplied reviewed certificate is incomplete or temporally invalid."""


def _index(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise CertificateError(f"{name}: missing or duplicate id {key!r}")
        result[key] = row
    return result


class EvidenceCompiler:
    def __init__(self, spec: dict[str, Any]):
        self.spec = copy.deepcopy(spec)
        self.frames = _index(spec["frames"], "frames")
        self.nodes = _index(spec["graph"]["nodes"], "nodes")
        self.edges = _index(spec["graph"]["edges"], "edges")
        self.proofs = _index(spec["proofs"], "proofs")
        self.next_step = spec["next_step"]
        self.fixed = set(spec["fixed_next_frame_ids"])
        history = spec["history_frame_ids"]
        if len(history) != len(set(history)):
            raise CertificateError("duplicate history frame")
        self.history = set(history)
        if not isinstance(self.next_step, int) or self.next_step < 0:
            raise CertificateError("next_step must be a nonnegative integer")
        for f in self.frames.values():
            if not isinstance(f.get("step"), int) or f["step"] < 0:
                raise CertificateError(f"invalid frame step: {f['id']}")
        if not self.fixed or not (self.fixed | self.history) <= self.frames.keys():
            raise CertificateError("unknown or empty fixed/history frame references")
        if self.fixed & self.history:
            raise CertificateError("next input cannot also be history")
        if any(self.frames[f]["step"] != self.next_step for f in self.fixed):
            raise CertificateError("fixed next frames must be at next_step")
        if any(self.frames[f]["step"] >= self.next_step for f in self.history):
            raise CertificateError("future/current frame appears in history candidates")
        self.proof_frames: dict[str, set[str]] = {}
        self._validate_graph()
        self.clauses: list[dict] = []
        for role in ("guards", "keep", "write"):
            for req in spec.get("requirements", {}).get(role, []):
                self._add_clause(role, req)
        if not self.clauses:
            raise CertificateError("no semantic requirements")
        self._add_latest_revision_clauses()
        self._check_label_consistency()
        for group in spec.get("selection_groups", []):
            ids = group["frame_ids"]
            lo, hi = group["min_count"], group["max_count"]
            if len(ids) != len(set(ids)) or not set(ids) <= self.history:
                raise CertificateError("invalid selection-group frame references")
            if not 0 <= lo <= hi <= len(ids):
                raise CertificateError("invalid selection-group cardinality")

    def _validate_graph(self) -> None:
        for node in self.nodes.values():
            if node.get("reviewed") is not True:
                raise CertificateError(f"unreviewed node: {node['id']}")
            fs = set(node.get("frame_ids", []))
            if not fs <= self.frames.keys():
                raise CertificateError(f"unknown source frame: {node['id']}")
            if node["kind"] == "witness" and not fs:
                raise CertificateError(f"witness lacks source frame: {node['id']}")
        for edge in self.edges.values():
            if edge.get("reviewed") is not True:
                raise CertificateError(f"unreviewed edge: {edge['id']}")
            if edge["source"] not in self.nodes or edge["target"] not in self.nodes:
                raise CertificateError(f"edge has unknown endpoint: {edge['id']}")
            if not set(edge.get("frame_ids", [])) <= self.frames.keys():
                raise CertificateError(f"edge has unknown source frame: {edge['id']}")
        for pid, proof in self.proofs.items():
            ns, es = set(proof["node_ids"]), set(proof["edge_ids"])
            if not ns <= self.nodes.keys() or not es <= self.edges.keys():
                raise CertificateError(f"proof {pid}: missing node or edge")
            root = proof["root"]
            if root not in ns or self.nodes[root]["kind"] != "claim":
                raise CertificateError(f"proof {pid}: invalid claim root")
            mandatory_identity = set(proof.get("identity_edge_ids", []))
            if not mandatory_identity <= es:
                raise CertificateError(f"proof {pid}: required identity edge omitted")
            if any(self.edges[e]["kind"] != "identity" for e in mandatory_identity):
                raise CertificateError(f"proof {pid}: wrong identity-edge type")
            adjacency = {n: [] for n in ns}
            for eid in es:
                edge = self.edges[eid]
                if edge["source"] not in ns or edge["target"] not in ns:
                    raise CertificateError(f"proof {pid}: detached edge endpoint")
                adjacency[edge["source"]].append(edge["target"])
            if adjacency[root]:
                raise CertificateError(f"proof {pid}: claim root is not a sink")
            done: dict[str, bool] = {}

            def reaches_root(n: str, visiting: set[str]) -> bool:
                if n == root:
                    return True
                if n in visiting:
                    raise CertificateError(f"proof {pid}: cyclic proof")
                if n not in done:
                    children = adjacency[n]
                    # Every declared premise branch must end at the claim.
                    done[n] = bool(children) and all(
                        reaches_root(c, visiting | {n}) for c in children
                    )
                return done[n]

            if not all(reaches_root(n, set()) for n in ns):
                raise CertificateError(f"proof {pid}: disconnected premise/identity chain")
            if not any(self.nodes[n]["kind"] == "witness" for n in ns):
                raise CertificateError(f"proof {pid}: no visual witness")
            self.proof_frames[pid] = set().union(
                *(set(self.nodes[n].get("frame_ids", [])) for n in ns),
                *(set(self.edges[e].get("frame_ids", [])) for e in es),
            )

    def _add_clause(self, role: str, req: dict) -> None:
        if req["fact_key"] in self.spec.get("unknown_fact_keys", []):
            raise CertificateError(f"{role}: unknown fact cannot become a supervised label")
        if req.get("epistemic_status", "known") != "known":
            raise CertificateError(f"{role}: only reviewed known facts are supported")
        pids = req.get("proof_ids", [])
        if not pids or len(pids) != len(set(pids)):
            raise CertificateError(f"{role}: missing or duplicate proof alternatives")
        options = []
        for pid in pids:
            if pid not in self.proofs:
                raise CertificateError(f"{role}: unknown proof {pid}")
            root = self.nodes[self.proofs[pid]["root"]]
            if (root.get("fact_key"), root.get("value")) != (req["fact_key"], req["value"]):
                raise CertificateError(f"{role}: proof {pid} supports a different fact")
            if root.get("confirmed_step") != req["confirmed_step"]:
                raise CertificateError(f"{role}: proof {pid} has the wrong confirmation time")
            fs = self.proof_frames[pid]
            allowed = self.history | self.fixed if role == "write" else self.history
            if not fs <= allowed:
                raise CertificateError(f"{role}: proof {pid} uses future/unavailable evidence")
            limit = self.next_step if role == "write" else self.next_step - 1
            if req["confirmed_step"] > limit:
                raise CertificateError(f"{role}: future confirmation time")
            if not any(self.frames[f]["step"] == req["confirmed_step"] for f in fs):
                raise CertificateError(f"{role}: confirmation time has no witness provenance")
            options.append({"proof_id": pid, "required_history": fs & self.history})
        self.clauses.append({"role": role, "label": copy.deepcopy(req), "options": options})

    def _add_latest_revision_clauses(self) -> None:
        protected = {
            c["label"]["fact_key"]: c["label"]["value"]
            for c in self.clauses if c["role"] in ("guards", "keep")
        }
        for key, value in protected.items():
            rows = [r for r in self.spec.get("revisions", [])
                    if r["fact_key"] == key and r["at_step"] < self.next_step]
            if not rows:
                raise CertificateError(f"no declared revision ledger for protected fact {key}")
            latest_time = max(r["at_step"] for r in rows)
            latest = [r for r in rows if r["at_step"] == latest_time]
            if any(r.get("reviewed") is not True or r["value"] != value for r in latest):
                raise CertificateError(f"latest revision contradicts protected fact {key}")
            for rev in latest:
                self._add_clause("latest_revision", {
                    "fact_key": key, "value": value, "confirmed_step": latest_time,
                    "proof_ids": rev["proof_ids"],
                })

    def _check_label_consistency(self) -> None:
        seen = {}
        for c in self.clauses:
            # A write can intentionally revise a guard. A keep cannot conflict
            # with a write at this update. Ledger consistency is checked above.
            if c["role"] not in ("keep", "write"):
                continue
            label = c["label"]
            key = label["fact_key"]
            if key in seen and seen[key] != label["value"]:
                raise CertificateError(f"K/W conflict on {key}")
            seen[key] = label["value"]

    def ordered(self, ids: set[str]) -> list[str]:
        return sorted(ids, key=lambda f: (self.frames[f]["step"], self.frames[f]["view"], f))

    def check(self, selection: list[str]) -> dict:
        selected = set(selection)
        errors, certificates = [], []
        if len(selected) != len(selection) or not selected <= self.history:
            errors.append("selection contains duplicate, future, or unknown history frames")
        if selection != self.ordered(selected):
            errors.append("selection is not in chronological frame/view order")
        for group in self.spec.get("selection_groups", []):
            count = len(selected & set(group["frame_ids"]))
            if not group["min_count"] <= count <= group["max_count"]:
                errors.append(f"selection-group cardinality violated: {group['frame_ids']}")
        for c in self.clauses:
            satisfied = [o for o in c["options"] if o["required_history"] <= selected]
            if not satisfied:
                errors.append(f"missing complete {c['role']} proof: {c['label']['fact_key']}")
            else:
                certificates.append({"role": c["role"], "fact_key": c["label"]["fact_key"],
                                     "proof_id": satisfied[0]["proof_id"]})
        return {"legal": not errors, "errors": errors, "certificates": certificates}

    def compile(self, max_subsets: int = 64, max_examined: int = 100000) -> dict:
        if max_subsets < 1 or max_examined < 1:
            raise ValueError("search/output budgets must be positive")
        # Union the alternative support sets clause by clause, retaining only
        # inclusion-minimal bases. Additional real frames can then be enumerated.
        bases = {frozenset()}
        for clause in self.clauses:
            unions = {b | frozenset(o["required_history"])
                      for b in bases for o in clause["options"]}
            bases = {b for b in unions if not any(a < b for a in unions)}
        subsets, visited, examined, exhausted = [], set(), 0, True
        for base in sorted(bases, key=lambda s: (len(s), self.ordered(set(s)))):
            optional = self.ordered(self.history - base)
            for count in range(len(optional) + 1):
                for extra in itertools.combinations(optional, count):
                    selected = base | frozenset(extra)
                    if selected in visited:
                        continue
                    visited.add(selected)
                    if examined >= max_examined or len(subsets) >= max_subsets:
                        exhausted = False
                        break
                    examined += 1
                    ordered = self.ordered(set(selected))
                    result = self.check(ordered)
                    if result["legal"]:
                        subsets.append({"loader_manifest": {
                            "history_frame_ids": ordered,
                            "fixed_next_frame_ids": self.ordered(self.fixed),
                        }, "certificates": result["certificates"]})
                if not exhausted:
                    break
            if not exhausted:
                break
        labels = {
            name: [{k: v for k, v in req.items() if k != "proof_ids"}
                   for req in self.spec["requirements"].get(role, [])]
            for name, role in (("G", "guards"), ("K", "keep"), ("W", "write"))
        }
        return {"schema_version": 1, "episode": self.spec.get("episode"),
                "labels": labels, "legal_history_subsets": subsets,
                "unknown_fact_keys": self.spec.get("unknown_fact_keys", []),
                "examined_candidates": examined, "enumeration_exhausted": exhausted,
                "scope": "Validity relative to supplied reviewed graph and revision ledger only.",
                "limitations": ["No automatic visual/identity annotation or model training.",
                                "Known facts need reviewed proofs; explicitly unknown facts are not supervised.",
                                "Future evidence is forbidden; fixed next views are allowed only for write proofs.",
                                "Image/frame IDs are loader metadata, never model tokens; native IDs are label-only.",
                                "Next means the fixed next sampled observation, not necessarily the adjacent native frame.",
                                "Latest revision means latest declared reviewed confirmation, not a dense visual audit.",
                                "No proof of continuous observability or unchanged hidden world state.",
                                "No counterfactual motor labels or formal retention guarantee.",
                                "Certificate compilation does not establish novelty of a training mechanism."]}


def _add_proof(spec: dict, pid: str, key: str, value: str, step: int,
               chain: list[str]) -> dict:
    """Fixture builder: each chain edge is an explicitly reviewed identity link."""
    nodes, edges = spec["graph"]["nodes"], spec["graph"]["edges"]
    nids, eids, identities = [], [], []
    for i, frame in enumerate(chain):
        nid = f"{pid}:w{i}"
        nodes.append({"id": nid, "kind": "witness", "reviewed": True,
                      "frame_ids": [frame]})
        nids.append(nid)
        if i:
            eid = f"{pid}:identity{i}"
            edges.append({"id": eid, "kind": "identity", "reviewed": True,
                          "source": nids[i - 1], "target": nid,
                          "frame_ids": [chain[i - 1], frame]})
            eids.append(eid)
            identities.append(eid)
    root, eid = f"{pid}:claim", f"{pid}:supports"
    nodes.append({"id": root, "kind": "claim", "reviewed": True, "frame_ids": [],
                  "fact_key": key, "value": value, "confirmed_step": step})
    edges.append({"id": eid, "kind": "support", "reviewed": True,
                  "source": nids[-1], "target": root, "frame_ids": []})
    spec["proofs"].append({"id": pid, "root": root, "node_ids": nids + [root],
                           "edge_ids": eids + [eid], "identity_edge_ids": identities})
    return {"fact_key": key, "value": value, "confirmed_step": step, "proof_ids": [pid]}


def synthetic_fixture() -> dict:
    spec = {"episode": "synthetic", "next_step": 3,
            "frames": [{"id": f"f{i}", "step": i, "view": "head"} for i in range(1, 5)],
            "history_frame_ids": ["f1", "f2"], "fixed_next_frame_ids": ["f3"],
            "graph": {"nodes": [], "edges": []}, "proofs": [],
            "requirements": {"guards": [], "keep": [], "write": []}, "revisions": []}
    old = _add_proof(spec, "old", "A.location", "table", 1, ["f1"])
    latest = _add_proof(spec, "latest", "A.location", "table", 2, ["f2"])
    identity = _add_proof(spec, "bound", "B.identity", "bound_object", 2, ["f1", "f2"])
    write = _add_proof(spec, "write", "B.location", "shelf_workarea", 3, ["f2", "f3"])
    spec["requirements"] = {"guards": [identity], "keep": [old], "write": [write]}
    spec["revisions"] = [
        {"fact_key": "A.location", "value": "table", "at_step": 2,
         "proof_ids": latest["proof_ids"], "reviewed": True},
        {"fact_key": "B.identity", "value": "bound_object", "at_step": 2,
         "proof_ids": identity["proof_ids"], "reviewed": True},
    ]
    return spec


def episode10710_fixture(verify_files: bool = True) -> dict:
    """Explicit human-reviewed fixture, NOT automatically extracted labels.

    Identity/support review supplied and independently rechecked by
    /root/read_robocop. Native object names
    below are label/scoring keys, never additional model inputs. At 6500 the book
    is still held near the shelving: this fixture does NOT assert shelf placement.
    Frame 6500 is the next sampled observation, not the native frame after 6000.
    """
    directory = Path(__file__).resolve().parent
    audit = json.loads((directory / "witness_frames/visual_audit.json").read_text())
    metadata = {r["file"]: r for r in audit["images"]}
    names = ["head_4101", "right_4101", "head_4440", "right_4440",
             "head_5700", "head_6000", "head_6500", "left_6500", "right_6500"]
    frames = []
    for name in names:
        view, step = name.split("_")
        path = directory / "witness_frames" / f"{name}.png"
        expected = metadata[path.name]["sha256"]
        if verify_files and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise CertificateError(f"source hash mismatch: {path}")
        frames.append({"id": name, "step": int(step), "view": view,
                       "source_path": str(path), "sha256": expected})
    spec = {"episode": 10710, "next_step": 6500, "frames": frames,
            "history_frame_ids": names[:6], "fixed_next_frame_ids": names[6:],
            "graph": {"nodes": [], "edges": []}, "proofs": [],
            "requirements": {"guards": [], "keep": [], "write": []}, "revisions": [],
            "selection_groups": [
                {"frame_ids": ["head_4101", "right_4101"], "min_count": 1, "max_count": 1},
                {"frame_ids": ["head_4440", "right_4440"], "min_count": 1, "max_count": 1}],
            "unknown_fact_keys": ["pot_plant_54.current_support", "hardback_52.placed_in_bookcase"],
            "review_note": "Explicit visual review; sampled frames only; native records are audit-only."}
    a = _add_proof(spec, "plant_support", "pot_plant_54.last_confirmed_support", "coffee_table",
                   6000, ["head_5700", "head_6000"])
    b = _add_proof(spec, "book_context", "hardback_52.last_confirmed_context", "coffee_table_workarea",
                   6000, ["head_5700", "head_6000"])
    hold = _add_proof(spec, "book_hold", "hardback_52.last_confirmed_hold", "left_gripper",
                      6000, ["head_5700", "head_6000"])
    new_context = _add_proof(spec, "book_new_context", "hardback_52.last_confirmed_context",
                             "shelving_workarea", 6500, ["head_5700", "head_6000", "head_6500"])
    new_hold = _add_proof(spec, "book_still_held", "hardback_52.last_confirmed_hold",
                          "left_gripper", 6500, ["head_5700", "head_6000", "left_6500"])
    spec["requirements"] = {"guards": [a, b, hold], "keep": [copy.deepcopy(a)],
                            "write": [new_context, new_hold]}
    for label in (a, b, hold):
        spec["revisions"].append({"fact_key": label["fact_key"], "value": label["value"],
                                  "at_step": 6000, "proof_ids": label["proof_ids"], "reviewed": True})
    return spec


def self_test() -> dict:
    tested = []
    spec = synthetic_fixture()
    compiler = EvidenceCompiler(spec)
    assert compiler.check(["f1", "f2"])["legal"]
    tested.append("complete proof accepted")
    # Give the future frame a correct label: chronology must still reject it.
    future = copy.deepcopy(spec)
    for node in future["graph"]["nodes"]:
        if node["id"] == "old:w0":
            node["frame_ids"] = ["f4"]
        if node["id"] == "old:claim":
            node["confirmed_step"] = 4
    future["requirements"]["keep"][0]["confirmed_step"] = 4
    _expect_invalid(future, "future witness")
    tested.append("future witness rejected")
    future_write = copy.deepcopy(spec)
    for node in future_write["graph"]["nodes"]:
        if node["id"] == "write:w1":
            node["frame_ids"] = ["f4"]
        if node["id"] == "write:claim":
            node["confirmed_step"] = 4
    future_write["graph"]["edges"][-2]["frame_ids"] = ["f2", "f4"]
    future_write["requirements"]["write"][0]["confirmed_step"] = 4
    _expect_invalid(future_write, "future write witness")
    tested.append("write proof cannot use a later observation beyond the fixed next input")
    missing_edge = copy.deepcopy(spec)
    missing_edge["graph"]["edges"] = [e for e in missing_edge["graph"]["edges"]
                                       if e["id"] != "bound:identity1"]
    _expect_invalid(missing_edge, "missing identity edge")
    tested.append("missing identity edge rejected")
    # Remove every requirement that already forces f2, isolating the revision
    # rule: f1 still proves K, but it may not replace the later confirmation.
    revision_only = copy.deepcopy(spec)
    revision_only["requirements"]["guards"] = []
    revision_only["requirements"]["write"] = []
    result = EvidenceCompiler(revision_only).check(["f1"])
    assert not result["legal"] and any("latest_revision" in e for e in result["errors"])
    tested.append("deleting latest required revision rejected despite older K proof")
    disconnected = copy.deepcopy(spec)
    p = next(p for p in disconnected["proofs"] if p["id"] == "bound")
    p["edge_ids"].remove("bound:identity1")
    p["identity_edge_ids"] = []
    _expect_invalid(disconnected, "detached identity witness")
    tested.append("detached witness cannot bypass identity chain")
    unknown = copy.deepcopy(spec)
    unknown["unknown_fact_keys"] = ["A.location"]
    _expect_invalid(unknown, "unknown fact")
    tested.append("explicitly unknown fact cannot become a known label")
    alternative = copy.deepcopy(spec)
    alternative["frames"].append({"id": "f2b", "step": 2, "view": "wrist"})
    alternative["history_frame_ids"].append("f2b")
    alt = _add_proof(alternative, "alternative", "A.location", "table", 2, ["f2b"])
    alt["proof_ids"].append("latest")
    alternative["requirements"] = {"guards": [], "keep": [alt], "write": []}
    alternative["revisions"][0]["proof_ids"] = alt["proof_ids"]
    assert EvidenceCompiler(alternative).check(["f2b"])["legal"]
    tested.append("one complete alternative proof suffices")
    assert not compiler.check(["f2", "f1"])["legal"]
    tested.append("reordered history rejected")
    compiled = compiler.compile()
    assert len(compiled["legal_history_subsets"]) == 1
    assert compiled["enumeration_exhausted"]
    tested.append("bounded enumeration yields only legal selections")
    real = EvidenceCompiler(episode10710_fixture())
    for selection in (["head_4101", "head_4440", "head_5700", "head_6000"],
                      ["right_4101", "right_4440", "head_5700", "head_6000"]):
        assert real.check(selection)["legal"]
    assert not real.check(["head_4101", "head_4440", "head_5700"])["legal"]
    assert not real.check(["head_4101", "head_4440", "head_6000"])["legal"]
    real_out = real.compile()
    assert len(real_out["legal_history_subsets"]) == 4
    tested.append("episode10710: hashes and two reviewed selections verified; 4 legal camera combinations")
    return {"passed": len(tested), "checks": tested,
            "scope": "Certificate compilation only; no visual annotation, training, or simulation."}


def _expect_invalid(spec: dict, name: str) -> None:
    try:
        EvidenceCompiler(spec)
    except CertificateError:
        return
    raise AssertionError(f"{name} unexpectedly accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--self-test", action="store_true")
    source.add_argument("--demo-episode10710", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-subsets", type=int, default=64)
    parser.add_argument("--max-examined", type=int, default=100000)
    args = parser.parse_args()
    try:
        if args.self_test:
            result = self_test()
        else:
            spec = episode10710_fixture() if args.demo_episode10710 else json.loads(args.input.read_text())
            result = EvidenceCompiler(spec).compile(args.max_subsets, args.max_examined)
    except (CertificateError, KeyError, OSError, json.JSONDecodeError, ValueError) as exc:
        parser.exit(2, f"certificate error: {exc}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
