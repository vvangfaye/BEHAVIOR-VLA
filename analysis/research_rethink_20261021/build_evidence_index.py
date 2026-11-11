"""Index existing method proposals and verdicts without running experiments.

This is a discovery aid. Excerpts are not independent evidence and do not
replace opening a source or checking a cited result's scope.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("method_evidence_index.json")
DIRECTORIES = (
    "analysis/training_boundary_audit_20261019",
    "analysis/research_rethink_20261021",
    "analysis/action_support_20261021",
    "analysis/relative_flow_20261021",
    "analysis/joint_outcome_critic_20261021",
    "analysis/likelihood_ratio_policy_20261021",
    "analysis/latent_posterior_credit_20261021",
    "analysis/interaction_residual_20261021",
    "analysis/event_feedback_20261021",
    "analysis/replanning_response_20261021",
    "analysis/fiber_credit_20261021",
)
SKIP = {"venv", "__pycache__", "episodes", "frames", "features", "receipts",
        "retrieved", "selected_frames_v1", "natural_order_frames"}
MARKERS = ("proposal", "candidate", "audit", "decision", "conclusion",
           "reassessment", "readiness", "status", "method_spec")
KEYS = ("name", "candidate", "working_name", "hypothesis", "status", "decision",
        "conclusion", "recommendation_strength", "next_method_boundary",
        "interpretation", "feasibility_gaps", "unresolved")


def summarize(path: Path) -> dict | None:
    if path == OUTPUT or not any(word in path.stem for word in MARKERS):
        return None
    if path.suffix not in {".json", ".md"}:
        return None
    raw = path.read_text()
    entry = {"path": str(path.relative_to(ROOT)),
             "urls": sorted(set(re.findall(r"https?://[^\s\"<>\\)]+", raw)))}
    if path.suffix == ".json":
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        summary = {key: value[key] for key in KEYS if key in value}
        if not summary:
            return None
        entry["summary"] = summary
    else:
        paragraphs = [part.strip() for part in raw.split("\n\n") if part.strip()]
        entry["excerpt"] = "\n\n".join(paragraphs[:3])[:1800]
    return entry


def main() -> None:
    entries = []
    for directory in DIRECTORIES:
        base = ROOT / directory
        for path in sorted(base.rglob("*")):
            if not path.is_file() or any(part in SKIP for part in path.relative_to(base).parts):
                continue
            item = summarize(path)
            if item is not None:
                entries.append(item)
    OUTPUT.write_text(json.dumps({
        "scope": "Existing local method evidence; not a new experiment or novelty certification",
        "entries": entries,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"entries": len(entries), "output": str(OUTPUT),
                      "source_paths_exist": all((ROOT / item["path"]).is_file() for item in entries)}))


if __name__ == "__main__":
    main()
