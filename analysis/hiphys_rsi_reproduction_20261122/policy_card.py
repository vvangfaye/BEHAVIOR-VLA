"""Freeze an episode routing card from paired, full-horizon train rollout results."""

import argparse
import json
from pathlib import Path
from statistics import mean


CONTEXTS = ("indicator_visible", "indicator_not_visible")
METHODS = ("pipeline", "vla")


def build(pipeline, vla):
    for method, report in zip(METHODS, (pipeline, vla)):
        if (report["candidate"] != method or report["task"] != "turning_on_radio" or report["mode"] != "train" or
                not report["complete"] or report["max_steps_override"] is not None):
            raise ValueError(f"Expected a completed full-horizon train run for {method}")
    for field in ("policy_sha256", "urdf_sha256", "instance_ids"):
        if pipeline[field] != vla[field]:
            raise ValueError(f"Training comparisons differ in {field}")
    if not pipeline["instance_ids"] or any(not 0 <= i < 300 for i in pipeline["instance_ids"]):
        raise ValueError("Use only the predeclared training instances")

    # The same initial scene and detector should define the context for both candidates.
    pairs = [(p, v) for p, v in zip(pipeline["results"], vla["results"])
             if p["instance_id"] == v["instance_id"] and p["context"] == v["context"]]
    if len(pairs) != len(pipeline["instance_ids"]):
        raise ValueError("Paired training episodes must agree in instance order and initial context")

    routes = {}
    evidence = {}
    for context in CONTEXTS:
        group = [pair for pair in pairs if pair[0]["context"] == context]
        candidates = group or pairs
        stats = {method: {"mean_q": mean(pair[k]["q"] for pair in candidates),
                          "success_rate": mean(pair[k]["success"] for pair in candidates)}
                 for k, method in enumerate(METHODS)}
        routes[context] = max(METHODS, key=lambda method: (stats[method]["mean_q"], stats[method]["success_rate"],
                                                           method == "vla"))
        evidence[context] = {"paired_instances": len(group), "uses_all_train_pairs": not group, "candidates": stats}
    return {"task": pipeline["task"], "policy_sha256": pipeline["policy_sha256"], "urdf_sha256": pipeline["urdf_sha256"],
            "routes": routes, "train_evidence_by_context": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-train", type=Path, required=True)
    parser.add_argument("--vla-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [args.pipeline_train.resolve(), args.vla_train.resolve()]
    card = build(*(json.loads(source.read_text()) for source in sources))
    card["training_evidence"] = [str(source) for source in sources]
    with args.output.open("x") as out:
        out.write(json.dumps(card, indent=2) + "\n")


if __name__ == "__main__":
    main()
