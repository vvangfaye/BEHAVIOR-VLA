"""Check historical-reference labels against already audited native split events.

This is a symbolic label-construction prototype, not a visual training dataset.
No images, object visibility, or counterfactual motor outcomes are fabricated.
Native entity names below belong to the training/scoring side, never policy input.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "BEHAVIOR-VLAs/eval_results/sco_migration_20261007/transition_native_source_inspection.json"
RAW = Path(__file__).with_name("lineage_data") / "episode_00950310.hdf5"
OUTPUT = Path(__file__).with_name("reference_intervention_prototype_results.json")
QUERY_FRAME = 9140
# These physical cue events stay fixed while their arbitrary names are permuted.
CUE_FRAMES = {
    "brussels_sprouts_93": 7290,
    "brussels_sprouts_95": 7729,
    "brussels_sprouts_94": 8298,
    "brussels_sprouts_92": 8873,
}


def recorded_action_event_evidence(source: dict) -> dict:
    """Check deictic cues and the factual pickup against existing native H5.

    A grasp packet establishes which object was held; it does not establish
    camera visibility, a successful placement, or a counterfactual outcome.
    """
    import hashlib

    import h5py
    import numpy as np

    sys.path.insert(0, str(PROJECT / "BEHAVIOR-VLAs/scripts"))
    from recorded_native_state import decode_start

    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != source["raw_sha256"]:
        raise ValueError("Raw recording differs from the audited transition source")
    with h5py.File(RAW, "r", locking=False) as recording:
        data = recording["data"]
        demo = data["demo_1"]
        transitions = json.loads(demo.attrs["transitions"])
        cache = {
            "recorded_config": json.loads(data.attrs["config"]),
            "scene_file": json.loads(data.attrs["scene_file"]),
            "states": np.asarray(demo["state"][:9256]),
            "state_sizes": np.asarray(demo["state_size"][:9256]),
            "transitions": transitions,
            "transition_by_step": {int(step): value for step, value in transitions.items()},
            "sha256": digest,
        }
    for event in source["transitions"]:
        actual = transitions[str(event["step"])]["objects"]
        assert actual["remove"] == event["removed"]
        assert {item["args"]["name"] for item in actual["add"]} == {
            item["name"] for item in event["added"]
        }

    def held(frame: int) -> dict:
        proof = decode_start(cache, {"start": frame, "sample_id": f"frame-{frame}"})
        return {
            "frame": frame,
            "robot_state_source_frame": proof["source_robot_last_written_frame"],
            "held_by_arm": {
                str(packet["arm_index"]): packet["target_object"]
                for packet in proof["recorded_grasps"]
            },
        }

    cues = []
    for entity, frame in sorted(CUE_FRAMES.items(), key=lambda item: item[1]):
        proof = held(frame)
        assert proof["held_by_arm"].get("0") == entity
        cues.append({**proof, "entity_scoring_only": entity})
    query = held(QUERY_FRAME)
    assert "0" not in query["held_by_arm"]
    next_grasp = next(
        proof for frame in range(QUERY_FRAME + 1, 9256)
        if "0" in (proof := held(frame))["held_by_arm"]
    )
    assert next_grasp["held_by_arm"]["0"] == "half_brussels_sprouts_92_1"
    return {
        "raw_file": str(RAW.relative_to(PROJECT)),
        "raw_sha256": digest,
        "demo": "demo_1",
        "cue_type": "action_event_binding",
        "cue_instruction_template": "The object currently held in your left hand is called {alias}.",
        "cue_events": cues,
        "query_state": query,
        "next_recorded_left_grasp": next_grasp,
        "rgb_visibility_verified": False,
        "placement_success_verified": False,
    }


def lifetimes(source: dict) -> dict[str, tuple[int, int | None]]:
    """Return creation/removal frames for identities supported by this audit."""
    out = {name: (-1, None) for name in source["task_mapping"].values()}
    for event in sorted(source["transitions"], key=lambda item: item["step"]):
        step = event["step"]
        for parent in event["removed"]:
            if parent not in out or out[parent][1] is not None:
                raise ValueError("Removal of an unknown or already removed entity")
            out[parent] = (out[parent][0], step)
        for child in event["added"]:
            name = child["name"]
            if name in out:
                raise ValueError("Duplicate entity creation")
            out[name] = (step, None)
    return out


def live_descendants(
    source: dict, root: str, frame: int, *, binding_frame: int | None = None
) -> set[str]:
    """Resolve a supported reference without inventing a pre-birth identity.

    Events execute after their recorded frame. A binding requires a living entity
    at its own creation time; that check still does not prove visual visibility.
    """
    if frame < 0:
        raise ValueError("Negative query frame")
    spans = lifetimes(source)
    if root not in spans:
        raise ValueError("Identity is not supported by the audited source")
    born, removed = spans[root]
    if binding_frame is not None:
        if not 0 <= binding_frame <= frame:
            raise ValueError("Binding must precede or coincide with the query")
        if born >= binding_frame or (removed is not None and binding_frame > removed):
            raise ValueError("Cannot bind an entity outside its lifetime")
    if born >= frame:
        return set()
    live = {root}
    for event in sorted(source["transitions"], key=lambda item: item["step"]):
        if event["step"] >= frame:
            break
        for parent in event["removed"]:
            if parent not in live:
                continue
            # Restrict this prototype to the verified one-parent, named-half events.
            if len(event["removed"]) != 1:
                raise ValueError("Ambiguous multi-parent event is unsupported")
            children = {
                child["name"] for child in event["added"]
                if child["name"].startswith("half_" + parent + "_")
            }
            if len(children) != 2:
                raise ValueError("Unverified split form")
            live.remove(parent)
            live.update(children)
    return live


def main() -> None:
    source = json.loads(SOURCE.read_text())
    evidence = recorded_action_event_evidence(source)
    roots = sorted({parent for event in source["transitions"] for parent in event["removed"]})
    aliases = ["A", "B", "C", "D"]
    assert len(roots) == len(aliases) == 4
    assert set(roots) == set(CUE_FRAMES)
    # The query starts with an empty left hand, and its factual suffix acquires
    # this target. Other referents receive grounding labels, never this action.
    target = evidence["next_recorded_left_grasp"]["held_by_arm"]["0"]
    assert target == "half_brussels_sprouts_92_1"

    records = []
    for permutation in itertools.permutations(roots):
        bindings = dict(zip(aliases, permutation))
        cue_records = [
            {
                "frame": CUE_FRAMES[entity],
                "alias": alias,
                "instruction": evidence["cue_instruction_template"].format(alias=alias),
                "bound_entity_scoring_only": entity,
            }
            for alias, entity in bindings.items()
        ]
        cue_records.sort(key=lambda item: item["frame"])
        # Validate each historical cue, not just the alias selected by the query.
        for entity in bindings.values():
            live_descendants(source, entity, QUERY_FRAME, binding_frame=CUE_FRAMES[entity])
        targets = live_descendants(
            source, bindings["A"], QUERY_FRAME, binding_frame=CUE_FRAMES[bindings["A"]]
        )
        compatible = target in targets
        records.append({
            "historical_bindings_scoring_only": bindings,
            "historical_action_event_cues": cue_records,
            "query": "Pick up any currently existing half originating from the object called A.",
            "local_operation": "pick_up",
            "current_frame": QUERY_FRAME,
            "query_left_hand_empty_verified": True,
            "current_target_set_scoring_only": sorted(targets),
            "factual_native_grasp_target_scoring_only": target,
            "factual_target_first_grasp_frame": evidence["next_recorded_left_grasp"]["frame"],
            "factual_action_semantically_compatible": compatible,
            "motor_supervision": "original_recording_only" if compatible else "none",
        })

    assert len(records) == 24
    assert sum(r["factual_action_semantically_compatible"] for r in records) == 6
    assert all(r["motor_supervision"] == "none" for r in records if not r["factual_action_semantically_compatible"])

    # After all four audited splits, the same query can refer to four disjoint
    # current target groups, determined exclusively by its past binding.
    later_frame = QUERY_FRAME
    groups = [live_descendants(source, root, later_frame, binding_frame=CUE_FRAMES[root]) for root in roots]
    assert all(len(group) == 2 for group in groups)
    assert len(set.union(*groups)) == 8
    assert all(not left & right for left, right in itertools.combinations(groups, 2))
    distinct_targets = {tuple(sorted(group)) for group in groups}
    assert len(distinct_targets) == 4

    # Names introduced after a birth refer to that child only, whereas a name
    # introduced for the original object still denotes both living descendants.
    parent = "brussels_sprouts_93"
    children = sorted(live_descendants(source, parent, QUERY_FRAME, binding_frame=CUE_FRAMES[parent]))
    inherited = live_descendants(source, parent, later_frame)
    child_local = [
        live_descendants(source, child, later_frame, binding_frame=7392)
        for child in children
    ]
    assert set.union(*child_local) == inherited
    assert all(len(group) == 1 for group in child_local)
    assert not child_local[0] & child_local[1]

    # Regression: the previous implementation returned the not-yet-created
    # child here. Also reject an alias attached after its physical parent died.
    child = "half_brussels_sprouts_93_0"
    assert live_descendants(source, child, 7000) == set()
    assert live_descendants(source, child, 7370) == set()
    assert live_descendants(source, child, 7371) == {child}
    rejected_bindings = 0
    for name, cue, query in [(child, 7000, 7392), (parent, 7371, 7392), (parent, 7400, 7392)]:
        try:
            live_descendants(source, name, query, binding_frame=cue)
        except ValueError:
            rejected_bindings += 1
    assert rejected_bindings == 3

    # Compute the memory-blind bound only for these balanced, disjoint classes.
    classes = Counter(tuple(r["current_target_set_scoring_only"]) for r in records)
    class_sets = [set(key) for key in classes]
    assert all(not left & right for left, right in itertools.combinations(class_sets, 2))
    proportions = [count / len(records) for count in classes.values()]
    blind_accuracy = max(proportions)
    blind_nll = -sum(p * math.log(p) for p in proportions)
    factual_classes = {
        tuple(r["current_target_set_scoring_only"])
        for r in records if r["factual_action_semantically_compatible"]
    }
    assert len(factual_classes) == 1

    result = {
        "kind": "symbolic_native_lineage_reference_intervention_prototype",
        "source": str(SOURCE.relative_to(PROJECT)),
        "source_raw_sha256": source["raw_sha256"],
        "action_event_evidence": evidence,
        "checks_passed": {
            "binding_permutations": len(records),
            "factual_motor_compatible_permutations": 6,
            "grounding_only_permutations": 18,
            "post_split_disjoint_reference_sets": len(distinct_targets),
            "post_split_current_entities": 8,
            "counterfactual_motor_targets_generated": 0,
            "ancestor_alias_targets": len(inherited),
            "post_birth_alias_targets": [len(group) for group in child_local],
            "sibling_alias_overlap": len(child_local[0] & child_local[1]),
            "invalid_lifetime_bindings_rejected": rejected_bindings,
            "full_grounding_memory_blind_accuracy_upper_bound": blind_accuracy,
            "full_grounding_memory_blind_set_nll_lower_bound": blind_nll,
            "factual_motor_subset_memory_blind_accuracy_upper_bound": 1.0,
            "all_alias_cues_precede_query": all(frame < QUERY_FRAME for frame in CUE_FRAMES.values()),
            "fixed_physical_cue_events_across_permutations": len({
                tuple((cue["frame"], cue["bound_entity_scoring_only"]) for cue in record["historical_action_event_cues"])
                for record in records
            }) == 1,
            "query_left_hand_empty": True,
            "factual_next_grasp_verified_in_raw_h5": True,
        },
        "limits": [
            "Historical cue frames and held identities are verified native action events; RGB visibility is NOT verified.",
            "No RGB examples or policy inputs have been generated.",
            "No model has been trained or evaluated; this is only label logic.",
            "Compatibility means the recorded target belongs to the reference set, not proof of physical success.",
            "A nonempty target set may contain more than one correct target; all are retained.",
            "Action-event aliases name the currently held object; they are not claimed to be visible crop bindings.",
            "Lifetime checks do not establish that an object was visible to the camera.",
            "Grounding-only permutations have no continuous action supervision.",
            "The grounding bound does not prove the motor expert uses its target input.",
            "An empty left hand makes pickup a consistent operation; alternative targets' reachability and successful grasps are not certified.",
        ],
        "examples": records[:4],
        "all_records": records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["checks_passed"], ensure_ascii=False))


if __name__ == "__main__":
    main()
