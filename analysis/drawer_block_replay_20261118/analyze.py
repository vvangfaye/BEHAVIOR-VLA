"""Summarize paired continuations at frozen steps on two recorded PPO trajectories."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

from source_inputs import HORIZON, NOISE_VERSION, digest, load_manifest, read_source, stream_name
from verify_source import verify as verify_source


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "drawer_regression_pilot_20261118"
sys.path.insert(0, str(PREVIOUS))
from stall_report import arm_summary


def read_json(path):
    return json.loads(path.read_text())


def exact_mcnemar(sft_only, ppo_only):
    discordant = sft_only + ppo_only
    smaller = min(sft_only, ppo_only)
    tail_count = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2 * tail_count / (2 ** discordant))


def holm_adjust(values):
    adjusted = [0.0] * len(values)
    running_max = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running_max = max(running_max, (len(values) - rank) * values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def summarize(pairs):
    counts = arm_summary(pairs)
    return {
        "pairs": counts["paired"],
        "sft_successes": counts["old_only"] + counts["both_success"],
        "ppo_successes": counts["new_only"] + counts["both_success"],
        "sft_only": counts["old_only"],
        "ppo_only": counts["new_only"],
        "both_successes": counts["both_success"],
        "both_failures": counts["neither"],
        "sft_minus_ppo": counts["old_minus_new"],
        "pointwise_ci95_conservative": counts["old_minus_new_95pct_conservative_interval"],
        "p_exact_two_sided": exact_mcnemar(counts["old_only"], counts["new_only"]),
    }


def validate_block(block, scene, step, phase, definition, reference, measured_state, source_mode, actor_digest):
    landmark = definition["landmarks"][str(scene)]
    identity = (block["seed"], block["step"], block["phase"], block["landmark_step"])
    if identity != (scene, step, phase, landmark):
        raise ValueError(f"Unexpected scene/step/phase/landmark: {identity}")
    if block["initial_hash"] != reference["initial_hash"] or block["start_metrics"]["step"] != step:
        raise ValueError(f"Initial state or start step mismatch at {identity}")
    if block.get("horizon") != HORIZON or block.get("ppo_actor_sha256") != actor_digest or block.get("noise_version") != NOISE_VERSION:
        raise ValueError(f"Unexpected horizon, weights or noise protocol at {identity}")
    if any(block.get(k) != measured_state[k] for k in ("initial_hash", "state_hash", "observation_hash")):
        raise ValueError(f"State did not come from the recorded PPO prefix at {identity}")
    if source_mode == "exact" and step == landmark and (block["state_hash"], block["observation_hash"]) != (
            reference["state_hash"], reference["observation_hash"]):
        raise ValueError(f"Landmark is not the recorded PPO state at {identity}")
    if block["start_metrics"]["success"]:
        raise ValueError(f"Continuation starts after task success at {identity}")
    pairs = block["pairs"]
    count = definition["phases"][phase]
    if sorted(pair["repeat"] for pair in pairs) != list(range(count)):
        raise ValueError(f"Unexpected repeat IDs at {identity}")
    for pair in pairs:
        for name in ("old", "new"):
            success, terminal = pair[f"{name}_success"], pair[f"{name}_success_step"]
            if type(success) is not bool or (terminal is not None) != success:
                raise ValueError(f"Invalid success or completion step at {identity}")
            if success and (type(terminal) is not int or not step < terminal <= HORIZON):
                raise ValueError(f"Completion outside the original remaining horizon at {identity}")
            end_metrics = pair[f"{name}_end_metrics"]
            if end_metrics["success"] != success or end_metrics["step"] != (terminal if success else HORIZON):
                raise ValueError(f"Native endpoint and stored outcome disagree at {identity}")
        if pair["noise_stream"] != stream_name(scene, phase, pair["repeat"]):
            raise ValueError(f"Unexpected paired noise stream ID at {identity}")
    streams = {p["repeat"]: p["noise_stream"] for p in pairs}
    if len(set(streams.values())) != count:
        raise ValueError(f"Repeated noise stream at {identity}")
    return pairs, streams


def read_grid(folder, references, anchors, definition, provenance):
    actor_digest = provenance["ppo_actor_sha256"]
    if set(definition["states"]) != set(provenance["scenes"]):
        raise ValueError("Statistical plan and source manifest must cover the same scenes")
    if set(definition["phases"]) != {"screen", "confirm"} or any(
        type(count) is not int or count < 1 for count in definition["phases"].values()
    ):
        raise ValueError("Statistical plan requires positive screen and confirm counts")
    rows = []
    for key, steps in definition["states"].items():
        scene = int(key)
        scene_provenance = provenance["scenes"][key]
        if (not steps or sorted(set(steps)) != steps
                or definition["landmarks"][key] != scene_provenance["landmark"]
                or steps[0] < scene_provenance["landmark"] - 80 or steps[-1] > scene_provenance["landmark"]):
            raise ValueError(f"Unexpected steps or landmark in the statistical plan for scene {scene}")
        prefix, reference = read_source(references, scene, scene_provenance)
        scene_directory = folder / f"seed_{scene}"
        source = read_json(scene_directory / "source.json")
        if (source["seed"] != scene or source["landmark_step"] != reference["landmark_step"]
                or source["ppo_actor_sha256"] != actor_digest or source["source_prefix_sha256"] != scene_provenance["prefix_sha256"]
                or source["source_record_sha256"] != scene_provenance["record_sha256"]
                or set(source["blocks"]) != {str(step) for step in steps}):
            raise ValueError(f"Source and grid provenance disagree for scene {scene}")
        mode, alternate_metrics = verify_source(
            source, scene_directory, anchors, prefix, reference, scene_provenance.get("reference_sha256s", {})
        )
        stream_sets = {}
        for step in steps:
            batches, phases = {}, {}
            state_identity = None
            for phase in ("screen", "confirm"):
                path = folder / f"seed_{scene}" / phase / f"block_{step}.json"
                block = read_json(path)
                pairs, streams = validate_block(block, scene, step, phase, definition, reference,
                                                source["blocks"][str(step)], mode, actor_digest)
                if alternate_metrics and block["start_metrics"] != alternate_metrics[str(step)]:
                    raise ValueError(f"Continuation does not match validated cross-host metrics at {scene}:{step}")
                if streams != stream_sets.setdefault(phase, streams):
                    raise ValueError(f"Streams not shared across steps for scene {scene} phase {phase}")
                identity = (block["state_hash"], block["observation_hash"], block["start_metrics"])
                if state_identity is not None and identity != state_identity:
                    raise ValueError(f"Screen and confirm states differ at {scene}:{step}")
                state_identity = identity
                batches[phase] = pairs
                phases[phase] = summarize(pairs)
            if set(stream_sets["screen"].values()) & set(stream_sets["confirm"].values()):
                raise ValueError(f"Screen and confirm have overlapping streams for scene {scene}")
            pooled = summarize(batches["screen"] + batches["confirm"])
            rows.append({
                "seed": scene, "step": step, "relative_to_stall": step - reference["landmark_step"],
                "state_hash": state_identity[0], "observation_hash": state_identity[1],
                "start_metrics": state_identity[2], "source_reconstruction_mode": mode,
                **phases, "pooled": pooled,
            })
    return rows


def decisions(rows):
    for which in ("pooled", "confirm"):
        adjusted = holm_adjust([r[which]["p_exact_two_sided"] for r in rows])
        for row, value in zip(rows, adjusted, strict=True):
            row[which]["p_holm"] = value
    for row in rows:
        screen = row["screen"]["sft_minus_ppo"]
        confirm = row["confirm"]["sft_minus_ppo"]
        row["both_phases_same_nonzero_direction"] = screen * confirm > 0
        row["stable_gap"] = row["pooled"]["p_holm"] <= 0.05 and screen * confirm > 0
        row["confirm_only_significant"] = row["confirm"]["p_holm"] <= 0.05
        row["direction"] = None if not row["stable_gap"] else (
            "SFT higher" if row["pooled"]["sft_minus_ppo"] > 0 else "PPO higher"
        )
    scenes = {}
    for scene in sorted({r["seed"] for r in rows}):
        individual = sorted((r for r in rows if r["seed"] == scene), key=lambda r: r["step"])
        runs = []
        for row in individual:
            if not row["stable_gap"]:
                continue
            if runs and row["direction"] == runs[-1]["direction"] and row["step"] == runs[-1]["last_step"] + 10:
                runs[-1]["last_step"] = row["step"]
                runs[-1]["n_blocks"] += 1
            else:
                runs.append({"first_step": row["step"], "last_step": row["step"],
                             "n_blocks": 1, "direction": row["direction"]})
        earliest = runs[0]["first_step"] if runs else None
        scenes[str(scene)] = {
            "source_reconstruction_mode": individual[0]["source_reconstruction_mode"],
            "earliest_detected_step": earliest,
            "earliest_detected_direction": runs[0]["direction"] if runs else None,
            "left_censored": earliest == individual[0]["step"] if earliest is not None else None,
            "qualifying_contiguous_ranges": runs,
            "tested_first": individual[0]["step"],
            "tested_last": individual[-1]["step"],
        }
    return scenes


def write_table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "step", "relative_to_stall", "source_reconstruction_mode", "screen_sft", "screen_ppo",
                         "confirm_sft", "confirm_ppo", "pooled_sft", "pooled_ppo", "pairs",
                         "sft_only", "ppo_only", "delta_sft_minus_ppo", "pointwise_ci95_low",
                         "pointwise_ci95_high", "p_exact", "p_holm", "confirm_p_holm",
                         "stable_gap", "direction"])
        for row in rows:
            a, b, p = row["screen"], row["confirm"], row["pooled"]
            writer.writerow([row["seed"], row["step"], row["relative_to_stall"], row["source_reconstruction_mode"],
                             a["sft_successes"], a["ppo_successes"], b["sft_successes"],
                             b["ppo_successes"], p["sft_successes"], p["ppo_successes"],
                             p["pairs"], p["sft_only"], p["ppo_only"], p["sft_minus_ppo"],
                             *p["pointwise_ci95_conservative"], p["p_exact_two_sided"], p["p_holm"],
                             b["p_holm"], row["stable_gap"], row["direction"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing seed_<id> replay directories")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Recorded prefix and paired landmark files")
    parser.add_argument("--anchor-dir", type=Path, help="Original camera frames and root batch for cross-host checks")
    parser.add_argument("--plan", type=Path, required=True, help="Prespecified analysis plan")
    parser.add_argument("--source-manifest", type=Path, required=True, help="Independent digest manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    definition = read_json(args.plan)
    provenance = load_manifest(args.source_manifest)
    rows = read_grid(args.input_dir, args.reference_dir, args.anchor_dir, definition, provenance)
    scenes = decisions(rows)
    result = {
        "statistical_plan_sha256": digest(args.plan),
        "source_manifest_sha256": digest(args.source_manifest),
        "scene_count": len(scenes), "state_count": len(rows),
        "paired_continuations": sum(r["pooled"]["pairs"] for r in rows),
        "scene_summaries": scenes, "blocks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        write_table(args.table, rows)
    print(json.dumps(scenes, indent=2))


if __name__ == "__main__":
    main()
