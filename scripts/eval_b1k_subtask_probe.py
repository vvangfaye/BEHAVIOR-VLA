#!/usr/bin/env python3
"""Validate selected frozen samples using the production exact-state rollout code."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import eval_b1k_subtask_samples as runner

KIND = "subtask_state_validation"
GRASP_DIAGNOSTIC_KIND = "subtask_grasp_restoration_diagnostic"
GRASP_RECONSTRUCTION_KIND = "subtask_grasp_reconstruction_diagnostic"
GRASP_HISTORY_KIND = "subtask_grasp_history_diagnostic"
NATIVE_MODEL_KIND = "subtask_native_snapshot_model_validation"
DIAGNOSTIC_KINDS = (GRASP_DIAGNOSTIC_KIND, GRASP_RECONSTRUCTION_KIND, GRASP_HISTORY_KIND)


def load_plan(path, panel):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 1 or plan.get("kind") not in (KIND, NATIVE_MODEL_KIND, *DIAGNOSTIC_KINDS):
        raise ValueError("Unsupported subtask validation plan")
    if plan.get("panel_fingerprint") != panel["fingerprint"]:
        raise ValueError("Validation plan differs from the frozen panel")
    ids = plan.get("sample_ids")
    if not isinstance(ids, list) or not ids or len(ids) > 12 or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Validation requires 1-12 unique sample IDs")
    steps = plan.get("max_steps")
    limit = 2048 if plan["kind"] == NATIVE_MODEL_KIND else 256
    if type(steps) is not int or not 1 <= steps <= limit:
        raise ValueError(f"Validation max_steps must be 1-{limit}")
    if plan["kind"] == NATIVE_MODEL_KIND:
        from validate_b1k_native_snapshot_model import validate_plan
        validate_plan(plan)
    if plan["kind"] in DIAGNOSTIC_KINDS and steps > 32:
        raise ValueError("Grasp restoration diagnosis is limited to 32 recorded-action steps per start")
    if plan['kind'] == GRASP_RECONSTRUCTION_KIND:
        if len(ids) > 2 or plan.get('paired_seed') != 1 or type(plan.get('paired_seed')) is not int:
            raise ValueError('Reconstruction diagnosis requires at most two starts with paired seed 1')
    by_id = {sample["sample_id"]: sample for sample in panel["samples"]}
    if set(ids) - set(by_id):
        raise ValueError("Validation contains samples outside the frozen panel")
    samples = [by_id[i] for i in ids]
    for task in {sample["task"] for sample in samples}:
        if len({sample["episode_index"] for sample in samples if sample["task"] == task}) != 1:
            raise ValueError("Validation supports one selected episode per task")
    if plan['kind'] == GRASP_RECONSTRUCTION_KIND and any(s['start'] < 64 for s in samples):
        raise ValueError('Reconstruction requires 64 preceding source frames')
    if plan['kind'] == GRASP_HISTORY_KIND:
        if len(samples) > 2 or plan.get('paired_seed') != 1 or type(plan.get('paired_seed')) is not int:
            raise ValueError('History diagnosis requires at most two original starts with paired seed 1')
        if any(not 64 <= s['start'] <= 2500 for s in samples):
            raise ValueError('History diagnosis requires a bounded 64-2500-frame original prefix')
    if "require_recorded_native_state" in plan and type(plan["require_recorded_native_state"]) is not bool:
        raise ValueError("Recorded native requirement must be boolean")
    if plan.get("require_recorded_native_state"):
        import re
        if plan["kind"] != KIND or type(plan.get("checkpoint_step")) is not int or plan["checkpoint_step"] <= 0:
            raise ValueError("Recorded native validation requires a standard plan and frozen checkpoint")
        def hashes(value, keys):
            return (isinstance(value, dict) and set(value) == keys and
                    all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in value.values()))
        paths = {f"task-{s['task_index']:04d}/episode_{s['raw_episode_id']:08d}.hdf5" for s in samples}
        if (not isinstance(plan.get("checkpoint_manifest_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", plan["checkpoint_manifest_sha256"]) or
                not hashes(plan.get("raw_source_sha256"), paths) or
                not hashes(plan.get("selected_state_sha256"), set(ids))):
            raise ValueError("Recorded native validation requires exact source/checkpoint hash scope")
    return plan, samples


def validate_results(root, panel, plan, samples):
    if plan.get('kind') == GRASP_HISTORY_KIND:
        from diagnose_b1k_grasp_history import validate_results as validate_history
        return validate_history(root, panel, plan, samples)
    if plan.get("kind") == NATIVE_MODEL_KIND:
        from validate_b1k_native_snapshot_model import validate_results as validate_native
        return validate_native(root, panel, plan, samples)
    if plan.get('kind') == GRASP_RECONSTRUCTION_KIND:
        from validate_b1k_grasp_reconstruction import validate_results as validate_reconstruction
        return validate_reconstruction(root, panel, plan, samples)
    if plan.get("kind") == GRASP_DIAGNOSTIC_KIND:
        from diagnose_b1k_grasp_restore import validate_results as validate_grasp_diagnostic
        return validate_grasp_diagnostic(root, panel, plan, samples)
    expected = {sample["sample_id"]: sample for sample in samples}
    paths = list(Path(root).glob("pi05/*/json/*.json"))
    seen, eligible = set(), 0
    for path in paths:
        result = json.loads(path.read_text())
        sample_id = result.get("sample_id")
        if sample_id not in expected or sample_id in seen:
            raise ValueError("Unexpected or duplicate validation result")
        sample = expected[sample_id]
        runner.EvaluationQueue._validate_result(result, sample["task"], sample["task_instance_id"], sample["rollout_id"])
        for key in ("task", "episode_index", "raw_episode_id", "rollout_id", "subtask"):
            if result.get(key) != sample[key]:
                raise ValueError(f"Validation result identity mismatch: {key}")
        if result.get("instance_id") != sample["task_instance_id"] or result.get("panel_fingerprint") != panel["fingerprint"]:
            raise ValueError("Validation result differs from the frozen sample")
        if result.get("restoration", {}).get("full_snapshot_format") != runner.FULL_SNAPSHOT_FORMAT:
            raise ValueError("Validation result lacks named state restoration evidence")
        if plan.get("require_recorded_native_state"):
            from recorded_native_state import validate_result_proof
            validate_result_proof(result, sample, plan)
            if result.get("metric_eligible") is True:
                video = path.parent.parent / "videos" / (path.stem + ".mp4")
                if not video.is_file() or runner.file_sha256(video) != result.get("video_sha256"):
                    raise ValueError("Recorded native model video is missing or changed")
        if result.get("validation_kind") != KIND or result.get("official_benchmark") is not False:
            raise ValueError("Validation output is not explicitly labeled")
        steps = result.get("steps")
        if type(steps) is not int or not 0 <= steps <= plan["max_steps"]:
            raise ValueError("Validation result has invalid step count")
        if result.get("metric_eligible") is True:
            video = path.parent.parent / "videos" / (path.stem + ".mp4")
            if steps == 0 or not video.is_file() or video.stat().st_size == 0:
                raise ValueError("Eligible validation requires actions and a nonempty video")
            if result.get("conditioning", {}).get("source") != "provided":
                raise ValueError("Validation lacks fixed-subtask action evidence")
            eligible += 1
        elif result.get("metric_eligible") is not False or result.get("result_state") != "setup_excluded" or not result.get("setup_exclusion_reason"):
            raise ValueError("Validation exclusion lacks evidence")
        seen.add(sample_id)
    if seen != set(expected):
        raise ValueError("Validation results are incomplete")
    if eligible == 0:
        raise ValueError("Validation has no eligible model rollout; restoration alone is insufficient")
    return {"kind": KIND, "complete": True, "official_benchmark": False,
            "panel_fingerprint": panel["fingerprint"], "samples": len(seen),
            "eligible_rollouts": eligible, "setup_excluded": len(seen)-eligible,
            "max_steps": plan["max_steps"], "full_snapshot_format": runner.FULL_SNAPSHOT_FORMAT}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--subtask-panel", type=Path, required=True)
    parser.add_argument("--probe-plan", type=Path, required=True)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--sample-max-steps", type=int, required=True)
    probe_args, base_argv = parser.parse_known_args(argv)
    base = runner._load_base_module()
    args = base.parse_args(base_argv)
    if args.mode != "train" or args.num_rollouts != 1 or not args.write_video or args.external_subtask is not None or args.external_subtask_selector is not None:
        raise ValueError("Validation requires training states, fixed panel prompts and videos")
    panel = runner.load_panel(probe_args.subtask_panel)
    plan, samples = load_plan(probe_args.probe_plan, panel)
    if probe_args.sample_max_steps != plan["max_steps"]:
        raise ValueError("Validation step limit differs from its frozen plan")
    samples = [sample for sample in samples if sample["task"] == args.task_name]
    if not samples:
        raise ValueError("Task is absent from the validation plan")
    args.sample_max_steps = plan["max_steps"]
    cache = runner.load_episode_cache(runner.raw_hdf5_path(probe_args.raw_data_root, samples[0]), samples,
        allow_unverified_grasp_state=plan["kind"] in (*DIAGNOSTIC_KINDS, NATIVE_MODEL_KIND))
    if plan.get("require_recorded_native_state"):
        from recorded_native_state import decode_start
        source_path = f"task-{samples[0]['task_index']:04d}/episode_{samples[0]['raw_episode_id']:08d}.hdf5"
        if cache["sha256"] != plan["raw_source_sha256"][source_path]:
            raise ValueError("Recorded native source differs from the frozen full-file hash")
        for sample in samples:
            if decode_start(cache, sample)["selected_state_sha256"] != plan["selected_state_sha256"][sample["sample_id"]]:
                raise ValueError("Recorded native selected row differs from its frozen source")
    base.gm.HEADLESS = args.headless
    seed = base.seed_everything(base.DEFAULT_EVAL_SEED)
    evaluator_args = copy.copy(args)
    evaluator_args.external_subtask = samples[0]["subtask"]
    evaluator_args.max_steps = args.sample_max_steps
    cfg = runner._build_evaluator_cfg(base, evaluator_args, seed)
    with runner.create_sample_evaluator(base, cfg, cache, samples[0], args.sample_max_steps) as evaluator:
        if plan['kind'] == GRASP_HISTORY_KIND:
            from diagnose_b1k_grasp_history import diagnose
            diagnose(base, evaluator, args, panel, plan, samples, cache)
            return
        if plan['kind'] == NATIVE_MODEL_KIND:
            from validate_b1k_native_snapshot_model import run, INPUT_DIRECTORY
            run(base, evaluator, args, panel, plan, samples, cache,
                probe_args.probe_plan.parent / INPUT_DIRECTORY)
            return
        if plan['kind'] == GRASP_RECONSTRUCTION_KIND:
            from validate_b1k_grasp_reconstruction import diagnose
            diagnose(base, evaluator, args, panel, plan, samples, cache)
            return
        if plan["kind"] == GRASP_DIAGNOSTIC_KIND:
            from diagnose_b1k_grasp_restore import diagnose
            diagnose(base, evaluator, args, panel, plan, samples, cache)
            return
        states = runner.materialize_episode_start_states(evaluator, cache, samples)
        for sample in samples:
            audit = None
            if plan.get("require_recorded_native_state"):
                from validate_b1k_native_snapshot_model import ActionAudit
                audit = ActionAudit()
            result = runner.run_sample(base, evaluator, args, sample, cache, states[sample["sample_id"]], Path(args.output_dir), panel["fingerprint"], trajectory_writer=audit)
            result.update(validation_kind=KIND, official_benchmark=False)
            if audit is not None:
                video = Path(args.output_dir)/"videos"/(runner._artifact_stem(sample)+".mp4")
                result.update(model_action_audit=audit.rows, actual_model_control_steps=len(audit.rows),
                              video_sha256=runner.file_sha256(video) if result["metric_eligible"] else None,
                              production_difficulty_baseline=False)
            runner.atomic_write_json(Path(args.output_dir)/"json"/(runner._artifact_stem(sample)+".json"), result)
            print(f"Validated sample {sample['sample_id']}: steps={result['steps']} eligible={result['metric_eligible']}", flush=True)


if __name__ == "__main__":
    main()
