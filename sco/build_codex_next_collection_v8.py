#!/usr/bin/env python3
"""Build one finite successor using already installed, audited code-v8.

No code installation, authentication, model probe or cloud submission. The same
canonical journal/ledger retains all code-v7 and code-v8 attempt reservations.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import build_codex_next_collection as shared
from build_codex_first_collection_v5 import _replace_once
from run_codex_collection import validate_payload

KIND = "codex_next_collection_v8"
CODE8_SHA = "b2f054f13ccc76f2db46fa990e5d40ef3bfc8ca4cfaaae18baf6b231f42803e0"
ALLOWED_CHANGES = {"scripts/collect_b1k_codex.py", "scripts/codex_collection_actions.py"}
TEACHER_INTERVALS = {"turning_on_radio": 256, "putting_shoes_on_rack": 320}


def worker_source():
    source = _replace_once(shared.WORKER,
        "manifest['version']=='v7' and str(code)==manifest['code']==str(Path(config['pilot_root'])/'code-v7')",
        "manifest['version']=='v8' and str(code)==manifest['code']==str(Path(config['pilot_root'])/'code-v8')")
    return _replace_once(source, "'kind':'codex_bounded_collection_gpu_v1'", "'kind':'codex_bounded_collection_gpu_v8'")


WORKER = worker_source()


def validate_frozen_config(frozen):
    """Pure source-bound audit/admission validation; no filesystem mutations."""
    ensure = shared.ensure
    ensure(frozen.get("builder_kind") == KIND and "installer" not in frozen, "Only installed code-v8 reuse is supported")
    collection = validate_payload(frozen["collection"])
    task, instance, attempt = (collection[key] for key in ("task", "instance_id", "attempt_id"))
    ensure(task in shared.TASKS and shared.NAME.fullmatch(attempt), "Unknown successor task or unsafe attempt ID")
    slug = ("radio" if task == "turning_on_radio" else "shoes") + f"-{instance}-{attempt}"
    ensure(collection["run_root"] == shared.AFS + "/collection/" + slug
           and frozen.get("report_root") == shared.AFS + "/next-code8-" + slug
           and frozen.get("pilot_root") == shared.AFS and frozen.get("python") == shared.PYTHON
           and collection["runtime"] + "/envs/b1k-eval/bin/python" == shared.PYTHON
           and collection["teacher_backend"] == "codex_local"
           and collection["codex_home"] == shared.AFS + "/private-codex-home-v1",
           "Successor output, runtime or selected account differs")
    ensure(all(collection[key] == value for key, value in dict(max_steps=shared.TASKS[task][0], max_decisions=32,
               teacher_interval=TEACHER_INTERVALS[task], max_seconds=7200, wall_timeout_seconds=8400, teacher_timeout=120).items()),
           "Successor finite limits differ")
    episode = frozen.get("demo_episode")
    ensure(type(episode) is int and episode in shared.TASKS[task][1] and frozen.get("episodes") == shared.EPISODES
           and collection["demonstration_manifest"] == shared.AFS + f"/original-reader-v4/demonstrations/{episode}/manifest.json",
           "Successor must use a prepared demonstration of the same task")
    manifest, code7, prep = (frozen[key] for key in ("manifest", "previous_manifest", "preparation_manifest"))
    inventory = manifest.get("source_files_sha256", {})
    ensure(manifest.get("version") == "v8" and manifest.get("code") == shared.AFS + "/code-v8"
           and manifest.get("bundle_sha256") == CODE8_SHA and frozen.get("expected_bundle_sha256") == CODE8_SHA
           and collection["code"] == manifest["code"] and collection["source_files_sha256"] == inventory
           and len(inventory) == 29 and inventory.keys() == code7.get("source_files_sha256", {}).keys()
           and code7.get("version") == "v7" and code7.get("code") == shared.AFS + "/code-v7"
           and code7.get("bundle_sha256") == shared.CODE_SHA
           and {name for name in inventory if inventory[name] != code7["source_files_sha256"][name]} == ALLOWED_CHANGES,
           "Successor must reuse the exact installed code-v8 and its two reviewed source changes")
    ensure(prep.get("version") == "v6" and prep.get("code") == shared.AFS + "/code-v6"
           and prep.get("bundle_sha256") == shared.OLD_BUNDLE_SHA256 and len(prep.get("source_files_sha256", {})) == 29
           and manifest["base_code"] == code7["base_code"] == prep["base_code"] == collection["base_code"]
           and manifest["base_source_manifest_sha256"] == code7["base_source_manifest_sha256"] == prep["base_source_manifest_sha256"]
           and inventory["scripts/codex_collection_teacher.py"] == prep["source_files_sha256"]["scripts/codex_collection_teacher.py"]
           and collection["robot_config"] == manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
           "Successor must preserve the verified preparation, base and teacher")
    admission = frozen.get("admission", {})
    sequence, task_number = admission.get("sequence"), admission.get("task_attempt_number")
    ensure(type(sequence) is int and 4 <= sequence <= 20 and type(task_number) is int and 1 <= task_number <= 10
           and admission.get("reserved_decisions") == 32 and admission.get("total_reserved_decisions") == sequence * 32
           and admission.get("max_teacher_decisions_total") == 640 and admission.get("max_attempts_per_task") == 10
           and admission.get("reservations_released") is False
           and admission.get("all_continue_action_ceiling") == 32 * TEACHER_INTERVALS[task]
           and admission.get("step_limit_may_exceed_teacher_budget") is (collection["max_steps"] > 32 * TEACHER_INTERVALS[task])
           and admission.get("correction_can_reduce_action_ceiling") is True,
           "Successor must preserve the shared finite cross-version reservation")
    predecessor = frozen.get("predecessor", {})
    ensure(predecessor.get("job", {}).get("state") in shared.TERMINAL
           and shared.HEX.fullmatch(str(predecessor.get("source_sha256")))
           and Path(predecessor.get("run_root", "")).parent == Path(shared.AFS) / "collection"
           and Path(predecessor.get("audit", {}).get("path", "")).parent.parent == Path(shared.AFS)
           and shared.HEX.fullmatch(str(predecessor.get("audit", {}).get("content_sha256"))),
           "Successor needs its exact terminal predecessor and complete audit evidence")
    return collection


def build(output, *, task, instance_id, attempt_id, demo_episode, prior_job_path, prior_submission_path,
          prior_source_path, prior_audit_path, ledger_path=shared.LEDGER_PATH, now=None):
    ensure = shared.ensure
    ensure(task in shared.TASKS and type(instance_id) is int and 311 <= instance_id <= 320
           and shared.NAME.fullmatch(str(attempt_id)) and type(demo_episode) is int and demo_episode in shared.TASKS[task][1],
           "Choose a supported task, 311..320 instance, new attempt ID and same-task demonstration")
    previous = shared.checked_previous(prior_job_path, prior_submission_path, prior_source_path, prior_audit_path, now=now)
    old = previous["config"]
    ensure(previous["collection"]["code"] == shared.AFS + "/code-v8"
           and old["manifest"]["bundle_sha256"] == CODE8_SHA,
           "Reuse begins only after an exact code-v8 attempt has a terminal audited result")
    # Copy immutable inputs from the audited predecessor, never its installer,
    # run directory, transient process state, private URL or recovery-only gates.
    slug = ("radio" if task == "turning_on_radio" else "shoes") + f"-{instance_id}-{attempt_id}"
    collection = copy.deepcopy(previous["collection"])
    collection.update(task=task, instance_id=instance_id, attempt_id=attempt_id,
                      run_root=shared.AFS + "/collection/" + slug, execution_name="codex-v8-" + slug,
                      max_steps=shared.TASKS[task][0], max_decisions=32, teacher_interval=TEACHER_INTERVALS[task],
                      max_seconds=7200, wall_timeout_seconds=8400, teacher_timeout=120,
                      demonstration_manifest=shared.AFS + f"/original-reader-v4/demonstrations/{demo_episode}/manifest.json")
    collection = validate_payload(collection)
    predecessor = dict(job=previous["job"], source_sha256=previous["source_sha256"],
                       run_root=previous["collection"]["run_root"], audit=previous["audit_ref"])
    config = dict(builder_kind=KIND, pilot_root=shared.AFS, report_root=shared.AFS + "/next-code8-" + slug,
                  python=shared.PYTHON, episodes=shared.EPISODES, demo_episode=demo_episode,
                  manifest=old["manifest"], previous_manifest=old["previous_manifest"],
                  expected_bundle_sha256=CODE8_SHA, preparation_manifest=old["preparation_manifest"],
                  old_bundle_sha256=shared.OLD_BUNDLE_SHA256, ciphertext_sha256=old["ciphertext_sha256"],
                  collection=collection, predecessor=predecessor, terminal_states=sorted(shared.TERMINAL))
    return shared.reserve_source(output, ledger_path=ledger_path, previous=previous, config=config, worker=WORKER,
        inputs=[prior_job_path, prior_submission_path, prior_source_path, prior_audit_path], validate_config=validate_frozen_config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=sorted(shared.TASKS), required=True)
    parser.add_argument("--instance-id", required=True, type=int)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--demo-episode", required=True, type=int)
    for flag in ("prior-job", "prior-submission", "prior-source", "prior-audit"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, task=args.task, instance_id=args.instance_id,
        attempt_id=args.attempt_id, demo_episode=args.demo_episode, prior_job_path=args.prior_job,
        prior_submission_path=args.prior_submission, prior_source_path=args.prior_source,
        prior_audit_path=args.prior_audit), indent=2))


if __name__ == "__main__":
    main()
