#!/usr/bin/env python3
"""Build one finite native-with-corrections attempt after an exact audited run.

Offline only: preserves the canonical cross-version reservation journal. The
first transition installs a separately frozen v9 bundle; later v9 attempts only
verify that same installation. No authentication, model probe or cloud submit.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

import build_codex_next_collection as shared
import build_codex_next_collection_v8 as code8
from build_codex_first_collection_v5 import _replace_once, private_bundle_url, private_installer
from run_codex_collection import validate_payload

KIND = "codex_native_collection_v9"
MODE = "native_with_corrections"
# Root-frozen r2 candidate: eight reviewed changes from code-v8. The unused
# seven-change candidate remains archived and is not admitted by this pin.
CODE9_SHA = "46bb77c810fde0cbdd5779a7cf49658b2066540edad71238fc86726d885fa512"
TEACHER_INTERVALS = dict(code8.TEACHER_INTERVALS)
ALLOWED_CHANGES = {
    "scripts/collect_b1k_codex.py", "scripts/codex_collection_teacher.py",
    "scripts/codex_collection_trajectory.py", "scripts/codex_teacher_bridge.py",
    "scripts/codex_teacher_mailbox.py", "scripts/run_codex_collection.sh",
    "sco/run_codex_collection.py", "scripts/codex_collection_actions.py",
}


def worker_source():
    source = _replace_once(shared.WORKER,
        "manifest['version']=='v7' and str(code)==manifest['code']==str(Path(config['pilot_root'])/'code-v7')",
        "manifest['version']=='v9' and str(code)==manifest['code']==str(Path(config['pilot_root'])/'code-v9')")
    source = _replace_once(source,
        "ensure(manifest['source_files_sha256'][teacher]==old['source_files_sha256'][teacher],'New code changed the previously audited teacher')",
        "ensure(collection.get('collection_mode')=='native_with_corrections','Native collection mode required')\n"
        " ensure(manifest['source_files_sha256'][teacher]==config['teacher_sha256'],'New teacher source differs')")
    source = _replace_once(source, "'kind':'codex_bounded_collection_gpu_v1'", "'kind':'codex_native_collection_gpu_v9'")
    source = _replace_once(source, "  report['predecessor']=verify_predecessor(config);report['gates']=gates(config);publish()", """  report['collection_mode']=collection['collection_mode']
  report['predecessor']=verify_predecessor(config)
  report['preparation_gates']=preparation_gates(preparation_config(config));publish()
  if config['installation_required']:
   report['stage']='code_install';publish()
   try:
    exec(compile(config['installer'],'code_v9_installer','exec'),{'__name__':'__cluster_installer__','outer_deadline':deadline})
   except BaseException as error:
    write(root/'code_install_report.json',{'state':'failed',**safe_error(error)})
    raise
   finally:
    remaining=int(deadline-time.monotonic())
    if remaining<1:raise WrapperDeadline('Finite collection wrapper deadline reached')
    signal.alarm(remaining)
  report['stage']='gates';report['gates']=gates(config);publish()""")
    return source


WORKER = worker_source()


def bounded_installer(manifest, url, report_root):
    return _replace_once(private_installer(manifest, url, report_root), "signal.alarm(600)",
        "remaining=int(outer_deadline-time.monotonic())\n"
        "if remaining<1:raise TimeoutError('No global time remains for code installation')\n"
        "signal.alarm(min(600,remaining))")


def validate_frozen_config(frozen):
    """Pure contract used by source-bound admission and the read-only auditor."""
    ensure = shared.ensure
    ensure(frozen.get("builder_kind") == KIND, "Unknown native collection worker")
    collection = validate_payload(frozen["collection"])
    task, instance, attempt = (collection[key] for key in ("task", "instance_id", "attempt_id"))
    ensure(task in shared.TASKS and shared.NAME.fullmatch(attempt), "Unknown task or unsafe attempt")
    slug = ("radio" if task == "turning_on_radio" else "shoes") + f"-{instance}-{attempt}"
    ensure(collection.get("collection_mode") == MODE
           and collection["run_root"] == shared.AFS + "/collection/" + slug
           and frozen.get("report_root") == shared.AFS + "/native-code9-" + slug
           and frozen.get("pilot_root") == shared.AFS and frozen.get("python") == shared.PYTHON
           and collection["runtime"] + "/envs/b1k-eval/bin/python" == shared.PYTHON
           and collection["teacher_backend"] == "codex_local"
           and collection["codex_home"] == shared.AFS + "/private-codex-home-v1",
           "Native mode, output, runtime or selected account differs")
    ensure(all(type(collection[key]) is int and collection[key] == value for key, value in dict(
        max_steps=shared.TASKS[task][0], max_decisions=32, teacher_interval=TEACHER_INTERVALS[task],
        max_seconds=7200, wall_timeout_seconds=8400, teacher_timeout=120).items()), "Native finite limits differ")
    episode = frozen.get("demo_episode")
    ensure(type(episode) is int and episode in shared.TASKS[task][1]
           and frozen.get("episodes") == shared.EPISODES
           and collection["demonstration_manifest"] == shared.AFS + f"/original-reader-v4/demonstrations/{episode}/manifest.json",
           "Use the verified same-task demonstration")
    manifest, prior, old = (frozen[key] for key in ("manifest", "previous_manifest", "preparation_manifest"))
    code7 = frozen["code7_manifest"]
    inventory, before = manifest.get("source_files_sha256", {}), prior.get("source_files_sha256", {})
    expected = frozen.get("expected_bundle_sha256")
    ensure(shared.HEX.fullmatch(str(CODE9_SHA)) and expected == CODE9_SHA and manifest.get("bundle_sha256") == expected
           and manifest.get("version") == "v9" and manifest.get("code") == shared.AFS + "/code-v9"
           and collection["code"] == manifest["code"] and collection["source_files_sha256"] == inventory
           and prior.get("version") == "v8" and prior.get("code") == shared.AFS + "/code-v8"
           and prior.get("bundle_sha256") == code8.CODE8_SHA
           and len(inventory) == len(before) == 29 and inventory.keys() == before.keys()
           and {name for name in inventory if inventory[name] != before[name]} == ALLOWED_CHANGES,
           "Only the eight reviewed native-mode files may differ from the exact code-v8 bundle")
    ensure(code7.get("version") == "v7" and code7.get("bundle_sha256") == shared.CODE_SHA
           and code7.get("code") == shared.AFS + "/code-v7"
           and before.keys() == code7.get("source_files_sha256", {}).keys()
           and {name for name in before if before[name] != code7["source_files_sha256"][name]} == code8.ALLOWED_CHANGES
           and old.get("version") == "v6" and old.get("code") == shared.AFS + "/code-v6"
           and old.get("bundle_sha256") == shared.OLD_BUNDLE_SHA256
           and len(old.get("source_files_sha256", {})) == 29
           and manifest["base_code"] == prior["base_code"] == code7["base_code"] == old["base_code"] == collection["base_code"]
           and manifest["base_source_manifest_sha256"] == prior["base_source_manifest_sha256"] == code7["base_source_manifest_sha256"] == old["base_source_manifest_sha256"]
           and before["scripts/codex_collection_teacher.py"] == old["source_files_sha256"]["scripts/codex_collection_teacher.py"]
           and frozen.get("teacher_sha256") == inventory["scripts/codex_collection_teacher.py"]
           and collection["robot_config"] == manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
           "Frozen preparation, original base, legacy teacher or native teacher proof differs")
    admission = frozen.get("admission", {})
    sequence, task_number = admission.get("sequence"), admission.get("task_attempt_number")
    ceiling = 33 * TEACHER_INTERVALS[task]
    ensure(type(sequence) is int and 4 <= sequence <= 20 and type(task_number) is int and 1 <= task_number <= 10
           and admission.get("reserved_decisions") == 32 and admission.get("total_reserved_decisions") == sequence * 32
           and admission.get("max_teacher_decisions_total") == 640 and admission.get("max_attempts_per_task") == 10
           and admission.get("reservations_released") is False and admission.get("all_continue_action_ceiling") == ceiling
           and admission.get("step_limit_may_exceed_teacher_budget") is (collection["max_steps"] > ceiling)
           and admission.get("correction_can_reduce_action_ceiling") is True,
           "Native mode must preserve the finite cross-version reservation")
    predecessor = frozen.get("predecessor", {})
    ensure(predecessor.get("job", {}).get("state") in shared.TERMINAL
           and shared.HEX.fullmatch(str(predecessor.get("source_sha256")))
           and Path(predecessor.get("run_root", "")).parent == Path(shared.AFS) / "collection"
           and Path(predecessor.get("audit", {}).get("path", "")).parent.parent == Path(shared.AFS)
           and shared.HEX.fullmatch(str(predecessor.get("audit", {}).get("content_sha256"))),
           "Native successor needs an exact terminal predecessor and completed audit")
    install = frozen.get("installation_required")
    ensure(type(install) is bool and predecessor.get("collection_code") == shared.AFS + ("/code-v8" if install else "/code-v9")
           and predecessor.get("bundle_sha256") == (code8.CODE8_SHA if install else expected),
           "Only code-v8 to v9 installation or identical audited v9 reuse is allowed")
    if install:
        source = frozen.get("installer")
        ensure(isinstance(source, str), "First native attempt requires its bounded installer")
        settings = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "settings"]
        ensure(len(settings) == 1, "Installer settings are not literal")
        url = ast.literal_eval(settings[0].value).get("url")
        ensure(isinstance(url, str) and source == bounded_installer(manifest, url, frozen["report_root"]),
               "Installer differs from the exact reviewed bounded implementation")
    else:
        ensure("installer" not in frozen, "Installed code-v9 reuse must not reinstall")
    return collection


def build(output, *, task, instance_id, attempt_id, demo_episode, manifest_path, expected_bundle_sha256,
          prior_job_path, prior_submission_path, prior_source_path, prior_audit_path,
          bundle_url_file=None, ledger_path=shared.LEDGER_PATH, now=None):
    ensure = shared.ensure
    ensure(task in shared.TASKS and type(instance_id) is int and 311 <= instance_id <= 320
           and shared.NAME.fullmatch(str(attempt_id)) and type(demo_episode) is int and demo_episode in shared.TASKS[task][1],
           "Choose a supported task, collection instance, fresh attempt and same-task demonstration")
    previous = shared.checked_previous(prior_job_path, prior_submission_path, prior_source_path, prior_audit_path, now=now)
    old = previous["config"]
    install = previous["collection"]["code"] == shared.AFS + "/code-v8"
    ensure(install or (previous["collection"]["code"] == shared.AFS + "/code-v9" and old.get("builder_kind") == KIND),
           "Native collection requires an audited code-v8 or identical native code-v9 predecessor")
    if install:
        if old.get("builder_kind") == code8.KIND:
            prior_worker = code8.WORKER
        else:
            from build_codex_radio_recovery_v8 import WORKER as prior_worker
        tree = ast.parse(Path(prior_source_path).read_text())
        ensure(ast.dump(ast.Module(body=tree.body[:-1], type_ignores=[])) == ast.dump(ast.parse(prior_worker)),
               "Code-v8 predecessor is not its exact reviewed collection worker")
    manifest = shared.load(manifest_path)
    ensure(shared.HEX.fullmatch(str(expected_bundle_sha256)) and manifest.get("bundle_sha256") == expected_bundle_sha256
           and shared.sha(manifest["bundle"]) == expected_bundle_sha256, "New bundle must match its independently frozen SHA256")
    if not install:
        ensure(manifest == old["manifest"] and expected_bundle_sha256 == old["expected_bundle_sha256"],
               "Reuse must retain the exact installed code-v9 manifest")
        ensure(bundle_url_file is None, "Reuse must not supply another installation URL")
    slug = ("radio" if task == "turning_on_radio" else "shoes") + f"-{instance_id}-{attempt_id}"
    report_root = shared.AFS + "/native-code9-" + slug
    collection = copy.deepcopy(previous["collection"])
    collection.update(code=manifest["code"], source_files_sha256=manifest["source_files_sha256"],
        collection_mode=MODE, task=task, instance_id=instance_id, attempt_id=attempt_id,
        run_root=shared.AFS + "/collection/" + slug, execution_name="codex-native-v9-" + slug,
        max_steps=shared.TASKS[task][0], max_decisions=32, teacher_interval=TEACHER_INTERVALS[task],
        max_seconds=7200, wall_timeout_seconds=8400, teacher_timeout=120,
        robot_config=manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
        demonstration_manifest=shared.AFS + f"/original-reader-v4/demonstrations/{demo_episode}/manifest.json")
    collection = validate_payload(collection)
    predecessor = dict(job=previous["job"], source_sha256=previous["source_sha256"],
        run_root=previous["collection"]["run_root"], audit=previous["audit_ref"],
        collection_code=previous["collection"]["code"], bundle_sha256=old["manifest"]["bundle_sha256"])
    config = dict(builder_kind=KIND, pilot_root=shared.AFS, report_root=report_root, python=shared.PYTHON,
        episodes=shared.EPISODES, demo_episode=demo_episode, manifest=manifest,
        previous_manifest=old["manifest"] if install else old["previous_manifest"],
        code7_manifest=old["previous_manifest"] if install else old["code7_manifest"],
        expected_bundle_sha256=expected_bundle_sha256, preparation_manifest=old["preparation_manifest"],
        old_bundle_sha256=shared.OLD_BUNDLE_SHA256, ciphertext_sha256=old["ciphertext_sha256"],
        teacher_sha256=manifest["source_files_sha256"]["scripts/codex_collection_teacher.py"],
        installation_required=install, collection=collection, predecessor=predecessor, terminal_states=sorted(shared.TERMINAL))
    inputs = [manifest_path, manifest["bundle"], prior_job_path, prior_submission_path, prior_source_path, prior_audit_path]
    if install:
        ensure(bundle_url_file is not None, "First native attempt needs the private frozen-bundle URL file")
        config["installer"] = bounded_installer(manifest, private_bundle_url(bundle_url_file), report_root)
        inputs.append(bundle_url_file)
    return shared.reserve_source(output, ledger_path=ledger_path, previous=previous, config=config, worker=WORKER,
                                 inputs=inputs, validate_config=validate_frozen_config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=sorted(shared.TASKS))
    parser.add_argument("--instance-id", required=True, type=int)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--demo-episode", required=True, type=int)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--bundle-url-file", type=Path)
    for flag in ("manifest", "prior-job", "prior-submission", "prior-source", "prior-audit"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, task=args.task, instance_id=args.instance_id,
        attempt_id=args.attempt_id, demo_episode=args.demo_episode, manifest_path=args.manifest,
        expected_bundle_sha256=args.expected_bundle_sha256, bundle_url_file=args.bundle_url_file,
        prior_job_path=args.prior_job, prior_submission_path=args.prior_submission,
        prior_source_path=args.prior_source, prior_audit_path=args.prior_audit), indent=2))


if __name__ == "__main__":
    main()
