#!/usr/bin/env python3
"""Build the finite radio311/pilot-003 recovery with a verified code-v8 install.

Offline only. The exact pilot-002 terminal/audit closes the previous reservation;
its 17 observed model calls remain in the shared experiment ledger. Installation
and collection share a 9000-second deadline, with 9600 seconds for the cloud job.
No fresh authentication, model probe, data transfer or old-run modification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import build_codex_next_collection as shared
from build_codex_first_collection_v5 import _replace_once, private_bundle_url, private_installer
from run_codex_collection import validate_payload

ALLOWED_CHANGES = {"scripts/collect_b1k_codex.py", "scripts/codex_collection_actions.py"}
PRIOR_SOURCE_SHA = "b8698f30fd16835d7ac4bd6f2db011c2a22f9d3b1708025e68aa6af924c1e7e3"
RUN_ROOT = shared.AFS + "/collection/radio-311-pilot003"
REPORT_ROOT = shared.AFS + "/radio311-recovery-gpu-v8"


def worker_source():
    source = _replace_once(shared.WORKER,
        "manifest['version']=='v7' and str(code)==manifest['code']==str(Path(config['pilot_root'])/'code-v7')",
        "manifest['version']=='v8' and str(code)==manifest['code']==str(Path(config['pilot_root'])/'code-v8')")
    source = _replace_once(source, "'kind':'codex_bounded_collection_gpu_v1'", "'kind':'codex_radio_recovery_gpu_v8'")
    source = _replace_once(source, "  report['predecessor']=verify_predecessor(config);report['gates']=gates(config);publish()", """  report['predecessor']=verify_predecessor(config)
  report['recovery_from']=config['recovery_from']
  report['preparation_gates']=preparation_gates(preparation_config(config));publish()
  report['stage']='code_install';publish()
  try:
   exec(compile(config['installer'],'code_v8_installer','exec'),{'__name__':'__cluster_installer__','outer_deadline':deadline})
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


def bounded_installer(manifest, url):
    source = private_installer(manifest, url, REPORT_ROOT)
    return _replace_once(source, "signal.alarm(600)",
        "remaining=int(outer_deadline-time.monotonic())\n"
        "if remaining<1:raise TimeoutError('No global time remains for code installation')\n"
        "signal.alarm(min(600,remaining))")


def validate_frozen_config(frozen):
    """Pure read-only contract for source-bound auditing and ledger settlement."""
    ensure = shared.ensure
    collection = validate_payload(frozen["collection"])
    manifest, code7, old = (frozen[key] for key in ("manifest", "previous_manifest", "preparation_manifest"))
    expected = frozen.get("expected_bundle_sha256")
    ensure(isinstance(expected, str) and shared.HEX.fullmatch(expected)
           and manifest.get("version") == "v8" and manifest.get("code") == shared.AFS + "/code-v8"
           and manifest.get("bundle_sha256") == expected
           and collection["code"] == manifest["code"] and collection["source_files_sha256"] == manifest["source_files_sha256"],
           "Frozen code-v8 manifest and payload differ")
    ensure(code7.get("version") == "v7" and code7.get("code") == shared.AFS + "/code-v7"
           and code7.get("bundle_sha256") == shared.CODE_SHA,
           "Frozen predecessor must be the known code-v7 bundle")
    ensure(old.get("version") == "v6" and old.get("code") == shared.AFS + "/code-v6"
           and old.get("bundle_sha256") == shared.OLD_BUNDLE_SHA256
           and len(old.get("source_files_sha256", {})) == 29,
           "Frozen preparation must remain the verified code-v6")
    inventory = manifest["source_files_sha256"]
    ensure(len(inventory) == 29 and inventory.keys() == code7["source_files_sha256"].keys()
           and {name for name in inventory if inventory[name] != code7["source_files_sha256"][name]} == ALLOWED_CHANGES,
           "Only the collector and bounded correction implementation may change")
    ensure(manifest["base_code"] == code7["base_code"] == old["base_code"] == collection["base_code"]
           and manifest["base_source_manifest_sha256"] == code7["base_source_manifest_sha256"] == old["base_source_manifest_sha256"]
           and inventory["scripts/codex_collection_teacher.py"] == old["source_files_sha256"]["scripts/codex_collection_teacher.py"],
           "Frozen immutable base or verified teacher changed")
    identity = dict(task="turning_on_radio", instance_id=311, attempt_id="pilot-003", run_root=RUN_ROOT)
    ensure(shared.row_identity(collection) == identity
           and all(collection[key] == value for key, value in dict(max_steps=4500, max_decisions=32,
                teacher_interval=256, max_seconds=7200, wall_timeout_seconds=8400, teacher_timeout=120).items())
           and collection["teacher_backend"] == "codex_local"
           and collection["codex_home"] == shared.AFS + "/private-codex-home-v1"
           and collection["runtime"] + "/envs/b1k-eval/bin/python" == shared.PYTHON
           and collection["demonstration_manifest"] == shared.AFS + "/original-reader-v4/demonstrations/28/manifest.json"
           and collection["robot_config"] == manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml"
           and frozen.get("pilot_root") == shared.AFS and frozen.get("report_root") == REPORT_ROOT
           and frozen.get("python") == shared.PYTHON and frozen.get("demo_episode") == 28
           and frozen.get("episodes") == shared.EPISODES,
           "Frozen recovery identity, paths, account or finite limits differ")
    recovery = frozen["recovery_from"]
    ensure(recovery.get("source_sha256") == PRIOR_SOURCE_SHA and recovery.get("observed_model_calls") == 17
           and recovery.get("executed_steps") == 3339 and recovery.get("failed_attempt_reserved_decisions") == 32
           and recovery.get("job", {}).get("name") == shared.PILOT2_NAME
           and recovery.get("job", {}).get("uid") == shared.PILOT2_UID
           and recovery.get("job", {}).get("state") in shared.TERMINAL
           and frozen["predecessor"]["source_sha256"] == PRIOR_SOURCE_SHA
           and frozen["predecessor"]["job"] == recovery["job"]
           and isinstance(frozen.get("installer"), str) and frozen["installer"],
           "Frozen recovery must retain exact terminal pilot-002 evidence and its 17 calls")
    admission = frozen.get("admission", {})
    ensure(all(type(admission.get(key)) is int and admission[key] == value for key, value in dict(
               sequence=3, reserved_decisions=32, total_reserved_decisions=96, max_teacher_decisions_total=640,
               task_attempt_number=3, max_attempts_per_task=10).items())
           and admission.get("reservations_released") is False,
           "Frozen recovery must preserve all three attempts and the conservative 96-decision reservation")
    return collection


def checked_recovery(previous):
    ensure = shared.ensure
    collection, audit = previous["collection"], previous["audit"]
    ensure(previous["job"]["uid"] == shared.PILOT2_UID and previous["job"]["name"] == shared.PILOT2_NAME
           and previous["source_sha256"] == PRIOR_SOURCE_SHA
           and shared.row_identity(collection) == dict(task="turning_on_radio", instance_id=311,
                attempt_id="pilot-002", run_root=shared.AFS + "/collection/radio-311-pilot002"),
           "Recovery requires the exact submitted radio311/pilot-002")
    teachers = audit["teachers"]
    ensure(audit.get("kind") == "codex_pilot002_readonly_audit_v5" and audit.get("training_eligible") is False
           and audit["actions"].get("rows") == 3339
           and teachers.get("requests") == teachers.get("model_calls_observed") == 17
           and teachers.get("requests_with_unknown_model_call") == 0
           and teachers.get("requests_failed_before_model") == 0
           and len(teachers.get("entries", [])) == 17
           and all(row.get("receipt_status") == "validated"
                   and all(row.get(key) is True for key in
                           ("decision_valid", "event_audit_valid", "pinned_model_verified", "chatgpt_login_verified"))
                   for row in teachers["entries"]),
           "Recovery requires the actual failed 3339-action, 17-model-call audit")
    return dict(job=previous["job"], source_sha256=previous["source_sha256"],
                audit=previous["audit_ref"], executed_steps=3339, observed_model_calls=17,
                failed_attempt_reserved_decisions=32, reason="bounded_cartesian_ik_rejection_before_env_step")


def build(output, *, manifest_path, expected_bundle_sha256, bundle_url_file, prior_job_path,
          prior_submission_path=shared.OUT / "first_collection_submission_v5.json",
          prior_source_path=shared.OUT / "first_collection_job_v5.py",
          prior_audit_path=shared.OUT / "pilot002_audit_submission_report.json",
          ledger_path=shared.LEDGER_PATH, now=None):
    ensure = shared.ensure
    previous = shared.checked_previous(prior_job_path, prior_submission_path, prior_source_path, prior_audit_path, now=now)
    recovery = checked_recovery(previous)
    old_config = previous["config"]
    old = old_config["preparation_manifest"]
    code7 = old_config["manifest"]
    manifest = shared.load(manifest_path)
    ensure(isinstance(expected_bundle_sha256, str) and shared.HEX.fullmatch(expected_bundle_sha256)
           and manifest.get("version") == "v8" and manifest.get("code") == shared.AFS + "/code-v8"
           and manifest.get("bundle_sha256") == expected_bundle_sha256
           and shared.sha(manifest["bundle"]) == expected_bundle_sha256,
           "New code-v8 bundle must match the independently supplied frozen SHA256")
    ensure(code7.get("source_files_sha256") == previous["collection"]["source_files_sha256"],
           "Predecessor source inventory differs from its actual collection payload")
    inventory = manifest.get("source_files_sha256", {})
    collection = copy.deepcopy(previous["collection"])
    ensure(collection.get("teacher_backend") == "codex_local"
           and collection.get("codex_home") == shared.AFS + "/private-codex-home-v1"
           and collection["runtime"] + "/envs/b1k-eval/bin/python" == shared.PYTHON,
           "Reuse the selected private account and verified runtime")
    collection.update(code=manifest["code"], source_files_sha256=inventory, attempt_id="pilot-003",
                      execution_name="codex-radio311-pilot003-20261011", run_root=RUN_ROOT,
                      robot_config=manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
                      demonstration_manifest=shared.AFS + "/original-reader-v4/demonstrations/28/manifest.json",
                      max_steps=4500, max_decisions=32, teacher_interval=256, max_seconds=7200,
                      wall_timeout_seconds=8400, teacher_timeout=120)
    collection = validate_payload(collection)
    predecessor = dict(job=previous["job"], source_sha256=previous["source_sha256"],
                       run_root=previous["collection"]["run_root"], audit=previous["audit_ref"])
    config = dict(pilot_root=shared.AFS, report_root=REPORT_ROOT, python=shared.PYTHON,
                  episodes=shared.EPISODES, demo_episode=28, manifest=manifest,
                  previous_manifest=code7, expected_bundle_sha256=expected_bundle_sha256,
                  preparation_manifest=old, old_bundle_sha256=shared.OLD_BUNDLE_SHA256,
                  ciphertext_sha256=old_config["ciphertext_sha256"], collection=collection,
                  predecessor=predecessor, terminal_states=sorted(shared.TERMINAL), recovery_from=recovery,
                  installer=bounded_installer(manifest, private_bundle_url(bundle_url_file)))
    return shared.reserve_source(output, ledger_path=ledger_path, previous=previous, config=config, worker=WORKER,
        inputs=[manifest_path, manifest["bundle"], bundle_url_file, prior_job_path,
                prior_submission_path, prior_source_path, prior_audit_path], validate_config=validate_frozen_config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=shared.OUT / "radio311_pilot003_job_v8.py")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--bundle-url-file", required=True, type=Path)
    parser.add_argument("--prior-job", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, manifest_path=args.manifest,
        expected_bundle_sha256=args.expected_bundle_sha256, bundle_url_file=args.bundle_url_file,
        prior_job_path=args.prior_job), indent=2))


if __name__ == "__main__":
    main()
