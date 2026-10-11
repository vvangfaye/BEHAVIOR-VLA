#!/usr/bin/env python3
"""Build one subsequent collection attempt offline, after a terminal audited one.

The append-only local ledger conservatively reserves 32 decisions forever for
every admitted attempt, including failed/unknown attempts. A reserved row must
be settled by its exact submitted source, fresh terminal observation and actual
read-only audit before another source can be built. This module never submits,
installs code, authenticates, accesses the network or calls a model.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time

from build_codex_first_collection_v4 import AFS, EPISODES, OUT, PYTHON, WORKER as V4_WORKER
from build_codex_first_collection_v5 import OLD_BUNDLE_SHA256, RECOVERY, TERMINAL, _replace_once
from run_codex_collection import validate_payload

KIND = "codex_collection_attempt_ledger_v1"
LEDGER_PATH = OUT / "collection_attempt_ledger.json"
CODE_SHA = "7054eabae8d7e566501d9894fbdc233f9ab638efdf3d09fa550f6af879edcf20"
PILOT2_UID = "37bf17cc0ef849d0901f5a3a396b44e1"
PILOT2_NAME = "behavior-codex-radio311-pilot-v5-1011-051558"
TASKS = {"turning_on_radio": (4500, (28, 68, 102)), "putting_shoes_on_rack": (10000, (4424, 4487, 4517))}
HEX = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_config(path):
    tree = ast.parse(Path(path).read_text())
    final = tree.body[-1]
    ensure(isinstance(final, ast.Assign) and len(final.targets) == 1
           and isinstance(final.targets[0], ast.Name) and final.targets[0].id == "BOOTSTRAP_REPORT"
           and isinstance(final.value, ast.Call) and isinstance(final.value.func, ast.Name)
           and final.value.func.id == "run" and len(final.value.args) == 1 and not final.value.keywords,
           "Submitted source has no literal collection configuration")
    return ast.literal_eval(final.value.args[0])


def checked_previous(job_path, submission_path, source_path, audit_path, *, now=None):
    job, submission, audit = load(job_path), load(submission_path), load(audit_path)
    observed = job.get("observed_at")
    now = time.time() if now is None else now
    ensure(type(observed) in (int, float) and 0 <= now - observed <= 900
           and job.get("state") in TERMINAL
           and all(isinstance(job.get(key), str) and job[key] == submission.get(key) for key in ("name", "uid")),
           "Need a fresh terminal observation of the exact submitted predecessor")
    source_sha = sha(source_path)
    ensure(submission.get("source_sha256") == source_sha, "Predecessor submitted source SHA256 differs")
    config = frozen_config(source_path)
    collection = config["collection"]
    native = collection.get("code") == AFS + "/code-v9" and config.get("builder_kind") == "codex_native_collection_v9"
    interval = (320 if ((collection.get("code") == AFS + "/code-v8"
                and config.get("builder_kind") == "codex_next_collection_v8") or native)
                and collection.get("task") == "putting_shoes_on_rack" else 256)
    ensure(collection.get("task") in TASKS and type(collection.get("instance_id")) is int
           and 311 <= collection["instance_id"] <= 320 and collection.get("code") in {AFS + "/code-v7", AFS + "/code-v8", AFS + "/code-v9"}
           and collection.get("max_decisions") == 32 and collection.get("teacher_interval") == interval
           and collection.get("collection_mode", "teacher_conditioned") == ("native_with_corrections" if native else "teacher_conditioned"),
           "Predecessor is outside the bounded collection experiment")
    if collection["code"] == AFS + "/code-v9":
        from build_codex_native_collection_v9 import WORKER as native_worker, validate_frozen_config
        validate_frozen_config(config)
        tree = ast.parse(Path(source_path).read_text())
        ensure(ast.dump(ast.Module(body=tree.body[:-1], type_ignores=[])) == ast.dump(ast.parse(native_worker)),
               "Predecessor is not the exact reviewed native worker")
    elif collection["code"] == AFS + "/code-v8":
        if config.get("builder_kind") == "codex_next_collection_v8":
            from build_codex_next_collection_v8 import validate_frozen_config
        else:
            from build_codex_radio_recovery_v8 import validate_frozen_config
        validate_frozen_config(config)
    ensure(audit.get("kind") in {"codex_pilot002_readonly_audit_v5", "codex_collection_readonly_audit_v1"}
           and audit.get("state") == "audit_complete" and audit.get("source_unchanged_during_audit") is True
           and audit.get("model_calls") == 0 and audit.get("network_calls") == 0
           and audit.get("model_calls_scope") == "audit_process_only"
           and audit.get("collection_source_sha256") == source_sha
           and audit.get("run_root") == collection["run_root"]
           and all(audit.get("source_job", {}).get(key) == job[key] for key in ("name", "uid", "state")),
           "Need the actual completed read-only audit of this exact predecessor")
    if native:
        ensure(audit.get("collection_mode") == "native_with_corrections", "Native predecessor audit mode differs")
    ensure(type(audit.get("training_eligible")) is bool and isinstance(audit.get("issues"), list)
           and isinstance(audit.get("teachers"), dict) and isinstance(audit.get("actions"), dict)
           and isinstance(audit.get("trajectory"), dict), "Predecessor audit is incomplete")
    accounting = audit["teachers"]
    ensure(all(type(accounting.get(key)) is int and 0 <= accounting[key] <= 32 for key in
               ("requests", "model_calls_observed", "requests_with_unknown_model_call", "requests_failed_before_model"))
           and accounting["model_calls_observed"] + accounting["requests_with_unknown_model_call"] <= 32,
           "Predecessor model usage is not contained by its 32-call reservation")
    audit_root = Path(audit.get("audit_root", ""))
    ensure(audit_root.is_absolute() and audit_root.parent == Path(AFS)
           and audit_root.name.startswith("audit-") and audit_root != Path(collection["run_root"]),
           "Predecessor audit must be in its independent pilot audit directory")
    # The cloud bootstrap serializes the returned audit again for AOSS. Its
    # compact bytes differ from the AFS pretty JSON despite identical evidence.
    audit_ref = dict(path=str(audit_root / "report.json"), content_sha256=fingerprint(audit),
                     downloaded_report_sha256=sha(audit_path), downloaded_report_bytes=Path(audit_path).stat().st_size)
    return dict(job={key: job[key] for key in ("name", "uid", "state", "observed_at")},
                source_sha256=source_sha, collection=collection, config=config, audit=audit, audit_ref=audit_ref)


def row_identity(collection):
    return {key: collection[key] for key in ("task", "instance_id", "attempt_id", "run_root")}


def seed_ledger(previous):
    """The first successor must account for both already-admitted pilot attempts."""
    collection, config = previous["collection"], previous["config"]
    ensure(previous["job"]["uid"] == PILOT2_UID and previous["job"]["name"] == PILOT2_NAME
           and (collection["task"], collection["instance_id"], collection["attempt_id"]) == ("turning_on_radio", 311, "pilot-002"),
           "A new ledger can only start from the exact audited pilot-002")
    recovery = config.get("recovery_from", {})
    old = recovery.get("report_identity", {})
    ensure(old.get("state") == "failed" and old.get("training_eligible") is False
           and (old.get("task"), old.get("instance_id"), old.get("attempt_id")) == ("turning_on_radio", 311, "pilot-001")
           and old.get("executed_steps_observed") == 0 and old.get("teacher_usage")
           and all(type(value) is int and value == 0 for value in old["teacher_usage"].values())
           and recovery.get("job", {}).get("state") in TERMINAL
           and HEX.fullmatch(str(recovery.get("source_sha256"))),
           "Initial zero-action attempt recovery evidence is missing")
    rows = [dict(task="turning_on_radio", instance_id=311, attempt_id="pilot-001",
                 run_root=AFS + "/collection/radio-311-pilot001", reserved_decisions=32,
                 state="historical_terminal", source_sha256=recovery["source_sha256"],
                 job=recovery["job"], observed_model_calls=0,
                 evidence="verified_zero_action_recovery_in_exact_pilot002_source"),
            dict(**row_identity(collection), reserved_decisions=32, state="reserved",
                 source_sha256=previous["source_sha256"])]
    return dict(kind=KIND, selected_auth="auth_weizhi.json", code_bundle_sha256=CODE_SHA,
                max_teacher_decisions_total=640, max_attempts_per_task=10,
                reservations_released=False, entries=rows)


def validate_ledger(ledger):
    ensure(ledger.get("kind") == KIND and ledger.get("selected_auth") == "auth_weizhi.json"
           and ledger.get("code_bundle_sha256") == CODE_SHA
           and ledger.get("max_teacher_decisions_total") == 640 and ledger.get("max_attempts_per_task") == 10
           and ledger.get("reservations_released") is False, "Ledger contract differs")
    rows = ledger.get("entries")
    ensure(isinstance(rows, list) and 2 <= len(rows) <= 20, "Ledger attempt count is invalid")
    identities, roots = set(), set()
    for index, row in enumerate(rows):
        ensure(row.get("task") in TASKS and type(row.get("instance_id")) is int
               and 311 <= row["instance_id"] <= 320 and NAME.fullmatch(str(row.get("attempt_id")))
               and type(row.get("reserved_decisions")) is int and row["reserved_decisions"] == 32
               and HEX.fullmatch(str(row.get("source_sha256"))), "Invalid ledger reservation")
        identity = (row["task"], row["instance_id"], row["attempt_id"])
        ensure(identity not in identities and row["run_root"] not in roots
               and Path(row["run_root"]).parent == Path(AFS) / "collection", "Repeated or unsafe ledger attempt")
        identities.add(identity); roots.add(row["run_root"])
        ensure(row.get("state") in {"historical_terminal", "audited", "reserved"}, "Unknown ledger state")
        ensure(row["state"] != "reserved" or index == len(rows) - 1, "Earlier attempt still has an unresolved reservation")
        ensure(row["state"] != "historical_terminal" or index == 0, "Only the verified startup pilot is historical")
    ensure(sum(row["reserved_decisions"] for row in rows) <= 640
           and all(sum(row["task"] == task for row in rows) <= 10 for task in TASKS), "Experiment budget exhausted")


def settle(ledger, previous):
    validate_ledger(ledger)
    last = ledger["entries"][-1]
    ensure(last["state"] == "reserved" and row_identity(last) == row_identity(previous["collection"])
           and last["source_sha256"] == previous["source_sha256"], "Audit does not settle the latest reserved attempt")
    teachers = previous["audit"]["teachers"]
    # Retain usage evidence; never infer missing usage as zero or release budget.
    last.update(state="audited", job=previous["job"], audit=previous["audit_ref"],
                training_eligible=previous["audit"]["training_eligible"],
                teacher_accounting={key: teachers.get(key) for key in
                    ("model_calls_observed", "requests_with_unknown_model_call", "requests_failed_before_model")})


EXTRA_GATES = r'''
class WrapperDeadline(BaseException):pass

def verify_predecessor(config):
 previous=config['predecessor'];ref=previous['audit'];path=Path(ref['path'])
 ensure(path.is_file(),'Previous audit report is missing')
 audit=load(path)
 digest=hashlib.sha256(json.dumps(audit,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 ensure(digest==ref['content_sha256'],'Previous audit evidence changed')
 ensure(previous['job']['state'] in config['terminal_states'] and audit.get('state')=='audit_complete'
        and audit.get('source_unchanged_during_audit') is True
        and audit.get('collection_source_sha256')==previous['source_sha256']
        and audit.get('run_root')==previous['run_root']
        and all(audit.get('source_job',{}).get(key)==previous['job'][key] for key in ('name','uid','state')),
        'Exact terminal predecessor and completed audit required')
 return previous
'''


def worker_source():
    source = _replace_once(V4_WORKER, "def gates(config):", "def preparation_gates(config):")
    source = _replace_once(source, "ensure(str(demo_path)==str(base/'original-reader-v4/demonstrations/28/manifest.json'),'Wrong radio reference')",
                          "ensure(str(demo_path)==str(base/'original-reader-v4/demonstrations'/str(config['demo_episode'])/'manifest.json'),'Wrong task reference')")
    source = _replace_once(source, "ensure(demo.get('metadata',{}).get('task')=='turning_on_radio' and demo['metadata'].get('episode_index')==28,'Radio reference identity differs')",
                          "ensure(demo.get('metadata',{}).get('task')==collection['task'] and demo['metadata'].get('episode_index')==config['demo_episode'],'Task reference identity differs')")
    source = _replace_once(source, "matching=next(row for row in demonstrations if row['episode_index']==28)",
                          "matching=next(row for row in demonstrations if row['episode_index']==config['demo_episode'])")
    lines = RECOVERY.splitlines(keepends=True)
    code_gates = "\n".join("".join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(RECOVERY).body
                           if isinstance(node, ast.FunctionDef) and node.name in {"preparation_config", "gates"})
    source = _replace_once(source, "def run(config):", EXTRA_GATES + "\n" + code_gates + "\ndef run(config):")
    source = _replace_once(source, "'kind':'codex_first_collection_gpu_v4'", "'kind':'codex_bounded_collection_gpu_v1'")
    source = _replace_once(source, "'task':'turning_on_radio','instance_id':311,\n         'attempt_id':'pilot-001'",
                          "'task':collection['task'],'instance_id':collection['instance_id'],\n         'attempt_id':collection['attempt_id']")
    source = _replace_once(source, "'max_steps':4500,'max_decisions':32,'max_seconds':7200}",
                          "'max_steps':collection['max_steps'],'max_decisions':32,'max_seconds':7200},'budget_admission':config['admission']")
    source = _replace_once(source, "def stop(signum,frame):raise TimeoutError('Finite collection wrapper deadline reached')",
                          "def stop(signum,frame):raise WrapperDeadline('Finite collection wrapper deadline reached')")
    source = _replace_once(source, "  report['gates']=gates(config);publish()",
                          "  report['predecessor']=verify_predecessor(config);report['gates']=gates(config);publish()")
    return source


WORKER = worker_source()


def write_private(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "w") as stream:
        stream.write(value)
        stream.flush(); os.fsync(stream.fileno())


def verify_journal(directory, ledger):
    """Refuse missing or rolled-back ledgers; never rewrite admission anchors."""
    ensure(directory.is_dir() and not directory.is_symlink(), "The existing ledger journal is missing")
    count = len(ledger.get("entries", []))
    expected = [f"{number:04d}.json" for number in range(3, count + 1)]
    ensure(sorted(path.name for path in directory.iterdir()) == expected and expected,
           "Ledger journal disagrees with the current reservation history")
    previous = None
    for number, name in enumerate(expected, 3):
        path = directory / name
        ensure(path.is_file() and not path.is_symlink(), "Ledger journal anchor is invalid")
        anchor = load(path)
        ensure(anchor.get("sequence") == number and anchor.get("previous_fingerprint") == previous
               and HEX.fullmatch(str(anchor.get("ledger_fingerprint"))), "Ledger journal chain changed")
        previous = anchor["ledger_fingerprint"]
    ensure(previous == ledger.get("fingerprint")
           and anchor.get("source_sha256") == ledger["entries"][-1]["source_sha256"],
           "The ledger was replaced or rolled back after a reservation")


def build(output, *, task, instance_id, attempt_id, demo_episode, ledger_path,
          prior_job_path, prior_submission_path, prior_source_path, prior_audit_path,
          manifest_path=OUT / "collection_code_v7.tar.gz.manifest.json",
          preparation_manifest_path=OUT / "collection_code_v6.tar.gz.manifest.json",
          collection_path=OUT / "initial_collection_payload_prepared.json",
          encryption_path=OUT / "auth_encryption_receipt.json", now=None):
    ensure(task in TASKS and type(instance_id) is int and 311 <= instance_id <= 320
           and NAME.fullmatch(str(attempt_id)) and type(demo_episode) is int and demo_episode in TASKS[task][1],
           "Choose one of the two tasks, instance311..320, safe attempt ID and same-task prepared demonstration")
    previous = checked_previous(prior_job_path, prior_submission_path, prior_source_path, prior_audit_path, now=now)
    manifest, old = load(manifest_path), load(preparation_manifest_path)
    ensure(previous["collection"]["code"] == manifest.get("code"),
           "A continuation must not silently revert the predecessor's collection code")
    ensure(manifest.get("version") == "v7" and manifest.get("code") == AFS + "/code-v7"
           and manifest.get("bundle_sha256") == CODE_SHA and len(manifest.get("source_files_sha256", {})) == 29,
           "Use the already installed frozen code-v7 bundle")
    ensure(old.get("version") == "v6" and old.get("code") == AFS + "/code-v6"
           and old.get("bundle_sha256") == OLD_BUNDLE_SHA256 and len(old.get("source_files_sha256", {})) == 29,
           "Use the verified original code-v6 preparation")
    teacher = "scripts/codex_collection_teacher.py"
    ensure(manifest["base_code"] == old["base_code"]
           and manifest["base_source_manifest_sha256"] == old["base_source_manifest_sha256"]
           and manifest["source_files_sha256"][teacher] == old["source_files_sha256"][teacher],
           "Immutable base or previously audited teacher differs")
    template, encryption = load(collection_path), load(encryption_path)
    ensure(Path(encryption["selected_auth_file"]).name == "auth_weizhi.json"
           and template.get("teacher_backend") == "codex_local"
           and template.get("codex_home") == AFS + "/private-codex-home-v1", "Reuse only the existing selected private account")
    slug = ("radio" if task == "turning_on_radio" else "shoes") + f"-{instance_id}-{attempt_id}"
    collection = {**template, "task": task, "instance_id": instance_id, "attempt_id": attempt_id,
                  "execution_name": "codex-" + slug, "run_root": AFS + "/collection/" + slug,
                  "code": manifest["code"], "source_files_sha256": manifest["source_files_sha256"],
                  "robot_config": manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
                  "demonstration_manifest": AFS + f"/original-reader-v4/demonstrations/{demo_episode}/manifest.json",
                  "max_steps": TASKS[task][0], "max_decisions": 32, "teacher_interval": 256,
                  "max_seconds": 7200, "wall_timeout_seconds": 8400, "teacher_timeout": 120}
    collection = validate_payload(collection)
    ensure(collection["runtime"] + "/envs/b1k-eval/bin/python" == PYTHON
           and collection["base_code"] == manifest["base_code"], "Runtime interpreter or immutable base differs")
    predecessor = dict(job=previous["job"], source_sha256=previous["source_sha256"],
                       run_root=previous["collection"]["run_root"], audit=previous["audit_ref"])
    config = dict(pilot_root=AFS, report_root=AFS + "/next-" + slug, python=PYTHON,
                  episodes=EPISODES, demo_episode=demo_episode, manifest=manifest,
                  preparation_manifest=old, old_bundle_sha256=OLD_BUNDLE_SHA256,
                  ciphertext_sha256=encryption["ciphertext_sha256"], collection=collection,
                  predecessor=predecessor, terminal_states=sorted(TERMINAL))
    return reserve_source(output, ledger_path=ledger_path, previous=previous, config=config, worker=WORKER,
                          inputs=[prior_job_path, prior_submission_path, prior_source_path, prior_audit_path,
                                  manifest_path, preparation_manifest_path, collection_path, encryption_path])


def reserve_source(output, *, ledger_path, previous, config, worker, inputs, validate_config=None):
    """One shared admission transaction for reviewed code-version builders.

    Callers verify their code/install contract first. The ledger's code hash is
    the original code-v7 experiment anchor, not permission to reset on an upgrade.
    Each reserved row records the exact source and collection code for that run.
    """
    collection, manifest = config["collection"], config["manifest"]
    task, instance_id, attempt_id = (collection[key] for key in ("task", "instance_id", "attempt_id"))
    demo_episode = config["demo_episode"]
    predecessor = config["predecessor"]
    output, ledger_path = Path(output).absolute(), Path(ledger_path).absolute()
    ensure(ledger_path.resolve() == LEDGER_PATH.resolve(), "Use the single canonical experiment ledger")
    receipt_path = output.with_suffix(".receipt.json")
    protected = {Path(path).resolve() for path in inputs}
    ensure(len({output.resolve(), receipt_path.resolve(), ledger_path.resolve()}) == 3
           and not ({output.resolve(), receipt_path.resolve(), ledger_path.resolve()} & protected), "Output path collides with an input")
    ensure(not any(path.exists() or path.is_symlink() for path in (output, receipt_path)), "Source and receipt must be fresh")
    ensure(not ledger_path.is_symlink(), "Ledger symlinks are forbidden")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    journal = ledger_path.with_name(ledger_path.name + ".journal")
    with os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger_path.exists():
            ledger = load(ledger_path)
            ensure(ledger.get("fingerprint") == fingerprint({k: v for k, v in ledger.items() if k != "fingerprint"}), "Ledger changed")
            verify_journal(journal, ledger)
            previous_fingerprint = ledger["fingerprint"]
            ledger.pop("fingerprint")
        else:
            ensure(not journal.exists() and not journal.is_symlink(), "An initialized ledger is missing; inspect preserved reservations")
            ledger = seed_ledger(previous)
            previous_fingerprint = None
        settle(ledger, previous)
        rows = ledger["entries"]
        ensure(sum(row["task"] == task for row in rows) < 10 and len(rows) < 20, "No attempt budget remains for this task")
        ensure(sum(row["task"] == task and row.get("training_eligible") is True for row in rows) < 5,
               "This task already reached its five verified-success target")
        ensure(not any((row["task"], row["instance_id"], row["attempt_id"]) == (task, instance_id, attempt_id)
                       or row["run_root"] == collection["run_root"] for row in rows), "Attempt identity or root has already been admitted")
        ceiling = (33 if collection.get("collection_mode") == "native_with_corrections" else 32) * collection["teacher_interval"]
        admission = dict(sequence=len(rows) + 1, reserved_decisions=32, total_reserved_decisions=32 * (len(rows) + 1),
                         max_teacher_decisions_total=640, task_attempt_number=sum(row["task"] == task for row in rows) + 1,
                         max_attempts_per_task=10, reservations_released=False,
                         all_continue_action_ceiling=ceiling, correction_can_reduce_action_ceiling=True,
                         step_limit_may_exceed_teacher_budget=collection["max_steps"] > ceiling)
        config = {**config, "admission": admission}
        if validate_config is not None:
            validate_config(config)
        source = worker + "\nBOOTSTRAP_REPORT = run(" + repr(config) + ")\n"
        compile(source, "<next-collection>", "exec")
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        rows.append(dict(**row_identity(collection), reserved_decisions=32, state="reserved", source_sha256=source_sha,
                         source=str(output), report_root=config["report_root"], demonstration_episode=demo_episode,
                         collection_code=collection["code"], collection_bundle_sha256=manifest["bundle_sha256"]))
        validate_ledger(ledger)
        ledger["fingerprint"] = fingerprint(ledger)
        temporary = ledger_path.with_name(ledger_path.name + ".pending")
        ensure(not temporary.exists() and not temporary.is_symlink(), "Inspect an interrupted ledger transaction before retrying")
        if previous_fingerprint is None:
            journal.mkdir(mode=0o700, exist_ok=False)
        # An interrupted commit fails closed. An older valid ledger cannot
        # erase these exclusive admission anchors or refund unknown calls.
        write_private(journal / f"{len(rows):04d}.json", json.dumps(dict(
            sequence=len(rows), ledger_fingerprint=ledger["fingerprint"],
            previous_fingerprint=previous_fingerprint, source_sha256=source_sha), indent=2) + "\n")
        write_private(temporary, json.dumps(ledger, indent=2) + "\n")
        os.replace(temporary, ledger_path)
        # A failure from here keeps the reservation and blocks automatic reuse.
        output.parent.mkdir(parents=True, exist_ok=True)
        write_private(output, source)
        receipt = dict(kind="codex_next_collection_job_source_v1", source=str(output), sha256=source_sha,
                       bytes=len(source.encode()), task=task, instance_id=instance_id, attempt_id=attempt_id,
                       code=manifest["code"], code_bundle_sha256=manifest["bundle_sha256"], code_installation="installer" in config,
                       collection_run_root=collection["run_root"], report_root=config["report_root"],
                       demonstration_episode=demo_episode, ledger=str(ledger_path), ledger_sha256=sha(ledger_path),
                       ledger_journal=str(journal),
                       budget_admission=admission, predecessor=predecessor, max_seconds=9000,
                       recommended_cloud_max_seconds=9600, python_binary=PYTHON,
                       bootstrap_report=True, actual_job_submitted=False)
        write_private(receipt_path, json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ledger", default=LEDGER_PATH, type=Path)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--instance-id", required=True, type=int)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--demo-episode", required=True, type=int)
    for flag in ("prior-job", "prior-submission", "prior-source", "prior-audit"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, task=args.task, instance_id=args.instance_id, attempt_id=args.attempt_id,
                           demo_episode=args.demo_episode, ledger_path=args.ledger, prior_job_path=args.prior_job,
                           prior_submission_path=args.prior_submission, prior_source_path=args.prior_source,
                           prior_audit_path=args.prior_audit), indent=2))


if __name__ == "__main__":
    main()
