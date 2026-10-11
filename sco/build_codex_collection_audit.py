"""Build a finite read-only audit of one admitted subsequent collection attempt.

This offline builder reuses the reviewed pilot-002 audit implementation. It does
not submit jobs, contact services, read authentication, or invoke models. Both
successful and failed terminal attempts can be audited; unknown use stays unknown.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import build_codex_next_collection as collection_builder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/codex_pilot_20261011"
KIND = "codex_collection_readonly_audit_v1"
spec = importlib.util.spec_from_file_location(
    "codex_pilot002_audit_shared", OUT / "build_pilot002_attempt_audit_v5.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Reviewed audit structure changed; review before rebuilding")
    return source.replace(before, after)


def audit_source():
    """Bind identity, budget and collection mode to reviewed worker evidence."""
    source = pilot.AUDIT
    changes = [
        ("def audit_teachers(attempt):", "def audit_teachers(attempt,expected_identity,collection_mode='teacher_conditioned'):"),
        ("if (request.get('task')!='turning_on_radio' or request.get('instance_id')!=311\n"
         "       or request.get('attempt_id')!='pilot-002'):",
         "if any(request.get(key)!=expected_identity[key] for key in ('task','instance_id','attempt_id')):"),
        ("   location=decision_location(path,request)",
         "   ensure(request.get('collection_mode','teacher_conditioned')==collection_mode,'Teacher collection mode differs')\n"
         "   location=decision_location(path,request)"),
        ("answer=validate_decision(load(path/'response.json'),identity)",
         "answer=(validate_decision(load(path/'response.json'),identity,collection_mode=collection_mode) if collection_mode=='native_with_corrections' else validate_decision(load(path/'response.json'),identity))"),
        ("if validate_decision(wire,identity)!=answer:",
         "if (validate_decision(wire,identity,collection_mode=collection_mode) if collection_mode=='native_with_corrections' else validate_decision(wire,identity))!=answer:"),
        ("def trajectory_summary(path):", "def trajectory_summary(path,expected_identity,collection_mode='teacher_conditioned'):"),
        ("ensure(identity['task']=='turning_on_radio' and identity['instance_id']==311 and identity['attempt_id']=='pilot-002',",
         "ensure(all(identity.get(key)==expected_identity[key] for key in ('task','instance_id','attempt_id')),"),
        ("  return {'present':True,'valid_complete_trajectory':True,'steps':info['steps'],",
         "  ensure(identity.get('collection_mode','teacher_conditioned')==collection_mode and identity.get('subtask_source')==('mixed_native_and_teacher' if collection_mode=='native_with_corrections' else 'provided'),'Trajectory collection mode differs')\n"
         "  return {'present':True,'valid_complete_trajectory':True,'steps':info['steps'],'collection_mode':collection_mode,'subtask_source':identity['subtask_source'],"),
        (" root=Path(config['run_root']);output=Path(config['audit_root']);attempt=root/'attempt'",
         " root=Path(config['run_root']);output=Path(config['audit_root']);attempt=root/'attempt'\n"
         " mode=audit_collection_mode(config.get('collection_mode','teacher_conditioned'))"),
        ("'kind':'codex_pilot002_readonly_audit_v5'", "'kind':'" + KIND + "'"),
        ("'max_seconds':600,'issues':[]", "'max_seconds':600,'identity':config['identity'],'max_steps':config['max_steps'],'max_decisions':32,'collection_mode':mode,'collection_code':config['code'],'collection_bundle_sha256':config.get('collection_bundle_sha256'),'collection_source_files_sha256':config['source_files_sha256'],'issues':[]"),
        ("before=snapshot(root);publish()", "before=snapshot(root);report.update(capture_last_teacher_pngs(attempt));publish()"),
        ("action_summary(attempt/'action_sources.jsonl')", "action_summary(attempt/'action_sources.jsonl',config['max_steps'],mode)"),
        ("audit_teachers(attempt);publish()", "audit_teachers(attempt,config['identity'],mode);publish()"),
        ("trajectory_summary(attempt/'trajectory.hdf5')", "trajectory_summary(attempt/'trajectory.hdf5',config['identity'],mode)"),
        ("execution_consistency(report['teachers'],report['actions'],report['trajectory'],report['correction_rejections'])",
         "execution_consistency(report['teachers'],report['actions'],report['trajectory'],report['correction_rejections'],mode,config.get('teacher_interval',256))"),
    ]
    for before, after in changes:
        source = replace_once(source, before, after)
    return source


AUDIT = audit_source()
IMPORTS = "import hashlib,json,os,re,signal,sys,time\nfrom collections import Counter\nfrom pathlib import Path\n"


def checked_inputs(job_receipt, source_sha256, submission_path, source_path, *, now=None):
    load, ensure = collection_builder.load, collection_builder.ensure
    job, submission = load(job_receipt), load(submission_path)
    now = time.time() if now is None else now
    observed = job.get("observed_at")
    ensure(type(observed) in (int, float) and 0 <= now - observed <= 900
           and job.get("state") in pilot.TERMINAL
           and all(isinstance(job.get(key), str) and job[key] and job[key] == submission.get(key)
                   for key in ("name", "uid")),
           "A fresh terminal observation of the exact submitted collection job is required")
    ensure(isinstance(source_sha256, str) and collection_builder.HEX.fullmatch(source_sha256)
           and submission.get("source_sha256") == source_sha256
           and collection_builder.sha(source_path) == source_sha256,
           "Original submitted collection source SHA256 differs")
    frozen = collection_builder.frozen_config(source_path)
    code_contract = None
    version = 'v7'
    is_recovery = False
    if frozen.get('collection', {}).get('code') == pilot.AFS + '/code-v8':
        version = 'v8'
        if frozen.get('builder_kind') == 'codex_next_collection_v8':
            import build_codex_next_collection_v8 as code_contract
        elif 'builder_kind' not in frozen:
            import build_codex_radio_recovery_v8 as code_contract
            is_recovery = True
        else:
            raise ValueError('Unknown code-v8 collection worker contract')
    elif frozen.get('collection', {}).get('code') == pilot.AFS + '/code-v9':
        ensure(frozen.get('builder_kind') == 'codex_native_collection_v9', 'Unknown code-v9 collection worker contract')
        import build_codex_native_collection_v9 as code_contract
        version = 'v9'
    if code_contract:code_contract.validate_frozen_config(frozen)
    # A config copied into another program is not a source emitted by this builder.
    tree = ast.parse(Path(source_path).read_text())
    ensure(ast.dump(ast.Module(body=tree.body[:-1], type_ignores=[]))
           == ast.dump(ast.parse(code_contract.WORKER if code_contract else collection_builder.WORKER)),
           "Submitted source is not a reviewed bounded collection worker")
    collection, manifest = frozen["collection"], frozen["manifest"]
    task, instance, attempt = (collection.get(key) for key in ("task", "instance_id", "attempt_id"))
    ensure(task in collection_builder.TASKS and type(instance) is int and 311 <= instance <= 320
           and isinstance(attempt, str) and collection_builder.NAME.fullmatch(attempt),
           "Submitted collection identity is outside this experiment")
    slug = ("radio" if task == "turning_on_radio" else "shoes") + f"-{instance}-{attempt}"
    mode = collection.get('collection_mode', 'teacher_conditioned')
    ensure(mode == ('native_with_corrections' if version == 'v9' else 'teacher_conditioned'),
           'Collection mode is not supported by its frozen worker version')
    teacher_interval = code_contract.TEACHER_INTERVALS[task] if code_contract and not is_recovery else 256
    ensure(collection.get("max_steps") == collection_builder.TASKS[task][0]
           and type(collection.get("max_steps")) is int
           and collection.get("max_decisions") == 32 and collection.get("teacher_interval") == teacher_interval
           and collection.get("max_seconds") == 7200 and collection.get("wall_timeout_seconds") == 8400,
           "Submitted task step or teacher budget differs")
    prefix = '/native-code9-' if version == 'v9' else '/next-code8-' if version == 'v8' else '/next-'
    expected_report = code_contract.REPORT_ROOT if is_recovery else pilot.AFS + prefix + slug
    ensure(collection.get("run_root") == (code_contract.RUN_ROOT if is_recovery else pilot.AFS + "/collection/" + slug)
           and frozen.get("report_root") == expected_report
           and collection.get("code") == pilot.AFS + "/code-" + version
           and frozen.get("python") == pilot.PYTHON,
           "Submitted collection code or evidence path differs")
    inventory = collection.get("source_files_sha256", {})
    ensure(manifest.get("version") == version and manifest.get("code") == collection["code"]
           and manifest.get("bundle_sha256") == (frozen['expected_bundle_sha256'] if code_contract else collection_builder.CODE_SHA)
           and manifest.get("source_files_sha256") == inventory and len(inventory) == 29
           and {"scripts/codex_collection_teacher.py", "scripts/codex_collection_trajectory.py"} <= inventory.keys()
           and all(not Path(name).is_absolute() and ".." not in Path(name).parts
                   and collection_builder.HEX.fullmatch(str(digest)) for name, digest in inventory.items()),
           "Submitted frozen collection code inventory differs")
    admission = frozen.get("admission", {})
    sequence = admission.get("sequence")
    ensure(type(sequence) is int and 3 <= sequence <= 20
           and admission.get("reserved_decisions") == 32
           and admission.get("total_reserved_decisions") == sequence * 32
           and admission.get("max_teacher_decisions_total") == 640
           and admission.get("reservations_released") is False,
           "Submitted attempt lacks its bounded reservation")
    return dict(job={key: job[key] for key in ("name", "uid", "state", "observed_at")},
                source_sha256=source_sha256, run_root=collection["run_root"],
                wrapper_root=frozen["report_root"], code=collection["code"], source_files_sha256=inventory,
                collection_mode=mode, teacher_interval=teacher_interval, collection_bundle_sha256=manifest['bundle_sha256'],
                audit_root=pilot.AFS + "/audit-" + slug, max_steps=collection["max_steps"],
                identity=dict(task=task, instance_id=instance, attempt_id=attempt))


def build(job_receipt, source_sha256, *, submission_path, source_path, output=None, now=None):
    config = checked_inputs(job_receipt, source_sha256, submission_path, source_path, now=now)
    source = IMPORTS + pilot.shared_helpers() + "\n" + AUDIT + "\nBOOTSTRAP_REPORT = run(" + repr(config) + ")\n"
    compile(source, "bounded_collection_audit", "exec")
    target = Path(output).absolute() if output else OUT / (Path(config["audit_root"]).name + ".py")
    receipt = target.with_suffix(".receipt.json")
    inputs = {Path(path).resolve() for path in (job_receipt, submission_path, source_path)}
    collection_builder.ensure(target.resolve() != receipt.resolve()
        and not ({target.resolve(), receipt.resolve()} & inputs), "Audit output collides with an input")
    collection_builder.ensure(not any(path.exists() or path.is_symlink() for path in (target, receipt)),
                              "Audit source and receipt must be fresh")
    target.parent.mkdir(parents=True, exist_ok=True)
    collection_builder.write_private(target, source)
    result = dict(kind="codex_collection_audit_source_v1", source=str(target),
        sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
        source_job=config["job"], collection_source_sha256=source_sha256, identity=config["identity"],
        audit_root=config["audit_root"], max_steps=config["max_steps"], max_decisions=32,
        max_seconds=600, recommended_cloud_max_seconds=660, python_binary=pilot.PYTHON,
        actual_job_submitted=False, model_calls=0, network_calls=0)
    collection_builder.write_private(receipt, json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("job-receipt", "submission", "source"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.job_receipt, args.source_sha256, submission_path=args.submission,
                           source_path=args.source, output=args.output), indent=2))


if __name__ == "__main__":
    main()
