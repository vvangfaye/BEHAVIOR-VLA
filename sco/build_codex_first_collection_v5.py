#!/usr/bin/env python3
"""Build, never submit, the isolated second radio311 collection attempt.

The caller supplies a fresh terminal observation and a private bundle URL file.
Building is offline; only the generated worker downloads its verified code-v7
bundle. Existing preparation, demonstrations, and the first attempt stay intact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib.parse import urlsplit

from build_codex_first_collection_v4 import AFS, EPISODES, OUT, PYTHON, WORKER as V4_WORKER
from build_codex_collection_bundle import installer_source

OLD_NAME = "behavior-codex-radio311-pilot-v4-1011-045818"
OLD_UID = "740f1d30fe2740f4b36bcf621e53cc3e"
OLD_BUNDLE_SHA256 = "60b685c67308179a61b607497af15c7a0af8c825127c45d583e1e90804277280"
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "TERMINATED", "CANCELLED"}


RECOVERY = r'''
class WrapperDeadline(BaseException):pass

def recovery_identity(report):
 ensure(report.get('state')=='failed' and report.get('training_eligible') is False,'Previous attempt was not a failed collection')
 ensure((report.get('task'),report.get('instance_id'),report.get('attempt_id'))==('turning_on_radio',311,'pilot-001'),'Previous attempt identity differs')
 observed=report.get('collector_progress',{});usage=report.get('teacher_usage',{})
 ensure(type(observed.get('executed_steps_observed')) is int and observed['executed_steps_observed']==0,'Previous attempt executed actions')
 keys=('request_directories','validated_decisions','failed_decisions','model_calls_with_usage','requests_without_usage')
 ensure(all(type(usage.get(key)) is int and usage[key]==0 for key in keys),'Previous attempt used a teacher')
 return {'state':'failed','training_eligible':False,'task':'turning_on_radio','instance_id':311,
         'attempt_id':'pilot-001','executed_steps_observed':0,'teacher_usage':{key:usage[key] for key in keys}}

def verify_recovery(config):
 recovery=config['recovery_from'];job=recovery['job'];base=Path(config['pilot_root'])
 ensure(job['name']==config['old_name'] and job['uid']==config['old_uid'] and job['state'] in config['terminal_states'],'Prior exact job is not terminal')
 old_root=base/'collection/radio-311-pilot001';old_report=base/'first-radio311-gpu-v4/report.json'
 ensure(recovery_identity(load(old_report))==recovery['report_identity'],'Previous failure report differs')
 actions=old_root/'attempt/action_sources.jsonl'
 ensure(not actions.exists() or (actions.is_file() and actions.stat().st_size==0),'Previous attempt action evidence is nonempty')
 ensure(not any((old_root/'attempt/teacher').glob('step-*')),'Previous attempt contains teacher evidence')
 ensure(not (old_root/'_COMPLETE').exists() and not (old_root/'attempt/completed.json').exists()
        and not (old_root/'attempt/trajectory.hdf5').exists(),'Previous attempt contains trajectory or completion evidence')
 return {**recovery,'afs_report_sha256':sha(old_report),'old_run_root':str(old_root)}

def preparation_config(config):
 old=config['preparation_manifest']
 ensure(old['version']=='v6' and old['code']==str(Path(config['pilot_root'])/'code-v6')
        and old['bundle_sha256']==config['old_bundle_sha256'] and len(old['source_files_sha256'])==29,'Wrong original preparation code')
 collection={**config['collection'],'code':old['code'],'source_files_sha256':old['source_files_sha256']}
 return {**config,'manifest':old,'collection':collection}

def gates(config):
 previous=preparation_gates(preparation_config(config))
 manifest=config['manifest'];collection=config['collection'];old=config['preparation_manifest']
 code=Path(collection['code']);ready_path=code/'_CODEX_PILOT_CODE_READY';ready=load(ready_path)
 ensure(manifest['version']=='v7' and str(code)==manifest['code']==str(Path(config['pilot_root'])/'code-v7'),'Wrong new collection code')
 ensure(ready.get('state')=='code_verified' and ready.get('bundle_sha256')==manifest['bundle_sha256'],'New code readiness differs')
 ensure(ready.get('source_files_sha256')==manifest['source_files_sha256']==collection['source_files_sha256'],'New frozen code inventory differs')
 ensure(manifest['base_code']==old['base_code']==collection['base_code'] and manifest['base_source_manifest_sha256']==old['base_source_manifest_sha256'],'New code changed immutable base')
 ensure(sha(Path(collection['base_code'])/'_PUBLIC_TEST_SOURCE_READY')==manifest['base_source_manifest_sha256'],'New code base receipt differs')
 teacher='scripts/codex_collection_teacher.py'
 ensure(manifest['source_files_sha256'][teacher]==old['source_files_sha256'][teacher],'New code changed the previously audited teacher')
 for name,digest in manifest['source_files_sha256'].items():ensure(sha(code/name)==digest,'New code source hash differs')
 return {**previous,'preparation_code_ready_sha256':previous['code_ready_sha256'],
         'code_ready_sha256':sha(ready_path),'collection_code':str(code),'collection_bundle_sha256':manifest['bundle_sha256']}
'''


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Original wrapper structure changed; review before rebuilding")
    return source.replace(before, after)


def worker_source():
    source = _replace_once(V4_WORKER, "def gates(config):", "def preparation_gates(config):")
    source = _replace_once(source, "def run(config):", RECOVERY + "\ndef run(config):")
    source = _replace_once(source, "'kind':'codex_first_collection_gpu_v4'", "'kind':'codex_first_collection_gpu_v5'")
    source = _replace_once(source, "'attempt_id':'pilot-001','limits'", "'attempt_id':'pilot-002','limits'")
    source = _replace_once(source, "def stop(signum,frame):raise TimeoutError('Finite collection wrapper deadline reached')",
                           "def stop(signum,frame):raise WrapperDeadline('Finite collection wrapper deadline reached')")
    source = _replace_once(source, "  report['gates']=gates(config);publish()", """  report['stage']='prior_attempt';report['recovery_from']=verify_recovery(config);publish()
  report['preparation_gates']=preparation_gates(preparation_config(config));publish()
  report['stage']='code_install';publish()
  try:
   exec(compile(config['installer'],'code_v7_installer','exec'),{'__name__':'__cluster_installer__'})
  except BaseException as error:
   write(root/'code_install_report.json',{'state':'failed',**safe_error(error)})
   raise
  finally:signal.alarm(max(1,int(deadline-time.monotonic())))
  report['stage']='gates';report['gates']=gates(config);publish()""")
    return source


WORKER = worker_source()


def checked_prior(job, submission, report, *, now=None):
    now = time.time() if now is None else now
    observed = job.get("observed_at")
    if (job.get("name") != OLD_NAME or job.get("uid") != OLD_UID or job.get("state") not in TERMINAL
            or any(submission.get(key) != job.get(key) for key in ("name", "uid"))
            or type(observed) not in (int, float) or not 0 <= now - observed <= 900):
        raise ValueError("Need a fresh terminal observation of the exact first collection job")
    namespace = {}
    exec("def ensure(condition,message):\n if not condition:raise ValueError(message)\n" + RECOVERY.split("def verify_recovery", 1)[0], namespace)
    identity = namespace["recovery_identity"](report)
    return {key: job[key] for key in ("name", "uid", "state", "observed_at")}, identity


def private_bundle_url(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("Bundle URL file must be an ordinary private 0600 file")
    url = path.read_text().strip()
    parsed = urlsplit(url)
    if (not 1 <= len(url) <= 8192 or parsed.scheme != "https" or not parsed.hostname
            or parsed.username or parsed.password or "\n" in url or "\r" in url):
        raise ValueError("Bundle URL must be one HTTPS URL")
    return url


def private_installer(manifest, url, report_root):
    source = installer_source(manifest, url)
    start = source.index("def publish(value):")
    end = source.index("base=Path(settings", start)
    publish = ("def publish(value):\n"
               " path=Path(" + repr(str(Path(report_root) / "code_install_report.json")) + ")\n"
               " temporary=path.with_suffix('.tmp')\n"
               " temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)\n"
               "def open_bundle(url):\n"
               " import socket,urllib.error\n"
               " for attempt in range(4):\n"
               "  try:return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url,timeout=30)\n"
               "  except urllib.error.URLError as error:\n"
               "   if not isinstance(error.reason,socket.gaierror) or error.reason.errno!=socket.EAI_AGAIN or attempt==3:raise\n"
               "   time.sleep(0.5)\n")
    source = source[:start] + publish + source[end:]
    source = _replace_once(source,
                           "urllib.request.build_opener(urllib.request.ProxyHandler({})).open(settings['url'],timeout=30)",
                           "open_bundle(settings['url'])")
    compile(source, "<private-code-v7-installer>", "exec")
    return source


def build(output, *, bundle_url_file, prior_job_path,
          manifest_path=OUT / "collection_code_v7.tar.gz.manifest.json",
          preparation_manifest_path=OUT / "collection_code_v6.tar.gz.manifest.json",
          collection_path=OUT / "initial_collection_payload_prepared.json",
          encryption_path=OUT / "auth_encryption_receipt.json",
          previous_submission_path=OUT / "first_collection_submission_v4.json",
          previous_source_path=OUT / "first_collection_job_v4.py",
          previous_report_path=OUT / "first_collection_submission_v4_report.json"):
    load = lambda path: json.loads(Path(path).read_text())
    manifest, old = load(manifest_path), load(preparation_manifest_path)
    if (old.get("version") != "v6" or old.get("code") != AFS + "/code-v6"
            or old.get("bundle_sha256") != OLD_BUNDLE_SHA256 or len(old.get("source_files_sha256", {})) != 29):
        raise ValueError("Only the verified code-v6 preparation bundle may be reused")
    if manifest.get("version") != "v7" or manifest.get("code") != AFS + "/code-v7":
        raise ValueError("The second GPU attempt requires a fresh code-v7 bundle")
    for value in (old, manifest):
        if hashlib.sha256(Path(value["bundle"]).read_bytes()).hexdigest() != value["bundle_sha256"]:
            raise ValueError("Code bundle differs from its frozen manifest")
    teacher = "scripts/codex_collection_teacher.py"
    if (manifest["base_code"] != old["base_code"] or manifest["base_source_manifest_sha256"] != old["base_source_manifest_sha256"]
            or manifest["source_files_sha256"][teacher] != old["source_files_sha256"][teacher]):
        raise ValueError("Previously verified base or teacher auditor changed")
    previous, previous_report = load(previous_submission_path), load(previous_report_path)
    job, report_identity = checked_prior(load(prior_job_path), previous, previous_report)
    source_digest = hashlib.sha256(Path(previous_source_path).read_bytes()).hexdigest()
    if previous.get("source_sha256") != source_digest:
        raise ValueError("Previously submitted source differs from its receipt")
    collection = copy.deepcopy(load(collection_path));encryption = load(encryption_path)
    if Path(encryption["selected_auth_file"]).name != "auth_weizhi.json":
        raise ValueError("Unexpected selected authentication account")
    expected = dict(task="turning_on_radio", instance_id=311, attempt_id="pilot-001", teacher_backend="codex_local",
                    max_steps=4500, max_decisions=32, max_seconds=7200, wall_timeout_seconds=8400, teacher_interval=256)
    if any(collection.get(key) != value for key, value in expected.items()):
        raise ValueError("Original collection scope or budget differs")
    collection.update(attempt_id="pilot-002", execution_name="codex-radio311-pilot002-20261011",
                      run_root=AFS + "/collection/radio-311-pilot002", code=manifest["code"],
                      source_files_sha256=manifest["source_files_sha256"],
                      robot_config=manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
                      demonstration_manifest=AFS + "/original-reader-v4/demonstrations/28/manifest.json")
    from run_codex_collection import validate_payload
    collection = validate_payload(collection)
    report_root = AFS + "/first-radio311-gpu-v5"
    recovery = dict(job=job, source_sha256=source_digest,
                    downloaded_report_sha256=hashlib.sha256(Path(previous_report_path).read_bytes()).hexdigest(),
                    report_identity=report_identity)
    config = dict(pilot_root=AFS, report_root=report_root, python=PYTHON, episodes=EPISODES,
                  manifest=manifest, preparation_manifest=old, collection=collection,
                  old_name=OLD_NAME, old_uid=OLD_UID, terminal_states=sorted(TERMINAL),
                  old_bundle_sha256=OLD_BUNDLE_SHA256, recovery_from=recovery,
                  ciphertext_sha256=encryption["ciphertext_sha256"],
                  installer=private_installer(manifest, private_bundle_url(bundle_url_file), report_root))
    source = WORKER + "\nBOOTSTRAP_REPORT = run(" + repr(config) + ")\n"
    compile(source, "<first-collection-v5>", "exec")
    output = Path(output);receipt_path = output.with_suffix(".receipt.json")
    if output.exists() or output.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Source and receipt outputs must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write(source)
    receipt = dict(kind="codex_first_collection_job_source_v5", source=str(output.resolve()),
                   sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
                   code_bundle_sha256=manifest["bundle_sha256"], code=manifest["code"],
                   preparation_bundle_sha256=old["bundle_sha256"], recovery_from=recovery,
                   report_root=report_root, collection_run_root=collection["run_root"], attempt_id="pilot-002",
                   original_data=AFS + "/original-reader-v4", dataset_root=AFS + "/original-v3",
                   max_seconds=9000, installer_max_seconds=600, recommended_cloud_max_seconds=9600,
                   python_binary=PYTHON, bootstrap_report=True, actual_job_submitted=False)
    with os.fdopen(os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(receipt, stream, indent=2);stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "first_collection_job_v5.py")
    parser.add_argument("--manifest", type=Path, default=OUT / "collection_code_v7.tar.gz.manifest.json")
    parser.add_argument("--bundle-url-file", required=True, type=Path)
    parser.add_argument("--prior-job", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, manifest_path=args.manifest, bundle_url_file=args.bundle_url_file,
                           prior_job_path=args.prior_job), indent=2))


if __name__ == "__main__":
    main()
