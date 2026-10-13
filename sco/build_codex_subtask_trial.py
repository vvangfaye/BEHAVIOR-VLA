"""Freeze an independent overlay and one bounded Codex subtask worker, offline."""
from __future__ import annotations

import argparse
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
AFS = "/mnt/afs/260010165/behavior_1k/codex_subtask_freeze_20261012"
BASE = "/mnt/afs/260010165/behavior_1k/code_snapshots/public-test-1688977-aoss-20261010"
RUNTIME = "/mnt/afs/260010165/behavior_1k/runtime_20261007_1306"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_reference_readiness(demo_path, sample):
    """Verify original readiness, including one explicit derived reference layer."""
    import hashlib
    import json
    from pathlib import Path
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def load(path):
        data = json.loads(Path(path).read_text())
        content = {key: value for key, value in data.items() if key != 'fingerprint'}
        actual = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if data.get('kind') != 'codex_collection_demonstrations_v1' or data.get('fingerprint') != actual:
            raise ValueError('Demonstration manifest identity differs')
        return data
    demo_path = Path(demo_path)
    if not demo_path.is_absolute() or '..' in demo_path.parts:
        raise ValueError('Demonstration path must be absolute')
    demo = load(demo_path)
    original_path, original = demo_path, demo
    parent = demo.get('derived_from_demonstration')
    if parent is not None:
        if demo.get('reference_frame_policy') != 'recorded_full_close_command_v1':
            raise ValueError('Unknown derived demonstration protocol')
        original_path = Path(parent['manifest'])
        if not original_path.is_absolute() or '..' in original_path.parts:
            raise ValueError('Original parent path must be absolute')
        original = load(original_path)
        if (original.get('derived_from_demonstration') is not None
                or parent.get('manifest_sha256') != digest(original_path)
                or parent.get('fingerprint') != original['fingerprint']
                or original['metadata'] != demo['metadata']):
            raise ValueError('Derived reference source differs from its original parent')
    elif demo.get('reference_frame_policy') is not None:
        raise ValueError('Derived phase reference needs an original readiness parent')
    original_root = original_path.parents[2]
    metadata = demo['metadata']
    if (metadata.get('source_kind') != 'original_train'
            or metadata.get('source_dataset') != str(original_root)
            or metadata.get('episode_index') != sample['episode_index']
            or metadata.get('task') != sample['task']
            or metadata.get('instance_id') != sample['task_instance_id']):
        raise ValueError('Original demonstration identity differs from the sample')
    ready_path = original_root/'_ORIGINAL_DATA_READY.json'
    ready = json.loads(ready_path.read_text())
    if (ready.get('state') != 'complete' or ready.get('original_reader_verified') is not True
            or ready.get('raw_success_verified') is not True
            or ready.get('selected_episodes') != [sample['episode_index']]
            or (original_root/'_FAILED').exists()):
        raise ValueError('Original demonstration staging is incomplete')
    references = ready.get('demonstrations', [])
    if (len(references) != 1 or references[0].get('task') != sample['task']
            or references[0].get('manifest') != str(original_path)
            or references[0].get('fingerprint') != original['fingerprint']
            or original.get('required_subtask') != sample['subtask']
            or demo.get('required_subtask') != sample['subtask']):
        raise ValueError('Original readiness binding or required target differs')
    for example in demo['examples']:
        image = demo_path.parent/example['image']
        if image.parent.resolve() != demo_path.parent.resolve() or digest(image) != example['image_sha256']:
            raise ValueError('Reference image bytes differ')
    return dict(episode=sample['episode_index'], task=sample['task'], fingerprint=demo['fingerprint'],
                ready_sha256=digest(ready_path), manifest_sha256=digest(demo_path),
                original_manifest=str(original_path), original_fingerprint=original['fingerprint'],
                derived_reference=parent is not None)


def build(output, *, revision="v1", demonstration_manifest=None, trial_name="trial001",
          plan_file=None, codex_home=None, experiment_root=AFS, quota_type="spot"):
    from run_codex_subtask_collection import REQUIRED_SOURCES, validate_payload
    from codex_subtask_collection import load_plan
    output = Path(output)
    if re.fullmatch(r"v[1-9][0-9]*", revision) is None:
        raise ValueError("Invalid isolated overlay revision")
    if re.fullmatch(r"trial[0-9]{3}", trial_name) is None:
        raise ValueError("Invalid independent trial name")
    if quota_type not in ("spot", "reserved"):
        raise ValueError("Invalid explicit trial quota")
    output.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / "eval_results/codex_subtask_freeze_20261012"
    paths = json.loads((evidence / "collection_dependencies_candidate.json").read_text())
    paths = sorted(set(paths) | REQUIRED_SOURCES)
    blobs = {}
    for name in paths:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid overlay path")
        raw = (ROOT / path).read_bytes()
        if path.suffix == ".py":
            compile(raw, name, "exec")
        blobs["code/" + name] = raw
    sources = {name: sha(blobs["code/" + name]) for name in paths}
    plan_path = Path(plan_file) if plan_file else evidence / "attempt001_plan.json"
    plan = load_plan(plan_path)
    plan_name = plan_path.name
    blobs[plan_name] = plan_path.read_bytes()
    blobs["checkpoint_upload_plan.json"] = (ROOT / "eval_results/sco_test2000_1688977_20261010/aoss_checkpoint_upload_plan.json").read_bytes()
    if sha(blobs["checkpoint_upload_plan.json"]) != plan["checkpoint_identity"]["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint inventory changed")
    if not experiment_root.startswith("/mnt/afs/260010165/behavior_1k/") or ".." in Path(experiment_root).parts:
        raise ValueError("Experiment root must stay in the owned project directory")
    code_root = experiment_root + "/code-" + trial_name + "-" + revision
    bundle_root = experiment_root + "/bundle-" + trial_name + "-" + revision
    payload = dict(code=code_root, base_code=BASE, runtime=RUNTIME,
        checkpoint="/mnt/afs/260010165/behavior_1k/checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010",
        run_root=experiment_root + "/" + trial_name + "-" + revision, plan_path=bundle_root + "/" + plan_name,
        plan_sha256=sha(blobs[plan_name]),
        checkpoint_upload_plan=bundle_root + "/checkpoint_upload_plan.json",
        raw_data_root=plan["raw_data_root"],
        demonstration_manifest=demonstration_manifest or AFS + "/original-10020/demonstrations/10020/manifest.json",
        robot_config=code_root + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
        codex_binary="/usr/local/bin/codex", codex_home=codex_home or "/mnt/afs/260010165/behavior_1k/codex_pilot_20261011/private-codex-home-v1",
        mihomo_binary="/usr/local/bin/mihomo", mihomo_config_dir="/mnt/afs/260010165/container-env/config/mihomo",
        execution_name="codex-subtask-" + plan["sample"]["task"] + "-" + trial_name[5:] + "-" + revision, teacher_backend="codex_local", port=8940,
        wall_timeout_seconds=9000, teacher_timeout=120, policy_startup_timeout=900,
        source_files_sha256=sources)
    validate_payload(payload)
    blobs["payload.json"] = (json.dumps(payload, indent=2) + "\n").encode()
    inventory = {name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in blobs.items()}
    blobs["manifest.json"] = (json.dumps(inventory, indent=2) + "\n").encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, raw in blobs.items():
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode, entry.mtime = len(raw), 0o600, 0
            archive.addfile(entry, io.BytesIO(raw))
    raw = stream.getvalue()
    (output / "bundle.tar.gz").write_bytes(raw)
    record = dict(kind="codex_subtask_trial_bundle_v1", bytes=len(raw), sha256=sha(raw),
        bundle_root=bundle_root, payload=payload,
        plan_fingerprint=plan["fingerprint"], source_files_sha256=sources,
        limits=dict(gpus=1, quota=quota_type, max_job_seconds=10200, max_attempts=1))
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


DRIVER = '''import hashlib,io,json,os,signal,shutil,subprocess,sys,tarfile,time,urllib.request
from pathlib import Path
REFERENCE_READINESS_VALIDATOR
config=CONFIG
root=Path(config['payload']['run_root'])
bundle=Path(config['bundle_root']);code=Path(config['payload']['code']);base=Path(config['payload']['base_code'])
deadline=time.monotonic()+9900
report={'kind':'codex_subtask_trial_gpu_v1','state':'preparing','plan_fingerprint':config['plan_fingerprint'],'run_root':str(root),'official_benchmark':False,'training_eligible':False}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def publish():
 raw=json.dumps(report,allow_nan=False).encode()
 try:
  with opener.open(urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'],data=raw,method='PUT'),timeout=20):pass
 except Exception as error:print(json.dumps({'report_upload_error_type':type(error).__name__}),flush=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
child=None
def interrupted(signum,frame):raise KeyboardInterrupt('Owned outer worker interrupted')
for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,interrupted)
try:
 publish()
 with opener.open(BUNDLE_URL,timeout=60) as response:raw=response.read(config['bytes']+1)
 if len(raw)!=config['bytes'] or hashlib.sha256(raw).hexdigest()!=config['sha256']:raise ValueError('Bundle identity differs')
 bundle.mkdir(parents=True,exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  for entry in archive.getmembers():
   p=Path(entry.name)
   if p.is_absolute() or '..' in p.parts or not entry.isfile():raise ValueError('Unsafe bundle member')
   target=bundle/p;target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as stream:stream.write(archive.extractfile(entry).read())
   target.chmod(0o600)
 for name,item in json.loads((bundle/'manifest.json').read_text()).items():
  path=bundle/name
  if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('Extracted bundle differs')
 if json.loads((bundle/'payload.json').read_text())!=config['payload']:raise ValueError('Frozen payload differs')
 demo_path=Path(config['payload']['demonstration_manifest'])
 selected_plan=json.loads((bundle/Path(config['payload']['plan_path']).name).read_text())
 selected_sample=selected_plan['sample']
 report['demonstration_readiness']=verify_reference_readiness(demo_path,selected_sample)
 shutil.copytree(base,code,symlinks=True,copy_function=lambda source,dest:os.symlink(str(Path(source).resolve()),dest))
 for name,expected in config['source_files_sha256'].items():
  source=bundle/'code'/name;target=code/name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.parent.resolve().is_relative_to(code.resolve()):raise ValueError('Overlay directory escapes owned code')
  if target.is_symlink():target.unlink()
  elif target.exists():raise ValueError('Unexpected nonlinked overlay destination')
  shutil.copyfile(source,target)
  if digest(target)!=expected:raise ValueError('Installed overlay differs')
 (code/'_CODEX_SUBTASK_OVERLAY_READY.json').write_text(json.dumps({'bundle_sha256':config['sha256'],'source_files_sha256':config['source_files_sha256']}))
 logdir=bundle/'worker-logs';logdir.mkdir()
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=str(code/'sco')+':'+str(code/'scripts'))
 python=Path(config['payload']['runtime'])/'envs/b1k-eval/bin/python'
 with (logdir/'worker.log').open('x',buffering=1) as log:
  child=subprocess.Popen([str(python),str(code/'sco/run_codex_subtask_collection.py'),str(bundle/'payload.json')],env=env,cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while True:
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:rc=None
   if (root/'status.json').is_file():
    report['worker']=json.loads((root/'status.json').read_text())
   report.update(state='running' if rc is None else 'worker_exited',worker_exit_code=rc,updated_at=time.time())
   publish()
   if rc is not None:break
   if time.monotonic()>deadline:raise TimeoutError('Bounded outer worker deadline')
 if rc!=0:raise RuntimeError('Subtask worker failed; inspect owned worker log')
 complete=json.loads((root/'_COMPLETE').read_text())
 if complete['state']!='complete' or complete['plan_fingerprint']!=config['plan_fingerprint']:raise ValueError('Completion proof differs')
 report.update(state='complete',worker=complete,training_eligible=complete['training_eligible'])
except BaseException as error:
 report.update(state='failed',error_type=type(error).__name__,training_eligible=False)
finally:
 if child is not None and child.poll() is None:
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=100)
  except subprocess.TimeoutExpired:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=10)
 publish()
 BOOTSTRAP_REPORT=report
 if report['state']=='failed':raise RuntimeError('Owned trial failed; private diagnostic evidence retained')
'''
DRIVER = DRIVER.replace('REFERENCE_READINESS_VALIDATOR', inspect.getsource(verify_reference_readiness))


def render(manifest, url_file, output):
    config = json.loads(Path(manifest).read_text())
    url = Path(url_file).read_text().strip()
    if not url.startswith("https://"):
        raise ValueError("Expected private HTTPS bundle URL")
    source = re.sub(r"\b(CONFIG|BUNDLE_URL)\b", lambda match: repr(config if match[0] == "CONFIG" else url), DRIVER)
    compile(source, "codex_subtask_trial_driver", "exec")
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--revision", default="v1")
    build_parser.add_argument("--demonstration-manifest")
    build_parser.add_argument("--trial-name", default="trial001")
    build_parser.add_argument("--plan-file")
    build_parser.add_argument("--codex-home")
    build_parser.add_argument("--experiment-root", default=AFS)
    build_parser.add_argument("--quota-type", choices=("spot", "reserved"), default="spot")
    render_parser = sub.add_parser("render-driver")
    render_parser.add_argument("--manifest", required=True)
    render_parser.add_argument("--bundle-url-file", required=True)
    render_parser.add_argument("--output-driver", required=True)
    args = parser.parse_args()
    if args.action == "build":
        result = build(args.output_dir, revision=args.revision, demonstration_manifest=args.demonstration_manifest,
                       trial_name=args.trial_name, plan_file=args.plan_file, codex_home=args.codex_home,
                       experiment_root=args.experiment_root, quota_type=args.quota_type)
        print(json.dumps({k: result[k] for k in ("kind", "sha256", "bytes", "bundle_root", "plan_fingerprint", "limits")}))
    else:
        render(args.manifest, args.bundle_url_file, args.output_driver)
        print(json.dumps(dict(state="private_driver_written", path=args.output_driver)))


if __name__ == "__main__":
    main()
