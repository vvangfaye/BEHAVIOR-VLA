"""Freeze one bounded ACP replay diagnostic against the existing collection."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r'''import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
config=CONFIG
root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=False)
code=Path(config['collection_payload']['code']);runtime=Path(config['collection_payload']['runtime'])
sys.path[:0]=[str(code/'sco'),str(code/'scripts')]
from run_codex_subtask_collection import verify_sources,environment
worker_source=SOURCE
if hashlib.sha256(worker_source.encode()).hexdigest()!=config['worker_sha256']:raise ValueError('Replay source differs')
(root/'replay_worker.py').write_text(worker_source)
(root/'config.json').write_text(json.dumps(config,indent=2))
report={'kind':'countertop_replay_job_v1','state':'preparing','diagnostic_only':True,'training_eligible':False,'model_calls':0,'output_root':str(root)}
def publish():
 import urllib.request
 raw=json.dumps(report,allow_nan=False).encode()
 (root/'driver_status.json').write_bytes(raw)
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'],data=raw,method='PUT'),timeout=20):pass
 except Exception as e:print(json.dumps({'report_upload_error':type(e).__name__}),flush=True)
child=None
def interrupted(signum,frame):raise KeyboardInterrupt('Owned diagnostic interrupted')
for n in (signal.SIGTERM,signal.SIGINT):signal.signal(n,interrupted)
try:
 publish()
 library,base_sha=verify_sources(config['collection_payload'])
 states=code/'BEHAVIOR-1K/OmniGibson/omnigibson/object_states'
 if hashlib.sha256((states/'on_top.py').read_bytes()).hexdigest()!=config['on_top_sha256']:raise ValueError('Original OnTop source changed')
 report['state_sources_sha256']={n:hashlib.sha256((states/n).read_bytes()).hexdigest() for n in ['on_top.py','touching.py','adjacency.py','aabb.py','tensorized_state.py','tensorized_relative_state.py']}
 report['base_source_sha256']=base_sha
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,timeout=30).strip().splitlines()
 if len(gpu)!=1:raise ValueError('Replay requires exactly one GPU')
 report['gpu']=gpu
 version=gpu[0].split(',')[1].strip()
 libraries=[p for parent in ['/usr/local/nvidia/lib64','/usr/lib/x86_64-linux-gnu','/usr/lib64','/lib/x86_64-linux-gnu','/lib64'] for p in Path(parent).glob('libEGL_nvidia.so.'+version)]
 if not libraries:raise ValueError('Matching graphics library unavailable')
 vk=root/'nvidia_icd.json';egl=root/'10_nvidia.json'
 vk.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(libraries[0]),'api_version':'1.3.242'}}))
 egl.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(libraries[0])}}))
 cache=Path('/var/tmp/behavior-1k-runtime')/root.name;cache.mkdir(parents=True,exist_ok=False)
 env=environment(config['collection_payload'],cache,vk,egl,library['library_path'])
 for key in list(env):
  if key.upper() in {'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY'} or key.startswith('CODEX_'):env.pop(key,None)
 env['PYTHONPATH']=':'.join(str(code/p) for p in ['scripts','sco','BEHAVIOR-1K/OmniGibson','BEHAVIOR-1K/bddl3','BEHAVIOR-1K/joylo','openpi/packages/openpi-client/src'])
 sim=cache/'sim'
 for n in ['appdata','cache','config','warp']:(sim/n).mkdir(parents=True)
 env.update(OMNIGIBSON_APPDATA_PATH=str(sim/'appdata'),OMNIGIBSON_GPU_ID='0',XDG_CACHE_HOME=str(sim/'cache'),XDG_CONFIG_HOME=str(sim/'config'),WARP_CACHE_PATH=str(sim/'warp'))
 report['state']='starting_replay';publish()
 with (root/'worker.log').open('x',buffering=1) as log:
  child=subprocess.Popen([str(runtime/'envs/b1k-eval/bin/python'),str(root/'replay_worker.py'),str(root/'config.json')],cwd=code/'BEHAVIOR-1K',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+4200
  while True:
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:rc=None
   if (root/'status.json').is_file():report['worker']=json.loads((root/'status.json').read_text())
   report.update(state='running' if rc is None else 'worker_exited',worker_exit_code=rc);publish()
   if rc is not None:break
   if time.monotonic()>deadline:raise TimeoutError('Finite replay timeout')
 if rc or (root/'failure.json').exists() or not (root/'summary.json').exists():raise RuntimeError('Replay failed; preserve diagnostic logs')
 report['summary']=json.loads((root/'summary.json').read_text())
 if report['summary']['executed_steps']!=960 or not report['summary']['byte_identical_actions']:raise ValueError('Replay not complete')
 report['state']='complete'
except BaseException as e:
 report.update(state='failed',error_type=type(e).__name__)
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
 if report['state']=='failed':raise RuntimeError('Bounded replay failed; original attempt unchanged')
'''


def build(output, quota="spot"):
    import re
    if quota not in ("spot", "reserved"):
        raise ValueError("Invalid quota")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / "eval_results/codex_subtask_freeze_20261012"
    manifest = json.loads((evidence / "trial002-v1/manifest.json").read_text())
    source = (ROOT / "scripts/replay_codex_subtask_contacts.py").read_text()
    config = dict(kind="countertop_contact_replay_v1", collection_payload=manifest["payload"],
        output_root="/mnt/afs/260010165/behavior_1k/codex_subtask_freeze_20261012/" + output.name,
        plan_path=manifest["payload"]["plan_path"], plan_fingerprint=manifest["plan_fingerprint"],
        attempt_root=manifest["payload"]["run_root"] + "/attempt",
        robot_config=manifest["payload"]["robot_config"],
        worker_sha256=hashlib.sha256(source.encode()).hexdigest(),
        on_top_sha256="5c05e3e5640bf3b5fc81decb629be3d9d4603df69b53ce040ce8f77862ba0efd",
        max_gpu_count=1, quota=quota, max_job_seconds=4500,
        diagnostic_only=True, model_calls=0, training_eligible=False)
    driver = re.sub(r"\b(CONFIG|SOURCE)\b", lambda m: repr(config if m[0] == "CONFIG" else source), DRIVER)
    compile(driver, "bounded_replay_driver", "exec")
    (output / "driver.py").write_text(driver)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--quota", choices=("spot", "reserved"), default="spot")
    args = p.parse_args()
    print(json.dumps(build(args.output, args.quota), indent=2))
