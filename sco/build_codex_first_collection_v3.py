#!/usr/bin/env python3
"""Build, never submit, the gated radio311 GPU collection wrapper v3."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/codex_pilot_20261011"
AFS = "/mnt/afs/260010165/behavior_1k/codex_pilot_20261011"
PYTHON = "/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-eval/bin/python"
EPISODES = [28, 68, 102, 4424, 4487, 4517]


WORKER = r'''import hashlib,json,os,re,signal,socket,subprocess,sys,threading,time,urllib.error,urllib.request
from pathlib import Path

def load(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
 temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def ensure(condition,message):
 if not condition:raise ValueError(message)
def safe_text(value,limit=4096):
 # A log line mentioning credentials is withheld in full. Also remove URLs,
 # JWTs and long opaque values even if the log omitted their field labels.
 text=str(value or '')
 lines=[]
 for line in text.splitlines():
  if re.search(r'(?i)token|auth|secret|password|bearer|credential|signature|cookie',line):
   lines.append('<sensitive log line withheld>');continue
  line=re.sub(r'[A-Za-z][A-Za-z0-9+.-]{1,20}://[^\s<>"\']+','<URL redacted>',line)
  line=re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?','<JWT redacted>',line)
  line=re.sub(r'(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])','<opaque value redacted>',line)
  lines.append(line)
 return '\n'.join(lines).encode('utf-8')[-limit:].decode('utf-8',errors='ignore')
def tail(path,limit=4096):
 path=Path(path)
 if not path.is_file():return {'present':False}
 with path.open('rb') as stream:
  size=path.stat().st_size;offset=max(0,size-65536);stream.seek(offset);raw=stream.read(65536)
 if offset:raw=raw.partition(b'\n')[2]
 return {'present':True,'bytes':size,'tail':safe_text(raw.decode('utf-8',errors='replace'),limit)}
def safe_error(error):
 result={'error_type':type(error).__name__,'error_detail':safe_text(str(error),2048)}
 reason=getattr(error,'reason',None)
 if reason is not None:
  result['reason_type']=type(reason).__name__
  if type(getattr(reason,'errno',None)) is int:result['reason_errno']=reason.errno
 return result
def direct_put(opener,url,raw,deadline):
 for attempt in range(4):
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('Report transport deadline reached')
  try:
   request=urllib.request.Request(url,data=raw,method='PUT')
   with opener.open(request,timeout=min(5,remaining)):pass
   return attempt+1
  except urllib.error.URLError as error:
   if not isinstance(error.reason,socket.gaierror) or error.reason.errno!=socket.EAI_AGAIN or attempt==3:raise
   threading.Event().wait(min(.5,max(0,deadline-time.monotonic())))
def gates(config):
 base=Path(config['pilot_root']);manifest=config['manifest'];collection=config['collection']
 prep_path=base/'preparation-v13/report.json';prep=load(prep_path)
 ensure(prep.get('state')=='complete' and prep.get('ready_for_collection') is True,'CPU preparation is not ready')
 validation_path=base/'cluster_validation_v13_report.json';validation=load(validation_path)
 ensure(validation.get('state')=='validation_complete','Cluster validation is not complete')
 ensure(validation.get('selected_auth')=='auth_weizhi.json' and validation.get('account_failover') is False,'Unexpected account selection')
 authentication=validation.get('authentication',{});model=validation.get('model',{})
 ensure(authentication.get('ciphertext_sha256')==config['ciphertext_sha256'],'Authentication handoff fingerprint differs')
 ensure(authentication.get('readback_verified') is True and authentication.get('live_login_verified') is True,'Cluster login not verified')
 ensure(model.get('status')=='verified' and model.get('requested_model')=='gpt-6.1-sol' and model.get('reasoning_effort')=='medium','Sol medium validation missing')
 audit=model.get('event_audit',{})
 ensure(audit.get('status')=='no_tool_events_observed' and audit.get('completed_turns')==1,'Model event audit missing')
 ensure(validation.get('new_model_calls')==0,'Unexpected validation inference scope')
 ensure(sha(base/'cluster_validation_v9_report.json')==validation.get('source_report_sha256'),'Prior validation source changed')
 ensure(sha(base/'connectivity-v9/stdout.log')==validation.get('event_log_sha256'),'Prior model event evidence changed')
 code=Path(collection['code']);ready_path=code/'_CODEX_PILOT_CODE_READY';ready=load(ready_path)
 ensure(str(code)==manifest['code'] and manifest['version']=='v5','Wrong code version')
 ensure(ready.get('state')=='code_verified' and ready.get('bundle_sha256')==manifest['bundle_sha256'],'Code readiness differs')
 ensure(ready.get('source_files_sha256')==manifest['source_files_sha256']==collection['source_files_sha256'],'Frozen code inventory differs')
 ensure(prep.get('code')=={'path':str(code),'bundle_sha256':manifest['bundle_sha256'],'files':len(manifest['source_files_sha256'])},'Preparation code differs')
 ensure(sha(Path(collection['base_code'])/'_PUBLIC_TEST_SOURCE_READY')==manifest['base_source_manifest_sha256'],'Base source differs')
 for name,digest in manifest['source_files_sha256'].items():ensure(sha(code/name)==digest,'Frozen code hash differs')
 ensure(validation.get('auditor_sha256')==manifest['source_files_sha256']['scripts/codex_collection_teacher.py'],'Event auditor differs')
 data_path=base/'original-v3/_ORIGINAL_DATA_READY.json';data=load(data_path)
 ensure(data.get('state')=='complete' and data.get('original_reader_verified') is True and data.get('raw_success_verified') is True,'Original demonstrations not verified')
 ensure(data.get('selected_episodes')==config['episodes'],'Original episode selection differs')
 ensure(prep.get('data_root')==str(base/'original-v3') and prep.get('data_ready_sha256')==sha(data_path),'Preparation data receipt differs')
 demonstrations=data.get('demonstrations',[])
 ensure(len(demonstrations)==6 and sorted(row['episode_index'] for row in demonstrations)==config['episodes'],'Six prepared demonstrations required')
 demo_path=Path(collection['demonstration_manifest']);demo=load(demo_path)
 ensure(str(demo_path)==str(base/'original-v3/demonstrations/28/manifest.json'),'Wrong radio reference')
 ensure(demo.get('metadata',{}).get('task')=='turning_on_radio' and demo['metadata'].get('episode_index')==28,'Radio reference identity differs')
 matching=next(row for row in demonstrations if row['episode_index']==28)
 ensure(matching.get('manifest')==str(demo_path) and matching.get('fingerprint')==demo.get('fingerprint'),'Radio manifest receipt differs')
 return {'preparation_report_sha256':sha(prep_path),'validation_report_sha256':sha(validation_path),
         'data_ready_sha256':sha(data_path),'code_ready_sha256':sha(ready_path),
         'selected_auth':'auth_weizhi.json','model':'gpt-6.1-sol','reasoning_effort':'medium',
         'selected_episodes':config['episodes'],'validated_model_event_audit':{
          'status':'no_tool_events_observed','completed_turns':1}}
def numeric_usage(value):
 if not isinstance(value,dict):return None
 keys=('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')
 result={key:value[key] for key in keys if key in value}
 if not set(keys[:3])<=set(result) or any(type(v) is not int or v<0 for v in result.values()):return None
 if result['cached_input_tokens']>result['input_tokens']:return None
 return result
def event_usage(path):
 if not path.is_file():return None
 latest=None
 with path.open(encoding='utf-8',errors='replace') as stream:
  for line in stream:
   try:value=json.loads(line)
   except ValueError:continue
   if isinstance(value,dict) and value.get('type')=='turn.completed':latest=numeric_usage(value.get('usage')) or latest
 return latest
def teacher_usage(attempt):
 rows=sorted((attempt/'teacher').glob('step-*/decision/*'))
 ensure(len(rows)<=32,'Teacher request budget unexpectedly exceeded')
 observed=[];validated=failed=0
 for path in rows:
  if not path.is_dir():continue
  receipt={}
  try:receipt=load(path/'receipt.json')
  except (OSError,ValueError):pass
  validated+=receipt.get('status')=='validated';failed+=receipt.get('status')=='failed'
  usage=numeric_usage(receipt.get('usage')) or event_usage(path/'stdout.log')
  if usage is not None:observed.append(usage)
 totals={key:sum(row[key] for row in observed if key in row) for key in
         ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens') if any(key in row for row in observed)}
 return {'request_directories':len(rows),'validated_decisions':validated,'failed_decisions':failed,
         'model_calls_with_usage':len(observed),'requests_without_usage':len(rows)-len(observed),
         'observed_token_totals':totals if observed else None,'missing_usage_is_zero':False}
def result_summary(value):
 keys=('steps','task_success','success','terminated','truncated','stop_reason','teacher_decisions','wall_seconds','official_benchmark')
 return {key:safe_text(value[key],1024) if isinstance(value[key],str) else value[key] for key in keys if key in value and isinstance(value[key],(int,float,bool,str))}
def progress(run_root):
 run_root=Path(run_root);attempt=run_root/'attempt';status={}
 try:
  raw=load(run_root/'status.json')
  status={key:safe_text(raw[key],1024) if isinstance(raw[key],str) else raw[key] for key in
          ('state','task','instance_id','attempt_id','training_eligible','error_type','driver_exit_code','started_at','updated_at')
          if key in raw and isinstance(raw[key],(int,float,bool,str))}
  if isinstance(raw.get('result'),dict):status['result']=result_summary(raw['result'])
 except (OSError,ValueError):status={'state':'not_yet_written'}
 current={'executed_steps_observed':0,'observation_source':'action_sources.jsonl'}
 path=attempt/'action_sources.jsonl'
 if path.is_file():
  with path.open('rb') as stream:
   offset=max(0,path.stat().st_size-65536);stream.seek(offset);lines=stream.read(65536).splitlines()
  if offset:lines=lines[1:]
  for line in reversed(lines):
   try:last=json.loads(line)
   except (ValueError,UnicodeError):continue
   if isinstance(last,dict) and type(last.get('step')) is int:
    current.update(executed_steps_observed=last['step']+1,last_action_source=safe_text(last.get('source'),1024));break
 for name in ('completed.json','failed.json'):
  try:
   item=load(attempt/name);current[name]=result_summary(item.get('result',item))
   if name=='failed.json':current[name]['error_type']=safe_text(item.get('error_type'),1024)
  except (OSError,ValueError):pass
 return {'root_status':status,'collector_progress':current,'teacher_usage':teacher_usage(attempt)}
def run(config):
 os.umask(0o077)
 started=time.monotonic();deadline=started+9000
 collection=config['collection'];root=Path(config['report_root']);run_root=Path(collection['run_root'])
 report={'kind':'codex_first_collection_gpu_v3','state':'starting','stage':'gates','official_benchmark':False,
         'training_eligible':False,'started_at':time.time(),'task':'turning_on_radio','instance_id':311,
         'attempt_id':'pilot-001','limits':{'outer_seconds':9000,'collection_wall_seconds':8400,
         'max_steps':4500,'max_decisions':32,'max_seconds':7200},'report_transport':'direct_nonfatal'}
 process=None;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));owned_root=False
 def publish(remote=True):
  report.update(updated_at=time.time(),elapsed_seconds=round(time.monotonic()-started,3))
  if owned_root:write(root/'report.json',report)
  url=os.environ.get('CODEX_PILOT_REPORT_PUT')
  if remote and url:
   try:
    direct_put(opener,url,json.dumps(report,allow_nan=False).encode(),min(deadline-15,time.monotonic()+8))
    report.pop('report_transport_error',None)
   except Exception as error:report['report_transport_error']={'error_type':type(error).__name__,'reason_type':type(getattr(error,'reason',None)).__name__}
   if owned_root:write(root/'report.json',report)
 def stop(signum,frame):raise TimeoutError('Finite collection wrapper deadline reached')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(9000)
 try:
  root.mkdir(mode=0o700,exist_ok=False);owned_root=True;publish()
  report['gates']=gates(config);publish()
  ensure(Path(config['python']).is_file(),'Verified b1k-eval Python is unavailable')
  ensure(not run_root.exists(),'Collection output must be fresh')
  payload_path=root/'collection_payload.json';write(payload_path,collection)
  report['collection_payload_sha256']=sha(payload_path)
  env=dict(os.environ,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
  for key in list(env):
   if key.upper() in {'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY'} or key in {'CODEX_API_KEY','OPENAI_API_KEY','PYTHONHOME'}:env.pop(key)
  env['PYTHONPATH']=str(Path(collection['code'])/'sco')+':'+str(Path(collection['code'])/'scripts')
  env['PATH']='/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:'+env.get('PATH','')
  # The collection driver owns its mihomo lifetime and private Codex-home lease.
  launcher="import runpy,signal,sys\ndef stop(signum,frame):raise TimeoutError('Owned driver terminated')\nsignal.signal(signal.SIGTERM,stop)\npath=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(path,run_name='__main__')"
  with (root/'collection_driver.log').open('x') as log:
   process=subprocess.Popen([config['python'],'-c',launcher,str(Path(collection['code'])/'sco/run_codex_collection.py'),str(payload_path)],
       env=env,cwd=str(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   report.update(state='running',stage='collection',owned_driver_pid=process.pid);publish()
   until=min(deadline-120,time.monotonic()+8760)
   next_sample=time.monotonic()+30
   while True:
    remaining=until-time.monotonic()
    if remaining<=0:raise TimeoutError('Collection cleanup reserve reached')
    try:returncode=process.wait(timeout=min(max(.1,next_sample-time.monotonic()),remaining))
    except subprocess.TimeoutExpired:returncode=None
    report.update(progress(run_root));report['driver_returncode']=returncode;publish()
    next_sample=max(next_sample+30,time.monotonic())
    if returncode is not None:
     ensure(returncode==0,'Collection driver failed');break
  completed=load(run_root/'_COMPLETE')
  ensure(completed.get('state')=='complete' and completed.get('official_benchmark') is False,'Invalid collection completion')
  ensure(type(completed.get('training_eligible')) is bool,'Invalid training eligibility')
  report.update(state='complete',stage='complete',training_eligible=completed['training_eligible'],
       result=result_summary(completed.get('result',{})),collection_receipt_sha256=sha(run_root/'_COMPLETE'),completed_at=time.time())
 except BaseException as error:
  report.update(state='failed',training_eligible=False,**safe_error(error))
 finally:
  if process is not None and process.poll() is None:
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:process.wait(timeout=40)
   except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:report['owned_driver_cleanup_timeout']=True
  signal.alarm(0)
  try:report.update(progress(run_root))
  except Exception as error:report['progress_read_error_type']=type(error).__name__
  if report['state']=='failed' and owned_root:
   outer=tail(root/'collection_driver.log',2048);inner=tail(run_root/'driver.log',2048)
   report['failure_tails']={'driver':{'present':outer['present'] or inner['present'],
       'tail':safe_text(outer.get('tail','')+'\n'+inner.get('tail',''),4096)},
       'policy':tail(run_root/'logs/policy.log'),'collector':tail(run_root/'logs/collector.log')}
  publish()
 return report
'''


def build(output, *, manifest_path=OUT / "collection_code_v5.tar.gz.manifest.json",
          collection_path=OUT / "initial_collection_payload_prepared.json",
          encryption_path=OUT / "auth_encryption_receipt.json"):
    manifest = json.loads(Path(manifest_path).read_text())
    collection = copy.deepcopy(json.loads(Path(collection_path).read_text()))
    encryption = json.loads(Path(encryption_path).read_text())
    if manifest.get("version") != "v5" or manifest.get("code") != AFS + "/code-v5":
        raise ValueError("The first GPU collection requires the frozen code-v5 manifest")
    if hashlib.sha256(Path(manifest["bundle"]).read_bytes()).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("Code-v5 bundle differs from the frozen manifest")
    if Path(encryption["selected_auth_file"]).name != "auth_weizhi.json":
        raise ValueError("Unexpected selected authentication account")
    expected = dict(task="turning_on_radio", instance_id=311, attempt_id="pilot-001", teacher_backend="codex_local",
                    max_steps=4500, max_decisions=32, max_seconds=7200, wall_timeout_seconds=8400, teacher_interval=256)
    if any(collection.get(key) != value for key, value in expected.items()):
        raise ValueError("Initial collection scope or budget differs")
    collection.update(code=manifest["code"], source_files_sha256=manifest["source_files_sha256"],
                      robot_config=manifest["code"] + "/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
                      demonstration_manifest=AFS + "/original-v3/demonstrations/28/manifest.json")
    from run_codex_collection import validate_payload
    collection = validate_payload(collection)
    config = dict(pilot_root=AFS, report_root=AFS + "/first-radio311-gpu-v3", python=PYTHON,
                  episodes=EPISODES, manifest=manifest, collection=collection,
                  ciphertext_sha256=encryption["ciphertext_sha256"])
    source = WORKER + "\nBOOTSTRAP_REPORT = run(" + repr(config) + ")\n"
    compile(source, "<first-collection-v3>", "exec")
    output = Path(output)
    receipt_path = output.with_suffix(".receipt.json")
    if output.exists() or receipt_path.exists():
        raise ValueError("Source and receipt outputs must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write(source)
    receipt = dict(kind="codex_first_collection_job_source_v3", source=str(output.resolve()),
                   sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()),
                   code_bundle_sha256=manifest["bundle_sha256"], code=manifest["code"],
                   report_root=config["report_root"], collection_run_root=collection["run_root"],
                   original_data=AFS + "/original-v3", max_seconds=9000,
                   python_binary=PYTHON, bootstrap_report=True, actual_job_submitted=False)
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2); stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "first_collection_job_v3.py")
    parser.add_argument("--manifest", type=Path, default=OUT / "collection_code_v5.tar.gz.manifest.json")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, manifest_path=args.manifest), indent=2))


if __name__ == "__main__":
    main()
