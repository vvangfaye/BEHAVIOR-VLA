"""Submit the prepared, finite four-GPU Codex pilot exactly once.

Only this explicit CLI submits. A successful CPU preparation report must bind
the complete frozen payload. Unknown submission outcomes retain a canonical
intent; changing the requested name or receipt cannot retry the experiment.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import zlib

import codex_pilot_cloud as cloud
import public_test_admission as admission
import run_codex_pilot_training as worker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/codex_pilot_20261011"
INTENT = OUT / "training_validation_submission_intent.json"
TEMPLATE = "pt-4ewqyd0e"
TEMPLATE_UID = "5d95f116745940c58a4523cc24c75041"
SKU = "n12lp.nn.i10a.4"
POOL = "computing-cluster-5090-01g"
IMAGE = "registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634"
MOUNT = "01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs"
RUNTIME = "/mnt/afs/260010165/behavior_1k/runtime_20261007_1306"
PYTHON = RUNTIME + "/envs/b1k-eval/bin/python"
DRIVER_SHA = "089aefc0151e2eb051734ba7561b3d5b7054de199b642bc01cbe4853f3619cf1"
WORKER_SECONDS = 13800
OUTER_SECONDS = 14400
NAME = re.compile(r"behavior-codex-training-[a-z0-9][a-z0-9-]{0,39}")


def now():
    return datetime.now(timezone.utc)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())


def persist(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_new(temporary, value);temporary.replace(path)


def pinned_ref(ref):
    if (not isinstance(ref, dict) or not isinstance(ref.get("path"), str)
            or not Path(ref["path"]).is_absolute() or ".." in Path(ref["path"]).parts
            or type(ref.get("size")) is not int or ref["size"] < 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256")))):
        raise ValueError("Preparation must provide a complete pinned file reference")
    if not Path(ref["path"]).is_relative_to(cloud.AFS):
        raise ValueError("Training artifacts must remain in this experiment's AFS root")


def prepare_inputs(payload_path, preparation_report):
    payload_raw = Path(payload_path).read_bytes();report_raw = Path(preparation_report).read_bytes()
    original = json.loads(payload_raw);report = json.loads(report_raw)
    payload = worker.validate_payload(original)
    if (report.get("state") != "complete" or report.get("ready_for_training") is not True
            or report.get("training_payload") != original
            or any(report.get(key) != original.get(key) for key in ("training_plan", "code_ready"))):
        raise ValueError("Actual CPU preparation must be complete and bind this exact training payload")
    if (payload["wall_timeout_seconds"] != WORKER_SECONDS or payload["runtime"] != RUNTIME
            or payload["driver_sha256"] != DRIVER_SHA):
        raise ValueError("Training runtime, finite wall limit or frozen driver differs")
    for key in ("training_plan", "code_ready"):pinned_ref(payload[key])
    for key in ("code", "report_root", "destination"):
        path = Path(payload[key])
        if ".." in path.parts or not path.is_relative_to(cloud.AFS):
            raise ValueError("Training paths must remain in this experiment's AFS root")
    source = Path(worker.__file__).read_bytes()
    if sha(source) != DRIVER_SHA:raise ValueError("Local frozen training driver changed")
    compile(source, "training_worker.py", "exec")
    return payload, source, dict(payload_sha256=sha(payload_raw), preparation_report_sha256=sha(report_raw))


def validate_resources(job, *, template=False, name=None):
    if (job.get("ownership", {}).get("user_name") != cloud.BUCKET
            or job.get("resource_pool", {}).get("name") != POOL
            or job.get("resource_pool", {}).get("zone") != "cn-sh-01g"
            or job.get("scheduling", {}).get("quota_type") != "RESERVED"
            or job.get("framework") != "PYTORCH"):
        raise ValueError("Four-GPU owner, pool, framework or quota differs")
    if template:
        if (job.get("name") != TEMPLATE or job.get("uid") != TEMPLATE_UID
                or job.get("state") not in {"SUCCEEDED", "FAILED", "TERMINATED", "CANCELED"}):
            raise ValueError("Exact terminal four-GPU template identity differs")
    elif job.get("name") != name or not isinstance(job.get("uid"), str) or not job["uid"]:
        raise ValueError("Submitted job identity is missing or differs")
    roles = job.get("roles")
    if not isinstance(roles, list) or len(roles) != 1:raise ValueError("Exactly one four-GPU worker is required")
    role = roles[0];specs = role.get("resource_spec")
    if (role.get("name") != "Worker" or role.get("total_replicas") != 1
            or not isinstance(specs, list) or len(specs) != 1):
        raise ValueError("Worker replica count differs")
    spec = specs[0]
    if spec.get("name") != SKU or spec.get("replicas") != 1:raise ValueError("Four-GPU SKU differs")
    for key in ("requests", "limits"):
        values = spec.get(key, {})
        if any(str(values.get(field)) != expected for field, expected in
               {"cpu": "56", "memory": "240", "device": "4", "nvidia.com/gpu": "4"}.items()):
            raise ValueError("Four GPUs / 56 CPU / 240 GiB resources differ")
    if not template and role.get("image_path") != IMAGE:raise ValueError("Submitted user image differs")


def validate_capacity(capacity, *, observed_now):
    observed = datetime.fromisoformat(capacity.get("observed_at", ""))
    if (capacity.get("owner") != cloud.BUCKET or capacity.get("quota_type") != "reserved"
            or capacity.get("fresh_query") is not True or observed.tzinfo is None
            or not 0 <= (observed_now - observed).total_seconds() <= 60
            or capacity.get("standard_limit") != 4 or type(capacity.get("used_gpus")) is not int
            or capacity["used_gpus"] != 0 or capacity.get("active_reserved") != []
            or set(capacity.get("queried_states", [])) != admission.ACTIVE_STATES
            or type(capacity.get("pages_read")) is not int or not 10 <= capacity["pages_read"] <= 200):
        raise ValueError("Four-GPU submission requires fresh, fully paginated, owner-wide zero standard usage")


BOOTSTRAP = r'''
import base64,hashlib,json,os,re,runpy,socket,sys,time,urllib.error,urllib.request,zlib
from pathlib import Path
os.umask(0o077);sys.dont_write_bytecode=True
for key in list(os.environ):
 if key.upper() in {'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY'}:os.environ.pop(key)
root=Path(CONFIG['bootstrap_root']);root.mkdir(parents=True,mode=0o700,exist_ok=False)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
report={'state':'bootstrap_started','ready_for_evaluation':False}
def save():
 temporary=root/'report.json.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(root/'report.json')
def upload():
 raw=json.dumps(report).encode()
 for attempt in range(4):
  try:
   with opener.open(urllib.request.Request(CONFIG['report_put'],data=raw,method='PUT'),timeout=30):pass
   return
  except urllib.error.URLError as error:
   if not isinstance(error.reason,socket.gaierror) or error.reason.errno!=socket.EAI_AGAIN or attempt==3:raise
   time.sleep(.5)
def transport_artifacts():
 entries=[];total=0
 first=report.get('training_receipt')
 expected=Path(CONFIG['payload']['training_plan']['path']).parent/'training_receipt.json'
 for name,ref,wanted in [('training_receipt.json',first,expected)]:
  if not isinstance(ref,dict) or ref.get('path')!=str(wanted):raise ValueError('Training receipt reference differs')
  path=Path(ref['path'])
  if path.is_symlink() or type(ref.get('size')) is not int or not 0<ref['size']<=2*1024**2:raise ValueError('Invalid small artifact')
  with path.open('rb') as stream:raw=stream.read(ref['size']+1)
  total+=len(raw)
  if len(raw)!=ref['size'] or hashlib.sha256(raw).hexdigest()!=ref.get('sha256'):raise ValueError('Training receipt changed')
  entries.append({'name':name,'ref':ref,'base64':base64.b64encode(raw).decode('ascii')})
  manifest=json.loads(raw).get('checkpoint_manifest')
 if not isinstance(manifest,dict) or manifest.get('path')!=CONFIG['payload']['destination']+'/SHA256SUMS.inference':raise ValueError('Inference manifest reference differs')
 path=Path(manifest['path'])
 if path.is_symlink() or type(manifest.get('size')) is not int or not 0<manifest['size']<=2*1024**2-total:raise ValueError('Small artifact transfer exceeds 2 MiB')
 with path.open('rb') as stream:raw=stream.read(manifest['size']+1)
 if len(raw)!=manifest['size'] or hashlib.sha256(raw).hexdigest()!=manifest.get('sha256'):raise ValueError('Inference manifest changed')
 entries.append({'name':'SHA256SUMS.inference','ref':manifest,'base64':base64.b64encode(raw).decode('ascii')})
 return entries
try:
 save()
 source=zlib.decompress(base64.b64decode(CONFIG['source_inline'],validate=True))
 if hashlib.sha256(source).hexdigest()!=CONFIG['source_sha256']:raise ValueError('Decoded worker source changed')
 path=root/'training_worker.py'
 with path.open('xb') as stream:stream.write(source)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=CONFIG['source_sha256']:raise ValueError('Written worker source changed')
 namespace=runpy.run_path(str(path),run_name='codex_training_worker')
 report=namespace['run'](CONFIG['payload'])
 if report.get('state')=='complete' and report.get('ready_for_evaluation') is True:
  report=dict(report)
  try:
   report['bootstrap_artifacts']=transport_artifacts();report['bootstrap_artifacts_state']='verified'
  except Exception as error:
   report['bootstrap_artifacts_state']='failed';report['artifact_transport_failed']=type(error).__name__
except BaseException as error:
 # Never publish command text or credentials from an unexpected bootstrap exception.
 report={'state':'bootstrap_failed','ready_for_evaluation':False,'error_type':type(error).__name__}
finally:
 save()
 try:upload()
 except Exception as error:
  report['report_transport_error']=type(error).__name__;save()
print(json.dumps({key:report.get(key) for key in ('state','stage','ready_for_evaluation','error_type')}),flush=True)
raise SystemExit(0 if report.get('state')=='complete' and report.get('ready_for_evaluation') is True else 1)
'''


def startup(source, payload, name, report_put):
    encoded = base64.b64encode(zlib.compress(source, 9)).decode("ascii")
    if len(encoded) > 1024**2:raise ValueError("Worker exceeds bounded inline source size")
    bootstrap_root = cloud.AFS + "/training-bootstrap/" + name
    worker.disjoint([bootstrap_root, payload["code"], payload["report_root"], payload["destination"],
                     str(Path(payload["training_plan"]["path"]).parent)])
    config = dict(bootstrap_root=bootstrap_root, source_inline=encoded, source_sha256=DRIVER_SHA,
                  payload=payload, report_put=report_put)
    text = "CONFIG=" + repr(config) + "\n" + BOOTSTRAP
    compile(text, "training_bootstrap.py", "exec")
    return "timeout --signal=TERM --kill-after=30 " + str(OUTER_SECONDS) + " " + shlex.quote(PYTHON) + " -c " + shlex.quote(text)


def submit(payload_path, preparation_report, receipt_path, *, name):
    if not isinstance(name, str) or not NAME.fullmatch(name):raise ValueError("Use a unique behavior-codex-training-* name")
    payload, source, input_proof = prepare_inputs(payload_path, preparation_report)
    receipt = Path(receipt_path).resolve()
    if receipt == INTENT.resolve():raise ValueError("Receipt must differ from the canonical intent")
    receipt.parent.mkdir(parents=True, exist_ok=True);INTENT.parent.mkdir(parents=True, exist_ok=True)
    with INTENT.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if INTENT.exists() or INTENT.is_symlink() or receipt.exists() or receipt.is_symlink():
            raise FileExistsError("Training already has an intent; inspect its exact job name before any further action")
        template = cloud.describe(TEMPLATE);validate_resources(template, template=True)
        key = cloud.PREFIX + "jobs/" + name + "/training_worker.py"
        record = dict(kind="codex_training_submission_v1", state="intent", name=name,
                      created_at=now().isoformat(), gpu_count=4, quota_type="reserved", max_seconds=OUTER_SECONDS,
                      worker_seconds=WORKER_SECONDS, source_sha256=DRIVER_SHA, source_key=key,
                      report_key=cloud.PREFIX + "jobs/" + name + "/report.json", python_binary=PYTHON,
                      container_image=IMAGE, template=dict(name=TEMPLATE, uid=TEMPLATE_UID),
                      training_plan=payload["training_plan"], code_ready=payload["code_ready"], **input_proof)
        # The canonical intent survives failed uploads, capacity checks and unknown outcomes.
        write_new(INTENT, record);write_new(receipt, record)
        def publish():
            persist(INTENT, record);persist(receipt, record)
        store = cloud.storage();store.put_object(Bucket=cloud.BUCKET, Key=key, Body=source)
        if sha(store.get_object(Bucket=cloud.BUCKET, Key=key)["Body"].read()) != DRIVER_SHA:
            raise ValueError("Uploaded training worker source differs on readback")
        report_put = store.generate_presigned_url("put_object", Params=dict(Bucket=cloud.BUCKET, Key=record["report_key"]), ExpiresIn=86400)
        command = startup(source, payload, name, report_put)
        capacity = admission.query_standard_capacity(cloud.CLI, cloud.environment())
        validate_capacity(capacity, observed_now=now());record["admission"] = capacity
        args = [cloud.CLI, "acp", "jobs", "copy", "--copy-job-name=" + TEMPLATE,
                "--worker-spec=" + SKU, "--env=NVIDIA_DRIVER_CAPABILITIES:all", "--workspace-name=share-space",
                "--name=" + name, "--job-name=" + name, "--worker-nodes=1", "--quota-type=reserved",
                "--storage-mount=" + MOUNT, "--container-image-url=" + IMAGE, "--command=" + command]
        record["state"] = "submission_unknown";publish()
        try:
            result = subprocess.run(args, env=cloud.environment(), capture_output=True, text=True, timeout=90)
        except (subprocess.TimeoutExpired, OSError):
            raise RuntimeError("Submission outcome unknown; inspect the saved exact name, do not resubmit") from None
        record["cli_exit_code"] = result.returncode;publish()
        if result.returncode != 0:raise RuntimeError("Submission outcome unknown; inspect the saved exact name, do not resubmit")
        try:
            job = cloud.describe(name);validate_resources(job, name=name)
        except Exception:
            raise RuntimeError("Submission requires exact-name readback; saved intent blocks retry") from None
        record.update(state="submitted", uid=job["uid"], platform_state=job["state"]);publish()
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    result = submit(args.payload, args.preparation_report, args.receipt, name=args.name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
