"""Submit one fixed native-evaluation shard, with ordered, durable admission.

This explicit submit command consumes actual training receipts, never starts a
teacher, and never retries an ambiguous submission. All five shard intents use
one canonical experiment directory; at most four one-GPU jobs may be active.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import zlib

import codex_pilot_cloud as cloud
import public_test_admission as admission
import run_codex_pilot_evaluation as worker
import submit_codex_training_validation as common

INTENTS = common.OUT / "native_evaluation_submissions"
CODE = "/mnt/afs/260010165/behavior_1k/code_snapshots/public-test-1688977-aoss-20261010"
TEMPLATE = "pt-y91th8k9"
TEMPLATE_UID = "c14d0892eaa148d381c7f4424684c9d3"
SKU = "n12lp.nn.i10a.1"
DRIVER_SHA = "0ff4d1a429cc6fdcc280f1ae2874099b7ce3b7586e778f0c7437f67b19abf4b8"
HELPER_SHA = "4ad0162ece010028e0ec89fb6776900595698ece0dfb9e29dcfd8128cbcfb59e"
TERMINAL = {"SUCCEEDED", "FAILED", "TERMINATED", "CANCELED", "CANCELLED", "STOPPED"}
NAME = re.compile(r"behavior-codex-native-[a-z0-9][a-z0-9-]{0,40}")
PANEL_KIND = "codex_native_eval_submission_panel_v2"
MAX_ACTIVE = 4


def ensure(condition, message):
    if not condition:raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def checked_copy(path, ref):
    common.pinned_ref(ref)
    raw = Path(path).read_bytes()
    ensure(len(raw) == ref["size"] and common.sha(raw) == ref["sha256"], "Downloaded artifact differs from actual training receipt")
    return raw


def prepare_inputs(training_report, training_receipt, checkpoint_manifest, *, shard, name):
    ensure(shard in worker.SHARD_IDS and NAME.fullmatch(name), "Use one declared shard and a unique behavior-codex-native-* name")
    report = load(training_report)
    ensure(report.get("kind") == common.worker.KIND and report.get("state") == "complete"
           and report.get("ready_for_evaluation") is True and report.get("model_teacher_enabled") is False
           and all(report.get("phases", {}).get(phase, {}).get("state") == "verified"
                   for phase in ("data_preflight", "jax_preflight", "smoke", "full", "stage")),
           "Actual finite training worker must have completed every phase")
    receipt_ref = report.get("training_receipt")
    train = json.loads(checked_copy(training_receipt, receipt_ref))
    common.pinned_ref(report.get("training_plan"))
    ensure(report["phases"]["stage"].get("evidence") == receipt_ref
           and train.get("kind") == "codex_pilot_trained_checkpoint_v1"
           and train.get("training_plan") == report["training_plan"]
           and train.get("parent_checkpoint_step") == 1688977
           and type(train.get("optimizer_steps")) is int and train["optimizer_steps"] == 256
           and all(train.get(field) is True for field in ("ready_for_evaluation", "real_lerobot_readback_verified",
                     "base_normalization_preserved", "inference_staging_complete")),
           "Training receipt lacks the exact completed 256-step native checkpoint")
    run_id = Path(report["training_plan"]["path"]).parent.name
    ensure(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id)
           and report["training_plan"]["path"] == cloud.AFS + "/training/" + run_id + "/training_plan.json"
           and receipt_ref["path"] == cloud.AFS + "/training/" + run_id + "/training_receipt.json"
           and train.get("checkpoint_id") == "pi05_b1k_codex_" + run_id + "_256"
           and train.get("repo_id") == "b1k_codex_" + run_id
           and report.get("checkpoint") == train.get("staged_checkpoint") == cloud.AFS + "/inference/" + run_id,
           "Training identity or inference destination differs")
    manifest_ref = train.get("checkpoint_manifest")
    raw = checked_copy(checkpoint_manifest, manifest_ref)
    ensure(manifest_ref["path"] == report["checkpoint"] + "/SHA256SUMS.inference"
           and train.get("output_checkpoint_manifest_sha256") == manifest_ref["sha256"],
           "Staged checkpoint manifest identity differs")
    entries = {}
    for line in raw.decode().splitlines():
        digest, separator, entry = line.partition("  ");relative = Path(entry)
        ensure(separator and re.fullmatch(r"[0-9a-f]{64}", digest) and entry and not relative.is_absolute()
               and ".." not in relative.parts and entry not in entries, "Malformed inference manifest")
        entries[entry] = digest
    norm = "assets/" + train["repo_id"] + "/norm_stats.json"
    ensure({"params/_METADATA", "assets/subtask_conditioning.json", norm} <= entries.keys()
           and entries[norm] == train.get("base_norm_sha256"), "Native assets or preserved normalization are missing")
    source = Path(worker.__file__).read_bytes()
    helper = common.ROOT / "scripts/evaluate_codex_pilot.py";helper_raw = helper.read_bytes()
    ensure(common.sha(source) == DRIVER_SHA and common.sha(helper_raw) == HELPER_SHA, "Frozen native worker or panel verifier changed")
    helper_ref = dict(path=cloud.AFS + "/code-v8/scripts/evaluate_codex_pilot.py", size=len(helper_raw), sha256=HELPER_SHA)
    payload = worker.validate_payload(dict(code=CODE, runtime=common.RUNTIME, checkpoint=report["checkpoint"],
        run_root=cloud.AFS + "/native-evaluation-" + run_id + "/" + shard,
        checkpoint_id=train["checkpoint_id"], repo_id=train["repo_id"], execution_name=name,
        source_manifest_sha256=worker.SOURCE_SHA256, driver_sha256=DRIVER_SHA, training_receipt=receipt_ref,
        report_helper=helper_ref, shard_ids=[shard], port=8930, startup_timeout_seconds=900, shard_timeout_seconds=13200))
    identity = dict(checkpoint_id=train["checkpoint_id"], checkpoint_manifest_sha256=manifest_ref["sha256"],
                    training_receipt=receipt_ref, training_plan=report["training_plan"])
    proof = dict(training_report_sha256=common.sha(Path(training_report).read_bytes()),
                 downloaded_training_receipt_sha256=common.sha(Path(training_receipt).read_bytes()),
                 checkpoint_manifest_sha256=manifest_ref["sha256"])
    return payload, source, identity, proof


def validate_capacity(capacity, panel_active=()):
    observed = datetime.fromisoformat(capacity.get("observed_at", ""))
    rows = capacity.get("active_reserved")
    ensure(capacity.get("owner") == cloud.BUCKET and capacity.get("quota_type") == "reserved"
           and capacity.get("fresh_query") is True and observed.tzinfo is not None
           and 0 <= (common.now() - observed).total_seconds() <= 60
           and capacity.get("standard_limit") == 4 and type(capacity.get("used_gpus")) is int
           and 0 <= capacity["used_gpus"] <= 3 and isinstance(rows, list)
           and set(capacity.get("queried_states", [])) == admission.ACTIVE_STATES
           and type(capacity.get("pages_read")) is int and 10 <= capacity["pages_read"] <= 200,
           "Need fresh complete ACP owner pagination and one free standard GPU")
    known, names = {}, {}
    def merge(row):
        ensure(isinstance(row, dict) and all(isinstance(row.get(key), str) and row[key].strip() for key in ("name", "uid"))
               and row.get("state") in admission.ACTIVE_STATES
               and type(row.get("gpu_count")) is int and row["gpu_count"] > 0,
               "Active GPU identity, state or resource count is unknown")
        uid = row["uid"]
        ensure(row["name"] not in names or names[row["name"]] == uid, "Active job name has conflicting UIDs")
        if uid in known:
            # Both observations must be active, but a normal lifecycle transition
            # between them does not change this UID's name or GPU allocation.
            ensure(all(row[key] == known[uid][key] for key in ("name", "gpu_count")),
                   "Active UID has conflicting identity or GPU allocation")
        else:
            known[uid] = {key:row[key] for key in ("name", "uid", "state", "gpu_count")}
            names[row["name"]] = uid
    for row in rows:merge(row)
    ensure(sum(row["gpu_count"] for row in known.values()) == capacity["used_gpus"], "Owner GPU total differs from unique active UIDs")
    for row in panel_active:
        ensure(isinstance(row, dict) and type(row.get("gpu_count")) is int and row["gpu_count"] == 1,
               "Known panel jobs must use exactly one GPU")
        merge(row)
    effective = sum(row["gpu_count"] for row in known.values())
    ensure(effective <= 3, "Known panel and owner GPU union leaves no free standard GPU")
    return {**capacity, "effective_used_gpus":effective,
            "effective_active_reserved":[known[uid] for uid in sorted(known)]}


def validate_resources(job, *, template=False, name=None):
    ensure(job.get("ownership", {}).get("user_name") == cloud.BUCKET
           and job.get("resource_pool", {}).get("name") == common.POOL
           and job.get("resource_pool", {}).get("zone") == "cn-sh-01g"
           and (template or job.get("scheduling", {}).get("quota_type") == "RESERVED") and job.get("framework") == "PYTORCH",
           "Native evaluation must use the owned standard 5090 pool")
    ensure((job.get("name"), job.get("uid")) == (TEMPLATE, TEMPLATE_UID) if template else
           job.get("name") == name and isinstance(job.get("uid"), str) and bool(job["uid"]), "Exact job identity differs")
    roles = job.get("roles")
    ensure(isinstance(roles, list) and len(roles) == 1, "One GPU worker is required")
    role = roles[0];specs = role.get("resource_spec")
    ensure(role.get("total_replicas") == 1 and isinstance(specs, list) and len(specs) == 1, "Worker count differs")
    spec = specs[0]
    ensure(spec.get("name") == SKU and spec.get("replicas") == 1, "Native evaluation SKU differs")
    for key in ("requests", "limits"):
        ensure(all(str(spec.get(key, {}).get(field)) == value for field, value in
                   {"cpu":"14", "memory":"60", "device":"1", "nvidia.com/gpu":"1"}.items()), "One-GPU resource shape differs")
    if not template:ensure(role.get("image_path") == common.IMAGE, "User image differs")


def check_panel(identity, shard, name):
    """Check ordered admission and every exact prior UID under the panel lock."""
    anchor_path = INTENTS / "panel.json"
    paths = sorted(INTENTS.glob("intent-*.json"))
    if not anchor_path.exists():
        ensure(not paths, "Intents exist without their panel anchor")
        anchor = dict(kind=PANEL_KIND, identity=identity, admitted=[], max_active=MAX_ACTIVE)
    else:
        anchor = load(anchor_path)
    ensure(anchor.get("kind") == PANEL_KIND and anchor.get("identity") == identity
           and anchor.get("max_active") == MAX_ACTIVE,
           "This experiment already selected a different trained checkpoint")
    admitted = anchor.get("admitted")
    ensure(isinstance(admitted, list) and len(set(admitted)) == len(admitted)
           and set(admitted) <= set(worker.SHARD_IDS)
           and {path.name for path in paths} == {"intent-" + item + ".json" for item in admitted},
           "An admitted shard intent is missing or the panel transaction was interrupted")
    ensure(shard not in admitted, "This fixed shard already has an intent; do not repeat it")
    ensure(admitted == list(worker.SHARD_IDS[:len(admitted)])
           and len(admitted) < len(worker.SHARD_IDS) and shard == worker.SHARD_IDS[len(admitted)],
           "Submit the five fixed shards in their declared order")
    active = []
    for path in paths:
        record = load(path)
        ensure(record.get("state") == "submitted" and isinstance(record.get("uid"), str) and record["uid"]
               and record.get("identity") == identity and path.name == "intent-" + record.get("shard_id", "") + ".json",
               "An earlier submission outcome is unresolved; inspect its exact saved job")
        ensure(record.get("name") != name, "Every shard needs a unique job name")
        job = cloud.describe(record["name"])
        ensure(job.get("name") == record["name"] and job.get("uid") == record["uid"]
               and job.get("state") in TERMINAL | admission.ACTIVE_STATES,
               "An earlier exact evaluation job has a mismatched UID or unknown state")
        validate_resources(job, name=record["name"])
        if job["state"] in admission.ACTIVE_STATES:
            active.append(dict(name=job["name"], uid=job["uid"], state=job["state"], gpu_count=1))
    ensure(len(active) < MAX_ACTIVE, "Four exact panel jobs are already active; no new intent")
    return anchor, active


BOOTSTRAP = r'''
import base64,hashlib,json,os,runpy,signal,socket,sys,time,urllib.error,urllib.request,zlib
from pathlib import Path
os.umask(0o077);sys.dont_write_bytecode=True
for key in list(os.environ):
 if key.upper() in {'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY'}:os.environ.pop(key)
root=Path(CONFIG['bootstrap_root']);root.mkdir(parents=True,mode=0o700,exist_ok=False)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
report={'state':'bootstrap_started','external_agent':False}
class EvalDeadline(BaseException):pass
def expire(*_):raise EvalDeadline('Finite evaluation deadline reached')
def save():
 temporary=root/'report.json.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(root/'report.json')
def upload():
 for attempt in range(4):
  try:
   with opener.open(urllib.request.Request(CONFIG['report_put'],data=json.dumps(report).encode(),method='PUT'),timeout=30):pass
   return
  except urllib.error.URLError as error:
   if not isinstance(error.reason,socket.gaierror) or error.reason.errno!=socket.EAI_AGAIN or attempt==3:raise
   time.sleep(.5)
signal.signal(signal.SIGALRM,expire);signal.signal(signal.SIGTERM,expire);signal.alarm(13800)
try:
 save();source=zlib.decompress(base64.b64decode(CONFIG['source_inline'],validate=True))
 if hashlib.sha256(source).hexdigest()!=CONFIG['source_sha256']:raise ValueError('Decoded native worker changed')
 path=root/'evaluation_worker.py'
 with path.open('xb') as stream:stream.write(source)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=CONFIG['source_sha256']:raise ValueError('Written native worker changed')
 namespace=runpy.run_path(str(path),run_name='codex_native_worker')
 report=namespace['run'](CONFIG['payload'])
except BaseException as error:
 report={'state':'bootstrap_failed','external_agent':False,'error_type':type(error).__name__}
finally:
 signal.alarm(0);save()
 try:upload()
 except Exception as error:report['report_transport_error']=type(error).__name__;save()
print(json.dumps({key:report.get(key) for key in ('state','external_agent','error_type')}),flush=True)
raise SystemExit(0 if report.get('state')=='complete' and report.get('external_agent') is False else 1)
'''


def startup(source, payload, name, report_put):
    encoded = base64.b64encode(zlib.compress(source, 9)).decode()
    ensure(len(encoded) <= 1024**2, "Native worker source is too large")
    config = dict(bootstrap_root=cloud.AFS + "/native-evaluation-bootstrap/" + name,
                  source_inline=encoded, source_sha256=DRIVER_SHA, payload=payload, report_put=report_put)
    code = "CONFIG=" + repr(config) + "\n" + BOOTSTRAP;compile(code, "native_evaluation_bootstrap", "exec")
    return "timeout --signal=TERM --kill-after=30 14400 " + shlex.quote(common.PYTHON) + " -c " + shlex.quote(code)


def submit(training_report, training_receipt, checkpoint_manifest, receipt_path, *, shard, name):
    payload, source, identity, proof = prepare_inputs(training_report, training_receipt, checkpoint_manifest, shard=shard, name=name)
    receipt = Path(receipt_path).resolve()
    ensure(not receipt.is_relative_to(INTENTS.resolve()), "External receipt must differ from canonical admission files")
    ensure(not receipt.exists() and not receipt.is_symlink(), "Receipt must be fresh")
    INTENTS.mkdir(parents=True, exist_ok=True);receipt.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(INTENTS / "panel.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        anchor, panel_active = check_panel(identity, shard, name)
        template = cloud.describe(TEMPLATE);validate_resources(template, template=True)
        preflight = validate_capacity(admission.query_standard_capacity(cloud.CLI, cloud.environment()), panel_active)
        key = cloud.PREFIX + "jobs/" + name + "/evaluation_worker.py"
        record = dict(kind="codex_native_eval_submission_v1", state="intent", name=name, shard_id=shard,
            identity=identity, payload=payload, source_sha256=DRIVER_SHA, source_key=key,
            report_key=cloud.PREFIX + "jobs/" + name + "/report.json", created_at=common.now().isoformat(),
            gpu_count=1, quota_type="reserved", max_seconds=14400, worker_seconds=13800,
            capacity_preflight=preflight, panel_active_before=panel_active, submission_attempted=False,
            container_image=common.IMAGE, python_binary=common.PYTHON, template=dict(name=TEMPLATE,uid=TEMPLATE_UID), **proof)
        intent = INTENTS / ("intent-" + shard + ".json")
        # An interrupted two-file admission is deliberately not auto-repaired.
        anchor = {**anchor, "admitted":anchor["admitted"] + [shard]}
        if (INTENTS / "panel.json").exists():common.persist(INTENTS / "panel.json", anchor)
        else:common.write_new(INTENTS / "panel.json", anchor)
        common.write_new(intent, record);common.write_new(receipt, record)
        def publish():common.persist(intent, record);common.persist(receipt, record)
        store = cloud.storage();store.put_object(Bucket=cloud.BUCKET, Key=key, Body=source)
        ensure(common.sha(store.get_object(Bucket=cloud.BUCKET, Key=key)["Body"].read()) == DRIVER_SHA,
               "Uploaded native worker differs on readback")
        report_put = store.generate_presigned_url("put_object", Params=dict(Bucket=cloud.BUCKET, Key=record["report_key"]), ExpiresIn=86400)
        command = startup(source, payload, name, report_put)
        try:
            capacity = validate_capacity(admission.query_standard_capacity(cloud.CLI, cloud.environment()), panel_active)
        except Exception as error:
            record.update(state="not_submitted", not_submitted_stage="final_capacity_check",
                          error_type=type(error).__name__)
            publish()
            raise ValueError("Final standard capacity check failed; intent retained, no submission attempted") from None
        record["admission"] = capacity
        args = [cloud.CLI,"acp","jobs","copy","--copy-job-name="+TEMPLATE,"--worker-spec="+SKU,
                "--env=NVIDIA_DRIVER_CAPABILITIES:all","--workspace-name=share-space","--name="+name,"--job-name="+name,
                "--worker-nodes=1","--quota-type=reserved","--storage-mount="+common.MOUNT,
                "--container-image-url="+common.IMAGE,"--command="+command]
        record.update(state="submission_unknown", submission_attempted=True);publish()
        try:result = subprocess.run(args, env=cloud.environment(), capture_output=True, text=True, timeout=90)
        except (subprocess.TimeoutExpired, OSError):
            raise RuntimeError("Submission outcome unknown; inspect the saved exact name, do not resubmit") from None
        record["cli_exit_code"]=result.returncode;publish()
        if result.returncode != 0:raise RuntimeError("Submission outcome unknown; inspect the saved exact name, do not resubmit")
        try:
            job=cloud.describe(name);validate_resources(job,name=name)
        except Exception:raise RuntimeError("Submission needs exact job readback; intent blocks retry") from None
        record.update(state="submitted",uid=job["uid"],platform_state=job["state"]);publish()
        return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("training-report","training-receipt","checkpoint-manifest","receipt"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--shard",choices=worker.SHARD_IDS,required=True)
    parser.add_argument("--name",required=True)
    args=parser.parse_args(argv)
    result=submit(args.training_report,args.training_receipt,args.checkpoint_manifest,args.receipt,shard=args.shard,name=args.name)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
