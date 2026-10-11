import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import submit_codex_training_validation as submitter


def ref(path):
    return dict(path=str(path), size=123, sha256="a" * 64)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(submitter, "INTENT", tmp_path / "canonical/intent.json")
    afs = Path(submitter.cloud.AFS)
    payload = dict(kind=submitter.worker.KIND, code=str(afs / "code-v8"), runtime=submitter.RUNTIME,
                   report_root=str(afs / "training-report"), destination=str(afs / "trained-model"),
                   training_plan=ref(afs / "training-run/training_plan.json"),
                   code_ready=ref(afs / "code-v8/_CODEX_PILOT_CODE_READY"),
                   driver_sha256=submitter.DRIVER_SHA, wall_timeout_seconds=13800)
    report = dict(state="complete", ready_for_training=True, training_payload=payload,
                  training_plan=payload["training_plan"], code_ready=payload["code_ready"])
    payload_path=tmp_path / "payload.json";report_path=tmp_path / "preparation.json"
    payload_path.write_text(json.dumps(payload));report_path.write_text(json.dumps(report))
    return payload_path, report_path, tmp_path / "receipt.json", payload, report


def template():
    return json.loads((submitter.OUT / "training_4gpu_template_verified.json").read_text())


def capacity():
    return dict(owner=submitter.cloud.BUCKET, quota_type="reserved", fresh_query=True,
                observed_at=submitter.now().isoformat(), standard_limit=4, used_gpus=0,
                active_reserved=[], queried_states=sorted(submitter.admission.ACTIVE_STATES), pages_read=10)


class Store:
    def __init__(self):self.values={};self.puts=0;self.corrupt=False
    def put_object(self, *, Bucket, Key, Body):
        assert Bucket == submitter.cloud.BUCKET
        self.values[Key]=Body;self.puts+=1
    def get_object(self, *, Bucket, Key):
        return {"Body":io.BytesIO(b"changed" if self.corrupt else self.values[Key])}
    def generate_presigned_url(self, *args, **kwargs):return "https://example.invalid/?signature=PRIVATE"


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    store=Store();calls=[];described=[]
    def describe(name):
        described.append(name);job=template()
        if name != submitter.TEMPLATE:
            job.update(name=name, uid="actual-job-uid", state="QUEUEING")
            job["roles"][0]["image_path"]=submitter.IMAGE
        return job
    def run(args, **kwargs):
        calls.append((args,kwargs));return subprocess.CompletedProcess(args, 0, stdout="do not print", stderr="signed URL")
    monkeypatch.setattr(submitter.cloud, "describe", describe)
    monkeypatch.setattr(submitter.cloud, "storage", lambda:store)
    monkeypatch.setattr(submitter.admission, "query_standard_capacity", lambda *args:capacity())
    monkeypatch.setattr(submitter.subprocess, "run", run)
    return store,calls,described


def submit(inputs):
    return submitter.submit(*inputs[:3], name="behavior-codex-training-pilot-test")


def test_success_exact_four_gpu_template_and_finite_command(inputs, cloud):
    record=submit(inputs);store,calls,described=cloud
    assert len(calls)==1 and store.puts==1
    args,kwargs=calls[0]
    for flag in ("--copy-job-name=pt-4ewqyd0e", "--worker-spec=n12lp.nn.i10a.4", "--worker-nodes=1",
                 "--quota-type=reserved", "--container-image-url="+submitter.IMAGE, "--storage-mount="+submitter.MOUNT):
        assert flag in args
    command=next(arg.removeprefix("--command=") for arg in args if arg.startswith("--command="))
    assert command.startswith("timeout --signal=TERM --kill-after=30 14400 "+submitter.PYTHON+" -c ")
    assert kwargs["timeout"]==90 and kwargs["capture_output"] is True
    assert record["gpu_count"]==4 and record["worker_seconds"]==13800 and record["state"]=="submitted"
    assert described==[submitter.TEMPLATE,record["name"]]
    assert "signature" not in json.dumps(record)
    assert json.loads(submitter.INTENT.read_text())==record==json.loads(inputs[2].read_text())


@pytest.mark.parametrize("patch", [
    {"state":"failed"},{"ready_for_training":False},{"training_payload":{}},{"code_ready":{}}, {"training_plan":{}}
])
def test_preparation_not_ready_before_cloud(inputs, cloud, patch):
    report=inputs[4];report.update(patch);inputs[1].write_text(json.dumps(report))
    with pytest.raises(ValueError):submit(inputs)
    assert cloud[0].puts==0 and cloud[1]==[] and cloud[2]==[]
    assert not submitter.INTENT.exists()


@pytest.mark.parametrize("field,value", [
    ("driver_sha256","b"*64),("runtime","/elsewhere/runtime"),("wall_timeout_seconds",14400),
    ("code","/unrelated/code"),("report_root",submitter.cloud.AFS+"/../other/report"),
    ("teacher", "codex"), ("devices",4), ("resume",False),
])
def test_payload_invariants_before_cloud(inputs, cloud, field, value):
    payload=inputs[3];payload[field]=value
    inputs[0].write_text(json.dumps(payload));inputs[1].write_text(json.dumps(inputs[4]))
    with pytest.raises(ValueError):submit(inputs)
    assert cloud[1]==[] and cloud[2]==[]


@pytest.mark.parametrize("patch", [
    {"used_gpus":1},{"used_gpus":False},{"fresh_query":False},{"owner":"other"},
    {"observed_at":(datetime.now(timezone.utc)-timedelta(seconds=120)).isoformat()},
    {"observed_at":(datetime.now(timezone.utc)+timedelta(seconds=120)).isoformat()},
    {"queried_states":["RUNNING"]},{"pages_read":9},{"active_reserved":[{"gpu_count":1}]},
])
def test_owner_wide_admission_fail_closed(inputs, cloud, monkeypatch, patch):
    value=capacity();value.update(patch)
    monkeypatch.setattr(submitter.admission,"query_standard_capacity",lambda *args:value)
    with pytest.raises(ValueError):submit(inputs)
    assert cloud[1]==[] and submitter.INTENT.exists()


@pytest.mark.parametrize("field,value", [("name","other"),("uid","other"),("state","RUNNING")])
def test_exact_template_identity(inputs, cloud, monkeypatch, field, value):
    job=template();job[field]=value
    monkeypatch.setattr(submitter.cloud,"describe",lambda name:job)
    with pytest.raises(ValueError):submit(inputs)
    assert cloud[1]==[] and not submitter.INTENT.exists()


@pytest.mark.parametrize("field,value", [("cpu","55"),("memory","239"),("nvidia.com/gpu","1"),("device","1")])
def test_template_actual_resource_shape(field,value):
    job=template();job["roles"][0]["resource_spec"][0]["limits"][field]=value
    with pytest.raises(ValueError):submitter.validate_resources(job,template=True)


def test_source_readback_mismatch_retains_intent(inputs, cloud):
    cloud[0].corrupt=True
    with pytest.raises(ValueError,match="readback"):submit(inputs)
    assert cloud[1]==[]
    with pytest.raises(FileExistsError):submit(inputs)
    assert cloud[0].puts==1


@pytest.mark.parametrize("outcome", ["timeout", "nonzero", "readback"])
def test_unknown_never_retries_even_new_receipt_name(inputs, cloud, monkeypatch, outcome):
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        assert json.loads(submitter.INTENT.read_text())["state"]=="submission_unknown"
        if outcome=="timeout":raise subprocess.TimeoutExpired(args,90)
        return subprocess.CompletedProcess(args,2 if outcome=="nonzero" else 0)
    monkeypatch.setattr(submitter.subprocess,"run",run)
    if outcome=="readback":
        original=submitter.cloud.describe
        def describe(name):
            job=original(name)
            if name!=submitter.TEMPLATE:job["roles"][0]["image_path"]="wrong"
            return job
        monkeypatch.setattr(submitter.cloud,"describe",describe)
    with pytest.raises(RuntimeError):submit(inputs)
    assert json.loads(submitter.INTENT.read_text())["state"]=="submission_unknown"
    with pytest.raises(FileExistsError):
        submitter.submit(*inputs[:2],inputs[2].with_name("different.json"),name="behavior-codex-training-another")
    assert len(calls)==1


def test_success_also_cannot_replay(inputs,cloud):
    submit(inputs)
    with pytest.raises(FileExistsError):submit(inputs)
    assert len(cloud[1])==1


def test_bootstrap_uses_real_hash_verified_file_and_reports_worker_failure(tmp_path):
    # Exercise the generated transport offline with a deliberately tiny worker.
    source=b"from pathlib import Path\nimport hashlib\ndef run(payload):\n assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==payload['driver_sha256']\n return {'state':'failed','ready_for_evaluation':False,'error_type':'ExpectedFixtureFailure'}\n"
    import base64,zlib
    root=tmp_path/"bootstrap"
    config=dict(bootstrap_root=str(root),source_inline=base64.b64encode(zlib.compress(source)).decode(),
                source_sha256=hashlib.sha256(source).hexdigest(),payload={"driver_sha256":hashlib.sha256(source).hexdigest()},
                report_put="unused")
    script="CONFIG="+repr(config)+"\n"+submitter.BOOTSTRAP
    # No HTTP request: the only outbound operation is replaced for this roundtrip.
    script=script.replace("try:upload()", "try:pass")
    result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True,timeout=10)
    assert result.returncode==1
    report=json.loads((root/"report.json").read_text())
    assert report["error_type"]=="ExpectedFixtureFailure"
    assert (root/"training_worker.py").read_bytes()==source
    assert (root.stat().st_mode & 0o777)==0o700
    assert ((root/"training_worker.py").stat().st_mode & 0o777)==0o600


def test_bootstrap_source_mismatch_never_executes(tmp_path):
    import base64,zlib
    root=tmp_path/"bootstrap"
    config=dict(bootstrap_root=str(root),source_inline=base64.b64encode(zlib.compress(b"raise RuntimeError('must not run')")).decode(),
                source_sha256="a"*64,payload={},report_put="unused")
    script=("CONFIG="+repr(config)+"\n"+submitter.BOOTSTRAP).replace("try:upload()", "try:pass")
    result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True,timeout=10)
    assert result.returncode==1 and not (root/"training_worker.py").exists()
    assert json.loads((root/"report.json").read_text())["error_type"]=="ValueError"


@pytest.mark.parametrize("fault", [None, "missing", "changed", "too_large", "wrong_path"])
def test_success_artifact_transport_is_bounded_original_bytes(tmp_path, fault):
    import base64,zlib
    training=tmp_path/"training";training.mkdir();checkpoint=tmp_path/"inference";checkpoint.mkdir()
    manifest=checkpoint/"SHA256SUMS.inference";manifest.write_bytes(b"a"*64+b"  params/_METADATA\n")
    def descriptor(path):return dict(path=str(path),size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_ref=descriptor(manifest)
    if fault=="wrong_path":manifest_ref["path"]=str(tmp_path/"do-not-read")
    if fault=="too_large":manifest_ref["size"]=2*1024**2
    receipt=training/"training_receipt.json"
    receipt.write_text(json.dumps({"checkpoint_manifest":manifest_ref},indent=2)+"\n")
    receipt_raw=receipt.read_bytes();receipt_ref=descriptor(receipt)
    if fault=="changed":manifest.write_text("changed")
    if fault=="missing":receipt.unlink()
    source=b"def run(payload):\n return payload['fixture_report']\n"
    root=tmp_path/"bootstrap"
    payload=dict(training_plan={"path":str(training/"training_plan.json")},destination=str(checkpoint),
                 fixture_report=dict(state="complete",ready_for_evaluation=True,training_receipt=receipt_ref))
    config=dict(bootstrap_root=str(root),source_inline=base64.b64encode(zlib.compress(source)).decode(),
                source_sha256=hashlib.sha256(source).hexdigest(),payload=payload,report_put="unused")
    script=("CONFIG="+repr(config)+"\n"+submitter.BOOTSTRAP).replace("try:upload()","try:pass")
    result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True,timeout=10)
    report=json.loads((root/"report.json").read_text())
    # Transfer failure preserves the completed training fact, but cannot claim
    # that the two artifacts reached the client successfully.
    assert result.returncode==0 and report["state"]=="complete"
    if fault is None:
        assert report["bootstrap_artifacts_state"]=="verified"
        artifacts=report["bootstrap_artifacts"]
        assert [row["name"] for row in artifacts]==["training_receipt.json","SHA256SUMS.inference"]
        assert base64.b64decode(artifacts[0]["base64"],validate=True)==receipt_raw
        assert base64.b64decode(artifacts[1]["base64"],validate=True)==manifest.read_bytes()
        assert sum(row["ref"]["size"] for row in artifacts)<=2*1024**2
    else:
        assert report["bootstrap_artifacts_state"]=="failed" and "artifact_transport_failed" in report
        assert "bootstrap_artifacts" not in report
    assert "base64" not in result.stdout and "bootstrap_artifacts" not in payload["fixture_report"]
