"""Finite, allowlisted native-evaluation transfer and offline paired comparison.

No evaluation, training, model calls or cloud-job submission. The generated CPU
worker verifies the five frozen shard receipts before uploading their exact bytes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import stat
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

AFS = "/mnt/afs/260010165/behavior_1k/codex_pilot_20261011"
BUCKET = "260010165"
PREFIX = "behavior1k/codex-pilot-20261011/"
ENDPOINT = "aoss.cn-sh-01b.sensecoreapi-oss.cn"
SHARDS = ("radio-pilot", "a000", "a001", "a088", "a089")
MAX_BYTES = 256 * 1024**2
MAX_SECONDS = 600
KIND = "codex_pilot_result_transfer_v1"
HEX = re.compile(r"[0-9a-f]{64}")


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def file_ref(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=sha(path), size=path.stat().st_size)


def ref_shape(ref, *, remote=False):
    ensure(isinstance(ref, dict) and isinstance(ref.get("path"), str)
           and Path(ref["path"]).is_absolute() and ".." not in Path(ref["path"]).parts
           and type(ref.get("size")) is int and 0 < ref["size"] <= MAX_BYTES
           and HEX.fullmatch(str(ref.get("sha256"))), "Invalid bounded pinned file reference")
    if remote:
        ensure(Path(ref["path"]).is_relative_to(AFS), "Transfer input is outside this experiment")


def checked_file(ref):
    ref_shape(ref)
    path = Path(ref["path"])
    ensure(not any(p.is_symlink() for p in (path, *path.parents)), "Transfer inputs cannot use symlinks")
    ensure(path.is_file() and stat.S_ISREG(path.stat().st_mode)
           and path.stat().st_size == ref["size"] and sha(path) == ref["sha256"], "Pinned artifact bytes differ")
    return path


def identity(ref):
    return {key: ref[key] for key in ("sha256", "size")}


def write_new(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write("\n")


def private_url(value):
    ensure(isinstance(value, str) and 1 <= len(value) <= 8192 and not any(c in value for c in "\r\n"), "Expected one private AOSS URL")
    parsed = urllib.parse.urlsplit(value);host = parsed.hostname or ""
    query = urllib.parse.parse_qs(parsed.query)
    ensure(parsed.scheme == "https" and not parsed.username and not parsed.password and not parsed.fragment
           and parsed.port in (None, 443) and host in {ENDPOINT, BUCKET + "." + ENDPOINT}
           and (query.get("X-Amz-Signature") or query.get("Signature")), "Expected an HTTPS presigned AOSS URL")
    prefix = "/" + BUCKET + "/" if host == ENDPOINT else "/"
    path = urllib.parse.unquote(parsed.path)
    ensure(path.startswith(prefix), "AOSS URL must address this user's bucket")
    key = path[len(prefix):]
    ensure(key.startswith(PREFIX) and key.endswith(".tar")
           and all(part not in {"", ".", ".."} for part in key.split("/"))
           and re.fullmatch(r"[A-Za-z0-9_./-]+", key), "Archive key must stay in the private experiment prefix")
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Archive transport redirects are forbidden")


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def upload_archive(path, url):
    private_url(url)
    for attempt in range(4):
        try:
            with Path(path).open("rb") as stream:
                request = urllib.request.Request(url, data=stream, method="PUT", headers={
                    "Content-Length": str(Path(path).stat().st_size), "Content-Type": "application/x-tar"})
                with opener().open(request, timeout=120) as response:
                    ensure(200 <= response.status < 300, "Archive upload was not accepted")
            return
        except urllib.error.URLError as error:
            if not isinstance(error.reason, socket.gaierror) or error.reason.errno != socket.EAI_AGAIN or attempt == 3:
                raise RuntimeError("Archive upload failed: " + type(error).__name__) from None
            time.sleep(.5)


def load_module(ref, name):
    path = checked_file(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_config(config):
    refs = config.get("run_receipts")
    ensure(isinstance(refs, list) and len(refs) == 5, "Exactly five ordered run receipts are required")
    for ref in refs + [config[key] for key in ("training_receipt", "evaluation_worker", "report_helper")]:
        ref_shape(ref, remote=True)
    ensure(len({ref["path"] for ref in refs}) == 5, "Run receipt references must be unique")
    root = Path(config["report_root"])
    ensure(root.is_absolute() and root.parent == Path(AFS) and root.name.startswith("result-transfer-")
           and re.fullmatch(r"[A-Za-z0-9_.-]+", root.name), "Use a fresh independent transfer directory")
    for ref in refs + [config[key] for key in ("training_receipt", "evaluation_worker", "report_helper")]:
        path = Path(ref["path"])
        ensure(not (root == path or root in path.parents or path in root.parents), "Transfer output overlaps immutable input")
    private_url(config["archive_put_url"])


def artifact_refs(panel, training):
    """The entire archive allowlist; never enumerate an evidence directory."""
    ensure([load(checked_file(ref))["shard_id"] for ref in panel["run_receipts"]] == list(SHARDS), "Shard order differs")
    refs = {"training/training_receipt.json": panel["checkpoint"]["training_receipt"],
            "training/SHA256SUMS.inference": training["checkpoint_manifest"]}
    refs.update({"runs/" + shard + "/run_receipt.json": ref for shard, ref in zip(SHARDS, panel["run_receipts"])})
    for row in panel["results"]:
        name = f"{row['task']}_{row['instance_id']}_0"
        refs["results/" + name + ".json"] = row["json"]
        if row.get("video"):
            ensure((row["task"], row["instance_id"]) == ("turning_on_radio", 301), "Unexpected video outside the fixed pilot")
            refs["videos/" + name + ".mp4"] = row["video"]
        ensure(not row.get("provenance_receipts"), "Candidate has unrecognized extra evidence")
    ensure(len(refs) == 28 and len({ref["path"] for ref in refs.values()}) == 28, "Exactly 28 distinct raw artifacts are required")
    return refs


def pack_archive(path, refs):
    ensure(len(refs) == 29 and sum(ref["size"] for ref in refs.values()) <= MAX_BYTES, "Transfer archive exceeds bounded inventory")
    with tarfile.open(path, "x", format=tarfile.USTAR_FORMAT) as archive:
        for name, ref in refs.items():
            source = checked_file(ref)
            with source.open("rb") as stream:raw = stream.read(ref["size"] + 1)
            ensure(len(raw) == ref["size"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "Artifact changed while packaging")
            info = tarfile.TarInfo(name);info.size = len(raw);info.mode = 0o600
            archive.addfile(info, io.BytesIO(raw))
    ensure(Path(path).stat().st_size <= MAX_BYTES, "Tar headers exceed the archive byte ceiling")
    return dict(sha256=sha(path), size=Path(path).stat().st_size)


class TransferDeadline(BaseException):
    pass


def transfer(config):
    os.umask(0o077)
    import sys
    sys.dont_write_bytecode = True
    validate_config(config)
    root = Path(config["report_root"])
    ensure(not any(p.is_symlink() for p in (root, *root.parents)), "Transfer output cannot use symlinks")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(kind=KIND, state="verifying", gpu_count=0, model_calls=0, max_seconds=MAX_SECONDS,
                  report_root=str(root), run_receipts=config["run_receipts"], training_receipt=config["training_receipt"],
                  evaluation_worker=config["evaluation_worker"], report_helper=config["report_helper"])
    started = time.monotonic()
    def publish():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        temporary = root / "report.tmp";temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(root / "report.json")
    def stop(signum, frame):
        raise TransferDeadline("Finite result transfer deadline reached")
    old_alarm = signal.signal(signal.SIGALRM, stop);old_term = signal.signal(signal.SIGTERM, stop)
    old_timer = signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        publish()
        for ref in config["run_receipts"] + [config["training_receipt"], config["report_helper"]]:checked_file(ref)
        evaluation = load_module(config["evaluation_worker"], "frozen_native_evaluation")
        ensure(tuple(evaluation.SHARD_IDS) == SHARDS, "Frozen evaluator shard schedule differs")
        training = load(checked_file(config["training_receipt"]))
        for ref in config["run_receipts"]:
            receipt = load(checked_file(ref))
            ensure(receipt.get("driver_sha256") == config["evaluation_worker"]["sha256"]
                   and receipt.get("report_helper") == config["report_helper"]
                   and receipt.get("training_receipt") == config["training_receipt"], "Run receipt does not bind the pinned helpers and training")
            for row in receipt.get("results", []):
                for key in ("json", "video"):
                    if row.get(key):ref_shape(row[key], remote=True);checked_file(row[key])
        ref_shape(training["checkpoint_manifest"], remote=True);checked_file(training["checkpoint_manifest"])
        payload = dict(checkpoint_id=training["checkpoint_id"], training_receipt=config["training_receipt"], report_helper=config["report_helper"])
        panel = evaluation.assemble_panel(payload, config["run_receipts"])
        refs = artifact_refs(panel, training)
        for ref in refs.values():ref_shape(ref, remote=True);checked_file(ref)
        panel_path = root / "candidate_remote_panel.json";write_new(panel_path, panel)
        refs["candidate_remote_panel.json"] = file_ref(panel_path)
        report.update(state="packaging", members=refs, candidate_remote_panel=refs["candidate_remote_panel.json"]);publish()
        archive = root / "native_eval_results.tar"
        report["archive"] = pack_archive(archive, refs);report.update(state="uploading", archive_path=str(archive));publish()
        upload_archive(archive, config["archive_put_url"])
        report.update(state="complete", uploaded=True, archive_key=private_url(config["archive_put_url"]), artifact_count=len(refs))
    except BaseException as error:
        report.update(state="failed", uploaded=False, error_type=type(error).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0);publish()
        signal.signal(signal.SIGALRM, old_alarm);signal.signal(signal.SIGTERM, old_term)
        if old_timer[0] > 0:
            remaining = old_timer[0] - (time.monotonic() - started)
            if remaining > 0:signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])
    return report


# LOCAL BUILDER AND CLIENT: never embedded into the CPU worker.
ROOT = Path(__file__).resolve().parents[1]


def build_transfer_source(output, *, run_receipts, training_receipt, evaluation_worker, report_helper,
                          report_root, archive_put_url=None, private_url_file=None):
    ensure((archive_put_url is None) != (private_url_file is None), "Supply one private archive PUT URL or 0600 URL file")
    if private_url_file is not None:
        path = Path(private_url_file)
        ensure(not path.is_symlink() and path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o600, "URL file must be private 0600")
        archive_put_url = path.read_text().strip()
    config = dict(run_receipts=run_receipts, training_receipt=training_receipt, evaluation_worker=evaluation_worker,
                  report_helper=report_helper, report_root=report_root, archive_put_url=archive_put_url)
    validate_config(config)
    for ref, local in ((evaluation_worker, ROOT / "sco/run_codex_pilot_evaluation.py"),
                       (report_helper, ROOT / "scripts/evaluate_codex_pilot.py")):
        ensure(identity(ref) == identity(file_ref(local)), "Frozen evaluation/helper bytes differ from the reviewed local implementation")
    source = Path(__file__).read_text().split("\n# LOCAL BUILDER AND CLIENT:", 1)[0]
    source += "\nBOOTSTRAP_REPORT = transfer(" + repr(config) + ")\n"
    compile(source, "codex_result_transfer", "exec")
    target = Path(output).absolute();receipt_path = target.with_suffix(".receipt.json")
    ensure(target != receipt_path and not any(p.exists() or p.is_symlink() for p in (target, receipt_path)), "Transfer source and receipt must be fresh")
    target.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:stream.write(source)
    result = dict(kind="codex_result_transfer_source_v1", source=file_ref(target), gpu_count=0,
                  max_seconds=MAX_SECONDS, recommended_cloud_max_seconds=660, max_archive_bytes=MAX_BYTES,
                  report_root=report_root, archive_key=private_url(archive_put_url), actual_job_submitted=False)
    write_new(receipt_path, result)
    return result


def archive_members(report):
    ensure(report.get("kind") == KIND and report.get("state") == "complete" and report.get("uploaded") is True
           and report.get("gpu_count") == 0 and report.get("model_calls") == 0, "Transfer must have a complete verified upload receipt")
    members = report.get("members", {})
    names = {"candidate_remote_panel.json", "training/training_receipt.json", "training/SHA256SUMS.inference",
             "videos/turning_on_radio_301_0.mp4"}
    names.update("runs/" + shard + "/run_receipt.json" for shard in SHARDS)
    names.update(f"results/{task}_{instance}_0.json" for task in ("turning_on_radio", "putting_shoes_on_rack") for instance in range(301, 311))
    ensure(isinstance(members, dict) and set(members) == names and report.get("artifact_count") == 29, "Archive inventory differs from the fixed 29 files")
    for ref in members.values():ref_shape(ref, remote=True)
    ensure(sum(ref["size"] for ref in members.values()) <= MAX_BYTES, "Unpacked evidence exceeds the byte ceiling")
    ensure(report.get("archive", {}).get("size", 0) > 0 and report["archive"]["size"] <= MAX_BYTES
           and HEX.fullmatch(str(report["archive"].get("sha256"))), "Missing archive hash and bounded size")
    return members


def download_archive(url, report, output):
    archive_members(report);private_url(url)
    ensure(private_url(url) == report.get("archive_key"), "Download key differs from the completed upload")
    target = Path(output)
    ensure(not target.exists() and not target.is_symlink(), "Download target must be fresh")
    try:
        with opener().open(url, timeout=120) as response, os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            remaining = report["archive"]["size"]
            while True:
                block = response.read(min(1024**2, remaining + 1))
                if not block:break
                ensure(len(block) <= remaining, "Download exceeds the pinned archive size")
                stream.write(block);remaining -= len(block)
            ensure(remaining == 0, "Download ended before the pinned archive size")
        ensure(sha(target) == report["archive"]["sha256"], "Downloaded archive SHA256 differs")
    except Exception as error:
        raise RuntimeError("Archive download failed: " + type(error).__name__) from None
    return file_ref(target)


def unpack_and_compare(archive_path, transfer_report, baseline_path, output_dir, *, report_helper=None):
    report = load(transfer_report) if not isinstance(transfer_report, dict) else transfer_report
    members = archive_members(report)
    archive = Path(archive_path);checked_file({"path": str(archive.absolute()), **identity(report["archive"])})
    helper_ref = report_helper or file_ref(ROOT / "scripts/evaluate_codex_pilot.py")
    ensure(identity(helper_ref) == identity(report["report_helper"]), "Local comparison helper differs from the frozen remote verifier")
    helper = load_module(helper_ref, "local_native_panel_verifier")
    output = Path(output_dir).absolute()
    ensure(not any(p.is_symlink() for p in (output, *output.parents)) and not output.exists(), "Extraction directory must be fresh and have no symlinks")
    # Validate every header before creating output; never use extract/extractall.
    with tarfile.open(archive, "r:") as packed:
        headers = []
        for header in packed:
            ensure(len(headers) < 29, "Tar exceeds the fixed file count")
            headers.append(header)
        ensure(len(headers) == 29 and len({h.name for h in headers}) == 29
               and {h.name for h in headers} == set(members), "Tar contains unexpected, duplicate or missing paths")
        for header in headers:
            ensure(header.isfile() and not header.issym() and not header.islnk() and not header.pax_headers
                   and PurePosixPath(header.name).as_posix() == header.name
                   and header.size == members[header.name]["size"], "Tar member type or size differs")
        output.mkdir(parents=True, mode=0o700)
        for header in headers:
            raw = packed.extractfile(header).read(header.size + 1)
            ensure(len(raw) == header.size and hashlib.sha256(raw).hexdigest() == members[header.name]["sha256"], "Tar member bytes differ")
            target = output / header.name;target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:stream.write(raw)
    panel = load(output / "candidate_remote_panel.json")
    # Only panel paths change; every downloaded JSON receipt retains its bytes.
    def relocated(ref, name):
        ensure(ref == members[name], "Remote panel reference differs from the transferred inventory")
        return {**ref, "path": str(output / name)}
    panel["checkpoint"]["training_receipt"] = relocated(panel["checkpoint"]["training_receipt"], "training/training_receipt.json")
    training = load(output / "training/training_receipt.json")
    panel["checkpoint"]["checkpoint_manifest"] = relocated(training["checkpoint_manifest"], "training/SHA256SUMS.inference")
    ensure(len(panel["run_receipts"]) == 5, "Remote panel lacks five receipts")
    panel["run_receipts"] = [relocated(ref, "runs/" + shard + "/run_receipt.json") for shard, ref in zip(SHARDS, panel["run_receipts"])]
    for row in panel["results"]:
        name = f"{row['task']}_{row['instance_id']}_0"
        row["json"] = relocated(row["json"], "results/" + name + ".json")
        if row.get("video"):row["video"] = relocated(row["video"], "videos/" + name + ".mp4")
    candidate = output / "candidate_local_panel.json";write_new(candidate, panel)
    comparison = helper.compare_panels(baseline_path, candidate)
    write_new(output / "paired_report.json", comparison)
    return comparison
