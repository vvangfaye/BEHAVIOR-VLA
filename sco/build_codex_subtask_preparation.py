#!/usr/bin/env python3
"""Build an isolated selected-episode CPU preparation bundle without networking.

The public archive contains code, metadata and frozen labels only. Expiring OSS
URLs are read from an explicit private JSON input and embedded only in a 0600
worker driver. A caller uploads the archive, signs its GET URL and submits that
driver separately; this module never authenticates, transfers or submits jobs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "eval_results/codex_subtask_freeze_20261012"
AFS = "/mnt/afs/260010165/behavior_1k/codex_subtask_freeze_20261012"
BASE_CODE = "/mnt/afs/260010165/behavior_1k/code_snapshots/public-test-1688977-aoss-20261010"
BASE_SOURCE_SHA256 = "7f2a16738961dfb1ca5b95d280ee61450f996757c5514b8c144e86dce3c982d4"
RUNTIME_PYTHON = "/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-eval/bin/python"
SOURCE = "oss://traffic-model/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542"
EPISODE = 10020
TASK = "freeze_fruit"
TARGET_SUBTASK = "place on tupperware to countertop"
RAW = "/mnt/afs/260010165/behavior_1k/raw_full100_20261007_1308/2026-challenge-rawdata/task-0050/episode_00500330.hdf5"
RAW_SHA256 = "3daf2242bae84e20e389caaa0274497269c3969e0520e4d2918eaa07e781586d"
KIND = "codex_subtask_demo_cpu_preparation_v1"
CODE_FILES = (
    "sco/prepare_codex_original_data.py",
    "scripts/codex_collection_demonstrations.py",
    "scripts/codex_collection_trajectory.py",
    "scripts/subtask_trajectory.py",
    "scripts/stage_subtask_replay.py",
    "scripts/subtask_sample_panel.py",
    "scripts/strip_depth_from_info.py",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_new(path, raw, *, private=False):
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)


def write_json(path, value, *, private=False):
    write_new(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), private=private)


def source_inventory(metadata_root, selection_path, source_spec=None):
    """Resolve exact shared files and verify existing training-list/label bytes."""
    import pyarrow.parquet as pq
    sys.path.insert(0, str(ROOT / "scripts"))
    from stage_subtask_replay import required_files, metadata_inventory
    from codex_collection_demonstrations import _segments

    spec = source_spec or dict(episode_index=EPISODE, task_name=TASK, task_instance_id=33,
        raw_path=RAW, raw_sha256=RAW_SHA256, raw_episode_id=500330, task_index=50,
        subtask=TARGET_SUBTASK)
    episode, task, instance = (spec[key] for key in ("episode_index", "task_name", "task_instance_id"))
    raw, raw_sha, target = (spec[key] for key in ("raw_path", "raw_sha256", "subtask"))
    if (type(episode) is not int or episode < 0 or type(instance) is not int or not 0 <= instance <= 300
            or not raw.startswith("/mnt/afs/260010165/behavior_1k/") or ".." in Path(raw).parts
            or re.fullmatch(r"[0-9a-f]{64}", raw_sha) is None):
        raise ValueError("Invalid selected training demonstration source")

    selection = json.loads(Path(selection_path).read_text())
    sources = {}
    for key in ("segments", "retained_episode_list"):
        row = selection[key]
        path = Path(row["path"])
        if not path.is_absolute() or sha(path.read_bytes()) != row["sha256"]:
            raise ValueError("Frozen labels or retained episode list changed")
        sources[key] = dict(path=str(path), sha256=row["sha256"], bytes=path.stat().st_size)
    retained = {int(line) for line in Path(sources["retained_episode_list"]["path"]).read_text().split()}
    if episode not in retained:
        raise ValueError("Selected episode is not retained training data")
    metadata_root = Path(metadata_root)
    paths, selected = required_files(metadata_root, [episode], [])
    if (len(selected) != 1 or selected[0]["task_name"] != task or selected[0]["task_instance_id"] != instance
            or (source_spec is None and selected[0]["length"] != 12613)):
        raise ValueError("Selected episode identity differs from the fixed source")
    episode_rows = [row for path in (metadata_root / "meta/episodes").glob("chunk-*/*.parquet")
                    for row in pq.read_table(path).to_pylist() if row["episode_index"] == episode]
    if (len(episode_rows) != 1 or episode_rows[0].get("raw_episode_id") != spec["raw_episode_id"]
            or episode_rows[0]["task_index"] != spec["task_index"]
            or Path(raw).name != f"episode_{spec['raw_episode_id']:08d}.hdf5"
            or Path(raw).parent.name != f"task-{spec['task_index']:04d}"):
        raise ValueError("Selected episode does not map to the expected published raw source")
    labels = [row for row in pq.read_table(sources["segments"]["path"]).to_pylist()
              if row["episode_index"] == episode]
    segments = _segments(labels, selected[0]["length"])
    target_segments = [dict(segment_index=i, **row) for i, row in enumerate(segments)
                       if row["subtask"] == target]
    if not target_segments or len(target_segments) > 6:
        raise ValueError("Fixed target is missing or exceeds the bounded demonstration examples")
    result = dict(kind=KIND, source=SOURCE, episode=selected[0], raw_path=raw,
                  raw_sha256=raw_sha, retained_episode_verified=True,
                  metadata_root=str(metadata_root.resolve()), metadata=metadata_inventory(metadata_root),
                  labels=sources, target_subtask=target, target_segments=target_segments,
                  required_files=[dict(relative_path=path, source=SOURCE + "/" + path) for path in paths],
                  reader_verified=False, raw_success_verified=False)
    result["fingerprint"] = sha(canonical(result).encode())
    return result


def signed_files(path, inventory):
    """Validate private URL input while returning no URL-bearing diagnostics."""
    rows = json.loads(Path(path).read_text())
    expected = {row["relative_path"] for row in inventory["required_files"]}
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) for row in rows)):
        raise ValueError("Signed inventory must cover exactly the four selected files")
    result = []
    seen = set()
    for row in rows:
        relative = row.get("relative_path")
        size, etag, url = row.get("bytes"), row.get("etag"), row.get("url")
        if relative not in expected or relative in seen:
            raise ValueError("Signed inventory contains an extra, repeated or missing source")
        if type(size) is not int or size <= 0 or not isinstance(etag, str) or not etag.strip('"'):
            raise ValueError("Signed files require positive bytes and an ETag")
        parsed = urlsplit(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Signed files require HTTPS bearer URLs")
        if row.get("sha256") is not None and re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None:
            raise ValueError("Invalid optional frozen source checksum")
        kept = dict(relative_path=relative, bytes=size, etag=etag, url=url)
        if row.get("sha256") is not None:
            kept["sha256"] = row["sha256"]
        result.append(kept)
        seen.add(relative)
    if sum(row["bytes"] for row in result) > 8 * 1024**3:
        raise ValueError("Selected original downloads exceed the fixed 8 GiB budget")
    return sorted(result, key=lambda row: row["relative_path"])


def render_driver(plan, signed, *, bundle_url=None):
    if plan.get("kind") != KIND or plan.get("fingerprint") != sha(canonical({k: v for k, v in plan.items() if k != "fingerprint"}).encode()):
        raise ValueError("Preparation plan changed")
    if bundle_url is not None:
        parsed = urlsplit(bundle_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Bundle GET URL must be HTTPS")
    driver = '''import hashlib,io,json,os,shutil,signal,socket,subprocess,tarfile,threading,time,urllib.request,urllib.error
from pathlib import Path
def terminate_owned_driver(signum,frame):
 raise KeyboardInterrupt('Preparation driver terminated')
for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,terminate_owned_driver)
plan=PLAN_VALUE
signed=SIGNED_VALUE
bundle_url=BUNDLE_URL_VALUE or os.environ['CODEX_SUBTASK_PREPARATION_BUNDLE_GET']
bundle=Path(plan['bundle_root']);code=Path(plan['code_root']);output=Path(plan['output_root'])
base=Path(plan['base_code'])
def digest(path):
 value=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while block:=stream.read(8*1024**2):value.update(block)
 return value.hexdigest()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def open_remote(request,timeout=30):
 for attempt in range(4):
  try:return opener.open(request,timeout=timeout)
  except urllib.error.URLError as error:
   if not isinstance(error.reason,socket.gaierror) or error.reason.errno!=socket.EAI_AGAIN or attempt==3:raise
   threading.Event().wait(2**attempt)
def publish(value):
 if os.environ.get('CODEX_PILOT_REPORT_PUT'):
  request=urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'],data=json.dumps(value).encode(),method='PUT')
  try:
   with open_remote(request,timeout=30):pass
  except Exception as error:print(json.dumps({'report_upload_error_type':type(error).__name__}),flush=True)
assert digest(base/'_PUBLIC_TEST_SOURCE_READY')==plan['base_source_sha256'],'Base source differs'
assert not any(path.exists() for path in (bundle,code,output)),'Preparation destinations must be fresh'
publish(dict(state='preparing_original_episode_code',output_root=str(output)))
with open_remote(bundle_url,timeout=60) as response:
 raw=response.read(plan['bundle']['bytes']+1)
assert len(raw)==plan['bundle']['bytes'] and hashlib.sha256(raw).hexdigest()==plan['bundle']['sha256'],'Bundle bytes differ'
bundle.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for entry in archive.getmembers():
  relative=Path(entry.name)
  assert entry.isfile() and not relative.is_absolute() and '..' not in relative.parts,'Invalid bundle member'
  target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(archive.extractfile(entry).read())
manifest=json.loads((bundle/'manifest.json').read_text())
for name,item in manifest.items():
 path=bundle/name
 assert path.is_file() and path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],'Bundle member differs'
shutil.copytree(base,code,symlinks=True,copy_function=lambda source,dest:os.symlink(str(Path(source).resolve()),dest))
for path in (bundle/'code').rglob('*'):
 if path.is_file():
  target=code/path.relative_to(bundle/'code')
  assert target.parent.resolve().is_relative_to(code.resolve()),'Overlay parent escaped owned code'
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.is_symlink():target.unlink()
  elif target.exists():raise RuntimeError('Unexpected nonlinked overlay destination')
  shutil.copyfile(path,target)
ready=dict(kind='codex_subtask_preparation_code_ready_v1',base_source_sha256=plan['base_source_sha256'],
 files={name.removeprefix('code/'):item for name,item in manifest.items() if name.startswith('code/')})
(code/'_PREPARATION_CODE_READY.json').write_text(json.dumps(ready,indent=2)+'\\n')
payload=plan['payload'];payload['signed_files']=signed
private=bundle/'private_staging_payload.json'
descriptor=os.open(private,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(descriptor,'w') as stream:json.dump(payload,stream)
env=dict(os.environ,PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
 PYTHONPATH=str(code/'scripts')+':'+str(code/'openpi/src')+':'+str(code/'openpi/packages/openpi-client/src'))
started=time.monotonic()
with (bundle/'staging.log').open('x') as log:
 process=subprocess.Popen([plan['runtime_python'],str(code/'sco/prepare_codex_original_data.py'),'--payload',str(private)],
  env=env,stdout=log,stderr=subprocess.STDOUT)
 try:
  while True:
   if time.monotonic()-started>plan['payload']['max_seconds']+60:
    process.terminate()
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
    raise TimeoutError('Finite original preparation deadline exceeded')
   try:rc=process.wait(timeout=20)
   except subprocess.TimeoutExpired:rc=None
   status=json.loads((output/'status.json').read_text()) if (output/'status.json').is_file() else dict(state='starting_stager')
   status.update(worker_exit_code=rc,code=str(code),bundle=str(bundle),output_root=str(output))
   publish(status)
   if rc is not None:break
 finally:
  if process.poll() is None:
   process.terminate()
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
assert rc==0,'Selected episode staging failed; inspect owned staging.log'
receipt=json.loads((output/'_ORIGINAL_DATA_READY.json').read_text())
assert receipt['selected_episodes']==[plan['episode']] and receipt['original_reader_verified'] and receipt['raw_success_verified']
demo_path=output/'demonstrations'/str(plan['episode'])/'manifest.json'
demo=json.loads(demo_path.read_text())
assert demo.get('required_subtask')==plan['target_subtask']
assert any(row['subtask']==plan['target_subtask'] for row in demo['examples']),'Required local target is absent'
BOOTSTRAP_REPORT=dict(state='original_episode_verified',code=str(code),output_root=str(output),
 demonstration_manifest=str(demo_path),
 demonstration_fingerprint=demo['fingerprint'],bytes_staged=receipt['bytes_staged'])
publish(BOOTSTRAP_REPORT)
print(json.dumps(BOOTSTRAP_REPORT),flush=True)
'''
    values = dict(PLAN_VALUE=plan, SIGNED_VALUE=signed, BUNDLE_URL_VALUE=bundle_url)
    driver = re.sub(r"\b(PLAN_VALUE|SIGNED_VALUE|BUNDLE_URL_VALUE)\b", lambda match: repr(values[match.group()]), driver)
    compile(driver, "codex_subtask_preparation_driver", "exec")
    return driver


def build(output_dir, signed_path, *, metadata_root=None, selection_path=None, source_spec=None, experiment_root=AFS):
    metadata_root = Path(metadata_root or ROOT / "eval_results/codex_pilot_20261011/original_metadata")
    selection_path = Path(selection_path or ROOT.parent / "analysis/codex_teacher_pilot_20261011/demonstration_selection.json")
    inventory = source_inventory(metadata_root, selection_path, source_spec)
    episode = inventory["episode"]["episode_index"]
    raw_path, raw_sha = inventory["raw_path"], inventory["raw_sha256"]
    target_subtask = inventory["target_subtask"]
    if not experiment_root.startswith("/mnt/afs/260010165/behavior_1k/") or ".." in Path(experiment_root).parts:
        raise ValueError("Preparation must stay in the owned experiment directory")
    signed = signed_files(signed_path, inventory)
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Preparation build needs a fresh local output directory")
    output.mkdir(parents=True)
    write_json(output / "source_inventory.json", inventory)
    objects = [{k: v for k, v in row.items() if k != "url"} for row in signed]
    write_json(output / "signed_object_identities.json", objects)
    blobs = {"code/" + name: (ROOT / name).read_bytes() for name in CODE_FILES}
    blobs.update({"metadata/" + row["path"]: (metadata_root / row["path"]).read_bytes() for row in inventory["metadata"]})
    blobs["labels/segments.parquet"] = Path(inventory["labels"]["segments"]["path"]).read_bytes()
    blobs["labels/retained.txt"] = Path(inventory["labels"]["retained_episode_list"]["path"]).read_bytes()
    blobs["labels/selected_episodes.txt"] = f"{episode}\n".encode()
    manifest = {name: dict(bytes=len(raw), sha256=sha(raw)) for name, raw in blobs.items()}
    blobs["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, raw in sorted(blobs.items()):
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode, entry.mtime = len(raw), 0o644, 0
                archive.addfile(entry, io.BytesIO(raw))
    raw = buffer.getvalue()
    archive_path = output / "preparation_bundle.tar.gz"
    write_new(archive_path, raw)
    bundle = experiment_root + "/bundle-preparation-v1"
    code = experiment_root + "/code-preparation-v1"
    original = experiment_root + f"/original-{episode}"
    payload = dict(kind="codex_original_pilot_staging_v1", output_root=original, code_root=code,
                   episodes=[episode], metadata_source=bundle + "/metadata",
                   segments_path=bundle + "/labels/segments.parquet",
                   retained_episodes_path=bundle + "/labels/retained.txt",
                   segments_sha256=inventory["labels"]["segments"]["sha256"],
                   retained_episodes_sha256=inventory["labels"]["retained_episode_list"]["sha256"],
                   demonstrations=[dict(episode_index=episode, raw_path=raw_path)], raw_source_sha256={raw_path: raw_sha},
                   required_demonstration_subtask=target_subtask,
                   max_seconds=3600, max_download_bytes=8 * 1024**3, reserve_bytes=8 * 1024**3)
    plan = dict(kind=KIND, episode=episode, target_subtask=target_subtask,
                bundle_root=bundle, code_root=code, output_root=original,
                base_code=BASE_CODE, base_source_sha256=BASE_SOURCE_SHA256, runtime_python=RUNTIME_PYTHON,
                bundle=dict(local_path=str(archive_path), bytes=len(raw), sha256=sha(raw)),
                source_inventory_fingerprint=inventory["fingerprint"], signed_objects=objects,
                download_bytes=sum(row["bytes"] for row in signed), payload=payload)
    plan["fingerprint"] = sha(canonical(plan).encode())
    write_json(output / "preparation_plan.json", plan)
    write_new(output / "private_driver.py", render_driver(plan, signed).encode(), private=True)
    return dict(state="offline_preparation_built", plan=str(output / "preparation_plan.json"),
                archive=str(archive_path), bytes=len(raw), sha256=sha(raw),
                download_bytes=plan["download_bytes"], private_driver=str(output / "private_driver.py"),
                output_root=original, code_root=code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--signed-files", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, default=LOCAL / "preparation-v1")
    create.add_argument("--metadata-root", type=Path)
    create.add_argument("--selection-path", type=Path)
    create.add_argument("--source-spec", type=Path)
    create.add_argument("--experiment-root", default=AFS)
    render = commands.add_parser("render-driver")
    for name in ("plan", "signed-files", "bundle-url-file", "output-driver"):
        render.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.output_dir, args.signed_files, metadata_root=args.metadata_root,
                       selection_path=args.selection_path,
                       source_spec=json.loads(args.source_spec.read_text()) if args.source_spec else None,
                       experiment_root=args.experiment_root)
    else:
        plan = json.loads(args.plan.read_text())
        inventory = json.loads((args.plan.parent / "source_inventory.json").read_text())
        signed = signed_files(args.signed_files, inventory)
        identities = [{k: v for k, v in row.items() if k != "url"} for row in signed]
        if identities != plan["signed_objects"]:
            raise ValueError("Signed object identities changed after preparation build")
        driver = render_driver(plan, signed, bundle_url=args.bundle_url_file.read_text().strip())
        write_new(args.output_driver, driver.encode(), private=True)
        result = dict(state="offline_private_driver_rendered", driver=str(args.output_driver),
                      driver_sha256=sha(driver.encode()), bytes=len(driver.encode()))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
