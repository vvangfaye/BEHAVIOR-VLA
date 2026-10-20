"""Summarize verified 2M receipts and assemble the exact 1000-episode package.

``summarize(root)`` is safe after every controller round and never contacts SCO.
``run(payload)`` is a CPU-only AFS operation; it refuses partial coverage. Optional
video_uploads publish each unchanged MP4 and verify its full remote bytes. Archive
object references from the worker are never advertised as directly usable MP4s.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import zipfile


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def validate_plan(plan):
    if plan.get("checkpoint_step") != 2000000:
        raise ValueError("Only checkpoint 2000000 supported")
    episodes = plan["episodes"]
    if len(episodes) != 1000 or {e["index"] for e in episodes} != set(range(1000)):
        raise ValueError("Exactly 1000 unique planned indices required")
    tasks = {e["task"] for e in episodes}
    pairs = {(e["task"], e["instance_id"]) for e in episodes}
    if len(tasks) != 100 or pairs != {(task, i) for task in tasks for i in range(301, 311)}:
        raise ValueError("Plan must cover 100 tasks x IDs301..310")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", task) for task in tasks):
        raise ValueError("Unsafe task name")
    if any(type(e["index"]) is not int or type(e["instance_id"]) is not int or e.get("rollout", 0) != 0 for e in episodes):
        raise ValueError("Invalid episode identity")
    return {e["index"]: e for e in episodes}


def inspect_report(plan, episode, report):
    name = plan["run_id"] + "-e" + str(episode["index"]).zfill(4)
    for field, value in dict(checkpoint_step=2000000, task=episode["task"], instance_id=episode["instance_id"],
                             rollout_id=0, run_id=name, execution_name=name).items():
        if report.get(field) != value:
            raise ValueError("Receipt identity mismatch: " + field)
    for field in ("source_manifest_sha256", "checkpoint_ready_sha256"):
        if report.get(field) != plan["immutable_inputs"][field]:
            raise ValueError("Receipt source differs from frozen plan")
    if report.get("state") != "complete" or any(report.get(k) is not True for k in (
        "verified", "archive_readback_verified", "full_video_decode_verified",
        "same_attempt_json_and_video", "canonical_results_kept_regardless_of_score")):
        raise ValueError("Receipt lacks verified original JSON/video evidence")
    steps, success, q = report["steps"], report["model_success"], report["q_score"]
    if type(steps) is not int or steps <= 0 or type(success) is not bool:
        raise ValueError("Invalid step count or model success")
    if type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1 or success != (q == 1):
        raise ValueError("Invalid final Q / task success pair")
    video = report["video_validation"]
    if (video.get("decoded_frames") != steps or video.get("advertised_frames") != steps or
            video.get("full_sequential_decode") is not True or video.get("frames_equal_simulator_steps") is not True):
        raise ValueError("Incomplete video decode evidence")
    archive = report["archive"]
    if (not re.fullmatch(r"[a-f0-9]{64}", archive["sha256"]) or type(archive["bytes"]) is not int or archive["bytes"] <= 0):
        raise ValueError("Invalid verified archive identity")
    files = {}
    for row in report["files"]:
        path = Path(row["path"])
        if (path.is_absolute() or ".." in path.parts or row["path"] in files or
                not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) or type(row["bytes"]) is not int or row["bytes"] < 0):
            raise ValueError("Invalid or repeated artifact manifest entry")
        files[row["path"]] = row
    stem = episode["task"] + "_" + str(episode["instance_id"]) + "_0"
    required = {"json/" + stem + ".json", "videos/" + stem + ".mp4", "config/r1pro.yaml", "wrapper/default_wrapper.py", "metadata/execution.json"}
    if not required <= files.keys():
        raise ValueError("Original JSON, video, wrapper or configuration missing")
    return files


def validate_recovery_evidence(plan, index, uid, generation, chain, evidence):
    """Verify each immutable predecessor before accepting a later generation."""
    if type(generation) is not int or generation < 1 or not chain or len(chain) != len(evidence) or len(chain) > generation:
        raise ValueError('Incomplete recovery chain')
    seen = {uid}
    first = generation - len(chain)
    episode = validate_plan(plan)[index]
    name = plan['run_id'] + '-e%04d' % index
    for offset, (entry, saved) in enumerate(zip(chain, evidence)):
        old, launch, proof = saved['submission'], saved['launch'], saved['proof']
        old_generation = first + offset
        if (old.get('uid') in seen or entry.get('old_uid') != old.get('uid')
                or old.get('index') != index or old.get('name') != name
                or old.get('task') != episode['task'] or old.get('instance_id') != episode['instance_id']
                or old.get('checkpoint_step') != 2000000
                or int(old.get('infrastructure_generation', 0)) != old_generation):
            raise ValueError('Recovery predecessor identity mismatch')
        seen.add(old['uid'])
        if old.get('inspection', {}).get('verified') or old.get('state') not in ('FAILED', 'STOPPED', 'DELETED', 'CANCELLED', 'CANCELED'):
            raise ValueError('Completed or active predecessor cannot be retried')
        if (entry.get('user_authorization') != '可以自动重提了'
                or entry.get('old_state') != old['state'] or entry.get('old_writer_stopped') is not True
                or entry.get('probe_state') != 'SUCCEEDED' or entry.get('probe_uid') != proof.get('probe_uid')
                or proof.get('probe_state') != 'SUCCEEDED' or proof.get('old_writer_stopped') is not True
                or proof.get('eligible') is not True or proof.get('protected_completed_outputs')):
            raise ValueError('Recovery lacks verified terminal incomplete-output proof')
        raw = json.dumps(proof, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest() != entry.get('proof_sha256'):
            raise ValueError('Recovery proof bytes changed')
        if (proof.get('old_uid') != old['uid'] or proof.get('old_state') != old['state']
                or proof.get('task') != episode['task'] or proof.get('instance_id') != episode['instance_id']
                or proof.get('checkpoint_step') != 2000000 or proof.get('gpu_count') != 0
                or not proof.get('worker_phases') or any(v not in ('Deleted', 'Failed', 'Succeeded') for v in proof['worker_phases'])
                or proof.get('attempt_count') != len(proof.get('attempts', []))
                or entry.get('attempt_count') != proof['attempt_count']):
            raise ValueError('Recovery probe identity or original attempt count differs')
        physical = launch.get('physical_name', name)
        expected_root = plan['afs_output_root'] + '/episodes/%04d' % index
        if old.get('isolated_generation'):
            expected_root += '-g%03d' % old_generation
        if (entry.get('old_name') != physical or proof.get('old_name') != physical
                or launch.get('name') != name or launch.get('worker_sha256') != plan['immutable_inputs']['worker_sha256']
                or launch.get('source_manifest_sha256') != plan['immutable_inputs']['source_manifest_sha256']
                or proof.get('assignment_root') != expected_root or entry.get('prior_assignment_root') != expected_root
                or old.get('recovery_chain', []) != chain[:offset]):
            raise ValueError('Recovery changes source, predecessor, path or prior chain')
    return True


def batch_version(policy, launch):
    digest = launch.get('batch_worker_sha256')
    versions = policy.get('worker_versions')
    if versions is None:
        if digest != policy.get('batch_worker_sha256'):
            raise ValueError('Unregistered batch source')
        return dict(source_file='frozen/video_eval_batch_worker.py',
                    max_episodes=policy.get('max_episodes_per_worker', 5))
    version = versions.get(digest)
    if not isinstance(version, dict) or version.get('max_episodes') not in (5, 10):
        raise ValueError('Unregistered batch source version')
    source = Path(version['source_file'])
    if source.parent != Path('frozen') or source.suffix != '.py':
        raise ValueError('Invalid frozen batch source path')
    return version


def validate_batch_provenance(plan, episode, report, launch, policy):
    """Validate shared-worker identity without weakening per-episode evidence."""
    if not launch or policy.get('plan_fingerprint') != hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False).encode()).hexdigest():
        raise ValueError('Missing or mismatching batch policy')
    ids = launch.get('batch_indices', [])
    episodes = validate_plan(plan)
    version = batch_version(policy, launch)
    if (not 2 <= len(ids) <= version['max_episodes'] or len(set(ids)) != len(ids) or episode['index'] not in ids
            or any(i not in episodes or episodes[i]['task'] != episode['task'] for i in ids)):
        raise ValueError('Batch mixes tasks or exceeds its finite reservation')
    context = report.get('batch_context', {})
    name = plan['run_id'] + '-b%04d-%04d' % (min(ids), max(ids))
    digest = launch.get('batch_worker_sha256')
    if (not re.fullmatch(r'[a-f0-9]{64}', str(digest))
            or launch.get('physical_name') != name or launch.get('batch_id') != name
            or context.get('batch_name') != name
            or context.get('root') != plan['afs_output_root'] + '/batches/' + name
            or launch.get('batch_root') != context.get('root')
            or context.get('batch_worker_sha256') != digest or launch.get('batch_worker_sha256') != digest
            or context.get('instance_ids') != [episodes[i]['instance_id'] for i in ids]):
        raise ValueError('Batch source, directory or episode identities changed')
    reuse = report.get('batch_reuse', {})
    if (reuse.get('single_model_server') is not True or reuse.get('single_evaluator_process') is not True
            or type(reuse.get('server_pid')) is not int or reuse['server_pid'] <= 0
            or type(reuse.get('evaluator_pid')) is not int or reuse['evaluator_pid'] <= 0
            or episode['instance_id'] not in reuse.get('reset_instance_ids', [])
            or len(reuse.get('loaded_instance_ids', [])) != len(set(reuse.get('loaded_instance_ids', [])))
            or episode['instance_id'] not in reuse.get('loaded_instance_ids', [])):
        raise ValueError('Missing independent reset or shared process evidence')
    files = {r['path']: r for r in report['files']}
    if (files.get('code/batch_worker.py', {}).get('sha256') != digest
            or not {'metadata/batch_reuse.json', 'metadata/batch_queue_snapshot.json'} <= files.keys()):
        raise ValueError('Missing frozen batch worker or raw reuse evidence')


def load_accepted(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_bytes())
    episodes = validate_plan(plan)
    accepted = []
    seen = set()
    for path in sorted((root / "submissions").glob("*.json")):
        record = json.loads(path.read_bytes())
        if record.get("inspection", {}).get("verified") is not True:
            continue
        index = record["index"]
        if index not in episodes or index in seen:
            raise ValueError("Duplicate or unknown accepted episode")
        expected_name = plan["run_id"] + "-e" + str(index).zfill(4)
        if record.get("name") != expected_name or record.get("task") != episodes[index]["task"] or record.get("instance_id") != episodes[index]["instance_id"]:
            raise ValueError("Controller acceptance identity mismatch")
        report_path = root / "artifacts" / (str(index).zfill(4) + ".json")
        report = json.loads(report_path.read_bytes())
        inspect_report(plan, episodes[index], report)
        launch_path = root / 'launches' / (str(index).zfill(4) + '.json')
        launch = json.loads(launch_path.read_bytes()) if launch_path.exists() else None
        if launch is not None:
            if launch['name'] != expected_name or launch['worker_sha256'] != plan['immutable_inputs']['worker_sha256']:
                raise ValueError('Launch provenance differs from accepted episode')
        batch_policy = None
        if record.get('batch_id'):
            batch_policy = json.loads((root / 'batch_policy.json').read_bytes())
            version = batch_version(batch_policy, launch)
            batch_source = (root / 'frozen/video_eval_worker.py').read_bytes() + b'\n' + (root / version['source_file']).read_bytes()
            if hashlib.sha256(batch_source).hexdigest() != launch['batch_worker_sha256']:
                raise ValueError('Frozen batch source differs from policy')
            validate_batch_provenance(plan, episodes[index], report, launch, batch_policy)
        recovery_evidence = []
        if record.get('isolated_generation'):
            chain = record.get('recovery_chain', [])
            generation = record.get('infrastructure_generation')
            if type(generation) is not int or generation < 1:
                raise ValueError('Invalid recovered generation')
            for old_generation in range(generation - len(chain), generation):
                directory = root / 'production_recoveries' / ('%04d' % index) / ('g%03d' % old_generation)
                recovery_evidence.append({key: json.loads((directory / (key + '.json')).read_bytes())
                                          for key in ('submission', 'launch', 'proof')})
            validate_recovery_evidence(plan, index, record.get('uid'), generation, chain, recovery_evidence)
        seen.add(index)
        accepted.append(dict(index=index, uid=record.get("uid"), report=report,
                             batch_policy=batch_policy,
                             recovery_evidence=recovery_evidence,
                             launch=launch,
                             approved_startup_recovery=record.get("approved_startup_recovery"),
                             assignment_root=record.get('assignment_root', plan['afs_output_root'] + '/episodes/' + str(index).zfill(4)),
                             infrastructure_generation=record.get('infrastructure_generation', 0),
                             isolated_generation=record.get('isolated_generation', False), recovery_chain=record.get('recovery_chain', []),
                             receipt_sha256=sha256(report_path)))
    return plan, accepted


def compute_summary(plan, accepted):
    episodes = validate_plan(plan)
    by_index = {}
    for item in accepted:
        index = item["index"]
        if index not in episodes or index in by_index:
            raise ValueError("Repeated or unexpected receipt")
        inspect_report(plan, episodes[index], item["report"])
        by_index[index] = item["report"]
    tasks = sorted({e["task"] for e in episodes.values()})
    per_task = []
    for task in tasks:
        planned = [e for e in episodes.values() if e["task"] == task]
        reports = [by_index[e["index"]] for e in planned if e["index"] in by_index]
        successes = sum(r["model_success"] for r in reports)
        q_sum = sum(r["q_score"] for r in reports)
        count = len(reports)
        per_task.append(dict(task=task, completed=count, successes=successes, unsuccessful=count-successes,
                             missing=10-count, success_rate_completed=successes/count if count else None,
                             success_rate_missing_zero=successes/10, q_completed=q_sum/count if count else None,
                             q_missing_zero=q_sum/10,
                             missing_instances=[e["instance_id"] for e in planned if e["index"] not in by_index]))
    count = len(by_index)
    successes = sum(r["model_success"] for r in by_index.values())
    q_sum = sum(r["q_score"] for r in by_index.values())
    return dict(schema_version=1, checkpoint_step=2000000, run_id=plan["run_id"],
                observed_at=time.time(), scope=dict(tasks=100, instance_ids=list(range(301, 311)), rollouts_per_instance=1),
                state="all_results_verified" if count == 1000 else "partial_results", expected=1000,
                completed=count, successes=successes, unsuccessful=count-successes, missing=1000-count,
                success_rate_completed=successes/count if count else None,
                success_rate_missing_zero=successes/1000, q_completed=q_sum/count if count else None,
                q_missing_zero=q_sum/1000, all_results_verified=count == 1000,
                submission_package_verified=False, per_task=per_task)


def _archive_index(plan, accepted):
    prefix = plan["object_prefix"].strip("/")
    rows = []
    for item in sorted(accepted, key=lambda x: x["index"]):
        report = item["report"]
        stem = f"{report['task']}_{report['instance_id']}_0"
        object_directory = prefix + "/episodes/" + str(item["index"]).zfill(4)
        if item.get("isolated_generation"):
            generation = item.get("infrastructure_generation")
            if type(generation) is not int or generation < 1:
                raise ValueError("Invalid archive generation")
            object_directory += "/generations/g%03d" % generation
        rows.append(dict(index=item["index"], task=report["task"], instance_id=report["instance_id"],
                         source_archive="aoss://260010165/" + object_directory + "/episode.tar.gz",
                         archive_member="episode/videos/" + stem + ".mp4",
                         archive_sha256=report["archive"]["sha256"], direct_mp4_uploaded=False))
    return rows


def summarize(root, output_dir=None):
    root = Path(root)
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    plan, accepted = load_accepted(root)
    summary = compute_summary(plan, accepted)
    write_json(output / "summary1000_current.json", summary)
    write_json(output / "video_archive_index_current.json", _archive_index(plan, accepted))
    target = output / "per_task_success_current.csv"
    with target.with_suffix(".tmp").open("w", newline="") as handle:
        fields = list(summary["per_task"][0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_task"]:
            writer.writerow(dict(row, missing_instances=" ".join(str(i) for i in row["missing_instances"])))
    os.replace(target.with_suffix(".tmp"), target)
    return summary


def make_finalization_payload(root, destination, readme, signer=None):
    """Build a private CPU-worker payload after all 1000 receipts are accepted.

    ``signer(operation, key)`` returns a signed URL for bucket 260010165, with
    operation ``put_object`` or ``get_object``. Keep the returned payload private;
    only the unsigned object key index belongs in public run metadata.
    """
    plan, accepted = load_accepted(root)
    if not compute_summary(plan, accepted)["all_results_verified"]:
        raise ValueError("Cannot finalize an incomplete evaluation")
    payload = dict(plan=plan, accepted=accepted, destination=str(destination), readme=readme)
    ready = json.loads((Path(root) / 'preparation_ready.json').read_text())
    payload['reproduction_source_root'] = ready['code']
    if signer is not None:
        prefix = plan["object_prefix"].strip("/") + "/submission"
        payload["video_uploads"] = {}
        for episode in plan["episodes"]:
            key = prefix + "/videos/" + f"{episode['task']}_{episode['instance_id']}_0.mp4"
            payload["video_uploads"][str(episode["index"]).zfill(4)] = dict(
                key=key, put_url=signer("put_object", key), get_url=signer("get_object", key))
        payload["final_artifact_keys"] = dict(archive=prefix + "/submission.zip", receipt=prefix + "/VERIFICATION.json")
        payload["final_artifact_puts"] = {k: signer("put_object", key) for k, key in payload["final_artifact_keys"].items()}
        payload["final_artifact_gets"] = dict(archive=signer("get_object", payload["final_artifact_keys"]["archive"]))
    return payload


def _copy_verified(source, target, row, *, link=False):
    if source.is_symlink() or not source.is_file() or source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]:
        raise ValueError("AFS original changed or missing: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    else:
        shutil.copyfile(source, target)
    if target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"]:
        raise ValueError("Copied original failed readback")


def copy_reproduction_sources(plan, payload, source_root, package):
    """Package the actual frozen evaluator source, not a moving upstream branch."""
    raw = (source_root / 'metadata/source_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['immutable_inputs']['source_manifest_sha256']:
        raise ValueError('Reproduction source manifest differs from evaluated source')
    manifest = json.loads(raw)
    code = Path(payload['reproduction_source_root']).resolve()
    if code != Path(manifest['code']).resolve():
        raise ValueError('Reproduction code root differs from frozen manifest')
    entries = []
    scripts = {'scripts/eval_b1k.sh', 'scripts/eval_b1k_queue.py', 'scripts/pi05_eval_queue.py'}
    for relative, digest in sorted(manifest['files_sha256'].items()):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe frozen source path')
        if not (relative.startswith('BEHAVIOR-1K/') or relative in scripts):
            continue
        if not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Invalid frozen source digest')
        original = code / path
        if code not in original.resolve().parents or original.is_symlink() or not original.is_file():
            raise ValueError('Frozen evaluator source is missing or escapes its root')
        row = dict(path=relative, bytes=original.stat().st_size, sha256=digest)
        _copy_verified(original, package / 'reproduction_source' / path, row)
        entries.append(row)
    required = scripts | {
        'BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval.py',
        'BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py',
        'BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/eval_utils.py',
        'BEHAVIOR-1K/OmniGibson/omnigibson/eval/wrappers/default_wrapper.py',
        'BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml',
    }
    if not required <= {row['path'] for row in entries}:
        raise ValueError('Frozen evaluator reproduction source is incomplete')
    (package / 'SOURCE_MANIFEST.json').write_bytes(raw)
    write_json(package / 'REPRODUCTION_SOURCE_SHA256.json', entries)
    return len(entries)


def upload_verified_videos(plan, rows, videos, uploads):
    """Upload original videos with four bounded workers and full-byte readback."""
    from video_eval_worker import put_file, verify_remote_archive
    if len(rows) != 1000 or {row['index'] for row in rows} != set(range(1000)):
        raise ValueError('Video delivery requires the complete unique 1000-row index')
    if set(uploads) != {str(i).zfill(4) for i in range(1000)}:
        raise ValueError('Video delivery destinations must cover all 1000 rows')
    prefix = plan['object_prefix'].rstrip('/') + '/submission/videos/'
    for row in rows:
        if Path(row['filename']).name != row['filename'] or not row['filename'].endswith('.mp4'):
            raise ValueError('Unsafe video filename')
        expected = prefix + row['filename']
        target = uploads[str(row['index']).zfill(4)]
        if target['key'] != expected or row['object_uri'] != 'aoss://260010165/' + expected:
            raise ValueError('Video object key does not match its verified episode')

    def upload_one(row):
        target = uploads[str(row['index']).zfill(4)]
        put_file(videos / row['filename'], target['put_url'])
        if verify_remote_archive(target['get_url'], row['bytes'], row['sha256']) is not True:
            raise ValueError('Video upload did not receive positive full-readback verification')
        return dict(row, uploaded=True, full_readback_verified=True)

    completed = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(upload_one, row) for row in rows]
        try:
            for future in as_completed(futures):
                row = future.result()
                completed[row['index']] = row
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return [completed[row['index']] for row in rows]


def run(payload):
    """Create a complete local ZIP and video tree; optional original MP4 upload.

    Required: plan, accepted (load_accepted entries), destination, readme. Each
    accepted entry retains index, report and optional UID/receipt_sha256. Optional
    video_uploads maps four-digit index to {put_url,get_url}; these secret URLs
    are never written to the package. The final public delivery mechanism remains
    separate from internal AOSS object references.
    """
    plan, accepted = payload["plan"], payload["accepted"]
    episodes = validate_plan(plan)
    summary = compute_summary(plan, accepted)
    if not summary["all_results_verified"]:
        raise ValueError("Full submission packaging requires all 1000 accepted JSON/video pairs")
    if not isinstance(payload.get("readme"), str) or not payload["readme"].strip():
        raise ValueError("Explicit reproducible serving/evaluation README required")
    uploads = payload.get("video_uploads")
    if uploads is not None and set(uploads) != {str(i).zfill(4) for i in range(1000)}:
        raise ValueError("Video uploads must cover exactly all 1000 instances")
    if uploads is not None and not all(payload.get(k) for k in ("final_artifact_puts", "final_artifact_gets", "final_artifact_keys")):
        raise ValueError("Uploaded-video finalization also requires ZIP upload/readback and final receipt destination")
    destination = Path(payload["destination"])
    destination.mkdir(parents=True, exist_ok=False)
    package = destination / "submission"
    package.mkdir()
    (package / 'recovery_evidence').mkdir()
    videos = destination / "videos"
    videos.mkdir()
    afs_root = Path(plan["afs_output_root"]).resolve()
    video_index, json_files, provenance = [], [], []
    common_hashes = None
    prefix = plan["object_prefix"].strip("/")
    for item in sorted(accepted, key=lambda x: x["index"]):
        index, report = item["index"], item["report"]
        episode = episodes[index]
        files = inspect_report(plan, episode, report)
        if item.get('batch_policy'):
            validate_batch_provenance(plan, episode, report, item.get('launch'), item['batch_policy'])
        directory = str(index).zfill(4)
        if item.get('isolated_generation'):
            generation = item.get('infrastructure_generation')
            if type(generation) is not int or generation <= 0 or not item.get('recovery_chain'):
                raise ValueError('Missing generation recovery provenance')
            validate_recovery_evidence(plan, index, item.get('uid'), generation,
                                       item['recovery_chain'], item.get('recovery_evidence', []))
            directory += '-g%03d' % generation
        assignment = afs_root / 'episodes' / directory
        if Path(item.get('assignment_root', str(assignment))) != assignment:
            raise ValueError('Unexpected accepted generation path')
        source_root = assignment / 'bundle'
        stem = f"{episode['task']}_{episode['instance_id']}_0"
        json_rel, video_rel = "json/" + stem + ".json", "videos/" + stem + ".mp4"
        # Verify all declared originals, including exact run config and observer.
        for relative, row in files.items():
            path = source_root / relative
            if source_root not in path.resolve().parents or path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("Declared AFS artifact failed final readback: " + relative)
        if item.get('batch_policy'):
            for relative in ('code/batch_worker.py', 'metadata/batch_reuse.json', 'metadata/batch_queue_snapshot.json'):
                _copy_verified(source_root / relative, package / 'batch_evidence' / str(index).zfill(4) / relative, files[relative])
            write_json(package / 'batch_evidence' / str(index).zfill(4) / 'policy.json', item['batch_policy'])
        if item.get('isolated_generation'):
            native_payload = json.loads((source_root / 'metadata/payload.json').read_bytes())
            if (native_payload.get('assignment_root') != str(assignment)
                    or native_payload.get('recovery_chain') != item['recovery_chain']):
                raise ValueError('Original worker payload differs from verified recovery chain')
            write_json(package / 'recovery_evidence' / ('%04d.json' % index), item['recovery_evidence'])
        raw_json = (source_root / json_rel).read_bytes()
        value = json.loads(raw_json)
        if ((value["task"], value["instance_id"], value["rollout_id"]) != (episode["task"], episode["instance_id"], 0)
                or value["success"] != report["model_success"] or value["q_score"]["final"] != report["q_score"]
                or value["steps"] != report["steps"]):
            raise ValueError("Original JSON disagrees with accepted receipt")
        _copy_verified(source_root / json_rel, package / json_rel, files[json_rel])
        _copy_verified(source_root / video_rel, videos / (stem + ".mp4"), files[video_rel], link=True)
        config_hashes = {name: files[name]["sha256"] for name in ("config/r1pro.yaml", "wrapper/default_wrapper.py")}
        if common_hashes is None:
            common_hashes = config_hashes
            for name in common_hashes:
                _copy_verified(source_root / name, package / name, files[name])
        elif common_hashes != config_hashes:
            raise ValueError("Robot or wrapper configuration differs between episodes")
        execution_target = package / "execution_configs" / (str(index).zfill(4) + ".json")
        _copy_verified(source_root / "metadata/execution.json", execution_target, files["metadata/execution.json"])
        video_entry = dict(index=index, task=episode["task"], instance_id=episode["instance_id"], rollout_id=0,
                           filename=stem + ".mp4", bytes=files[video_rel]["bytes"], sha256=files[video_rel]["sha256"],
                           object_uri="aoss://260010165/" + prefix + "/submission/videos/" + stem + ".mp4", uploaded=False)
        video_index.append(video_entry)
        json_files.append(dict(path=json_rel, bytes=files[json_rel]["bytes"], sha256=files[json_rel]["sha256"]))
        provenance.append(dict(index=index, uid=item.get("uid"), run_id=report["run_id"],
                               launch=item.get('launch'),
                               approved_startup_recovery=item.get("approved_startup_recovery"), recovery_chain=item.get("recovery_chain", []),
                               receipt_sha256=item.get("receipt_sha256"), archive=report["archive"],
                               original_afs_root=str(source_root), source_manifest_sha256=report["source_manifest_sha256"]))
    (package / "README.md").write_text(payload["readme"].rstrip() + "\n")
    reproduction_count = None
    if payload.get('reproduction_source_root'):
        reproduction_count = copy_reproduction_sources(plan, payload, source_root, package)
    if uploads is not None:
        video_index = upload_verified_videos(plan, video_index, videos, uploads)
    summary.update(submission_package_verified=True, direct_videos_uploaded=uploads is not None,
                   state="outputs_packaged_with_uploaded_videos" if uploads is not None else "local_package_complete")
    write_json(package / "SCORES.json", summary)
    write_json(package / "PROVENANCE.json", provenance)
    write_json(package / "VIDEO_INDEX.json", video_index)
    write_json(destination / "VIDEO_INDEX.json", video_index)
    write_json(package / "ORIGINAL_JSON_SHA256.json", json_files)
    archive = destination / "submission.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                output.write(path, arcname=path.relative_to(package).as_posix())
    # Reopen ZIP and check every original JSON as bytes, not a reserialized object.
    with zipfile.ZipFile(archive) as packed:
        for row in json_files:
            raw = packed.read(row["path"])
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("ZIP contains a changed evaluation output")
        if len([x for x in packed.namelist() if x.startswith("json/")]) != 1000:
            raise ValueError("ZIP does not contain exactly 1000 original JSON files")
    final = dict(state=summary["state"], checkpoint_step=2000000, run_id=plan["run_id"], verified=True,
                 source_manifest_sha256=plan["immutable_inputs"]["source_manifest_sha256"],
                 checkpoint_ready_sha256=plan["immutable_inputs"]["checkpoint_ready_sha256"],
                 completed=1000, successes=summary["successes"], success_rate=summary["success_rate_missing_zero"],
                 q_score=summary["q_missing_zero"], zip_path=str(archive), zip_bytes=archive.stat().st_size,
                 zip_sha256=sha256(archive), original_json_verified=True, direct_videos_uploaded=uploads is not None,
                 video_index=str(destination / "VIDEO_INDEX.json"), completed_at=time.time())
    if reproduction_count is not None:
        final.update(reproduction_source_files=reproduction_count, reproduction_source_verified=True)
    if uploads is not None:
        from video_eval_worker import put_file, verify_remote_archive
        archive_key = prefix + "/submission/submission.zip"
        receipt_key = prefix + "/submission/VERIFICATION.json"
        if payload["final_artifact_keys"] != dict(archive=archive_key, receipt=receipt_key):
            raise ValueError("Final artifact destinations differ from the frozen run prefix")
        put_file(archive, payload["final_artifact_puts"]["archive"])
        verify_remote_archive(payload["final_artifact_gets"]["archive"], final["zip_bytes"], final["zip_sha256"])
        final.update(zip_full_readback_verified=True,
                     zip_object_uri="aoss://260010165/" + archive_key,
                     verification_object_uri="aoss://260010165/" + receipt_key)
    write_json(destination / "VERIFICATION.json", final)
    if uploads is not None:
        put_file(destination / "VERIFICATION.json", payload["final_artifact_puts"]["receipt"])
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-request", type=Path)
    parser.add_argument("--destination", help="AFS destination for a later CPU-only package operation")
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()
    summary = summarize(args.root, args.output_dir)
    if args.prepare_request:
        if not summary["all_results_verified"] or not args.destination or not args.readme:
            parser.error("preparing finalization requires all results, --destination and --readme")
        plan, accepted = load_accepted(args.root)
        write_json(args.prepare_request, dict(plan=plan, accepted=accepted, destination=args.destination,
                                             readme=args.readme.read_text()))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}))


if __name__ == "__main__":
    main()
