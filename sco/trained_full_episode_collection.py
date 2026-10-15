"""Isolated step-20000 collection and compact native competition metrics."""
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

MODEL = '/mnt/afs/260010165/behavior_1k/paired_quick20_reused1688977_step20000_20261015/checkpoint20000'
MANIFEST_SHA = '16b3e949105f759451d7ce2f43608e516b360d9b99380b2f954ebee07c1cd80e'
EXPORT_REPORT = Path('/Users/faye/code/Behavior-1k Challenge/analysis/retraining_20261013/paired_quick20_reused1688977_step20000_20261015/reports/export.json')


def validate_model_plan(plan):
    export = plan['trained_model_export']
    assert plan['sampling_round'] == 4
    assert plan['checkpoint'] == MODEL
    assert plan['checkpoint_identity']['checkpoint_step'] == 20000
    assert plan['checkpoint_identity']['parent_checkpoint_step'] == 1688977
    assert export['checkpoint'] == MODEL and export['manifest_sha256'] == MANIFEST_SHA
    assert export['state'] == 'accepted' and export['completed_updates'] == 20000
    assert export['full_afs_sha256_readback'] is True
    assert plan['checkpoint_identity']['checkpoint_files'] == export['files']
    assert plan['metrics_protocol'] == 'native_competition_q_score_v1'
    assert plan['cpu_thread_limits'] == {'blas': 1, 'openmp': 8, 'torch_intraop': 8, 'torch_interop': 1}
    return plan


def verify_checkpoint(checkpoint, export):
    """Run on the worker before policy startup; verify the actual mounted bytes."""
    import hashlib
    import json
    from pathlib import Path
    checkpoint = Path(checkpoint)
    assert str(checkpoint) == export['checkpoint']
    assert export['completed_updates'] == 20000 and export['parent_checkpoint'] == 1688977
    assert (checkpoint / '_READY').is_file()
    manifest = (checkpoint / 'SHA256SUMS.inference').read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == export['manifest_sha256']
    entries = {}
    for line in manifest.decode().splitlines():
        digest, separator, name = line.partition('  ')
        relative = Path(name)
        assert separator and len(digest) == 64 and not relative.is_absolute() and '..' not in relative.parts
        assert name not in entries
        entries[name] = digest
    expected = {r['relative_path']: r for r in export['files']}
    assert set(entries) == set(expected)
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()}
    assert actual == set(entries) | {'_READY', 'SHA256SUMS.inference'}
    for name, digest in entries.items():
        path = checkpoint / name
        assert not path.is_symlink() and path.stat().st_size == expected[name]['bytes']
        assert digest == expected[name]['sha256']
        value = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024**2), b''):
                value.update(chunk)
        assert value.hexdigest() == digest, name
    assert json.loads((checkpoint / 'assets/subtask_conditioning.json').read_text())['action_conditioning'] == 'task_and_subtask'
    print('TRAINED_CHECKPOINT_FULL_SHA_VERIFIED step=20000 manifest=' + export['manifest_sha256'], flush=True)


def metric_summary(rows, task_names, target=1000):
    """Infrastructure failures and unpublished episodes are missing, not successes or zeros."""
    tasks = {name: [] for name in task_names}
    seen = set()
    for row in rows:
        identity = row['task'], row['instance_id']
        assert identity not in seen and row['task'] in tasks and 301 <= row['instance_id'] <= 310
        seen.add(identity)
        assert type(row['success']) is bool and type(row['terminated']) is bool and type(row['truncated']) is bool
        q = row['q_score_final']
        assert type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1
        assert row['steps'] > 0 and (row['terminated'] or row['truncated'])
        tasks[row['task']].append(row)
    def summarize(batch):
        n = len(batch)
        return dict(completed_episodes=n, successful_episodes=sum(r['success'] for r in batch),
                    success_rate=sum(r['success'] for r in batch) / n if n else None,
                    competition_sr=sum(r['q_score_final'] == 1.0 for r in batch) / n if n else None,
                    mean_q_score=sum(r['q_score_final'] for r in batch) / n if n else None,
                    bc_eligible_episodes=sum(r['success'] and r['terminated'] and not r['truncated'] for r in batch))
    out = summarize(rows)
    out.update(target_episodes=target, missing_episodes=target - len(rows),
               denominator='completed_published_episodes_only', final_competition_score=len(rows) == target,
               per_task={t: summarize(v) for t, v in tasks.items()})
    represented = [v for v in out['per_task'].values() if v['completed_episodes']]
    out['task_macro_mean_q_score_observed'] = sum(v['mean_q_score'] for v in represented) / len(represented) if represented else None
    return out


def observe_metrics(campaign, storage):
    from restore_full_episode_collection import read_plan, prefix_for
    from full_episode_collection import save
    campaign = Path(campaign)
    auth = json.loads((campaign / 'authorization.json').read_text())
    root = Path(auth['roots'][0]); plan = read_plan(root); validate_model_plan(plan)
    prefix = prefix_for(plan)
    cache_path = campaign / 'metrics_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    objects = {}
    for page in storage.get_paginator('list_objects_v2').paginate(Bucket='260010165', Prefix=prefix + '/'):
        objects.update({v['Key']: v for v in page.get('Contents', []) if '/private/' not in v['Key']})
    keys = [key for key in objects if key.startswith(prefix + '/metrics/') and key.endswith('.json')]
    def fetch(key):
        version = objects[key]['ETag']
        if key in cache and cache[key]['etag'] == version:
            return key, cache[key]
        assert 0 < objects[key]['Size'] < 32768
        response = storage.get_object(Bucket='260010165', Key=key)
        try:
            body = response['Body'].read(32769); assert len(body) < 32768
        finally:
            response['Body'].close()
        return key, dict(etag=version, row=json.loads(body))
    with ThreadPoolExecutor(max_workers=8) as pool:
        fresh = dict(pool.map(fetch, keys))
    rows = []
    for item in fresh.values():
        row = item['row']
        assert row['kind'] == 'native_competition_episode_metrics_v1'
        assert row['checkpoint_manifest_sha256'] == MANIFEST_SHA
        assert row['collection_run_id'] == plan['run_id'] and row['sampling_round'] == 4
        assert objects[row['receipt_key']]['Size'] > 0
        assert objects[row['archive_key']]['Size'] == row['archive_bytes'] > 0
        assert objects[row['archive_key']]['ETag'] == row['archive_etag']
        rows.append(row)
    progress = metric_summary(rows, plan['task_prompts'])
    progress.update(observed_at=time.time(), checkpoint_step=20000, checkpoint_manifest_sha256=MANIFEST_SHA,
                    sampling_round=4, ready_for_training=False)
    save(cache_path, fresh); save(campaign / 'metrics_progress_current.json', progress)
    return progress


def freeze(campaign, parent):
    from full_episode_collection import digest, save
    from restore_full_episode_collection import read_plan
    import inspect
    campaign, parent = Path(campaign).resolve(), Path(parent).resolve()
    old = read_plan(parent)
    export_raw = EXPORT_REPORT.read_bytes(); export = json.loads(export_raw)
    assert export['checkpoint'] == MODEL and export['manifest_sha256'] == MANIFEST_SHA
    assert export['state'] == 'accepted' and export['full_afs_sha256_readback']
    files = {name: (parent / 'overlay_sources' / name).read_bytes() for name in old['overlay_files_sha256']}
    for name, body in files.items():
        assert digest(body) == old['overlay_files_sha256'][name]
    def patch(name, before, after):
        text = files[name].decode(); assert text.count(before) == 1, (name, before)
        files[name] = text.replace(before, after, 1).encode()
    files['sco/trained_checkpoint_verify.py'] = inspect.getsource(verify_checkpoint).encode()
    patch('sco/run_public_test_assignment.py',
          "    assert json.loads((checkpoint / '_READY').read_text())['checkpoint_step'] == 1688977",
          "    from trained_checkpoint_verify import verify_checkpoint\n    verify_checkpoint(checkpoint, payload['trained_model_export'])")
    patch('sco/run_public_test_assignment.py', "    for row in provenance['checkpoint_files']:",
          "    for row in payload['checkpoint_identity']['checkpoint_files']:")
    text = files['sco/run_public_test_assignment.py'].decode()
    text = text.replace("'checkpoint_step': 1688977", "'checkpoint_step': payload['checkpoint_identity']['checkpoint_step']")
    anchor = "    status = {'state': 'running'"
    assert text.count(anchor) == 1
    text = text.replace(anchor,
        "    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',\n"
        "               NUMEXPR_NUM_THREADS='8', VECLIB_MAXIMUM_THREADS='8', BLIS_NUM_THREADS='1',\n"
        "               PI05_EVAL_TORCH_THREADS='8', PI05_EVAL_TORCH_INTEROP_THREADS='1')\n"
        "    print('CPU_THREAD_LIMITS openmp=8 blas=1 torch=8 interop=1 quota_vcpu=14', flush=True)\n" + anchor)
    files['sco/run_public_test_assignment.py'] = text.encode()
    patch('sco/run_full_episode_collection.py', 'checkpoint_step=1688977,', "checkpoint_step=payload['checkpoint_identity']['checkpoint_step'],")
    patch('sco/run_full_episode_collection.py',
          "        object_storage=payload['object_storage'], official_benchmark=False,",
          "        object_storage=payload['object_storage'], checkpoint_manifest_sha256=payload['trained_model_export']['manifest_sha256'],\n"
          "        metrics_object_storage=payload['metrics_object_storage'], official_benchmark=False,")
    patch('scripts/full_episode_repeat_seeding.py', 'round_id not in (1, 2, 3):', 'round_id not in (1, 2, 3, 4):')
    patch('scripts/eval_b1k_queue.py', 'def _load_base_module() -> ModuleType:\n    return',
          "def _load_base_module() -> ModuleType:\n"
          "    import os\n"
          "    if os.environ.get('PI05_EVAL_TORCH_THREADS'):\n"
          "        import torch\n"
          "        torch.set_num_threads(int(os.environ['PI05_EVAL_TORCH_THREADS']))\n"
          "        torch.set_num_interop_threads(int(os.environ['PI05_EVAL_TORCH_INTEROP_THREADS']))\n"
          "        print('EVALUATOR_CPU_THREADS', torch.get_num_threads(), torch.get_num_interop_threads(), flush=True)\n"
          "    return")
    patch('scripts/full_episode_checkpoint.py', "    upload_json(receipt, storage['receipt'])\n    return receipt",
          "    upload_json(receipt, storage['receipt'])\n"
          "    q = result['q_score']['final']\n"
          "    import math\n"
          "    assert type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1\n"
          "    metrics = dict(kind='native_competition_episode_metrics_v1',\n"
          "        collection_run_id=manifest['run_id'].rsplit('-a', 1)[0], run_id=manifest['run_id'],\n"
          "        task=result['task'], instance_id=result['instance_id'], rollout_id=result['rollout_id'],\n"
          "        sampling_round=manifest['sampling_round'], checkpoint_manifest_sha256=manifest['checkpoint_manifest_sha256'],\n"
          "        success=result['success'], terminated=result['terminated'], truncated=result['truncated'],\n"
          "        q_score_final=q, steps=result['steps'], receipt_key=storage['receipt']['key'],\n"
          "        archive_key=stored['object_key'], archive_bytes=stored['bytes'], archive_etag=stored['etag'],\n"
          "        source_manifest_sha256=manifest['source_manifest_sha256'], repeat_sampling=result['repeat_sampling'])\n"
          "    upload_json(metrics, manifest['metrics_object_storage'][key])\n"
          "    return receipt")
    campaign.mkdir(exist_ok=False); root = campaign / 'round-4'; root.mkdir(); source = root / 'overlay_sources'
    for name, body in files.items():
        if name.endswith('.py'): compile(body, name, 'exec')
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
    hashes = {name: digest(body) for name, body in files.items()}
    with tarfile.open(root / 'overlay.tar.gz', 'w:gz') as stream:
        for name, body in sorted(files.items()):
            member = tarfile.TarInfo(name); member.size = len(body); stream.addfile(member, io.BytesIO(body))
        body = json.dumps(hashes, sort_keys=True).encode(); member = tarfile.TarInfo('_FULL_EPISODE_OVERLAY.json'); member.size = len(body); stream.addfile(member, io.BytesIO(body))
    plan = dict(old, run_id='native-full1000-quick20k-20261015-r4', sampling_round=4,
        parent_plan_fingerprint=old['fingerprint'], checkpoint=MODEL,
        checkpoint_identity=dict(checkpoint_step=20000, parent_checkpoint_step=1688977,
            checkpoint_source=MODEL, checkpoint_files=export['files'], manifest_sha256=MANIFEST_SHA),
        trained_model_export=export, model_export_report_sha256=digest(export_raw),
        seed_policy='paired_same_policy_seeds_as_round3',
        metrics_protocol='native_competition_q_score_v1',
        cpu_thread_limits=dict(blas=1, openmp=8, torch_intraop=8, torch_interop=1),
        overlay_files_sha256=hashes, overlay_sha256=digest((root / 'overlay.tar.gz').read_bytes()))
    plan.pop('fingerprint'); plan['fingerprint'] = digest(json.dumps(plan, sort_keys=True).encode())
    validate_model_plan(plan)
    save(root / 'plan.json', plan); save(root / 'held_out_instances.json', old['held_out_instances']); (root / 'submissions').mkdir()
    deadline = json.loads((parent.parent / 'authorization.json').read_text())['deadline_epoch']
    auth = dict(kind='native_full_episode_independent_final1000', total_episodes=1000, additional_episodes=1000,
        total_shards=201, roots=[str(root)], fingerprints=[plan['fingerprint']], rounds=[4],
        deadline_epoch=deadline, inherited_deadline_epoch=deadline, max_active_gpus=None, max_pending_tasks=5,
        controller_poll_seconds=60, automatic_failure_recovery=False, checkpoint_step=20000,
        admission='bounded_spot_queue', all_public_instances_training_seen=True,
        validation_instances=old['held_out_instances'], old_campaigns_remain_stopped=True,
        user_instruction='用这个模型重新采一波数据，同时考虑计算成功率和比赛score指标；应用CPU线程限制', authorized_at=time.time())
    save(campaign / 'authorization.json', auth)
    save(root / 'controller_authorization.json', dict(kind='native_full_episode_finite_controller',
        plan_fingerprint=plan['fingerprint'], expected_episodes=1000, deadline_epoch=deadline,
        max_active_gpus=None, automatic_failure_recovery=False))
    save(campaign / 'pilot_acceptance_waiver.json', dict(kind='user_waived_pilot_production_gate',
        authorization_sha256=digest((campaign / 'authorization.json').read_bytes()), root=str(root),
        plan_fingerprint=plan['fingerprint'], deadline_epoch=deadline, total_episodes=1000, max_pending_tasks=5,
        allow_production_without_pilot_acceptance=True, artifact_acceptance_complete=False,
        user_instruction='沿用本聊天已确认的不需要训练导出验收；每个worker仍完整核验新模型SHA'))
    policy = json.loads((parent.parent / 'queue_policy.json').read_text()); policy.update(roots=[str(root)])
    save(campaign / 'queue_policy.json', policy)
    save(campaign / 'HANDOFF.json', dict(state='frozen_new_model_waiting_submission', expected_episodes=1000,
        checkpoint_step=20000, checkpoint=MODEL, sampling_round=4, controller_required='watch_full_episode_repeats.py',
        controller_poll_seconds=60, queue_scope='authorized_campaign_only', max_pending_tasks=5,
        old_campaigns_remain_stopped=True, metrics='metrics_progress_current.json', deadline_epoch=deadline,
        metrics_definition='native environment success and competition q_score.final, completed episodes denominator',
        cpu_thread_limits=plan['cpu_thread_limits'], speedup_verified=False, ready_for_training=False))
    return plan, auth
