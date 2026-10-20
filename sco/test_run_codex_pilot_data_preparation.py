import ast
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_codex_pilot_data_preparation as worker
import run_codex_pilot_training as shared


@pytest.fixture(autouse=True)
def preserve_process_settings():
    bytecode, search_path = sys.dont_write_bytecode, sys.path[:]
    mask = os.umask(0o077); os.umask(mask)
    yield
    sys.dont_write_bytecode = bytecode
    sys.path[:] = search_path
    os.umask(mask)


@pytest.fixture
def original_chain(tmp_path, monkeypatch):
    """Tiny byte fixtures with the real, complete v14 receipt schema."""
    monkeypatch.setattr(worker, "AFS", str(tmp_path))
    dataset, reader = tmp_path / "original-v3", tmp_path / "original-reader-v4"
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path
    def row(path, relative):
        return dict(path=relative, bytes=path.stat().st_size, sha256=shared.sha(path))
    episodes = sorted(ep for values in worker.TASK_EPISODES.values() for ep in values)
    transfers, per_episode = [], {}
    for task, values in worker.TASK_EPISODES.items():
        data_name = 'data/' + task + '.parquet'
        transfers.append(row(write(dataset / data_name, b'fixture parquet ' + task.encode()), data_name))
        for episode in values:
            per_episode[episode] = [data_name]
            for camera in range(3):
                name = f'videos/{episode}-{camera}.mp4'
                transfers.append(row(write(dataset / name, f'fixture video {episode}-{camera}'.encode()), name))
                per_episode[episode].append(name)
    info = dict(features=dict(rgb=dict(dtype='video', info=dict(is_depth_map=False)),
                              depth=dict(dtype='video', info=dict(is_depth_map=True))))
    original_info = write(dataset / 'meta/info.json.orig', info)
    metadata = [row(original_info, 'meta/info.json')]
    del info['features']['depth']
    write(dataset / 'meta/info.json', json.dumps(info, indent=4).encode())
    metadata.append(row(write(dataset / 'meta/episodes.jsonl', b'fixture metadata\n'), 'meta/episodes.jsonl'))
    labels = {name: write(dataset / 'labels' / name, ('fixture labels ' + name).encode())
              for name in ('subtask_segments.parquet', 'retained_episodes.txt')}
    write(dataset / 'selected_episodes.txt', ''.join(str(ep) + '\n' for ep in episodes).encode())
    write(dataset / '_FAILED', b'prior stopped reader failure')
    raw_paths, demos = [], []
    for episode in episodes:
        task = next(task for task, values in worker.TASK_EPISODES.items() if episode in values)
        raw = write(tmp_path / 'raw' / f'episode_{episode}.hdf5', f'fixture raw {episode}'.encode())
        raw_paths.append(raw)
        directory = reader / 'demonstrations' / str(episode)
        image = write(directory / 'sample.png', b'fixture image ' + str(episode).encode())
        manifest = dict(metadata=dict(source_kind='original_train', task=task, episode_index=episode,
            source_dataset=str(dataset), source_raw=str(raw), source_raw_sha256=shared.sha(raw),
            expected_raw_sha256=shared.sha(raw), identity_source='published_metadata_raw_id_and_sha',
            success_source='original_behavior_task_terminated_without_truncation',
            termination_evidence=dict(final_terminated=True, final_truncated=False, terminated_count=1),
            segments_sha256=shared.sha(labels['subtask_segments.parquet']),
            retained_episodes_sha256=shared.sha(labels['retained_episodes.txt']),
            source_files=[dict(path=item['path'], sha256=item['sha256']) for item in transfers
                          if item['path'] in per_episode[episode]]),
            examples=[dict(image=image.name, image_sha256=shared.sha(image))])
        manifest['fingerprint'] = shared.digest(manifest)
        path = write(directory / 'manifest.json', manifest)
        demos.append(dict(episode_index=episode, task=task, manifest=str(path),
                          manifest_sha256=shared.sha(path), fingerprint=manifest['fingerprint']))
    status = write(dataset / 'status.json', dict(state='failed', failure_phase='real_reader_and_demonstrations',
        root=str(dataset), selected_episodes=episodes, transfers=transfers,
        required_files=[item['path'] for item in transfers], metadata_files=metadata))
    payload = write(dataset / 'payload.json', b'private signed payload fixture: never parsed or logged')
    ready = write(reader / '_ORIGINAL_DATA_READY.json', dict(kind='codex_original_pilot_staging_v1', state='complete',
        root=str(reader), dataset_root=str(dataset), selected_episodes=episodes, original_reader_verified=True,
        raw_success_verified=True, source_receipt_sha256=dict(status=shared.sha(status), payload=shared.sha(payload)),
        transfers=transfers, demonstrations=demos))
    monkeypatch.setattr(worker, 'ORIGINAL_SHA', shared.sha(ready))
    return SimpleNamespace(dataset=dataset, reader=reader, transfers=transfers, metadata=metadata,
                           labels=labels, raw_paths=raw_paths, demos=demos, ready=ready)


def test_full_original_chain_is_hashed_without_loading_private_payload(original_chain, monkeypatch):
    loads, hashs = [], []
    helper = SimpleNamespace(**vars(shared))
    def load(path):
        loads.append(Path(path)); return shared.load(path)
    def sha(path):
        hashs.append(Path(path)); return shared.sha(path)
    helper.load, helper.sha = load, sha
    proof = worker.verify_original(helper, time.monotonic() + 600)
    assert proof['verified_files'] == 20 and proof['verified_raw_demonstrations'] == 6
    assert proof['verified_metadata_files'] == 2 and proof['downloaded_bytes'] == 0
    assert original_chain.dataset / 'payload.json' in hashs
    assert original_chain.dataset / 'payload.json' not in loads
    assert set(original_chain.raw_paths).issubset(hashs)
    assert set(original_chain.labels.values()).issubset(hashs)


@pytest.mark.parametrize('damage', ['data', 'video', 'metadata', 'metadata_backup', 'rgb_metadata',
    'segments', 'retained', 'raw', 'manifest', 'image', 'status', 'payload', 'ready', 'selection',
    'running', 'reader_running', 'symlink'])
def test_any_original_chain_change_rejects_before_export(original_chain, damage):
    source = original_chain
    files = dict(data=source.dataset / source.transfers[0]['path'], video=source.dataset / source.transfers[1]['path'],
        metadata=source.dataset / 'meta/episodes.jsonl', metadata_backup=source.dataset / 'meta/info.json.orig',
        rgb_metadata=source.dataset / 'meta/info.json', segments=source.labels['subtask_segments.parquet'],
        retained=source.labels['retained_episodes.txt'], raw=source.raw_paths[0], manifest=Path(source.demos[0]['manifest']),
        image=Path(source.demos[0]['manifest']).parent / 'sample.png', status=source.dataset / 'status.json',
        payload=source.dataset / 'payload.json', ready=source.ready, selection=source.dataset / 'selected_episodes.txt')
    if damage == 'running': (source.dataset / '_RUNNING').write_text('active writer')
    elif damage == 'reader_running': (source.reader / '_RUNNING').write_text('active reader writer')
    elif damage == 'symlink':
        target = files['video']; backup = target.with_suffix('.backup'); target.rename(backup); target.symlink_to(backup)
    else: files[damage].write_bytes(files[damage].read_bytes() + b'changed')
    with pytest.raises(ValueError): worker.verify_original(shared, time.monotonic() + 600)


def test_original_hashing_respects_cleanup_reserve(original_chain):
    with pytest.raises(shared.WorkerDeadline): worker.verify_original(shared, time.monotonic() + 29)


@pytest.fixture
def success(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "AFS", str(tmp_path))
    root = tmp_path / "collection/radio-311-test"
    path = root / "attempt/trajectory.hdf5"
    path.parent.mkdir(parents=True);path.write_bytes(b"test fixture; never actual success")
    identity = dict(task="turning_on_radio", instance_id=311, attempt_id="test")
    info = dict(metadata={**identity, "checkpoint": "pi05_b1k_full100_joint_1688977"}, steps=3,
                content_sha256="a" * 64, subtask_sha256="b" * 64)
    directory = tmp_path / "audit-radio-311-test";directory.mkdir()
    audit = dict(kind="codex_collection_readonly_audit_v1", state="audit_complete", training_eligible=True,
                 source_unchanged_during_audit=True, issues=[], official_benchmark=False, model_calls=0,
                 network_calls=0, model_calls_scope="audit_process_only", audit_root=str(directory),
                 run_root=str(root), identity=identity,
                 trajectory=dict(valid_complete_trajectory=True, environment_goal_success=True,
                                 content_sha256=info["content_sha256"], subtask_sha256=info["subtask_sha256"], steps=3))
    report = directory / "report.json";report.write_text(json.dumps(audit))
    ref = dict(path=str(report), content_sha256=shared.digest(audit))
    return ref, info, audit


def checker(info):
    def inspect(path, *, require_success):
        assert require_success is True
        assert path.is_file()
        return copy.deepcopy(info)
    return inspect


def test_success_selection_binds_current_trajectory_and_identity(success):
    ref, info, _ = success
    selected = worker.audit_successes([ref], shared, checker(info))
    assert selected[0]["steps"] == 3
    assert selected[0]["identity"]["instance_id"] == 311
    assert selected[0]["content_sha256"] == info["content_sha256"]


def native_success(success, monkeypatch):
    ref, info, audit = success
    inventory = {"fixture-" + str(i): "f" * 64 for i in range(29)}
    monkeypatch.setattr(worker, "CODE9_BUNDLE_SHA", "9" * 64)
    monkeypatch.setattr(worker, "EXPECTED_CODE_INVENTORY", inventory)
    mode = "native_with_corrections"
    info["metadata"].update(collection_mode=mode, subtask_source="mixed_native_and_teacher")
    audit.update(collection_mode=mode, collection_code=worker.AFS + "/code-v9",
                 collection_bundle_sha256=worker.CODE9_BUNDLE_SHA, collection_source_files_sha256=inventory)
    audit["trajectory"].update(collection_mode=mode, subtask_source="mixed_native_and_teacher")
    Path(ref["path"]).write_text(json.dumps(audit)); ref["content_sha256"] = shared.digest(audit)
    return ref, info, audit


def test_native_success_requires_frozen_v9_audit_and_actual_reader_metadata(success, monkeypatch):
    ref, info, audit = native_success(success, monkeypatch)
    selected = worker.audit_successes([ref], shared, checker(info), code_version="v9")
    assert selected[0]["collection_mode"] == "native_with_corrections"
    assert selected[0]["content_sha256"] == audit["trajectory"]["content_sha256"]
    with pytest.raises(ValueError, match="code-v9"):
        worker.audit_successes([ref], shared, checker(info), code_version="v8")


@pytest.mark.parametrize("damage", ["missing_mode", "reader_mode", "reader_source", "audit_source", "code",
    "bundle", "inventory", "trajectory_mode", "unfrozen"])
def test_native_preparation_rejects_cross_version_or_unbound_evidence(success, monkeypatch, damage):
    ref, info, audit = native_success(success, monkeypatch)
    if damage == "missing_mode": audit.pop("collection_mode")
    elif damage == "reader_mode": info["metadata"]["collection_mode"] = "teacher_conditioned"
    elif damage == "reader_source": info["metadata"]["subtask_source"] = "provided"
    elif damage == "audit_source": audit["trajectory"]["subtask_source"] = "provided"
    elif damage == "code": audit["collection_code"] = worker.AFS + "/code-v8"
    elif damage == "bundle": audit["collection_bundle_sha256"] = worker.BUNDLE_SHA
    elif damage == "inventory": audit["collection_source_files_sha256"] = {"different": "f" * 64}
    elif damage == "trajectory_mode": audit["trajectory"]["collection_mode"] = "teacher_conditioned"
    else: monkeypatch.setattr(worker, "CODE9_BUNDLE_SHA", None)
    Path(ref["path"]).write_text(json.dumps(audit)); ref["content_sha256"] = shared.digest(audit)
    with pytest.raises(ValueError): worker.audit_successes([ref], shared, checker(info), code_version="v9")


def test_v9_reader_also_accepts_unchanged_legacy_audit_contract(success):
    ref, info, _ = success
    selected = worker.audit_successes([ref], shared, checker(info), code_version="v9")
    assert selected[0]["collection_mode"] == "teacher_conditioned"


def test_version_selection_is_explicit_and_unfrozen_native_cannot_build(success, monkeypatch):
    ref, _, _ = success
    payload = dict(kind=worker.KIND, run_id="pilot001", success_audits=[ref])
    assert worker.validate_payload(payload)["code_version"] == "v8"
    native = {**payload, "code_version": "v9"}
    assert worker.paths_for("pilot001", "v9")["code"] == worker.AFS + "/code-v9"
    monkeypatch.setattr(worker, "CODE9_BUNDLE_SHA", None)
    with pytest.raises(ValueError, match="reviewed and frozen"): worker.build_source(native)
    with pytest.raises(ValueError, match="version"): worker.validate_payload({**payload, "code_version": "v10"})


@pytest.mark.parametrize("damage", ["not_success", "issues", "changed_audit", "changed_h5", "wrong_checkpoint",
                                   "evaluation_instance", "duplicate", "wrong_identity", "invalid_complete", "wrong_scope"])
def test_reject_unaccepted_or_changed_success_before_export(success, damage):
    ref, info, audit = success
    if damage == "changed_h5":info["content_sha256"] = "c" * 64
    elif damage == "wrong_checkpoint":info["metadata"]["checkpoint"] = "old"
    elif damage == "wrong_identity":info["metadata"]["attempt_id"] = "different"
    elif damage == "evaluation_instance":audit["identity"]["instance_id"] = 301
    elif damage == "not_success":audit["training_eligible"] = False
    elif damage == "issues":audit["issues"] = ["executed_rejected_action"]
    elif damage == "invalid_complete":audit["trajectory"]["valid_complete_trajectory"] = False
    elif damage == "wrong_scope":audit["model_calls_scope"] = "all_collection_calls"
    elif damage == "changed_audit":audit["unexpected"] = True
    Path(ref["path"]).write_text(json.dumps(audit))
    if damage != "changed_audit":ref["content_sha256"] = shared.digest(audit)
    with pytest.raises(ValueError):
        worker.audit_successes([ref, ref] if damage == "duplicate" else [ref], shared, checker(info))


@pytest.mark.parametrize("damage", ["no_success", "override", "unsafe_run", "duplicate_audit", "wrong_root"])
def test_preparation_payload_rejects_unbounded_or_unsafe_inputs(success, damage):
    ref, _, _ = success
    payload = dict(kind=worker.KIND, run_id="pilot001", success_audits=[ref])
    if damage == "no_success":payload["success_audits"] = []
    elif damage == "override":payload["max_seconds"] = 999999
    elif damage == "unsafe_run":payload["run_id"] = "../../old"
    elif damage == "duplicate_audit":payload["success_audits"] *= 2
    else:ref["path"] = "/unrelated/report.json"
    with pytest.raises(ValueError):worker.validate_payload(payload)


def test_commands_only_include_audited_sources_and_fresh_original_selection(success):
    ref, info, _ = success
    selected = worker.audit_successes([ref], shared, checker(info))
    paths = worker.paths_for("pilot001")
    commands = worker.commands(paths, "pilot001", selected, "/runtime/python", Path(paths["report_root"]) / "original_episodes.txt")
    assert [row[0] for row in commands] == ["export", "verify", "prepare"]
    assert all(row[2] <= 4200 for row in commands)
    assert selected[0]["trajectory"] in commands[0][1]
    assert paths["run_root"] in commands[2][1]
    assert commands[2][1][-6:] == ["--devices", "4", "--batch-size", "4", "--steps", "256"]
    assert "original-v2" not in repr(commands) and "original-reader-v4" not in repr(commands)


@pytest.mark.parametrize("version", ["v8", "v9"])
def test_source_embedding_requires_exact_reviewed_training_helper(version):
    payload = dict(kind=worker.KIND, run_id="pilot001", success_audits=[dict(
        path=worker.AFS + "/audit-radio-311-pilot003/report.json", content_sha256="a" * 64)], code_version=version)
    if version == "v9" and worker.CODE9_BUNDLE_SHA is None:
        with pytest.raises(ValueError, match="reviewed and frozen"): worker.build_source(payload)
        return
    source = worker.build_source(payload)
    compile(source, "prepared_job", "exec")
    assert "EXPECTED_CODE_INVENTORY" in source and "BOOTSTRAP_REPORT" in source
    assert ast.literal_eval(ast.parse(source).body[-1].value.args[0])["code_version"] == version
    with pytest.raises(ValueError, match="helpers changed"):
        worker.build_source(payload, training_source="print('not the reviewed helper')")


def test_native_freeze_changes_only_the_eight_reviewed_interfaces():
    import build_codex_native_collection_v9 as native
    assert worker.NATIVE_CHANGED_FILES == native.ALLOWED_CHANGES
    assert worker.CODE9_BUNDLE_SHA == native.CODE9_SHA
    assert len(worker.NATIVE_CHANGED_FILES) == 8
    assert "scripts/codex_collection_actions.py" in worker.NATIVE_CHANGED_FILES


def test_superseded_seven_file_native_bundle_cannot_prepare_training(monkeypatch):
    monkeypatch.setattr(worker, "CODE9_BUNDLE_NAME", "collection_code_v9.tar.gz")
    monkeypatch.setattr(worker, "CODE9_BUNDLE_SHA", "bb92c47c5ba669fdd7e054d84b4bcd759473ef32085e55594e981e7d139cac3d")
    payload = dict(kind=worker.KIND, run_id="pilot001", code_version="v9", success_audits=[dict(
        path=worker.AFS + "/audit-radio-311-pilot003/report.json", content_sha256="a" * 64)])
    with pytest.raises(ValueError, match="reviewed collector/reader interface"): worker.build_source(payload)


def test_never_reuses_existing_output(tmp_path, monkeypatch, success):
    ref, _, _ = success
    paths = worker.paths_for("pilot001")
    Path(paths["dataset_root"]).mkdir()
    with pytest.raises(ValueError, match="fresh"):
        worker.run(dict(kind=worker.KIND, run_id="pilot001", success_audits=[ref]), helpers=shared)
    assert not Path(paths["report_root"]).exists()


def test_failed_gate_never_produces_training_ready(tmp_path, monkeypatch, success):
    ref, _, _ = success
    # Missing original readiness is a hard pre-export failure, with durable evidence.
    report = worker.run(dict(kind=worker.KIND, run_id="pilot001", success_audits=[ref]), helpers=shared)
    assert report["state"] == "failed" and report["ready_for_training"] is False
    assert "training_payload" not in report
    assert not Path(worker.paths_for("pilot001")["dataset_root"]).exists()


@pytest.fixture
def pipeline(tmp_path, monkeypatch, success, original_chain):
    ref, info, _ = success
    runtime = tmp_path / "separate-runtime"
    monkeypatch.setattr(worker, "RUNTIME", str(runtime))
    python = runtime / "envs/b1k-eval/bin/python"
    python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
    paths = worker.paths_for("pilot001")
    code = Path(paths["code"]);code.mkdir()
    inventory = {"fixture-" + str(i): "f" * 64 for i in range(29)}
    monkeypatch.setattr(worker, "EXPECTED_CODE_INVENTORY", inventory)
    (code / "_CODEX_PILOT_CODE_READY").write_text(json.dumps(dict(bundle_sha256=worker.BUNDLE_SHA, source_files_sha256=inventory)))
    manifest = tmp_path / "bundle-v2/checkpoint_upload_plan.json"
    manifest.parent.mkdir();manifest.write_text("fixture checkpoint manifest")
    monkeypatch.setattr(worker, "CHECKPOINT_SHA", shared.sha(manifest))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda root: SimpleNamespace(free=300 * 1024**3))
    # Only this test substitutes I/O helpers; generated production source uses the pinned real helper.
    helper = SimpleNamespace(**vars(shared))
    helper.verify_code = lambda payload: {"fixture": True}
    helper.verify_plan = lambda payload: {"tasks": ["turning_on_radio"]}
    calls, deadlines = [], []
    def bounded(command, *, env, cwd, log_path, seconds, deadline, heartbeat):
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["JAX_PLATFORMS"] == "cpu"
        assert "codex_collection_teacher" not in repr(command)
        calls.append(command[2]);deadlines.append(deadline);log_path.write_text("fixture child completed")
        if command[2] == "export":Path(paths["dataset_root"]).mkdir()
        elif command[2] == "verify":(Path(paths["report_root"]) / "mixed_reader.json").write_text('{"complete":true}')
        else:
            run_root = Path(paths["run_root"]);run_root.mkdir(parents=True)
            (run_root / "training_plan.json").write_text('{"fixture":true}')
    helper.bounded_process = bounded
    monkeypatch.setitem(sys.modules, "codex_collection_trajectory", SimpleNamespace(inspect_codex_collection_trajectory=checker(info)))
    return SimpleNamespace(payload=dict(kind=worker.KIND, run_id="pilot001", success_audits=[ref]),
        helper=helper, calls=calls, deadlines=deadlines, paths=paths, source=original_chain)


def test_successful_pipeline_emits_exact_training_payload_without_gpu_or_model(pipeline):
    report = worker.run(pipeline.payload, helpers=pipeline.helper)
    assert pipeline.calls == ["export", "verify", "prepare"]
    assert report["state"] == "complete" and report["ready_for_training"] is True
    assert report["training_payload"] == json.loads(Path(report["training_payload_ref"]["path"]).read_text())
    assert report["training_payload"]["driver_sha256"] == worker.TRAINING_DRIVER_SHA
    assert report["training_payload"]["wall_timeout_seconds"] == 13800
    assert Path(report["original_episodes"]["path"]).read_text() == "28\n68\n102\n"
    assert not Path(pipeline.paths["training_report_root"]).exists()
    assert report['original_sources_before'] == report['original_sources_after']
    assert report['original_sources_after']['verified_files'] == 20
    assert sys.dont_write_bytecode is True


def test_native_pipeline_selects_v9_helpers_and_keeps_training_worker_unchanged(pipeline, success, monkeypatch):
    native_success(success, monkeypatch)
    pipeline.payload["code_version"] = "v9"
    original = Path(pipeline.paths["code"])
    pipeline.paths.update(worker.paths_for("pilot001", "v9"))
    code = Path(pipeline.paths["code"]); original.rename(code)
    (code / "_CODEX_PILOT_CODE_READY").write_text(json.dumps(dict(
        bundle_sha256=worker.CODE9_BUNDLE_SHA, source_files_sha256=worker.EXPECTED_CODE_INVENTORY)))
    original_bounded = pipeline.helper.bounded_process
    def checked(command, **kwargs):
        assert command[1].startswith(str(code) + "/scripts/")
        original_bounded(command, **kwargs)
    pipeline.helper.bounded_process = checked
    report = worker.run(pipeline.payload, helpers=pipeline.helper)
    assert report["state"] == "complete" and report["code_version"] == "v9"
    assert report["training_payload"]["code"] == str(code)
    assert report["training_payload"]["driver_sha256"] == worker.TRAINING_DRIVER_SHA
    assert report["selected_successes"][0]["collection_mode"] == "native_with_corrections"
    assert report["original_sources_before"] == report["original_sources_after"]


def test_changed_original_after_export_never_emits_training_ready(pipeline):
    bounded = pipeline.helper.bounded_process
    def change_after_prepare(command, **kwargs):
        bounded(command, **kwargs)
        if command[2] == 'prepare': pipeline.source.labels['subtask_segments.parquet'].write_bytes(b'changed in flight')
    pipeline.helper.bounded_process = change_after_prepare
    report = worker.run(pipeline.payload, helpers=pipeline.helper)
    assert pipeline.calls == ['export', 'verify', 'prepare']
    assert report['state'] == 'failed' and report['ready_for_training'] is False
    assert report['stage'] == 'verify_original_after'
    assert 'training_payload' not in report
    assert not (Path(pipeline.paths['report_root']) / 'training_payload.json').exists()


@pytest.mark.parametrize('outer_seconds,expected_budget', [(90, 90), (9000, 7800), (0, 7800)])
def test_shorter_outer_timer_and_child_deadlines_are_preserved(pipeline, monkeypatch, outer_seconds, expected_budget):
    timers, handlers = [], []
    clock = [1000.0]
    monkeypatch.setattr(worker.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(worker.signal, 'getitimer', lambda _: (outer_seconds, 7.0))
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: timers.append(args))
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: handlers.append(args) or 0)
    bounded = pipeline.helper.bounded_process
    def elapsed_child(*args, **kwargs):
        bounded(*args, **kwargs)
        clock[0] += 1
    pipeline.helper.bounded_process = elapsed_child
    report = worker.run(pipeline.payload, helpers=pipeline.helper)
    assert report['ready_for_training'] is True
    assert report['effective_wall_timeout_seconds'] == expected_budget
    assert pipeline.deadlines == [1000 + expected_budget] * 3
    assert timers[0] == (worker.signal.ITIMER_REAL, expected_budget)
    if outer_seconds:
        assert timers[-1] == (worker.signal.ITIMER_REAL, outer_seconds - 3, 7.0)
    else:
        assert timers[-1] == (worker.signal.ITIMER_REAL, 0)
    assert len(handlers) == 4


def test_expired_short_outer_timer_stops_before_export(pipeline, monkeypatch):
    monkeypatch.setattr(worker.time, 'monotonic', lambda: 1000.0)
    monkeypatch.setattr(worker.signal, 'getitimer', lambda _: (20.0, 0.0))
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: None)
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: 0)
    report = worker.run(pipeline.payload, helpers=pipeline.helper)
    assert report['state'] == 'failed' and report['error_type'] == 'WorkerDeadline'
    assert report['ready_for_training'] is False and not pipeline.calls
