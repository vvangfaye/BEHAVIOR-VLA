"""Metadata-only recovery, immutable identities and action-boundary regressions."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training import recovery_index as recovery


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "original"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(dict(codebase_version="v3.0", fps=30)))
    (root / "meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0, task_name="task", task="Do the task.")) + "\n")
    metadata = [dict(episode_index=ep, length=length, task_index=0, task_instance_id=instance)
                for ep, length, instance in ((42, 100, 3), (71, 96, 9), (305, 100, 5))]
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    rows = [dict(episode_index=42, start=0, end=5, subtask=None, annotated=True),
            dict(episode_index=42, start=5, end=37, subtask="pick", annotated=True),
            dict(episode_index=42, start=37, end=50, subtask="short", annotated=True),
            dict(episode_index=42, start=50, end=100, subtask="place", annotated=True),
            dict(episode_index=71, start=0, end=1, subtask="", annotated=True),
            dict(episode_index=71, start=1, end=96, subtask="pick", annotated=True),
            dict(episode_index=305, start=0, end=100, subtask="open", annotated=True)]
    labels = tmp_path / "segments.parquet"
    pq.write_table(pa.Table.from_pylist(rows), labels)
    return root, labels


def prepare(source, ids=(42, 71), **kwargs):
    return recovery.prepare_index(*source, ids, repo_id="source/old", root_id="old", source_kind=kwargs.pop("source_kind", "recovered"), **kwargs)


def test_recovery_keeps_original_frame_ids_does_not_need_any_video(source):
    root, labels = source; original = labels.read_bytes()
    index = prepare(source, held_out_instances=[["task", 9]])
    assert [row["episode_index"] for row in index["episodes"]] == [42]
    assert [(row["start"], row["end"]) for row in index["intervals"]] == [(5, 37), (50, 100)]
    assert [recovery.valid_anchor_count(row, 32) for row in index["intervals"]] == [1, 19]
    stats = index["provenance"]["statistics"]
    assert stats["valid_anchors"] == 20 and stats["short_segments"] == 1
    assert stats["held_out_excluded_episodes"] == 1
    assert not index["provenance"]["videos_read"] and labels.read_bytes() == original
    recovery.verify_source_files(index)


def test_indices_are_order_independent_and_holdout_removes_entire_instance(source):
    assert prepare(source, (42, 71)) == prepare(source, (71, 42))
    index = prepare(source, held_out_instances=[["task", 3]])
    assert [row["episode_index"] for row in index["episodes"]] == [71]
    assert index["intervals"][0]["start"] == 1


def test_explicit_leading_fill_includes_prefix_without_rewriting_human_labels(source, tmp_path):
    original = source[1].read_bytes()
    index = prepare(source, fill_leading_with_first_subtask=True)
    first = index["intervals"][0]
    assert (first["start"], first["end"], first["subtask"]) == (0, 37, "pick")
    assert first["imputed_leading_end"] == first["original_label_start"] == 5
    stats = index["provenance"]["statistics"]
    assert stats["imputed_leading_episodes"] == 2 and stats["imputed_leading_frames"] == 6
    assert stats["trusted_labelled_frames"] == 190 and stats["valid_anchors"] == 90
    assert index["provenance"]["leading_fill_records"][0]["is_human_annotated"] is False
    assert source[1].read_bytes() == original
    recovery.verify_source_files(index)
    # Prefix samples actually reach the existing training sampler.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    from prepare_subtask_mixture import build_manifest, make_source
    path = tmp_path / "filled-index.json"
    path.write_text(json.dumps(index))
    _, plan = build_manifest([make_source("recovered", [path], 1.0)], end_step=100)
    samples = [plan.reference(i) for i in range(12800)]
    assert recovery.valid_anchor_count(first, 32) == 6
    assert any(row["episode_index"] == 42 and row["frame"] < 5 for row in samples)


def test_leading_fill_preserves_holdout_and_never_fills_middle_or_tail(source):
    index = prepare(source, held_out_instances=[["task", 9]], fill_leading_with_first_subtask=True)
    assert [r["episode_index"] for r in index["provenance"]["leading_fill_records"]] == [42]
    rows = pq.read_table(source[1]).to_pylist()
    rows[2]["subtask"] = ""
    pq.write_table(pa.Table.from_pylist(rows), source[1])
    with pytest.raises(ValueError, match="gaps or tails"):
        prepare(source, (42,), fill_leading_with_first_subtask=True)


def test_leading_fill_refuses_untrusted_source_and_wrong_source_kind(source):
    with pytest.raises(ValueError, match="only supported"):
        prepare(source, (305,), source_kind="replay", fill_leading_with_first_subtask=True)
    rows = pq.read_table(source[1]).to_pylist()
    rows[1]["annotated"] = False
    pq.write_table(pa.Table.from_pylist(rows), source[1])
    with pytest.raises(ValueError, match="trusted first"):
        prepare(source, (42,), fill_leading_with_first_subtask=True)


@pytest.mark.parametrize("damage", ["gap", "overlap", "past_end", "missing", "bad_type", "flag_conflict"])
def test_invalid_source_intervals_fail_closed(source, damage):
    root, labels = source; rows = pq.read_table(labels).to_pylist()
    if damage == "gap": rows[1]["start"] = 6
    elif damage == "overlap": rows[1]["start"] = 4
    elif damage == "past_end": rows[3]["end"] = 101
    elif damage == "missing": rows.pop(1)
    elif damage == "bad_type": rows[0]["start"] = True
    else:
        for row in rows: row["is_annotated"] = row["annotated"]
        rows[1]["is_annotated"] = False
    if damage == "bad_type":
        metadata = [dict(episode_index=42, length=100, task_name="task", task_instance_id=3)]
        with pytest.raises(ValueError):
            recovery.build_index(metadata, rows, dataset_root=root, repo_id="r", root_id="root", source_kind="recovered", episode_indices=[42])
    else:
        pq.write_table(pa.Table.from_pylist(rows), labels)
        with pytest.raises(ValueError): prepare(source)


@pytest.mark.parametrize("ids", [(42, 42), (999,), (305,)])
def test_recovery_rejects_duplicate_unknown_or_wrong_pool_episodes(source, ids):
    with pytest.raises(ValueError): prepare(source, ids)


def test_replay_rejects_empty_source_and_recovers_only_trusted_segments(source):
    with pytest.raises(ValueError): prepare(source, (42,), source_kind="replay")
    replay = prepare(source, (305,), source_kind="replay")
    assert replay["provenance"]["statistics"]["valid_anchors"] == 69
    root, labels = source; rows = pq.read_table(labels).to_pylist()
    rows[3]["annotated"] = False
    pq.write_table(pa.Table.from_pylist(rows), labels)
    recovered = prepare(source, (42,))
    assert recovered["provenance"]["statistics"]["valid_anchors"] == 1
    assert recovered["provenance"]["statistics"]["unannotated_frames"] == 50


def test_source_hashes_are_relative_to_root_and_changed_metadata_is_rejected(source, tmp_path):
    index = prepare(source)
    for item in index["provenance"]["source_files"]:
        if item["role"] == "metadata": assert not Path(item["path"]).is_absolute()
    target = tmp_path / "index.json"; target.write_text(json.dumps(index))
    assert recovery.load_index(target) == index
    (source[0] / "meta/info.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"): recovery.load_index(target)
    assert recovery.load_index(target, verify_sources=False) == index


def test_external_labels_may_be_unavailable_after_relocation_but_metadata_is_checked(source):
    index = prepare(source)
    source[1].unlink()
    recovery.verify_source_files(index, include_labels=False)
    with pytest.raises(ValueError): recovery.verify_source_files(index)


def test_fingerprint_tampering_and_rehashed_illegal_anchor_fail(source):
    index = prepare(source)
    damaged = copy.deepcopy(index); damaged["intervals"][0]["start"] = 6
    with pytest.raises(ValueError, match="fingerprint"): recovery.validate_index(damaged)
    damaged["fingerprint"] = recovery.canonical_digest({k: v for k, v in damaged.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="horizon"): recovery.validate_index(damaged)


def test_collected_requires_content_identity_and_success_export_evidence(source):
    with pytest.raises(ValueError, match="trajectory provenance"):
        prepare(source, (305,), source_kind="collected")
    metadata = [dict(episode_index=305, length=100, task_name="task", task_instance_id=5)]
    labels = [dict(episode_index=305, start=0, end=100, subtask="open")]
    args = dict(dataset_root=source[0], repo_id="new", root_id="new", source_kind="collected", episode_indices=[305])
    with pytest.raises(ValueError, match="content identity"): recovery.build_index(metadata, labels, **args)
    proof = dict(episode_index=305, length=100, task_name="task", task_instance_id=5, source_kind="collected",
                 content_sha256="c" * 64, trial_id="trial-1", source_episode_index=891)
    index = recovery.build_index(metadata, labels, provenance=dict(collected_episodes=[proof]), **args)
    assert index["episodes"][0]["content_sha256"] == "c" * 64
    assert index["episodes"][0]["source_episode_index"] == 891


@pytest.mark.parametrize("fill_leading", [False, True])
def test_cli_reads_metadata_only_and_refuses_overwrite(source, tmp_path, fill_leading):
    root, labels = source
    ids = tmp_path / "episodes.txt"; ids.write_text("42\n71\n")
    holdout = tmp_path / "holdout.json"; holdout.write_text('[["task",9]]')
    target = tmp_path / "out.json"
    script = Path(__file__).resolve().parents[4] / "scripts/prepare_subtask_recovery.py"
    cmd = [sys.executable, str(script), "--dataset-root", str(root), "--labels", str(labels),
           "--episodes-file", str(ids), "--held-out-instances", str(holdout), "--source-kind", "recovered",
           "--root-id", "old", "--repo-id", "source/old", "--output", str(target)]
    if fill_leading:
        cmd.append("--fill-leading-with-first-subtask")
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid_anchors"] == (25 if fill_leading else 20)
    original = target.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    assert target.read_bytes() == original


def test_metadata_index_accepts_actual_existing_standard_export_if_available(tmp_path):
    """Use an already verified real export; do not open its video/trajectory bytes."""
    import tarfile
    repo = Path(__file__).resolve().parents[4]
    archive = repo / "eval_results/sco_subtask_collection_20261011/gate4_native_dataset.tar"
    if not archive.is_file(): pytest.skip("Existing standard-resource export is not present on this checkout")
    root = tmp_path / "actual_export"; root.mkdir()
    wanted = {"_EXPORTED", "_VERIFIED_EXPORT", "format_verification.json", "export_report.json"}
    with tarfile.open(archive) as source:
        candidates = [member for member in source.getmembers() if member.isfile() and Path(member.name).name == "_VERIFIED_EXPORT"]
        assert len(candidates) == 1
        prefix = str(Path(candidates[0].name).parent) + "/"
        for member in source.getmembers():
            if not member.isfile() or not member.name.startswith(prefix): continue
            relative = Path(member.name[len(prefix):])
            if not relative.parts or ".." in relative.parts or relative.is_absolute(): raise ValueError("Unsafe fixture archive")
            if relative.parts[0] != "meta" and relative.name not in wanted: continue
            destination = root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
            assert member.size < 32 * 1024**2
            with source.extractfile(member) as stream: destination.write_bytes(stream.read())
    proof = json.loads((root / "meta/collection_provenance.json").read_text())
    report = json.loads((root / "export_report.json").read_text())
    ids = [row["episode_index"] for row in proof if row["source_kind"] == "collected"]
    index = recovery.prepare_index(root, root / "meta/subtask_segments.parquet", ids,
        repo_id=report["repo_id"], root_id="actual_native", source_kind="collected")
    recovery.verify_source_files(index)
    assert index["episodes"][0]["length"] == 131
    assert index["provenance"]["statistics"]["valid_anchors"] == 100
    assert not (root / "videos").exists() and not (root / "data").exists()


@pytest.mark.parametrize("kind,expected", [("recovered", [42, 71]), ("replay", [305])])
def test_cli_audit_selection_and_consistent_campaign_holdout(source, tmp_path, kind, expected):
    root, labels = source
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(status="passed", dropped_episode_indices=[42, 71],
        selected_episode_indices=[305], training_contract=dict(annotation_sha256=recovery.file_sha256(labels)))))
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps(dict(groups=[dict(plan=dict(held_out_instances=[])), dict(plan=dict(held_out_instances=[]))])))
    target = tmp_path / "selected.json"
    script = Path(__file__).resolve().parents[4] / "scripts/prepare_subtask_recovery.py"
    cmd = [sys.executable, str(script), "--dataset-root", str(root), "--labels", str(labels),
           "--episode-audit-report", str(audit), "--held-out-instances", str(campaign), "--source-kind", kind,
           "--root-id", "old", "--repo-id", "old/original", "--output", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    index = recovery.load_index(target)
    assert [row["episode_index"] for row in index["episodes"]] == expected
    assert index["provenance"]["selection_input"]["sha256"] == recovery.file_sha256(audit)


@pytest.mark.parametrize("damage", ["inconsistent_split", "collected_audit", "labels_changed", "duplicate_id", "failed_audit"])
def test_cli_audit_inputs_fail_closed(source, tmp_path, damage):
    root, labels = source
    audit = tmp_path / "audit.json"
    data = dict(status="passed", dropped_episode_indices=[42, 71], selected_episode_indices=[305],
                training_contract=dict(annotation_sha256=recovery.file_sha256(labels)))
    if damage == "labels_changed": data["training_contract"]["annotation_sha256"] = "f" * 64
    if damage == "duplicate_id": data["dropped_episode_indices"] = [42, 42]
    if damage == "failed_audit": data["status"] = "failed"
    audit.write_text(json.dumps(data))
    held = tmp_path / "campaign.json"
    held.write_text(json.dumps(dict(groups=[dict(plan=dict(held_out_instances=[])),
        dict(plan=dict(held_out_instances=[["task", 9]] if damage == "inconsistent_split" else []))])))
    target = tmp_path / "fail.json"
    script = Path(__file__).resolve().parents[4] / "scripts/prepare_subtask_recovery.py"
    cmd = [sys.executable, str(script), "--dataset-root", str(root), "--labels", str(labels),
           "--episode-audit-report", str(audit), "--held-out-instances", str(held),
           "--source-kind", "collected" if damage == "collected_audit" else "recovered",
           "--root-id", "old", "--repo-id", "old/original", "--output", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0 and not target.exists()


def test_rgb_view_manifest_and_actual_lerobot_task_metadata_are_hashed(source, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    from prepare_subtask_rgb_view import RGB_KEYS, prepare_rgb_view
    root, labels = source
    (root / "data").mkdir(); (root / "videos").mkdir()
    info = dict(codebase_version="v3.0", fps=30,
                features={key: dict(dtype="video", shape=[224, 224, 3]) for key in RGB_KEYS})
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([dict(task_index=0, task="task")]), root / "meta/tasks.parquet")
    view = tmp_path / "rgb"
    prepare_rgb_view(root, view)
    index = recovery.prepare_index(view, labels, [42], repo_id="expert", root_id="expert_rgb", source_kind="recovered")
    sources = {row["path"]: row for row in index["provenance"]["source_files"] if row["role"] == "metadata"}
    assert {"meta/tasks.parquet", "rgb_view_manifest.json"} <= sources.keys()
    recovery.verify_source_files(index)
    (view / "rgb_view_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"): recovery.verify_source_files(index)
