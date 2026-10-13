"""Offline preparation builder tests; no credentials, URLs or network access."""
import ast
import json
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sco import build_codex_subtask_preparation as builder


def signed_fixture(tmp_path):
    inventory = {"required_files": [{"relative_path": f"data-{i}"} for i in range(4)]}
    rows = [dict(relative_path=f"data-{i}", bytes=i + 1, etag=f"etag-{i}",
                 url=f"https://example.invalid/object-{i}?signature=PRIVATE") for i in range(4)]
    path = tmp_path / "signed.json"
    path.write_text(json.dumps(rows))
    return inventory, path, rows


@pytest.mark.parametrize("damage", ["extra", "duplicate", "missing", "size", "etag", "url"])
def test_private_signed_inventory_has_exact_bounded_source_coverage(tmp_path, damage):
    inventory, path, rows = signed_fixture(tmp_path)
    if damage == "extra": rows[0]["relative_path"] = "unselected-file"
    elif damage == "duplicate": rows[0]["relative_path"] = rows[1]["relative_path"]
    elif damage == "missing": rows.pop()
    elif damage == "size": rows[0]["bytes"] = False
    elif damage == "etag": rows[0]["etag"] = ""
    else: rows[0]["url"] = "http://example.invalid/?signature=PRIVATE"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError) as caught:
        builder.signed_files(path, inventory)
    assert "PRIVATE" not in str(caught.value)


def test_private_driver_compiles_with_verified_bundle_and_no_authentication(tmp_path):
    inventory, path, rows = signed_fixture(tmp_path)
    signed = builder.signed_files(path, inventory)
    plan = dict(kind=builder.KIND, bundle={"bytes": 1, "sha256": "a" * 64},
                payload={"max_seconds": 3600}, target_subtask=builder.TARGET_SUBTASK)
    plan["fingerprint"] = builder.sha(builder.canonical(plan).encode())
    driver = builder.render_driver(plan, signed, bundle_url="https://example.invalid/bundle?signature=PRIVATE")
    ast.parse(driver)
    assert "CODEX_PILOT_REPORT_PUT" in driver and "BOOTSTRAP_REPORT" in driver
    assert "assert len(raw)==plan['bundle']['bytes']" in driver
    assert "oss2.Auth" not in driver and "ossutil" not in driver
    assert "_ORIGINAL_DATA_READY.json" in driver and "required_subtask" in driver
    assert "receipt['selected_episodes']==[plan['episode']]" in driver
    assert "output/'demonstrations'/str(plan['episode'])" in driver
    assert "[10020]" not in driver
    output = tmp_path / "driver.py"
    builder.write_new(output, driver.encode(), private=True)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        builder.write_new(output, b"overwrite", private=True)


def test_changed_plan_cannot_render_private_driver():
    plan = {"kind": builder.KIND, "fingerprint": "a" * 64}
    with pytest.raises(ValueError, match="changed"):
        builder.render_driver(plan, [])


def test_selected_camera_source_uses_published_metadata_and_original_labels():
    root = builder.ROOT
    metadata = root / "eval_results/codex_pilot_20261011/original_metadata"
    selection = root.parent / "analysis/codex_teacher_pilot_20261011/demonstration_selection.json"
    if not metadata.exists() or not selection.exists():
        pytest.skip("optional local published metadata fixture")
    spec = dict(episode_index=7003, task_name="attach_a_camera_to_a_tripod", task_instance_id=4,
        raw_path="/mnt/afs/260010165/behavior_1k/raw_full100_20261007_1308/2026-challenge-rawdata/task-0035/episode_00350040.hdf5",
        raw_sha256="1b73188eac4db6035dcd6573af4870c4d70d447a29c9087e4b40ec1d80101dcb",
        raw_episode_id=350040, task_index=35, subtask="pick up from digital camera to floors nlswvt")
    inventory = builder.source_inventory(metadata, selection, spec)
    assert inventory["episode"]["length"] == 3551
    assert inventory["target_segments"][0]["start"] == 520
    assert inventory["raw_path"] == spec["raw_path"]
    with pytest.raises(ValueError, match="identity"):
        builder.source_inventory(metadata, selection, {**spec, "task_instance_id": 5})
