"""RGB views preserve original data, offsets and metadata without bulk copying."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_subtask_rgb_view import RGB_KEYS, prepare_rgb_view


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "original"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    features = {key: dict(dtype="video", shape=[240, 240, 3], info=dict(is_depth_map=False)) for key in RGB_KEYS}
    features.update({f"depth-{i}": dict(dtype="video", shape=[240, 240, 1], info=dict(is_depth_map=True)) for i in range(3)})
    features.update({"action": dict(dtype="float32", shape=[23]), "observation.state": dict(dtype="float32", shape=[57])})
    info = dict(codebase_version="v3.0", fps=30, features=features,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    (root / "meta/info.json").write_text(json.dumps(info))
    # No dependency on pyarrow is needed to prove these identities remain byte-exact.
    (root / "meta/episodes/chunk-000/file-000.parquet").write_bytes(b"opaque source metadata: episode42 from900 to1000")
    (root / "meta/tasks.jsonl").write_bytes(b'{"task_index":0,"task_name":"task","task":"Do task"}\n')
    (root / "data/chunk-000/file-000.parquet").write_bytes(b"opaque controls, never read by preparation")
    (root / "videos/original.mp4").write_bytes(b"opaque video, never read by preparation")
    return root


def test_six_stream_view_has_only_three_rgb_and_original_metadata_is_unchanged(source, tmp_path):
    before = {p.relative_to(source): p.read_bytes() for p in (source / "meta").rglob("*") if p.is_file()}
    manifest = prepare_rgb_view(source, tmp_path / "view")
    view = Path(manifest["dataset_root"])
    info = json.loads((view / "meta/info.json").read_text())
    assert {key for key, value in info["features"].items() if value["dtype"] == "video"} == RGB_KEYS
    assert manifest["removed_depth_video_keys"] == ["depth-0", "depth-1", "depth-2"]
    for relative, contents in before.items():
        assert (source / relative).read_bytes() == contents
        if relative != Path("meta/info.json"): assert (view / relative).read_bytes() == contents
    assert manifest["source_info_sha256"] != manifest["derived_info_sha256"]


def test_data_video_paths_and_original_episode_offsets_are_preserved(source, tmp_path):
    view = tmp_path / "view"; manifest = prepare_rgb_view(source, view)
    assert (view / "data").is_symlink() and (view / "videos").is_symlink()
    assert (view / "data/chunk-000/file-000.parquet").samefile(source / "data/chunk-000/file-000.parquet")
    assert (view / "videos/original.mp4").samefile(source / "videos/original.mp4")
    before, after = [json.loads((root / "meta/info.json").read_text()) for root in (source, view)]
    assert before["data_path"] == after["data_path"] and before["video_path"] == after["video_path"]
    assert before["fps"] == after["fps"]
    assert manifest["original_episode_ids_and_offsets_preserved"]


def test_already_rgb_info_stays_byte_exact_in_fresh_outputs(source, tmp_path):
    path = source / "meta/info.json"; info = json.loads(path.read_text())
    info["features"] = {key: value for key, value in info["features"].items() if not key.startswith("depth-")}
    path.write_text(json.dumps(info, indent=4))
    one = prepare_rgb_view(source, tmp_path / "one")
    two = prepare_rgb_view(source, tmp_path / "two")
    assert not one["removed_depth_video_keys"] and not two["removed_depth_video_keys"]
    assert one["source_info_sha256"] == one["derived_info_sha256"] == two["derived_info_sha256"]
    assert path.read_bytes() == (tmp_path / "one/meta/info.json").read_bytes()


def test_output_collision_and_nested_output_fail_without_touching_sources(source, tmp_path):
    output = tmp_path / "view"; prepare_rgb_view(source, output)
    manifest = (output / "rgb_view_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="exists"): prepare_rgb_view(source, output)
    assert (output / "rgb_view_manifest.json").read_bytes() == manifest
    with pytest.raises(ValueError, match="separate"): prepare_rgb_view(source, source / "view")
    dangling = tmp_path / "dangling"; dangling.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="exists"): prepare_rgb_view(source, dangling)


@pytest.mark.parametrize("damage", ["missing_rgb", "unmarked_extra_video", "depth_bad_type", "missing_raw", "collected"])
def test_invalid_or_collected_sources_fail_closed(source, tmp_path, damage):
    path = source / "meta/info.json"; info = json.loads(path.read_text())
    if damage == "missing_rgb": del info["features"][next(iter(RGB_KEYS))]
    elif damage == "unmarked_extra_video": info["features"]["other"] = dict(dtype="video", shape=[240, 240, 3])
    elif damage == "depth_bad_type": info["features"]["depth-0"]["info"]["is_depth_map"] = "true"
    elif damage == "missing_raw": (source / "videos/original.mp4").unlink(); (source / "videos").rmdir()
    else: (source / "meta/collection_provenance.json").write_text("[]")
    path.write_text(json.dumps(info))
    output = tmp_path / "bad"
    with pytest.raises(ValueError): prepare_rgb_view(source, output)
    assert not output.exists()


def test_cli_is_stdlib_and_does_not_materialize_raw_data(source, tmp_path):
    output = tmp_path / "cli"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_subtask_rgb_view.py")),
        "--dataset-root", str(source), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["removed_depth_video_keys"] == ["depth-0", "depth-1", "depth-2"]
    assert (output / "data").is_symlink() and (output / "videos").is_symlink()
