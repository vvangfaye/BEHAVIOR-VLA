"""Inspect targeted training windows without downloading archive video members."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "formal_stage/bundle"
OUT = ROOT / "loss_spike_action_audit"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def parquet_members(read, length):
    """Walk tar headers by range; never read payloads merely to skip them."""
    offset, pending_name, result = 0, None, []
    for _ in range(512):
        if offset + 512 > length:
            raise ValueError("Truncated tar header")
        header = read(offset, 512)
        if header == bytes(512):
            return result
        info = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        start = offset + 512
        if info.size < 0 or start + info.size > length:
            raise ValueError("Tar member extends past object")
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
            if info.size > 1024 * 1024:
                raise ValueError("Oversized extended header")
            payload = read(start, info.size)
            if info.type == tarfile.GNUTYPE_LONGNAME:
                pending_name = payload.rstrip(b"\0").decode("utf-8")
            else:
                position = 0
                while position < len(payload):
                    space = payload.index(b" ", position)
                    size = int(payload[position:space])
                    if size <= space - position + 1 or position + size > len(payload):
                        raise ValueError("Invalid PAX record")
                    key, value = payload[space + 1:position + size - 1].split(b"=", 1)
                    if key == b"path":
                        pending_name = value.decode("utf-8")
                    position += size
        else:
            name = pending_name or info.name
            pending_name = None
            parts = Path(name).parts
            if info.isfile() and "data" in parts and name.endswith(".parquet"):
                if ".." in parts or Path(name).is_absolute():
                    raise ValueError("Unsafe member path")
                result.append({"name": name, "offset": start, "size": info.size})
        offset = start + ((info.size + 511) // 512) * 512
    raise ValueError("Too many tar headers")


def self_test():
    for format_ in (tarfile.PAX_FORMAT, tarfile.GNU_FORMAT):
        stream = io.BytesIO()
        name = "dataset/" + "nested/" * 25 + "data/chunk-000/file-000.parquet"
        with tarfile.open(fileobj=stream, mode="w", format=format_) as archive:
            for member, payload in [("dataset/videos/test.mp4", b"v" * 2_000_000), (name, b"PARQUET_TEST")]:
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        data = stream.getvalue()
        reads = []
        def read(offset, size):
            reads.append((offset, size))
            return data[offset:offset + size]
        members = parquet_members(read, len(data))
        assert len(members) == 1 and members[0]["name"] == name
        assert read(members[0]["offset"], members[0]["size"]) == b"PARQUET_TEST"
        assert sum(size for _, size in reads) < 10_000
        assert not any(512 <= offset < 2_000_512 for offset, _ in reads)
    print(json.dumps({"pax_and_gnu_long_names": True, "video_payloads_skipped": True}))


def source_function(path, name, owner=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if owner is None else next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == owner).body
    function = next(x for x in nodes if isinstance(x, ast.FunctionDef) and x.name == name)
    function.decorator_list = []
    scope = {"np": np}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def main():
    plan_path = ROOT / "loss_spike_data_access_plan.json"
    plan = json.loads(plan_path.read_text())
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    files = {key: BUNDLE / value for key, value in {
        "config": "openpi/src/openpi/training/config.py",
        "transforms": "openpi/src/openpi/transforms.py",
        "policy": "openpi/src/openpi/policies/b1k_policy.py",
        "robot": "openpi/src/openpi/configs/robots/b1k.py",
        "robot_base": "openpi/src/openpi/configs/robots/base_config.py"}.items()}
    for path in files.values():
        if sha(path.read_bytes()) != manifest[str(path.relative_to(BUNDLE))]:
            raise ValueError("Frozen transformation source changed")
    sys.path.insert(0, str(BUNDLE / "openpi/src"))
    from openpi.configs.robots.b1k import R1Pro
    build_mappings = source_function(files["config"], "_build_delta_mappings", "LeRobotB1KDataConfig")
    mappings = build_mappings(None, R1Pro)
    extract_state = source_function(files["policy"], "extract_state_from_proprio")
    delta = source_function(files["transforms"], "__call__", "MappedDeltaActions")
    normalize = source_function(files["transforms"], "_normalize", "Normalize")
    stats_path = ROOT / "cpu/bundle/inputs/norm_assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json"
    if sha(stats_path.read_bytes()) != "52d63e89f8359143d793ca80d0661c7facb0daff43c7d8c08f0d05572dd528fb":
        raise ValueError("Normalization identity changed")
    stats = json.loads(stats_path.read_text())["norm_stats"]["actions"]
    stats = SimpleNamespace(mean=np.array(stats["mean"]), std=np.array(stats["std"]))
    sys.path.insert(0, str(ROOT / "heldout20_baseline_vs_expert1k"))
    import controller
    client = controller.transport.aoss_client()
    lock, transferred = threading.Lock(), [0]

    def inspect(spec):
        source, root = spec["source"], spec["root_id"]
        head = client.head_object(Bucket=source["bucket"], Key=source["key"])
        etag = head["ETag"]
        if head["ContentLength"] != source["bytes"] or etag.strip('"').lower() != source["etag"].strip('"').lower():
            raise ValueError("Source object identity changed")
        def read(offset, size):
            if size == 0:
                return b""
            with lock:
                if transferred[0] + size > plan["maximum_parquet_and_metadata_bytes"]:
                    raise ValueError("Range read budget exceeded")
                transferred[0] += size
            response = client.get_object(Bucket=source["bucket"], Key=source["key"],
                Range=f"bytes={offset}-{offset+size-1}", IfMatch=etag)
            try:
                if response.get("ContentLength") != size or response.get("ContentRange") != f"bytes {offset}-{offset+size-1}/{source['bytes']}":
                    raise ValueError("Server did not honor the bounded byte range")
                data = response["Body"].read(size + 1)
            finally:
                response["Body"].close()
            if len(data) != size:
                raise ValueError("Short range read")
            return data
        members = parquet_members(read, source["bytes"])
        if not members:
            raise ValueError("No action Parquet member")
        tables, member_proofs = [], []
        for member in members:
            if member["size"] > plan["maximum_single_member_bytes"]:
                raise ValueError("Parquet member exceeds bound")
            data = read(member["offset"], member["size"])
            table = pq.read_table(io.BytesIO(data), columns=["episode_index", "frame_index", "action", "observation.state"])
            tables.append(table)
            member_proofs.append({**member, "sha256": sha(data)})
        data = pa.concat_tables(tables).to_pydict()
        episodes, frames = np.array(data["episode_index"]), np.array(data["frame_index"])
        actions = np.asarray(data["action"], dtype=np.float32)
        states = np.asarray(data["observation.state"], dtype=np.float32)
        windows = []
        for ref in spec["candidate_windows"]:
            selection = np.flatnonzero((episodes == ref["episode_index"]) & (frames >= ref["frame"]) & (frames < ref["frame"] + 32))
            selection = selection[np.argsort(frames[selection])]
            if len(selection) != 32 or not np.array_equal(frames[selection], np.arange(ref["frame"], ref["frame"] + 32)):
                raise ValueError("Candidate window missing or duplicated frames")
            raw = actions[selection].copy()
            state = extract_state(states[selection[0]], R1Pro)
            transformed = delta(SimpleNamespace(mappings=mappings), {"state": state, "actions": raw.copy()})["actions"]
            normalized = normalize(None, transformed, stats)
            if not all(np.isfinite(x).all() for x in (raw, state, transformed, normalized)):
                raise ValueError("Nonfinite action or state")
            per_dim = np.max(np.abs(normalized), axis=0)
            windows.append({"global_index": ref["global_index"], "frame": ref["frame"],
                "task": ref["task_name"], "episode_index": ref["episode_index"],
                "normalized_abs_max": float(np.max(per_dim)),
                "normalized_mean_square": float(np.mean(np.square(normalized))),
                "largest_dimension": int(np.argmax(per_dim)), "normalized_abs_max_by_dimension": per_dim.tolist(),
                "raw_abs_max": float(np.max(np.abs(raw))), "delta_abs_max": float(np.max(np.abs(transformed)))})
        result = {"root_id": root, "source_etag": etag, "source_bytes": source["bytes"],
                  "whole_archive_sha256_recomputed": False, "members": member_proofs, "windows": windows}
        save(OUT / "roots" / (root + ".json"), result)
        print(json.dumps({"root_completed": root, "windows": len(windows)}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, plan["source_objects"]))
    windows = [{"root_id": result["root_id"], **window} for result in results for window in result["windows"]]
    if len(windows) != plan["candidate_windows"]:
        raise ValueError("Incomplete candidate coverage")
    result = {"complete": True, "window_count": len(windows), "archive_count": len(results),
              "range_bytes": transferred[0], "normalization_sha256": sha(stats_path.read_bytes()),
              "plan_sha256": sha(plan_path.read_bytes()),
              "source_proofs": {name: sha(path.read_bytes()) for name, path in files.items()},
              "delta_mappings": mappings, "normalization_kind": "mean_std",
              "windows_ranked_by_magnitude": sorted(windows, key=lambda w: w["normalized_abs_max"], reverse=True),
              "data_modified": False, "videos_downloaded": False,
              "limitations": ["Only the full-episode spike cohort is covered.",
                  "Large values do not by themselves establish corrupted data or the cause of the observed loss.",
                  "Object ranges were pinned by size and ETag; whole archive hashes were not recomputed."]}
    save(OUT / "result.json", result)
    print(json.dumps({key: result[key] for key in ("complete", "window_count", "archive_count", "range_bytes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        main()
