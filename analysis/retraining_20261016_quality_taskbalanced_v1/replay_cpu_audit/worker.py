"""Bounded CPU-only inspection of the replay windows, using frozen transforms."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap():
    cfg = json.loads((HERE / "runtime.private.json").read_text())
    output = Path("/data/oss_bucket_0") / cfg["output_key"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name, expected in json.loads((HERE / "bundle_manifest.json").read_text()).items():
            if digest(HERE / name) != expected:
                raise ValueError("Frozen audit bundle changed")
        deadline = time.monotonic() + cfg["maximum_worker_seconds"]
        scratch = Path("/var/tmp/wangfaye/replay-action-audit") / cfg["run_id"]
        scratch.mkdir(parents=True, exist_ok=False)
        save(output / "started.json", {"run_id": cfg["run_id"], "pid": os.getpid(),
            "training": False, "gpus": 0, "started_at": time.time()})
        item = cfg["environment"]
        archive = scratch / "environment.tar.gz"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(item["get_url"], headers={"If-Match": '"' + item["etag"] + '"'})
        size, checksum = 0, hashlib.sha256()
        with opener.open(request, timeout=60) as response, archive.open("wb") as target:
            if int(response.headers["Content-Length"]) != item["bytes"]:
                raise ValueError("Environment size changed")
            for block in iter(lambda: response.read(8 * 1024**2), b""):
                if time.monotonic() > deadline or size + len(block) > item["bytes"]:
                    raise TimeoutError("Environment staging budget exceeded")
                target.write(block)
                checksum.update(block)
                size += len(block)
        if size != item["bytes"]:
            raise ValueError("Incomplete environment archive")
        environment = scratch / "environment"
        environment.mkdir()
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(environment)], check=True,
                       timeout=max(1, deadline - time.monotonic()))
        archive.unlink()
        python = environment / "bin/python"
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
        unpack = environment / "bin/conda-unpack"
        if unpack.exists():
            subprocess.run([str(python), "-E", "-s", str(unpack)], env=env, check=True,
                           timeout=max(1, deadline - time.monotonic()))
        save(output / "runtime.json", {"bytes": size, "sha256": checksum.hexdigest(),
            "source_etag": item["etag"], "training": False, "gpus": 0})
        subprocess.run([str(python), str(HERE / "worker.py"), "audit", str(output),
                        str(max(1, deadline - time.monotonic()))], env=env, check=True,
                       timeout=max(1, deadline - time.monotonic()))
    except BaseException as exc:
        if not (output / "failure.json").exists():
            save(output / "failure.json", {"phase": "bootstrap", "error_type": type(exc).__name__,
                "observed_at": time.time(), "training": False})
        raise SystemExit(1)


class RangeSource(io.RawIOBase):
    def __init__(self, item, reserve, deadline):
        self.item, self.reserve, self.deadline = item, reserve, deadline
        self.position = 0
        tail_bytes = max(65536, item.get("footer_bytes", 0) + 8)
        self.allowed = [(0, 4), (max(0, item["bytes"] - tail_bytes), item["bytes"])]
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.item["bytes"] + offset
        if not 0 <= position <= self.item["bytes"]:
            raise ValueError("Invalid Parquet seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.item["bytes"] - self.position if size < 0 else min(size, self.item["bytes"] - self.position)
        if size == 0: return b""
        start, end = self.position, self.position + size
        if not any(lo <= start and end <= hi for lo, hi in self.allowed):
            raise ValueError("Attempt to read unrequested Parquet columns")
        if time.monotonic() > self.deadline:
            raise TimeoutError("Audit deadline reached")
        self.reserve(size)
        request = urllib.request.Request(self.item["get_url"], headers={
            "Range": f"bytes={start}-{end-1}", "If-Match": '"' + self.item["etag"] + '"'})
        with self.opener.open(request, timeout=60) as response:
            expected = f"bytes {start}-{end-1}/{self.item['bytes']}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Object identity or requested range changed")
            block = response.read(size + 1)
        if len(block) != size:
            raise ValueError("Incomplete Parquet range")
        self.position = end
        return block

    def readinto(self, target):
        block = self.read(len(target))
        target[:len(block)] = block
        return len(block)


def audit(output, duration):
    import numpy as np
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    import transform_helpers
    transform_helpers.np = np
    sys.path.insert(0, str(HERE / "source/openpi/src"))
    from openpi.configs.robots.b1k import R1Pro
    cfg = json.loads((HERE / "runtime.private.json").read_text())
    plan = json.loads((HERE / "column_plan.json").read_text())
    estimates = {x["relative_path"]: x for x in json.loads((HERE / "footer_probe.json").read_text())["objects"]}
    urls = {x["relative_path"]: x["get_url"] for x in cfg["objects"]}
    src = HERE / "source/openpi/src/openpi"
    get = transform_helpers.source_function
    mappings = get(src / "training/config.py", "_build_delta_mappings", "LeRobotB1KDataConfig")(None, R1Pro)
    extract = get(src / "policies/b1k_policy.py", "extract_state_from_proprio")
    delta = get(src / "transforms.py", "__call__", "MappedDeltaActions")
    normalize = get(src / "transforms.py", "_normalize", "Normalize")
    stats = json.loads((HERE / "norm_stats.json").read_text())["norm_stats"]["actions"]
    stats = SimpleNamespace(mean=np.asarray(stats["mean"]), std=np.asarray(stats["std"]))
    deadline = time.monotonic() + duration
    lock, transferred = threading.Lock(), [0]
    all_windows, completed = [], []

    def reserve(size):
        with lock:
            if transferred[0] + size > cfg["maximum_data_range_bytes"]:
                raise ValueError("Data transfer budget exceeded")
            transferred[0] += size

    def inspect(item):
        source = dict(item["source"], get_url=urls[item["source"]["relative_path"]])
        estimate = estimates[source["relative_path"]]
        source["footer_bytes"] = estimate["footer_bytes"]
        stream = RangeSource(source, reserve, deadline)
        parquet = pq.ParquetFile(stream, pre_buffer=False)
        if parquet.metadata.num_row_groups != estimate["row_group_count"]:
            raise ValueError("Parquet footer changed")
        for i in estimate["selected_row_groups"]:
            group = parquet.metadata.row_group(i)
            for j in range(group.num_columns):
                column = group.column(j)
                if any(column.path_in_schema == name or column.path_in_schema.startswith(name + ".")
                       for name in plan["required_columns"]):
                    start = column.dictionary_page_offset if column.has_dictionary_page else column.data_page_offset
                    stream.allowed.append((start, start + column.total_compressed_size))
        table = parquet.read_row_groups(estimate["selected_row_groups"], columns=plan["required_columns"], use_threads=False).to_pydict()
        episodes, frames = np.asarray(table["episode_index"]), np.asarray(table["frame_index"])
        indices = np.asarray(table["index"])
        actions, states = np.asarray(table["action"], dtype=np.float32), np.asarray(table["observation.state"], dtype=np.float32)
        rows = []
        for ref in item["windows"]:
            selection = np.flatnonzero((episodes == ref["episode_index"]) & (frames >= ref["frame"]) & (frames < ref["frame"] + 32))
            selection = selection[np.argsort(frames[selection])]
            if len(selection) != 32 or not np.array_equal(frames[selection], np.arange(ref["frame"], ref["frame"] + 32)) or not np.array_equal(indices[selection], np.arange(ref["dataset_index_start"], ref["dataset_index_end_exclusive"])):
                raise ValueError("Window identity or frame order differs")
            raw = actions[selection].copy()
            state = extract(states[selection[0]], R1Pro)
            transformed = delta(SimpleNamespace(mappings=mappings), {"actions": raw.copy(), "state": state})["actions"]
            normalized = normalize(None, transformed, stats)
            if not np.isfinite(normalized).all():
                raise ValueError("Nonfinite normalized target")
            dim = int(np.argmax(np.max(np.abs(normalized), axis=0)))
            rows.append({"global_index": ref["global_index"], "episode_index": ref["episode_index"],
                "frame": ref["frame"], "task": ref["task_name"], "largest_dimension": dim,
                "normalized_abs_max": float(np.max(np.abs(normalized))),
                "normalized_mean_square": float(np.mean(np.square(normalized))),
                "raw_action_min_max_for_dimension": [float(raw[:, dim].min()), float(raw[:, dim].max())],
                "delta_min_max_for_dimension": [float(transformed[:, dim].min()), float(transformed[:, dim].max())]})
        stream.close()
        return {"relative_path": source["relative_path"], "source_etag": source["etag"], "windows": rows}

    try:
        with (output / "partial_results.jsonl").open("w") as partial, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(inspect, item) for item in plan["objects"]]
            for future in as_completed(futures):
                result = future.result()
                partial.write(json.dumps(result) + "\n")
                partial.flush()
                completed.append(result["relative_path"])
                all_windows.extend(result["windows"])
                if len(completed) % 10 == 0:
                    save(output / "progress.json", {"files": len(completed), "windows": len(all_windows),
                        "range_bytes": transferred[0], "observed_at": time.time()})
                    print(json.dumps({"files": len(completed), "windows": len(all_windows)}), flush=True)
        if len(all_windows) != plan["candidate_window_count"] or len(completed) != len(plan["objects"]):
            raise ValueError("Incomplete replay cohort")
        result = {"complete": True, "files": len(completed), "windows": len(all_windows),
            "range_bytes": transferred[0], "normalization_sha256": digest(HERE / "norm_stats.json"),
            "column_plan_sha256": digest(HERE / "column_plan.json"),
            "ranked_windows": sorted(all_windows, key=lambda x: x["normalized_abs_max"], reverse=True),
            "training": False, "data_modified": False, "model_loaded": False, "gpus": 0}
        save(output / "result.json", result)
        save(output / "_SUCCESS.json", {"complete": True, "result_sha256": digest(output / "result.json"),
            "files": len(completed), "windows": len(all_windows), "range_bytes": transferred[0]})
    except BaseException as exc:
        save(output / "failure.json", {"phase": "audit", "error_type": type(exc).__name__,
            "files": len(completed), "windows": len(all_windows), "range_bytes": transferred[0],
            "observed_at": time.time(), "training": False})
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        audit(Path(sys.argv[2]), float(sys.argv[3]))
    else:
        bootstrap()
