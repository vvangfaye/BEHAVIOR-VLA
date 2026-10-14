"""Bounded local OSS cache for the exact two smoke-test batches.

The frozen mixture owns sample selection. Caching changes only file placement;
it never rewrites parquet rows, videos, labels, or sampling indices.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
from pathlib import Path
import shutil


def selected_original_files(plan, metadata, source, *, rank=0, process_count=1):
    if process_count not in (1, 2) or rank not in range(process_count):
        raise ValueError("Invalid bounded smoke rank")
    if plan.global_batch_size != 192:
        raise ValueError("This cache is limited to two global batches of 192")
    # DistributedSampler(shuffle=False) partitions global indices by stride.
    indices = list(range(rank, 2 * plan.global_batch_size, process_count))
    episodes = sorted({r["episode_index"] for i in indices
                       if (r := plan.reference(i))["root_id"] == "expert_original"})
    rgb = sorted(k for k in metadata.video_keys if k.startswith("observation.rgb."))
    if len(rgb) != 3:
        raise ValueError("Expected the three original RGB streams")
    relative = set()
    for episode in episodes:
        relative.add(metadata.get_data_file_path(episode).as_posix())
        relative.update(metadata.get_video_file_path(episode, k).as_posix() for k in rgb)
    objects = {x["relative_path"]: x for x in source["objects"]}
    files = []
    for name in sorted(relative):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in ("data", "videos"):
            raise ValueError("Unsafe original dataset cache path")
        item = objects[name]
        if item["key"] != source["prefix"].rstrip("/") + "/" + name or item["bytes"] <= 0:
            raise ValueError("Original object manifest identity changed")
        files.append(item)
    total = sum(x["bytes"] for x in files)
    if total > 256 * 1024**3:
        raise ValueError("Two-batch original cache exceeds the 256 GiB bound")
    return dict(rank=rank, process_count=process_count, global_indices=indices,
                episodes=episodes, files=files, bytes=total,
                mixture_fingerprint=plan.fingerprint, formal_training=False)


def selected_training_files(plan, metadata, source, *, rank, process_count):
    if process_count != 2 or rank not in (0, 1) or plan.global_batch_size != 192:
        raise ValueError("Formal training requires two eight-GPU workers and batch192")
    rgb = {k for k in metadata.video_keys if k.startswith("observation.rgb.")}
    if len(rgb) != 3:
        raise ValueError("Expected three baseline RGB streams")
    files = []
    for item in source["objects"]:
        name = item["relative_path"]
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe original dataset cache path")
        if path.parts[0] == "data" or (len(path.parts) > 2 and path.parts[0] == "videos" and path.parts[1] in rgb):
            if item["key"] != source["prefix"].rstrip("/") + "/" + name or item["bytes"] <= 0:
                raise ValueError("Original training object manifest identity changed")
            files.append(item)
    total = sum(x["bytes"] for x in files)
    if not files or total > 1200 * 1024**3:
        raise ValueError("Original training cache exceeds the frozen 1200GiB bound")
    return dict(rank=rank, process_count=process_count, global_indices=None,
                files=files, bytes=total, formal_training=True,
                mixture_fingerprint=plan.fingerprint,
                storage_covers_all_original_files=True,
                training_selection="unchanged mixture excludes heldout instances")


def cache_original(root, cache_plan, source, mount, *, copy_verified, write, log, output):
    root = Path(root)
    if shutil.disk_usage(root).free < cache_plan["bytes"] + 48 * 1024**3:
        raise ValueError("Not enough local disk for bounded original cache and smoke outputs")
    # Never remove or write through an OSS symlink. Only replace these two links
    # inside this fresh, owned staging directory; source objects stay untouched.
    for name in ("data", "videos"):
        link = root / name
        expected = Path(mount) / source["prefix"] / name
        if not link.is_symlink() or link.readlink() != expected:
            raise ValueError("Expected the fresh original-data mount link: " + str(link))
    write(Path(output) / "original_cache_plan.json", cache_plan)
    for name in ("data", "videos"):
        (root / name).unlink()
        (root / name).mkdir()
    log("Caching exact original files: " + str(len(cache_plan["files"])) +
        " files, " + str(round(cache_plan["bytes"] / 1024**3, 2)) + " GiB")

    def one(item):
        target = root / item["relative_path"]
        copy_verified(Path(mount) / item["key"], target, item["bytes"])
        # ossutil's default CRC64 download check remains enabled. In addition,
        # single-part objects have an MD5 ETag that we verify against the frozen
        # source-object inventory after the exact-byte download.
        etag = item["etag"].strip('"').lower()
        if len(etag) == 32 and all(c in "0123456789abcdef" for c in etag):
            digest = hashlib.md5()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != etag:
                raise ValueError("Original cache MD5 mismatch: " + item["relative_path"])
        return item["bytes"]

    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for count, size in enumerate(pool.map(one, cache_plan["files"]), 1):
            total += size
            if count % 50 == 0:
                log("Cached original files " + str(count) + "/" + str(len(cache_plan["files"])))
    write(Path(output) / "original_cache_verified.json",
          dict(files=len(cache_plan["files"]), bytes=total,
               global_samples=None if cache_plan["global_indices"] is None else len(cache_plan["global_indices"]),
               rank=cache_plan["rank"], formal_training=cache_plan.get("formal_training", False),
               process_count=cache_plan["process_count"],
               mixture_fingerprint=cache_plan["mixture_fingerprint"],
               exact_source_bytes=True, single_part_md5_verified=True,
               source_objects_modified=False, training_samples_changed=False))
