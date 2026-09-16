#!/usr/bin/env python3
import json
import os
from pathlib import Path

import jax
from jax.experimental import multihost_utils
import numpy as np
import torch

from openpi.training import distributed
from openpi.training import sharding


def main() -> None:
    expected_process_count = int(os.environ.get("PROCESS_COUNT", "1"))
    expected_local_devices = int(os.environ.get("EXPECTED_LOCAL_DEVICES", "8"))
    expected_global_devices = int(os.environ.get("EXPECTED_GLOBAL_DEVICES", str(expected_local_devices)))
    config = distributed.initialize_from_env()
    if config is None and expected_process_count != 1:
        raise RuntimeError("Multi-worker preflight requires OPENPI_PROCESS_COUNT.")
    if jax.process_count() != expected_process_count:
        raise RuntimeError(f"Process count mismatch: got {jax.process_count()}, expected {expected_process_count}.")
    if jax.local_device_count() != expected_local_devices:
        raise RuntimeError(f"Local device mismatch: got {jax.local_device_count()}, expected {expected_local_devices}.")
    if jax.device_count() != expected_global_devices:
        raise RuntimeError(f"Global device mismatch: got {jax.device_count()}, expected {expected_global_devices}.")
    if any(device.platform != "gpu" for device in jax.local_devices()):
        raise RuntimeError(f"Preflight requires GPUs, got local devices {jax.local_devices()}.")

    fsdp_devices = int(os.environ.get("FSDP_DEVICES", "8"))
    expected_global_batch = int(os.environ.get("BATCH_SIZE", "64"))
    if expected_global_batch % jax.process_count() != 0:
        raise RuntimeError("Global batch is not divisible by process count.")

    mesh = sharding.make_mesh(fsdp_devices)
    actual_mesh = (mesh.shape[sharding.BATCH_AXIS], mesh.shape[sharding.FSDP_AXIS])
    expected_mesh = (jax.process_count(), fsdp_devices)
    if actual_mesh != expected_mesh:
        raise RuntimeError(f"Mesh mismatch: got {actual_mesh}, expected {expected_mesh}.")

    mesh_processes = [[int(device.process_index) for device in row] for row in np.asarray(mesh.devices)]
    expected_processes = [[process] * fsdp_devices for process in range(jax.process_count())]
    if mesh_processes != expected_processes:
        raise RuntimeError(f"Mesh is not host-contiguous: {mesh_processes}")

    dataset = list(range(expected_global_batch))
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=jax.process_count(),
        rank=jax.process_index(),
        shuffle=False,
        drop_last=True,
    )
    local_indices = np.asarray(list(sampler), dtype=np.int32)
    if jax.process_count() == 1:
        gathered = local_indices[np.newaxis, :]
    else:
        gathered = np.asarray(multihost_utils.process_allgather(local_indices, tiled=False))
    flattened = gathered.reshape(-1)
    if len(flattened) != expected_global_batch or sorted(flattened.tolist()) != dataset:
        raise RuntimeError(f"Distributed sampler is incomplete or overlapping: {gathered.tolist()}")

    report = {
        "available_local_bytes": int(os.environ["OPENPI_AVAILABLE_LOCAL_BYTES"]),
        "coordinator": config.coordinator_address if config is not None else None,
        "global_batch": expected_global_batch,
        "global_devices": jax.device_count(),
        "gpu_names": os.environ["OPENPI_GPU_NAMES"].split(";"),
        "local_batch": len(local_indices),
        "local_devices": jax.local_device_count(),
        "mesh": dict(mesh.shape),
        "mesh_processes": mesh_processes,
        "process_count": jax.process_count(),
        "process_id": jax.process_index(),
        "sample_indices": local_indices.tolist(),
    }
    report_dir = Path(os.environ["OPENPI_PREFLIGHT_DIR"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"rank-{jax.process_index()}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    multihost_utils.sync_global_devices("openpi_distributed_preflight_written")
    if jax.process_index() == 0:
        (report_dir / "_SUCCESS").write_text(json.dumps({"mesh": dict(mesh.shape)}, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
