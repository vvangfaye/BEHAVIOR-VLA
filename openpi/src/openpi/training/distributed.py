from __future__ import annotations

import dataclasses
import logging
import os

import jax


@dataclasses.dataclass(frozen=True)
class DistributedConfig:
    coordinator_address: str
    process_count: int
    process_id: int
    local_device_ids: tuple[int, ...]
    expected_global_devices: int | None = None


def config_from_env() -> DistributedConfig | None:
    process_count = os.environ.get("OPENPI_PROCESS_COUNT")
    if process_count is None:
        return None

    required = ("OPENPI_COORDINATOR_ADDRESS", "OPENPI_PROCESS_ID", "OPENPI_LOCAL_DEVICE_IDS")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Missing distributed environment variables: {', '.join(missing)}")

    count = int(process_count)
    process_id = int(os.environ["OPENPI_PROCESS_ID"])
    local_device_ids = tuple(int(value) for value in os.environ["OPENPI_LOCAL_DEVICE_IDS"].split(","))
    if count < 2:
        raise ValueError("OPENPI_PROCESS_COUNT must be at least 2 when distributed mode is enabled.")
    if process_id not in range(count):
        raise ValueError(f"OPENPI_PROCESS_ID={process_id} is outside [0, {count}).")
    if not local_device_ids or len(set(local_device_ids)) != len(local_device_ids):
        raise ValueError("OPENPI_LOCAL_DEVICE_IDS must contain unique device indices.")

    expected = os.environ.get("OPENPI_EXPECTED_GLOBAL_DEVICES")
    return DistributedConfig(
        coordinator_address=os.environ["OPENPI_COORDINATOR_ADDRESS"],
        process_count=count,
        process_id=process_id,
        local_device_ids=local_device_ids,
        expected_global_devices=int(expected) if expected else None,
    )


def initialize_from_env() -> DistributedConfig | None:
    config = config_from_env()
    if config is None:
        return None
    if jax.distributed.is_initialized():
        raise RuntimeError("JAX distributed runtime was initialized before OpenPI configured it.")

    jax.distributed.initialize(
        coordinator_address=config.coordinator_address,
        num_processes=config.process_count,
        process_id=config.process_id,
        local_device_ids=config.local_device_ids,
        cluster_detection_method="deactivate",
        initialization_timeout=int(os.environ.get("OPENPI_DISTRIBUTED_TIMEOUT", "300")),
    )

    if jax.process_count() != config.process_count or jax.process_index() != config.process_id:
        raise RuntimeError(
            f"JAX topology mismatch: process={jax.process_index()}/{jax.process_count()}, "
            f"expected={config.process_id}/{config.process_count}."
        )
    if jax.local_device_count() != len(config.local_device_ids):
        raise RuntimeError(
            f"JAX local device mismatch: got {jax.local_device_count()}, expected {len(config.local_device_ids)}."
        )
    if config.expected_global_devices is not None and jax.device_count() != config.expected_global_devices:
        raise RuntimeError(
            f"JAX global device mismatch: got {jax.device_count()}, expected {config.expected_global_devices}."
        )
    if any(device.platform != "gpu" for device in jax.local_devices()):
        raise RuntimeError(f"Distributed training requires GPUs, got local devices {jax.local_devices()}.")

    logging.info(
        "Initialized JAX distributed runtime: host=%s process=%d/%d local_devices=%d global_devices=%d coordinator=%s",
        os.uname().nodename,
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
        config.coordinator_address,
    )
    return config
