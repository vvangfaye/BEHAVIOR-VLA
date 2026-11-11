#!/usr/bin/env python3
"""One finite GPU job: verify existing weights, then evaluate the raw cache.

The worker neither trains a model nor interacts with a simulator. Each model
runs in a fresh process so its device allocations are released before the next.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
import urllib.request


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def check_files(root: Path, rows: list[dict]) -> dict:
    total = 0
    for row in rows:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Weight manifest path escapes checkpoint")
        path = root / relative
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Checkpoint content differs: {relative}")
        total += row["bytes"]
    return {"checkpoint": str(root), "files_verified": len(rows), "bytes_verified": total,
            "manifest_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def run(bundle: Path, output: Path) -> None:
    started = time.monotonic()
    plan = json.loads((bundle / "plan.json").read_text())
    if plan["kind"] != "finite_cached_boundary_probe" or not 1 <= len(plan["checkpoints"]) <= 4:
        raise ValueError("Unexpected finite probe plan")
    if not 60 <= plan["max_seconds"] <= 14400:
        raise ValueError("Probe wall-time bound is missing")
    output.mkdir(parents=True, exist_ok=False)
    environment = Path(plan["runtime"]) / "envs/b1k-pi05"
    python = environment / "bin/python"
    project = bundle / "project"
    script = project / "analysis/training_boundary_audit_20261019/eval_cached_probe.py"
    cache = project / "analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1/cache_manifest.json"
    env = dict(os.environ)
    env.pop("BOUNDARY_PROBE_RESULT_PUT_URL", None)
    env.pop("PYTHONHOME", None)
    env.pop("JAX_PLATFORMS", None)
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(project / "BEHAVIOR-VLAs/openpi/src"),
                                          str(project / "BEHAVIOR-VLAs/openpi/packages/openpi-client/src")]),
               OPENPI_DATA_HOME=str(Path(plan["runtime"]) / "openpi-cache"),
               LD_LIBRARY_PATH=str(environment / "lib") + ":" + env.get("LD_LIBRARY_PATH", ""),
               XLA_PYTHON_CLIENT_MEM_FRACTION="0.85")
    preflight = subprocess.run([str(python), "-c", "import json,jax; d=jax.devices(); "
                               "print(json.dumps([{'platform':x.platform,'device':str(x)} for x in d])); "
                               "assert any(x.platform=='gpu' for x in d), 'No usable JAX GPU'"],
                              env=env, text=True, capture_output=True, timeout=180)
    (output / "device_preflight.log").write_text(preflight.stdout + preflight.stderr)
    if preflight.returncode:
        raise RuntimeError("GPU preflight failed; inspect device_preflight.log")
    progress = {"kind": plan["kind"], "started_at": time.time(), "training_executed": False,
                "simulator_executed": False, "results": []}
    save(output / "progress.json", progress)
    for model in plan["checkpoints"]:
        name = model["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid model identity")
        checkpoint = Path(model["path"])
        receipt = check_files(checkpoint, model["files"])
        save(output / f"{name}.weights_verified.json", receipt)
        remaining = int(plan["max_seconds"] - (time.monotonic() - started))
        if remaining <= 0:
            raise TimeoutError("Finite probe time budget exhausted")
        result_path = output / f"{name}.json"
        command = [str(python), str(script), "--cache-manifest", str(cache),
                   "--norm-asset-id", plan["norm_asset_id"],
                   "--pi05-ckpt", str(checkpoint), "--out", str(result_path)]
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=remaining)
        record = {"name": name, "returncode": process.returncode, "finished_at": time.time()}
        if process.returncode == 0:
            result = json.loads(result_path.read_text())
            if result["samples"] != plan["sample_count"]:
                raise ValueError("Evaluation sample count differs from frozen plan")
            record.update(result_sha256=sha(result_path), samples=result["samples"])
        progress["results"].append(record)
        save(output / "progress.json", progress)
        if process.returncode:
            raise RuntimeError(f"Evaluation failed for {name}; inspect its log")
    progress.update(complete=True, finished_at=time.time(), elapsed_seconds=time.monotonic() - started)
    save(output / "progress.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failure = None
    try:
        run(args.bundle.resolve(), args.output.resolve())
    except BaseException as error:
        failure = error
        args.output.mkdir(parents=True, exist_ok=True)
        save(args.output / "failure.json", {"type": type(error).__name__,
             "message": re.sub(r"https?://\S+", "<redacted URL>", str(error))[:1500]})
    finally:
        upload = os.environ.get("BOUNDARY_PROBE_RESULT_PUT_URL")
        if upload:
            archive = args.output.with_suffix(".tar.gz")
            with tarfile.open(archive, "w:gz") as target:
                for path in sorted(args.output.iterdir()):
                    if path.is_file():
                        target.add(path, arcname=path.name)
            if archive.stat().st_size > 64 * 1024**2:
                raise RuntimeError("Results exceed the bounded archive size")
            request = urllib.request.Request(upload, data=archive.read_bytes(), method="PUT")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(request, timeout=120) as response:
                    if response.status not in (200, 201, 204):
                        raise RuntimeError("Result publication failed")
            except Exception:
                raise RuntimeError("Result publication failed; AFS results are retained") from None
    if failure is not None:
        raise SystemExit("Probe failed; inspect saved failure.json")
