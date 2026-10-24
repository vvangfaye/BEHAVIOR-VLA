"""Load a verified inline code bundle and reuse the frozen AFS model files."""

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import traceback
import urllib.request
from pathlib import Path


def write_inline_bundle(item, encoded, path):
    body = base64.b64decode(encoded, validate=True)
    assert len(body) == item["bytes"] and len(body) < 1024**2
    assert hashlib.sha256(body).hexdigest() == item["sha256"]
    with path.open("xb") as stream:
        stream.write(body)


def run(config):
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=False)
    # Stay within the allocated 14 CPU cores before JAX creates its pools.
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:14]))
    config["phase"] = "unpack_inline_patch"

    if shutil.disk_usage(root).free < 40 * 1024**3:
        raise RuntimeError("Less than 40 GiB free for this bounded run")
    code = root / "bundle.tar.gz"
    # This bundle is small enough to travel in the immutable job command.
    # Computation no longer depends on DNS resolving during bootstrap.
    write_inline_bundle(config["bundle"], config["bundle_inline"], code)
    with tarfile.open(code, "r:gz") as tar:
        for member in tar:
            relative = Path(member.name)
            assert (
                not relative.is_absolute()
                and ".." not in relative.parts
                and member.isfile()
            )
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, target.open("xb") as out:
                shutil.copyfileobj(stream, out)
    base = Path(config["base"])

    def verify(path, expected):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(chunk)
        assert h.hexdigest() == expected, str(path)

    config["phase"] = "verify_reused_files"
    missing = [
        str(base / rel) for rel in config["reuse_sha256"] if not (base / rel).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing reusable artifacts: " + repr(missing))
    for relative, expected in config["reuse_sha256"].items():
        verify(base / relative, expected)
    plan = json.loads((root / "run/plan.json").read_text())
    resume=Path(plan['resume_run'])
    completed=[str(p.relative_to(resume)) for p in sorted((resume/'training').glob('round_*/episode_*/rollout.pt'))
               if (p.parent/'result.json').is_file()]
    assert completed==['training/round_01/episode_01/rollout.pt'], completed
    assert not list((resume/'training').glob('round_*/result.json')), 'Previous PPO updates require optimizer restore'
    for instance,seed in plan['evaluation']['cases']:
        assert (resume/'previous_final'/f'i{instance}_s{seed}'/'result.json').is_file()
    evidence={'completed_episodes':completed,'previous_status':json.loads((resume/'status.json').read_text()),
              'previous_failure':json.loads((resume/'failure.json').read_text()) if (resume/'failure.json').exists() else None,
              'previous_critic_warmup':json.loads((resume/'critic_warmup.json').read_text()),
              'first_episode':json.loads((resume/'training/round_01/episode_01/result.json').read_text())}
    second=resume/'training/round_01/episode_02/rollout/simulator.log'
    if second.exists():evidence['incomplete_episode_log_tail']=second.read_text(errors='replace')[-4000:]
    (root/'resume_preflight.json').write_text(json.dumps(evidence,indent=2))
    safe_status=json.dumps({'state':'running','stage':'resume_preflight','completed_training_episodes':len(completed),
                            'previous_failure_type':(evidence['previous_failure'] or {}).get('error_type')}).encode()
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(config['status_put'],data=safe_status,method='PUT'),timeout=30):pass
    except Exception:pass
    config["phase"] = "verify_original_checkpoint"
    original = Path(plan["original"])
    restored = json.loads((original / "_RESTORE_VERIFIED.json").read_text())
    assert restored["passed"] is True and len(restored["files"]) == 21
    for item in restored["files"]:
        path = original / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError("Restored original checkpoint incomplete: " + str(path))
    runtime = Path(plan["runtime"])
    env = dict(os.environ)
    for name in ("PYTHONHOME", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(name, None)
    for name in list(env):
        if name.startswith(
            ("PI05_", "SUBTASK_", "EXTERNAL_SUBTASK", "GOALKEEPER_", "VIDEO_EVAL_")
        ):
            env.pop(name)
    previous = Path(plan["previous_run"])
    assert (previous/"checkpoints/round_05/trainable.pt").is_file()
    assert all((previous/"training"/f"round_{i:02d}"/"rollout.pt").is_file() for i in range(1,6))
    instances_root = runtime/"omnigibson-data/2026-challenge-task-instances/scenes"
    instance_files = {}
    for instance in plan["train_instances"]:
        matches = list(instances_root.glob(f"*/json/*turning_on_radio*/*turning_on_radio_0_{instance}_template-tro_state.json"))
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely verify training instance {instance}: {len(matches)} matches")
        instance_files[str(instance)] = str(matches[0])
    (root/"training_instances_verified.json").write_text(json.dumps(instance_files,indent=2))
    assert Path(plan["accepted_gate"]).is_file()
    config["phase"] = "graphics_setup"
    gpu = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(gpu) != 1 or not any(x in gpu[0] for x in ("5090", "N12LP")):
        raise RuntimeError("Expected exactly one RTX 5090")
    driver = gpu[0].split(",")[1].strip()
    libraries = [
        p
        for directory in [
            "/usr/local/nvidia/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/lib/x86_64-linux-gnu",
        ]
        for p in Path(directory).glob("libEGL_nvidia.so." + driver)
    ]
    assert libraries
    graphics = root / "graphics"
    graphics.mkdir()
    vk = graphics / "nvidia_icd.json"
    egl = graphics / "10_nvidia.json"
    vk.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "ICD": {"library_path": str(libraries[0]), "api_version": "1.3.242"},
            }
        )
    )
    egl.write_text(
        json.dumps(
            {"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0])}}
        )
    )
    repair = Path(plan["supplemental_library"])
    ready = json.loads((repair / "_READY").read_text())
    assert (
        hashlib.sha256((repair / "_READY").read_bytes()).hexdigest()
        == plan["supplemental_library_sha256"]
    )
    library_path = ready["library_path"]
    for name, expected in ready["libraries_sha256"].items():
        assert (
            hashlib.sha256((Path(library_path) / name).read_bytes()).hexdigest()
            == expected
        )
    for relative, expected in plan["original_source_sha256"].items():
        assert (
            hashlib.sha256((Path(plan["code"]) / relative).read_bytes()).hexdigest()
            == expected
        )
    original = Path(plan["original"])
    assert (
        hashlib.sha256((original / "params/_METADATA").read_bytes()).hexdigest()
        == plan["original_metadata_sha256"]
    )
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONHASHSEED=str(plan["seed"]),
        PYTHONPATH=":".join(
            [
                str(base / "site"),
                str(root / "run"),
                str(base / "adapter"),
                str(base / "upstream"),
                plan["code"] + "/openpi/src",
                plan["code"] + "/openpi/packages/openpi-client/src",
            ]
        ),
        OPENPI_DATA_HOME=str(runtime / "openpi-cache"),
        OMNIGIBSON_DATA_PATH=str(runtime / "omnigibson-data"),
        OMNIGIBSON_HEADLESS="1",
        OMNI_KIT_ACCEPT_EULA="YES",
        OMNIGIBSON_DISABLE_TORCH_COMPILE="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.35",
        VK_ICD_FILENAMES=str(vk),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
        LD_LIBRARY_PATH=":".join(
            [
                library_path,
                str(runtime / "graphics-runtime/usr/lib/x86_64-linux-gnu"),
                str(runtime / "envs/b1k-eval/lib"),
                env.get("LD_LIBRARY_PATH", ""),
            ]
        ),
        B1K_STATUS_PUT=config["status_put"],
        B1K_RESULT_PUT=config["result_put"],
        B1K_ARTIFACT_PUTS=json.dumps(config["artifact_puts"]),
        JAX_PLATFORMS="cpu",
        RLINF_ROOT=str(base / "upstream"),
    )
    (root / "runtime_receipt.json").write_text(
        json.dumps({"gpu": gpu, "root": str(root), "source_verified": True}, indent=2)
    )
    config["phase"] = "start_driver"
    command = [str(runtime / "envs/b1k-pi05/bin/python"), str(root / "run/driver.py")]
    return subprocess.call(command, cwd=root, env=env)


def guarded_run(config):
    try:
        code = run(config)
    except BaseException as error:
        details = re.sub(r"https?://[^\s\"']+", "[URL omitted]", traceback.format_exc())
        report = {
            "state": "bootstrap_failed",
            "phase": config.get("phase", "create_root"),
            "error_type": type(error).__name__,
            "details": details,
        }
        body = json.dumps(report).encode()
        try:
            root = Path(config["root"])
            if root.is_dir():
                (root / "bootstrap_error.json").write_bytes(body)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(
                urllib.request.Request(config["status_put"], data=body, method="PUT"),
                timeout=90,
            ):
                pass
        except BaseException:
            pass
        print(json.dumps(report), flush=True)
        raise SystemExit(1)
    raise SystemExit(code)
