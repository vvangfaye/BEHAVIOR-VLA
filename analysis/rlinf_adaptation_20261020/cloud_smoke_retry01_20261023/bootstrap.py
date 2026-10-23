"""Download only the named immutable artifacts into this run's new directory."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def run(config):
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def download(item, path):
        digest = hashlib.sha256()
        count = 0
        with opener.open(item["url"], timeout=180) as response, path.open("xb") as out:
            while True:
                data = response.read(8 * 1024**2)
                if not data:
                    break
                out.write(data)
                digest.update(data)
                count += len(data)
        assert count == item["bytes"] and digest.hexdigest() == item["sha256"]

    if shutil.disk_usage(root).free < 40 * 1024**3:
        raise RuntimeError("Less than 40 GiB free for this bounded run")
    code = root / "bundle.tar.gz"
    download(config["bundle"], code)
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

    for relative, expected in config["reuse_sha256"].items():
        verify(base / relative, expected)
    plan = json.loads((root / "run/plan.json").read_text())
    runtime = Path(plan["runtime"])
    env = dict(os.environ)
    for name in ("PYTHONHOME", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(name, None)
    for name in list(env):
        if name.startswith(
            ("PI05_", "SUBTASK_", "EXTERNAL_SUBTASK", "GOALKEEPER_", "VIDEO_EVAL_")
        ):
            env.pop(name)
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
        RLINF_ROOT=str(base / "upstream"),
    )
    (root / "runtime_receipt.json").write_text(
        json.dumps({"gpu": gpu, "root": str(root), "source_verified": True}, indent=2)
    )
    command = [str(runtime / "envs/b1k-pi05/bin/python"), str(root / "run/driver.py")]
    raise SystemExit(subprocess.call(command, cwd=root, env=env))
