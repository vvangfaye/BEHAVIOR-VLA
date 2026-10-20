import json,os,subprocess
from pathlib import Path
def _environment(payload, source, root, cache, hook_dir):
    runtime = Path(payload["runtime"])
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                        "--format=csv,noheader"], text=True).strip().splitlines()
    if len(gpu_rows) != 1 or not any(name in gpu_rows[0] for name in ("5090", "N12LP")):
        raise ValueError("Exactly one compatible RTX5090 required")
    driver = gpu_rows[0].split(",")[2].strip()
    libraries = [p for parent in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/lib/x86_64-linux-gnu", "/lib64")
                 for p in Path(parent).glob("libEGL_nvidia.so." + driver)]
    if not libraries:
        raise ValueError("Matching host graphics driver missing")
    graphics = root / "graphics"
    graphics.mkdir()
    vk, egl = graphics / "nvidia_icd.json", graphics / "10_nvidia.json"
    write_json(vk, {"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0]), "api_version": "1.3.242"}})
    write_json(egl, {"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0])}})
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("PI05_", "SUBTASK_", "EXTERNAL_SUBTASK", "PROVIDED_SUBTASK", "GOALKEEPER_", "VIDEO_EVAL_")):
            env.pop(name)
    for name in ("PYTHONHOME", "MAX_STEPS", "EVAL_EXTRA_ARGS", "ENV_WRAPPER", "ROBOT_CONFIG"):
        env.pop(name, None)
    library_paths = [str(runtime / "graphics-runtime/usr/lib/x86_64-linux-gnu"), str(runtime / "envs/b1k-eval/lib")]
    if "supplemental_library" in source:
        library = source["supplemental_library"]
        ready = Path(library["root"]) / "_READY"
        if sha256(ready) != library["ready_manifest_sha256"]:
            raise ValueError("Supplemental runtime manifest changed")
        library_ready = json.loads(ready.read_bytes())
        for name, expected in library_ready["libraries_sha256"].items():
            if sha256(Path(library_ready["library_path"]) / name) != expected:
                raise ValueError("Supplemental library changed")
        library_paths.insert(0, library_ready["library_path"])
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(hook_dir),
               OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_HEADLESS="1", OMNIGIBSON_DISABLE_TORCH_COMPILE="1",
               OPENPI_DATA_HOME=str(runtime / "openpi-cache"), OMNIGIBSON_DATA_PATH=str(runtime / "omnigibson-data"),
               PI05_ENV=str(runtime / "envs/b1k-pi05"), EVAL_ENV=str(runtime / "envs/b1k-eval"),
               PI05_CKPT=payload["checkpoint"], PI05_REPO_ID="b1k_full100_joint_subtask_v1_drop_episode_s1",
               SUBTASK_PREDICTION="1", MAX_TOKEN_LEN="512", MAX_SUBTASK_LEN="64", EXECUTE_HORIZON="16",
               PI05_MEM_FRACTION="0.35", XLA_PYTHON_CLIENT_PREALLOCATE="false", MODEL="pi05",
               SERVER_GPU="0", EVAL_GPU="0", PORT="8910", MODE="public_test", PROMPT_MODE="whole_task",
               NUM_ROLLOUTS="1", INSTANCE_INDICES=str(payload["instance_id"] - 301), WRITE_VIDEO="1",
               OUTPUT_ROOT=str(root / "results"), SCO_EPHEMERAL_RUNTIME_ROOT=str(cache),
               VK_ICD_FILENAMES=str(vk), __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
               LD_LIBRARY_PATH=":".join(library_paths) + ":" + env.get("LD_LIBRARY_PATH", ""),
               PI05_EVAL_QUEUE_DB=str(root / "queue.sqlite3"), PI05_EVAL_QUEUE_WORKER="gpu0",
               PI05_EVAL_QUEUE_RUN_ID=payload["run_id"], PI05_EVAL_QUEUE_GENERATION="0",
               PI05_EVAL_QUEUE_ATTEMPT_ROOT=str(root / "attempts"),
               VIDEO_EVAL_PROGRESS=str(cache / "native_progress.json"), VIDEO_EVAL_RUN_ID=payload["run_id"])
    return env, gpu_rows
