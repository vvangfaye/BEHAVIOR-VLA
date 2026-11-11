#!/usr/bin/env python3
"""H100 rendering diagnostic for LIBERO (MuJoCo / robosuite EGL). Local tool, 2026-11-05; not part of RLinf.

Earlier H100 runs (colocated, 64 envs on 4 GPUs) died or hung in the first rollout: an env subprocess exited without
an error after 6 min (LIBERO), or 48+ env processes sat at ~18% CPU with D-state processes and nvidia-smi not
answering (CALVIN). RLinf 91a302ec already maps each worker's CUDA device to its EGL index (#1458), so this script
measures instead of guessing:
  1. EGL devices vs CUDA devices (EGL_CUDA_DEVICE_NV, DRM files, EGL vendor) and the MUJOCO_EGL_DEVICE_ID that RLinf's
     scheduler gives the workers of each GPU (NvidiaGPUManager.get_accelerator_env_var).
  2. Real LIBERO envs (libero_10 task 0, the RLinf init_params: 256x256 agentview + wrist), one process per env as in
     RLinf's SubprocEnv: 1/8/16/32 envs on one GPU, 16 envs next to a CUDA compute load (100% and 50% duty, i.e. a
     colocated rollout model), and 16 envs on every GPU at once (64 envs, the colocated cluster4 layout).
  3. Hangs: every test has a hard timeout; workers still alive are reported with their /proc state (D = uninterruptible)
     and kernel wait channel, then killed. After every test nvidia-smi must answer within 20 s, otherwise the remaining
     GPU tests are skipped (the driver is stuck and further tests would only add noise).
Output: <out>/egl_diag.json, plus one 'EGL_DIAG {json}' line per result on stdout (the console log is mirrored).
Usage:  python rlinf/b1k_tools/egl_diag.py --out DIR [--quick] [--steps 100] [--tests n1,n8,...]
"""
import argparse
import ctypes
import json
import os
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # .../RLinf


def emit(kind, **kw):
    print("EGL_DIAG " + json.dumps({"kind": kind, **kw}, default=str), flush=True)


def sh(cmd, timeout=20):
    """Run a shell command; returns (rc, output) with rc='timeout' when it does not return in time."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[-6000:]
    except subprocess.TimeoutExpired:
        return "timeout", ""


# ------------------------------------------------------------------------------------------------ system information
def sysinfo():
    info = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
                                           "MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "EGL_DEVICE_ID",
                                           "LD_LIBRARY_PATH", "__EGL_VENDOR_LIBRARY_DIRS", "__EGL_VENDOR_LIBRARY_FILENAMES",
                                           "OMP_NUM_THREADS", "NCCL_P2P_DISABLE")}
    info["cpu_count"] = os.cpu_count()
    try:
        info["affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/memory.max",
              "/proc/driver/nvidia/version"):
        try:
            info[p] = open(p).read().strip()[:300]
        except Exception:
            pass
    for name, cmd in [
        ("smi_L", "nvidia-smi -L"),
        ("smi_query", "nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version,compute_mode,persistence_mode,"
                      "clocks.max.graphics,clocks.max.sm,power.limit --format=csv"),
        ("smi_q", "nvidia-smi -q | grep -i -E 'GSP Firmware|Compute Mode|MIG Mode|Current  *:|Virtualization Mode|"
                  "Addressing Mode|Persistence' | sort | uniq -c"),
        ("topo", "nvidia-smi topo -m"),
        ("vendor_files", "ls -la /usr/share/glvnd/egl_vendor.d/ /etc/glvnd/egl_vendor.d/ 2>&1; "
                         "cat /usr/share/glvnd/egl_vendor.d/*.json 2>&1"),
        ("egl_libs", "ls -la /usr/lib/x86_64-linux-gnu/ | grep -i -E 'egl|glvnd|osmesa|libGLX|libGL\\.' ; "
                     "ls -la /opt/sys_extra/lib 2>&1 | head -30"),
        ("dmesg_xid", "dmesg -T 2>&1 | grep -i -E 'xid|nvrm|gsp' | tail -20"),
        ("free", "free -g"),
    ]:
        info[name] = sh(cmd, timeout=30)
    return info


EGL_EXTENSIONS, EGL_VENDOR, EGL_VERSION = 0x3055, 0x3053, 0x3054
EGL_DRM_DEVICE_FILE_EXT, EGL_CUDA_DEVICE_NV, EGL_PLATFORM_DEVICE_EXT = 0x3233, 0x323A, 0x313F


def egl_info_child():
    """EGL enumeration + RLinf's per-GPU env vars; runs in its own process (a stuck driver must not hang the parent)."""
    out = {"devices": []}
    lib = ctypes.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    lib.eglGetProcAddress.restype = ctypes.c_void_p
    dev_t = ctypes.c_void_p

    def fn(name, restype, *argtypes):
        ptr = lib.eglGetProcAddress(name.encode())
        if not ptr:
            raise RuntimeError(f"{name} unavailable")
        return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)

    query_devices = fn("eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int, ctypes.POINTER(dev_t), ctypes.POINTER(ctypes.c_int))
    query_string = fn("eglQueryDeviceStringEXT", ctypes.c_char_p, dev_t, ctypes.c_int)
    query_attrib = fn("eglQueryDeviceAttribEXT", ctypes.c_uint, dev_t, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))
    get_display = fn("eglGetPlatformDisplayEXT", ctypes.c_void_p, ctypes.c_int, dev_t, ctypes.c_void_p)
    lib.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.eglInitialize.restype = ctypes.c_uint
    lib.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.eglQueryString.restype = ctypes.c_char_p
    lib.eglTerminate.argtypes = [ctypes.c_void_p]
    devs = (dev_t * 64)()
    n = ctypes.c_int()
    if not query_devices(64, devs, ctypes.byref(n)):
        raise RuntimeError("eglQueryDevicesEXT failed")
    for i in range(n.value):
        d = {"egl_index": i}
        ext = (query_string(devs[i], EGL_EXTENSIONS) or b"").decode()
        d["extensions"] = ext
        if "EGL_EXT_device_drm" in ext:
            d["drm"] = (query_string(devs[i], EGL_DRM_DEVICE_FILE_EXT) or b"").decode()
        v = ctypes.c_ssize_t(-1)
        if "EGL_NV_device_cuda" in ext and query_attrib(devs[i], EGL_CUDA_DEVICE_NV, ctypes.byref(v)):
            d["cuda_ordinal"] = v.value
        dpy = get_display(EGL_PLATFORM_DEVICE_EXT, devs[i], None)
        if dpy:
            ma, mi = ctypes.c_int(), ctypes.c_int()
            if lib.eglInitialize(dpy, ctypes.byref(ma), ctypes.byref(mi)):
                d["vendor"] = (lib.eglQueryString(dpy, EGL_VENDOR) or b"").decode()
                d["version"] = (lib.eglQueryString(dpy, EGL_VERSION) or b"").decode()
                lib.eglTerminate(dpy)
            else:
                d["init_failed"] = True
        out["devices"].append(d)
    try:  # what RLinf's scheduler exports for the workers of each GPU
        sys.path.insert(0, REPO)
        from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

        n_cuda = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True,
                                    timeout=30).stdout.strip() or 0)
        out["rlinf_env_by_gpu"] = {
            str(g): {k: v for k, v in NvidiaGPUManager.get_accelerator_env_var([str(g)]).items()
                     if "EGL" in k or k == "CUDA_VISIBLE_DEVICES"} for g in range(n_cuda)}
    except Exception as e:  # noqa: BLE001
        out["rlinf_env_error"] = repr(e)
    print(json.dumps(out))


# ------------------------------------------------------------------------------------------------ worker processes
def _install_shim():
    """robosuite's import-time assert rejects an EGL index that is not also a CUDA id; RLinf hides it during that
    import (rlinf/utils/robosuite_compat.py). Loaded from its file so the rlinf package __init__ is not imported."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_rc", os.path.join(REPO, "rlinf/utils/robosuite_compat.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.install_robosuite_egl_device_shim()


def worker(a):
    import numpy as np

    t0, c0 = time.time(), time.process_time()
    rec = {"pid": os.getpid(), "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "egl": os.environ.get("MUJOCO_EGL_DEVICE_ID")}

    def dump(**kw):
        rec.update(kw, t=round(time.time() - t0, 2))
        with open(a.result + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(a.result + ".tmp", a.result)

    dump(phase="import")
    _install_shim()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(a.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    dump(phase="create")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(a.seed)
    create_s = time.time() - t0
    dump(phase="reset", create_s=round(create_s, 2))
    t1 = time.time()
    env.reset()
    init = suite.get_task_init_states(a.task)
    obs = env.set_init_state(init[a.seed % len(init)])
    reset_s = time.time() - t1
    act = np.zeros(7)
    act[-1] = -1.0
    dts = []
    t2, c2 = time.time(), time.process_time()
    for k in range(a.steps):
        t = time.time()
        obs, _, _, _ = env.step(act)
        dts.append(time.time() - t)
        if k % 20 == 19:
            dump(phase=f"step{k + 1}", reset_s=round(reset_s, 2))
    cpu_frac = (time.process_time() - c2) / max(time.time() - t2, 1e-6)
    rts = []  # render-only time of one 256x256 camera (separates rendering from physics)
    try:
        sim = env.env.sim
        for _ in range(20):
            t = time.time()
            sim.render(camera_name="agentview", width=256, height=256)
            rts.append(time.time() - t)
    except Exception as e:  # noqa: BLE001
        rec["render_only_error"] = repr(e)[:200]
    img, wrist = obs["agentview_image"], obs["robot0_eye_in_hand_image"]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] * 1000 if xs else None  # noqa: E731
    dump(phase="done", ok=True, reset_s=round(reset_s, 2), steps=len(dts),
         step_ms_med=round(q(dts, 0.5), 2), step_ms_p90=round(q(dts, 0.9), 2), step_ms_max=round(max(dts) * 1000, 1),
         render_ms_med=round(q(rts, 0.5), 2) if rts else None, cpu_frac=round(cpu_frac, 2),
         threads=len(os.listdir("/proc/self/task")), img_mean=round(float(img.mean()), 1),
         img_std=round(float(img.std()), 1), wrist_mean=round(float(wrist.mean()), 1),
         cpu_s=round(time.process_time() - c0, 1))
    env.close()


def load_worker(a):
    """CUDA compute load on the one visible GPU: bf16 GEMMs for `duty` of every 100 ms until killed."""
    import torch

    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    n = 0
    while True:
        t = time.time()
        while time.time() - t < 0.1 * a.duty:
            for _ in range(4):
                x @ x
            torch.cuda.synchronize()
            n += 4
        if a.duty < 1.0:
            time.sleep(0.1 * (1 - a.duty))
        if n % 2000 < 4:
            print(f"load gemms={n}", flush=True)


# ------------------------------------------------------------------------------------------------ orchestration
def proc_state(pid):
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except Exception:
        return {"state": "gone"}
    out = {"state": st}
    try:
        out["wchan"] = open(f"/proc/{pid}/wchan").read().strip()
    except Exception:
        pass
    return out


def descendants(pid):
    rc, txt = sh(f"ps -o pid= --ppid {pid}", timeout=10)
    kids = [int(x) for x in txt.split() if x.isdigit()] if rc == 0 else []
    return kids + [d for k in kids for d in descendants(k)]


def driver_ok():
    t = time.time()
    rc, txt = sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader", timeout=20)
    _, dstate = sh("ps -eo stat=,pid=,comm= | awk '$1 ~ /^D/'", timeout=10)
    return {"smi_rc": rc, "smi_s": round(time.time() - t, 1), "smi": txt.strip()[:400], "d_state": dstate.strip()[:600]}


def run_test(a, name, placement, load_gpus, duty, egl_of, tmp):
    loads, workers = [], []
    for g in load_gpus:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        loads.append(subprocess.Popen([sys.executable, __file__, "--load", "--duty", str(duty)], env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
    if loads:
        time.sleep(8)  # CUDA context up and GEMMs running before the envs start
    t0 = time.time()
    for g, n in placement:
        for j in range(n):
            res = os.path.join(tmp, f"{name}_g{g}_{j}.json")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), MUJOCO_EGL_DEVICE_ID=str(egl_of(g)),
                       EGL_DEVICE_ID=str(egl_of(g)), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
            log = open(os.path.join(tmp, f"{name}_g{g}_{j}.log"), "w")
            p = subprocess.Popen([sys.executable, __file__, "--worker", "--task", "0", "--seed", str(j), "--steps",
                                  str(a.steps), "--result", res], env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            workers.append((g, j, p, res, log))
    while time.time() - t0 < a.test_timeout and any(w[2].poll() is None for w in workers):
        time.sleep(1)
    wall = time.time() - t0
    recs, hung, failed = [], [], []
    for g, j, p, res, log in workers:
        r = json.load(open(res)) if os.path.exists(res) else {"phase": "none"}
        r.update(gpu=g, j=j)
        if p.poll() is None:
            r["proc"] = proc_state(p.pid)
            r["children"] = [proc_state(c) for c in descendants(p.pid)][:4]
            hung.append(r)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
        elif p.returncode != 0 or not r.get("ok"):
            log.flush()
            r["rc"] = p.returncode
            try:
                r["log_tail"] = open(log.name).read()[-800:]
            except Exception:
                pass
            failed.append(r)
        else:
            recs.append(r)
        log.close()
    for p in loads:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
    agg = lambda k: round(statistics.median([r[k] for r in recs if r.get(k) is not None]), 2) if recs else None  # noqa
    out = {"test": name, "placement": placement, "load_gpus": load_gpus, "duty": duty if load_gpus else None,
           "n": len(workers), "ok": len(recs), "failed": len(failed), "hung": len(hung), "wall_s": round(wall, 1),
           "create_s_med": agg("create_s"), "reset_s_med": agg("reset_s"), "step_ms_med": agg("step_ms_med"),
           "step_ms_p90": agg("step_ms_p90"), "step_ms_max": max([r["step_ms_max"] for r in recs], default=None),
           "render_ms_med": agg("render_ms_med"), "cpu_frac_med": agg("cpu_frac"), "threads_med": agg("threads"),
           "img_mean_med": agg("img_mean"), "img_std_med": agg("img_std"),
           # env-steps/s of the whole test while all workers were stepping (approx.: steps / slowest stepping span)
           "env_steps_per_s": round(sum(r["steps"] for r in recs) / max(
               max((r["t"] - r["create_s"] - r["reset_s"]) for r in recs), 1e-6), 1) if recs else None,
           "hung_detail": hung[:6], "failed_detail": failed[:4]}
    time.sleep(3)
    out["driver"] = driver_ok()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="local smoke test: n1 + n2 only, 20 steps")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tests", default="")
    ap.add_argument("--test-timeout", type=float, default=360)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--egl-info", action="store_true")
    ap.add_argument("--driver-check", action="store_true", help="exit 3 if nvidia-smi does not answer in 20 s")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--result", default="")
    a = ap.parse_args()
    if a.worker:
        worker(a)
        return 0
    if a.load:
        load_worker(a)
        return 0
    if a.egl_info:
        egl_info_child()
        return 0
    if a.driver_check:  # between GPU stages: a stuck driver (D-state, nvidia-smi hang) makes later stages pointless
        d = driver_ok()
        emit("driver_check", **d)
        return 0 if d["smi_rc"] == 0 else 3

    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "workers")
    os.makedirs(tmp, exist_ok=True)
    report = {"start": time.strftime("%F %T"), "host": os.uname().nodename}
    path = os.path.join(a.out, "egl_diag.json")

    def save():
        with open(path, "w") as f:
            json.dump(report, f, indent=1, default=str)

    report["sys"] = sysinfo()
    emit("sys", **{k: report["sys"].get(k) for k in ("cpu_count", "affinity", "/sys/fs/cgroup/cpu.max",
                                                       "NVIDIA_DRIVER_CAPABILITIES", "smi_L", "smi_q", "dmesg_xid")})
    save()
    try:
        r = subprocess.run([sys.executable, __file__, "--egl-info"], capture_output=True, text=True, timeout=120)
        report["egl"] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-2000:]}
    except Exception as e:  # noqa: BLE001
        report["egl"] = {"error": repr(e)}
    emit("egl", devices=[{k: d.get(k) for k in ("egl_index", "cuda_ordinal", "drm", "vendor")}
                         for d in report["egl"].get("devices", [])],
         rlinf_env_by_gpu=report["egl"].get("rlinf_env_by_gpu"), error=report["egl"].get("error",
                                                                                           report["egl"].get("rlinf_env_error")))
    save()
    rl = report["egl"].get("rlinf_env_by_gpu") or {}
    egl_of = lambda g: int(rl.get(str(g), {}).get("MUJOCO_EGL_DEVICE_ID", g))  # noqa: E731
    n_gpu = max(len(rl), 1)
    if a.quick:
        a.steps = min(a.steps, 20)
        plan = [("n1", [(0, 1)], [], 1.0), ("n2_load50", [(0, 2)], [0], 0.5)]
    else:
        allg = [(g, 16) for g in range(n_gpu)]
        plan = [("n1", [(0, 1)], [], 1.0), ("n8", [(0, 8)], [], 1.0), ("n16", [(0, 16)], [], 1.0),
                ("n16_load100", [(0, 16)], [0], 1.0), ("n16_load50", [(0, 16)], [0], 0.5), ("n32", [(0, 32)], [], 1.0),
                (f"all{16 * n_gpu}", allg, [], 1.0), (f"all{16 * n_gpu}_load50", allg, list(range(n_gpu)), 0.5)]
    if a.tests:
        keep = set(a.tests.split(","))
        plan = [p for p in plan if p[0] in keep]
    report["tests"] = []
    for name, placement, load_gpus, duty in plan:
        emit("start", test=name, at=time.strftime("%T"))
        res = run_test(a, name, placement, load_gpus, duty, egl_of, tmp)
        report["tests"].append(res)
        emit("result", **{k: v for k, v in res.items() if k not in ("hung_detail", "failed_detail")},
             hung_detail=res["hung_detail"][:2], failed_detail=res["failed_detail"][:2])
        save()
        if res["driver"]["smi_rc"] != 0:
            report["aborted"] = f"nvidia-smi did not answer after {name}; remaining GPU tests skipped"
            emit("abort", reason=report["aborted"])
            break
    report["end"] = time.strftime("%F %T")
    report["dmesg_xid_end"] = sh("dmesg -T 2>&1 | grep -i -E 'xid|nvrm|gsp' | tail -20", timeout=30)
    save()
    emit("done", path=path)
    # exit code for the stage runner: 3 = driver stuck (later GPU stages are pointless), 2 = no env rendered at all
    if report.get("aborted"):
        return 3
    if not report["tests"] or report["tests"][0]["ok"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
