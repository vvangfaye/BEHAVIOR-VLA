"""Show the latest numbered worker receipt, including numbers beyond 99."""

import json
from pathlib import Path


root = Path(__file__).resolve().parent
paths = list(root.glob("live_[0-9]*.json"))
if not paths:
    print("No live receipt yet")
    raise SystemExit(0)
path = max(paths, key=lambda p: int(p.stem.split("_")[1]))
data = json.loads(path.read_text())
files = data.get("files", {})
status = files.get("results/status.json", {})
stage = status.get("stage")
out = {"receipt": path.name, "observed_at": data.get("observed_at"), "status": status}
if "results/reference_progress.json" in files:
    out["reference_inputs"] = files["results/reference_progress.json"]
out["latest_subtasks"] = data.get("latest_probe_subtasks", {})
out["completed"] = [
    value for name, value in files.items()
    if name.startswith(("results/jax/", "results/torch/")) and name.endswith("/result.json")
]
if stage in ("baseline", "adapted"):
    backend = "jax" if stage == "baseline" else "torch"
    for instance in (301, 302):
        prefix = f"results/{backend}/{instance}/"
        if prefix + "result.json" in files:
            continue
        out["active"] = {"backend": backend, "instance": instance}
        for name in ("initialization.json", "inference.json", "progress.json", "failure.json"):
            if prefix + name in files:
                out["active"][name.removesuffix(".json")] = files[prefix + name]
        break
for name in ("results/gate.json", "results/ppo/progress.json", "results/ppo/result.json"):
    if name in files:
        out[name] = files[name]
print(json.dumps(out, ensure_ascii=False))
