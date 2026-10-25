"""Freeze a minimal task-22 inference release, without demonstration downloads."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
public = json.loads((HERE / "public_asset_manifest.json").read_text())
extra = json.loads((HERE / "extra_asset_manifest.json").read_text())
paired = json.loads((HERE / "paired_instances.json").read_text())
items = []


def add(repo, revision, record, dest, dataset=False):
    name = record.get("rfilename", record.get("path"))
    lfs = record.get("lfs") or {}
    digest = lfs.get("sha256", lfs.get("oid"))
    items.append({
        "url": f"https://huggingface.co/{'datasets/' if dataset else ''}{repo}/resolve/{revision}/{name}",
        "destination": dest,
        "size": record["size"],
        "sha256": digest,
        "git_blob_sha1": None if digest else record.get("blobId", record.get("oid")),
    })


model = public["model"]["data"]
for record in model["siblings"]:
    if "task-0022/" in record["rfilename"]:
        add("byeonghyunpak/SERF-VLA", model["sha"], record, "checkpoints/" + record["rfilename"])

map_revision = extra["map_info"]["sha"]
map_paths = {
    f"exported_neural_points/task-0022/eval/scene_{i}.hdf5" for i in paired["instance_ids"]
} | {
    "map_models/task-0022/neural_points/robot/sampler.npz",
    "map_models/task-0022/neural_points/robot/robot_neural_points.pt",
    "map_models/task-0022/config.yaml",
    "map_models/task-0022/instance_id_to_name.json",
}
for group in ("maps_task22", "points_task22"):
    for record in public[group]["data"]:
        if record["path"] in map_paths:
            add("suk063/SERF", map_revision, record, "maps/" + record["path"], dataset=True)

# Metadata and task-specific state templates only. Existing shared object/scene
# assets are separately verified against the pinned OmniGibson version.
instances = extra["instances"]
for record in instances["siblings"]:
    name = record["rfilename"]
    if name.startswith("metadata/") or "putting_shoes_on_rack" in name:
        add("behavior-1k/2025-challenge-task-instances", instances["sha"], record,
            "instances/" + name, dataset=True)

assert len({x["destination"] for x in items}) == len(items)
assert all(x["sha256"] or x["git_blob_sha1"] for x in items)
manifest = {"items": items, "total_bytes": sum(x["size"] for x in items),
            "paired_instances": paired, "checkpoint_revision": model["sha"],
            "map_revision": map_revision,
            "excluded": ["training demonstrations", "training maps", "other task checkpoints"],
            "external_prerequisites": ["compatible OmniGibson shared assets", "R1Pro URDF and meshes", "CoTracker3 offline weights", "tokenizer"]}
(HERE / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"files": len(items), "total_bytes": manifest["total_bytes"], "GiB": manifest["total_bytes"] / 2**30,
                  "instances": paired["instance_ids"]}))
