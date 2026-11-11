"""Build a CPU diagnostic bundle; no remote operations or job submission."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEST = HERE / "bundle"


def main():
    if DEST.exists():
        raise FileExistsError("Never overwrite a prepared diagnostic bundle")
    DEST.mkdir()
    source = ROOT / "formal_stage/bundle"
    manifest = json.loads((source / "bundle_manifest.json").read_text())
    paths = ["openpi/src/openpi/training/config.py", "openpi/src/openpi/transforms.py",
             "openpi/src/openpi/policies/b1k_policy.py", "openpi/src/openpi/configs/robots/__init__.py",
             "openpi/src/openpi/configs/robots/base_config.py", "openpi/src/openpi/configs/robots/b1k.py"]
    for relative in paths:
        p = source / relative
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest[relative]:
            raise ValueError("Frozen source changed")
        target = DEST / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    for relative in ["source/openpi/src/openpi/__init__.py", "source/openpi/src/openpi/configs/__init__.py"]:
        (DEST / relative).touch()
    shutil.copy2(HERE / "worker.py", DEST / "worker.py")
    shutil.copy2(ROOT / "audit_action_ranges.py", DEST / "transform_helpers.py")
    shutil.copy2(ROOT / "replay_loss_column_plan.json", DEST / "column_plan.json")
    shutil.copy2(ROOT / "replay_footer_probe.json", DEST / "footer_probe.json")
    shutil.copy2(ROOT / "cpu/bundle/inputs/norm_assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json", DEST / "norm_stats.json")
    shutil.copy2(ROOT / "dispatch_support.py", HERE / "dispatch_support.py")
    shutil.copy2(source / "nebula/submit_oss.py", HERE / "submit_oss.py")
    cluster = {"worker": {"cpu": 800, "gpu": 0, "memory": 64000}}
    (DEST / "cluster.json").write_text(json.dumps(cluster, indent=2) + "\n")
    plan = json.loads((HERE / "plan.json").read_text())
    old = json.loads((source / "nebula/worker_job.json").read_text())
    base = {"run_id": "b1k-replay-loss-audit-20261017-g0", "output_key": plan["output_prefix"],
            "maximum_worker_seconds": plan["maximum_worker_seconds"],
            "maximum_data_range_bytes": plan["maximum_data_range_bytes"],
            "environment": old["environment"], "download_endpoint": old["download_endpoint"]}
    (HERE / "runtime_base.json").write_text(json.dumps(base, indent=2) + "\n")
    files = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in DEST.rglob("*") if p.is_file()}
    (HERE / "public_bundle_manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"bundle_files": len(files), "gpus": 0, "submitted": False}))


if __name__ == "__main__":
    main()
