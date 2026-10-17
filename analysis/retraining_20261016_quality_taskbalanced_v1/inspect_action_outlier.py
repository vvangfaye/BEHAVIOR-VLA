"""Inspect raw values underlying the largest observed full-episode target."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "heldout20_baseline_vs_expert1k"))
import controller


def main():
    output = ROOT / "loss_spike_action_audit"
    result = json.loads((output / "result.json").read_text())
    window = result["windows_ranked_by_magnitude"][0]
    root = window["root_id"]
    proof = json.loads((output / "roots" / (root + ".json")).read_text())
    plan = json.loads((ROOT / "loss_spike_data_access_plan.json").read_text())
    source = next(item["source"] for item in plan["source_objects"] if item["root_id"] == root)
    if len(proof["members"]) != 1:
        raise ValueError("Expected one proven data member for this focused inspection")
    member = proof["members"][0]
    if member["size"] > plan["maximum_single_member_bytes"]:
        raise ValueError("Member exceeds read bound")
    client = controller.transport.aoss_client()
    response = client.get_object(Bucket=source["bucket"], Key=source["key"], IfMatch=proof["source_etag"],
        Range=f"bytes={member['offset']}-{member['offset']+member['size']-1}")
    try:
        if response.get("ContentLength") != member["size"] or response.get("ContentRange") != f"bytes {member['offset']}-{member['offset']+member['size']-1}/{source['bytes']}":
            raise ValueError("Server ignored byte range")
        data = response["Body"].read(member["size"] + 1)
    finally:
        response["Body"].close()
    if hashlib.sha256(data).hexdigest() != member["sha256"]:
        raise ValueError("Focused inspection differs from the audited Parquet bytes")
    table = pq.read_table(io.BytesIO(data), columns=["episode_index", "frame_index", "action", "observation.state"]).to_pydict()
    actions = np.asarray(table["action"], dtype=np.float32)
    states = np.asarray(table["observation.state"], dtype=np.float32)
    frames = np.asarray(table["frame_index"])
    dim = window["largest_dimension"]
    mapping = next((a, s) for a, s in result["delta_mappings"] if dim in a)
    state_index = mapping[1][mapping[0].index(dim)]
    # State extraction in this run maps action dimension 6 to raw proprio index 56.
    if (dim, state_index) != (6, 6):
        raise ValueError("Recheck the frozen robot mapping for a different dimension")
    raw_state_index = 56
    stats_path = ROOT / "cpu/bundle/inputs/norm_assets/b1k_full100_joint_subtask_v1_drop_episode_s1/norm_stats.json"
    stats = json.loads(stats_path.read_text())["norm_stats"]["actions"]
    mean, std = stats["mean"][dim], stats["std"][dim]
    anchor = np.flatnonzero((np.asarray(table["episode_index"]) == window["episode_index"]) & (frames == window["frame"]))
    if len(anchor) != 1:
        raise ValueError("Ambiguous anchor")
    i = int(anchor[0])
    raw = actions[i:i+32, dim]
    state = float(states[i, raw_state_index])
    normalized = (raw - state - mean) / (std + 1e-6)
    if not np.isclose(np.max(np.abs(normalized)), window["normalized_abs_max"], rtol=1e-5):
        raise ValueError("Raw value reconstruction does not reproduce the audited target")
    instantaneous = (actions[:, dim] - states[:, raw_state_index] - mean) / (std + 1e-6)
    peak = int(np.argmax(np.abs(instantaneous)))
    result = {
        "root_id": root, "task": window["task"], "global_index": window["global_index"],
        "frame": window["frame"], "dimension": dim, "raw_state_index": raw_state_index,
        "parquet_sha256": member["sha256"], "range_bytes": len(data),
        "anchor_action_min_max": [float(raw.min()), float(raw.max())],
        "anchor_state": state, "normalization_mean": mean, "normalization_std": std,
        "normalized_target_min_max": [float(normalized.min()), float(normalized.max())],
        "episode_action_min_max": [float(actions[:, dim].min()), float(actions[:, dim].max())],
        "episode_state_min_max": [float(states[:, raw_state_index].min()), float(states[:, raw_state_index].max())],
        "instantaneous_abs_max": float(np.abs(instantaneous[peak])), "instantaneous_peak_frame": int(frames[peak]),
        "frames_above_abs_50_std": int(np.count_nonzero(np.abs(instantaneous) > 50)),
        "total_episode_rows": len(frames), "data_modified": False,
        "interpretation": "The target magnitude is reproduced from recorded action minus anchor state and frozen std. This identifies numerical amplification, but does not prove the physical state is corrupt or quantify the causal contribution to the training loss."
    }
    (output / "focused_outlier.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
