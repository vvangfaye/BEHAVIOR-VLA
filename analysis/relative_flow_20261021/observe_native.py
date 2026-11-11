"""Read bounded live evidence without modifying a frozen native experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "closed_loop_exploratory_ada01"


def main():
    state_path = ROOT / "state.json"
    if not state_path.exists():
        print(json.dumps(dict(stage="not_launched")))
        return
    state = json.loads(state_path.read_text())
    active = state.get("running")
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(),
                  active=active, complete_episode=False)
    if not active:
        result["completed_episodes"] = state.get("completed_episodes", 0)
        print(json.dumps(result, indent=2))
        return
    folder = ROOT / "episodes" / active["method"] / f'{active["task"]}-{active["instance"]}'
    metadata_path = folder / "policy_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        fields = ["model_horizon", "action_horizon", "subtask_generation",
                  "supplied_subtasks", "experimental_method", "study_type",
                  "full_episode_seed_protocol", "policy_rng_backend"]
        result["policy_metadata"] = {key: metadata[key] for key in fields if key in metadata}
    initial_path = folder / "initial_state.json"
    if initial_path.exists():
        initial = json.loads(initial_path.read_text())
        result["initial_physics_sha256"] = initial["physics_sha256"]
    action_path = folder / "actions.f32"
    if action_path.exists():
        # The unchanged R1Pro config fixes 23 float32 physical action values.
        # A live stream may contain a partial final write, and buffered records
        # may not yet reach disk. Report a lower bound, not an exact live step.
        record_bytes = 23 * 4
        size = action_path.stat().st_size
        complete_bytes = (min(size, 4 * 1024 * 1024) // record_bytes) * record_bytes
        with action_path.open("rb") as stream:
            raw = stream.read(complete_bytes)
        raw = raw[:len(raw) // record_bytes * record_bytes]
        actions = np.frombuffer(raw, dtype="<f4").reshape(-1, 23)
        result["persisted_action_records_lower_bound"] = len(actions)
        result["persisted_actions_finite"] = bool(np.isfinite(actions).all())
        result["closed_loop_execution_observed"] = len(actions) > 0
    receipt_path = folder / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        result["complete_episode"] = receipt.get("accepted", False)
        for key in ["record", "error", "error_type"]:
            if key in receipt:
                result[key] = receipt[key]
    result["seed_receipt_note"] = "The frozen evaluator writes seed_ack.json only at episode termination."
    result["scope"] = "Live execution evidence only; no task success is inferred before an accepted episode receipt."
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
