"""Read-only strict acceptance for the two-step expert-control smoke."""
import json
import math
from pathlib import Path

def verify_smoke(client, prefix):
    success = json.loads(client.read(prefix + "/_SMOKE_SUCCESS.json"))
    if success.get("completed_updates") != 2 or success.get("gpus") != 16 or not success.get("checkpoint_restore"):
        raise ValueError("Actual16GPU smoke proof invalid")
    state = json.loads(client.read(prefix + "/checkpoint_restore_ok.json"))
    if state.get("restored_step") != 2 or state.get("global_devices") != 16:
        raise ValueError("Smoke restored state/topology invalid")
    reports = [json.loads(x) for x in client.read(prefix + "/smoke_metrics.jsonl").decode().splitlines()]
    updates = [x for x in reports if x["kind"] == "optimizer_update"]
    restores = [x for x in reports if x["kind"] == "checkpoint_restore"]
    batches = [x for x in reports if x["kind"] == "real_expert_only_batch"]
    if len(updates) != 2 or len(restores) != 1 or not restores[0].get("all_state_leaves_exact") or restores[0].get("optimizer_clock") != 2:
        raise ValueError("Smoke exact two-update/optimizer restore proof missing")
    if len(batches) != 2 or [x.get("step") for x in batches] != [0, 1]:
        raise ValueError("Smoke must contain exactly two real global batches")
    for batch in batches:
        counts = batch.get("source_counts", {})
        if (counts != {"0": 192}
                or sum(counts.values()) != 192 or batch.get("actions_shape") != [192, 32, 32]
                or batch.get("full_episode_text_target_tokens") != 0
                or batch.get("supervised_text_target_tokens", 0) <= 0
                or len(batch.get("images", {})) != 3
                or any(shape != [192, 224, 224, 3] for shape in batch["images"].values())):
            raise ValueError("Actual GPU batch/source/text-mask proof invalid")
    fingerprints = {batch["mixture_fingerprint"] for batch in batches}
    if len(fingerprints) != 1 or restores[0].get("next_global_sample") != 384 or restores[0].get("third_update_executed") is not False:
        raise ValueError("Smoke sampler continuity or finite two-update bound invalid")
    # This experiment has a deterministic finite sample stream. Prove the new
    # global sampler was actually used, rather than only checking both sources
    # appeared in a batch that reported the expected manifest fingerprint.
    expected = json.loads((Path(__file__).parent / "smoke/bundle/nebula/smoke_job.json").read_text())
    if fingerprints != {expected["mixture_fingerprint"]} or success.get("run_id") != expected["run_id"]:
        raise ValueError("Smoke differs from the exact expert-control identity")
    for i, record in enumerate(updates):
        metrics = record["metrics"]
        expected = (4.975124378109452e-9, 9.950248756218905e-9)[i]
        if (record.get("step") != i or not all(math.isfinite(v) for v in metrics.values())
                or metrics.get("lr_clock_matches_step") != 1
                or not math.isclose(metrics["learning_rate"], expected, rel_tol=2e-5)):
            raise ValueError("Applied GPU learning rate or metrics invalid")
    for rank in (0, 1):
        record = json.loads(client.read(prefix + f"/rank-{rank}/_RANK_SUCCESS.json"))
        if record.get("completed_updates") != 2:
            raise ValueError("Both smoke workers must finish")
    return dict(success=success, checkpoint_state=state, reports=reports, source_prefix=prefix,
                verified_mixture_fingerprint=next(iter(fingerprints)))

