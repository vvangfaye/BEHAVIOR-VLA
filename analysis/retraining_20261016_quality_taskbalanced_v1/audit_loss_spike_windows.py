"""Reconstruct candidate windows for two observed loss spikes; never label them corrupt."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "formal_stage/bundle"
sys.path.insert(0, str(BUNDLE / "openpi/src"))
from openpi.training.subtask_mixture import MixturePlan


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metrics_path = ROOT / "training_metrics.snapshot.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    plan = MixturePlan.load(BUNDLE / "inputs/compiled/mixture.json")
    if any(row["mixture_fingerprint"] != plan.fingerprint for row in rows):
        raise ValueError("Metrics and sampler do not describe the same training run")
    logging_source = BUNDLE / "openpi/scripts/b1k/train_b1k.py"
    loader_source = BUNDLE / "openpi/src/openpi/training/data_loader.py"
    frozen = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    for path in (logging_source, loader_source):
        if sha(path) != frozen[str(path.relative_to(BUNDLE))]:
            raise ValueError("Audited logging/data-order source changed")
    previous, spans = 0, []
    for row in rows:
        end = row["completed_updates"]
        if end <= previous:
            raise ValueError("Metric update clock is not increasing")
        if end in (240, 820):
            indices = range(previous * plan.global_batch_size, end * plan.global_batch_size)
            references = [plan.reference(index) for index in indices]
            counts = Counter(ref["source_kind"] for ref in references)
            steps = end - previous
            for source in ("replay", "full_episode"):
                observed = row["metrics"]["source_count/" + source] * steps
                if abs(counts[source] - observed) > 1e-3:
                    raise ValueError("Reconstructed source counts differ from actual metrics")
            dominant = "full_episode" if end == 240 else "replay"
            suspects = [ref for ref in references if ref["source_kind"] == dominant]
            spans.append({
                "completed_updates": [previous + 1, end],
                "global_indices": [indices.start, indices.stop - 1],
                "source_counts": dict(counts),
                "dominant_source_by_reported_loss": dominant,
                "reported_metrics": row["metrics"],
                "candidate_windows_in_dominant_source": suspects,
                "all_windows": references,
                "interpretation": "Ten-update aggregate loss does not identify an offending example. These are candidates for raw/normalized action inspection, not confirmed bad data."
            })
        previous = end
    if len(spans) != 2:
        raise ValueError("Both observed spike intervals are required")
    result = {
        "kind": "loss_spike_candidate_window_audit",
        "metrics_sha256": sha(metrics_path), "mixture_fingerprint": plan.fingerprint,
        "source_proofs": {str(p.relative_to(ROOT)): sha(p) for p in (logging_source, loader_source)},
        "mapping": "No-shuffle distributed sampling over two ranks reconstructs one contiguous global batch of 192 indices per optimizer update; metrics after warmup average the intervening ten updates.",
        "spans": spans, "data_modified": False, "cloud_jobs_created": 0,
        "corrupted_examples_identified": False,
        "next_check": "Inspect finite raw and normalized action ranges for the candidate windows, compare with normal windows, and separate target outliers from model prediction errors before changing data."
    }
    (ROOT / "loss_spike_candidate_windows.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"spans": [{"updates": s["completed_updates"],
        "source_counts": s["source_counts"],
        "dominant_source_candidates": len(s["candidate_windows_in_dominant_source"])} for s in spans],
        "corrupted_examples_identified": False, "data_modified": False}))


if __name__ == "__main__":
    main()
