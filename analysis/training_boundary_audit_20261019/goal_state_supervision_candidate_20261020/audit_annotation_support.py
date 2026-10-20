"""Inventory annotation ordering; never interpret placement labels as achieved goals."""
from collections import Counter
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
META = HERE.parent / "pilot_three_tasks_cache_v1/dataset/meta"
LABELS = {
    "place on tissue dispenser to countertop": "dispenser",
    "place on toilet tissue to toilet": "toilet_tissue",
    "place on bar soap to soap dish": "soap",
    "place in cork to trash can": "cork",
}

def main():
    episodes = pd.concat([
        pd.read_parquet(p, columns=["episode_index", "task_index"])
        for p in sorted((META / "episodes").glob("chunk-*/*.parquet"))
    ], ignore_index=True)
    ids = set(episodes.loc[episodes.task_index.eq(84), "episode_index"].tolist())
    segments = pd.read_parquet(META / "original_subtask_segments.parquet")
    segments = segments[segments.episode_index.isin(ids)].sort_values(["episode_index", "start"])
    orders = Counter()
    for _, rows in segments.groupby("episode_index"):
        seen = []
        for label in rows.subtask:
            if label in LABELS and LABELS[label] not in seen:
                seen.append(LABELS[label])
        orders[tuple(seen)] += 1
    result = {
        "task": "tidying_bathroom", "task_index": 84,
        "episodes": len(ids), "annotation_segments": len(segments),
        "first_occurrence_placement_orders": [
            {"order": list(order), "episodes": count}
            for order, count in orders.most_common()
        ],
        "label_counts": {label: int(segments.subtask.eq(label).sum()) for label in LABELS},
        "interpretation": "Annotation-order concentration only. Does not measure achieved predicates, establish an incorrect order, or show failure to recover.",
        "data_gap": "Existing segment timestamps are not physical goal-state labels. Construct and validate those separately; do not infer them by cumulative annotation counts.",
    }
    (HERE / "annotation_support.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
