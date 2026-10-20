"""Find annotation-order variation without treating annotations as physical truth."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
META = HERE.parent / "pilot_three_tasks_cache_v1/dataset/meta"

def main():
    tasks = {t["task_index"]: t for t in map(json.loads, (META / "tasks.jsonl").read_text().splitlines())}
    episodes = pd.concat([
        pd.read_parquet(p, columns=["episode_index", "task_index"])
        for p in sorted((META / "episodes").glob("chunk-*/*.parquet"))
    ], ignore_index=True).set_index("episode_index")
    segments = pd.read_parquet(META / "original_subtask_segments.parquet")
    segments = segments.sort_values(["episode_index", "start"])
    groups = defaultdict(Counter)
    members = defaultdict(list)
    for episode, rows in segments.groupby("episode_index"):
        if episode not in episodes.index:
            continue
        task = int(episodes.loc[episode, "task_index"])
        order = tuple(dict.fromkeys(rows.subtask.tolist()))
        groups[(task, frozenset(order))][order] += 1
        members[(task, order)].append(int(episode))
    qualifying = []
    for (task, labels), counts in groups.items():
        top = counts.most_common()
        if len(top) < 2 or top[1][1] < 10 or sum(counts.values()) < 30:
            continue
        qualifying.append({
            "task_index": task,
            "task_name": tasks[task]["task_name"],
            "episodes_in_same_label_set": sum(counts.values()),
            "unique_labels": len(labels),
            "distinct_first_occurrence_orders": len(counts),
            "top_two": [{
                "labels": list(order), "episodes": count,
                "example_episode_indices": members[(task, order)][:5]
            } for order, count in top[:2]],
        })
    qualifying.sort(key=lambda x: x["top_two"][1]["episodes"], reverse=True)
    result = {
        "scope": "Full local original_subtask_segments metadata and episode-to-task mapping; no images, actions or simulator outcomes inspected.",
        "criterion": "Same task and same set of distinct labels; >=30 episodes total; at least two first-occurrence orders with >=10 episodes each.",
        "qualifying_task_labelset_groups": len(qualifying),
        "qualifying_distinct_tasks": len({g["task_index"] for g in qualifying}),
        "groups": qualifying,
        "limits": [
            "A label-set group is not a task count; one task can have multiple groups.",
            "First occurrence discards repetitions; variation can reflect navigation, retries or annotation choices rather than alternate completed-goal order.",
            "This does not establish matched physical states, actual task completion, or causal action equivalence.",
            "Demonstration metadata availability does not imply corresponding full videos and action files are already cached locally."
        ],
        "implication": "The bathroom fixed-order observation must not be generalized to the entire dataset. Natural order variants are candidates for validating action-composition support before generating new data.",
    }
    HERE.joinpath("order_variation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, ensure_ascii=False, indent=2))
    for row in qualifying[:2]:
        print(json.dumps({"task": row["task_name"], "counts": [x["episodes"] for x in row["top_two"]]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
