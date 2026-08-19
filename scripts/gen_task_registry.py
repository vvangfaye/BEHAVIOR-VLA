#!/usr/bin/env python
"""Regenerate openpi's b1k task prompt registry from the released dataset.

Upstream ships only ``turning_on_radio``, so serving any other task raises
KeyError in scripts/b1k/serve_b1k.py. The prompts must also match what training
saw: the b1k data config sets ``prompt_from_task=True``, so the model was
conditioned on the dataset's own task descriptions. meta/tasks.jsonl is that
authoritative source (its turning_on_radio text is byte-identical to upstream's).

Usage:
  ./scripts/gen_task_registry.py [--dataset /var/tmp/behavior-1k/2026-challenge-demos]
"""
from __future__ import annotations

import argparse
import json
import pathlib

DEFAULT_DATASET = "/var/tmp/behavior-1k/2026-challenge-demos"
TARGET = "openpi/src/openpi/configs/tasks/b1k.py"


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--target", default=str(repo_root / TARGET))
    args = ap.parse_args()

    src = pathlib.Path(args.dataset) / "meta" / "tasks.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no tasks found in {src}")

    lines = [
        "from . import TASK_REGISTRY",
        "",
        "# Task prompts for the 2026 BEHAVIOR Challenge, generated from the released",
        "# dataset's meta/tasks.jsonl so serving uses the exact prompts seen in training",
        "# (the b1k data config sets prompt_from_task=True).",
        "TASKS = {",
    ]
    for r in sorted(rows, key=lambda x: x["task_index"]):
        desc = r["task"].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    "{r["task_name"]}": "{desc}",')
    lines += ["}", "", "# Register in global registry", 'TASK_REGISTRY["b1k"] = TASKS', ""]

    target = pathlib.Path(args.target)
    target.write_text("\n".join(lines))
    print(f"wrote {len(rows)} task prompts to {target}")


if __name__ == "__main__":
    main()
