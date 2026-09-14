#!/usr/bin/env python
"""Build a per-frame sub-task language table for BEHAVIOR-1K from the dataset's
skill annotations, so training/inference can condition on the CURRENT sub-task
instead of one whole-episode prompt.

Verified alignment:
  * Each dataset episode carries an authoritative ``annotation_path`` column in
    meta/episodes/*.parquet that names its annotation JSON directly, so no
    positional pairing heuristic is needed (an earlier per-task rank pairing was
    checked to agree with it on all 20000 episodes).
  * A dataset episode shares the annotation's frame origin (frame 0 == frame 0),
    so an annotation skill segment [start,end) maps onto dataset LOCAL frame
    indices, and frames at/after task_duration inherit the last skill.
  * Usually the dataset episode is >= the annotated duration (extra trailing
    settle frames), but 53 of 20000 episodes are SHORTER than their annotation
    (worst: episode_index 544, length 148 vs task_duration 8790). Intervals are
    clipped to the episode length, so those demos are labelled only up to where
    the recorded data actually ends.

Output: a compact run-length-encoded segment table, one row per contiguous sub-task
run, written to <dataset>/meta/subtask_segments.parquet:
  episode_index (int64), start (int64), end (int64, exclusive), subtask (str)
The runs tile [0, episode_length) so a frame lookup is an interval search. This is
~1000x smaller than a per-frame table, so every dataloader worker can hold it.

  openpi/.venv/bin/python scripts/gen_subtask_prompts.py \
      --dataset /var/tmp/behavior-1k/2026-challenge-demos \
      --annotations /var/tmp/behavior-1k/2026-challenge-demos/annotations
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _obj_phrase(seg: dict) -> str:
    """Human-readable object phrase from a skill/primitive segment."""
    def leaves(x):
        out = []
        if isinstance(x, (list, tuple)):
            for e in x:
                out.extend(leaves(e))
        elif x is not None:
            out.append(str(x))
        return out

    manip = leaves(seg.get("manipulating_object_id"))
    objs = leaves(seg.get("object_id"))

    def clean(x):
        s = re.sub(r"_[0-9]+$", "", str(x))     # drop trailing instance number
        return s.replace("_", " ").strip()

    target = manip[0] if manip else (objs[0] if objs else None)
    ct = clean(target) if target else ""
    # a reference object only if it is genuinely different from the target
    ref = None
    for o in objs:
        if o != target and clean(o) and clean(o) != ct:
            ref = o
            break
    if ct and ref:
        return f"{ct} to {clean(ref)}"
    return ct


def _subtask_text(seg: dict) -> str:
    desc = seg.get("skill_description") or seg.get("primitive_description") or []
    desc = desc[0] if isinstance(desc, list) and desc else (desc if isinstance(desc, str) else "")
    obj = _obj_phrase(seg)
    return f"{desc} {obj}".strip() if obj else str(desc).strip()


def load_episode_meta(dataset: str):
    """[(episode_index, length, annotation_path)] sorted by episode_index.

    Uses the dataset's authoritative ``annotation_path`` column, which links each
    episode directly to its annotation JSON (no positional pairing heuristic).
    """
    rows = []
    for f in sorted(glob.glob(f"{dataset}/meta/episodes/chunk-*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "length", "annotation_path"]).to_pydict()
        for i, e in enumerate(t["episode_index"]):
            rows.append((int(e), int(t["length"][i]), t["annotation_path"][i]))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/var/tmp/behavior-1k/2026-challenge-demos")
    ap.add_argument("--annotations", default=None,
                    help="annotations root; default <dataset>/annotations")
    ap.add_argument("--use-primitive", action="store_true",
                    help="use primitive_annotation instead of skill_annotation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-annotated-coverage", type=float, default=0.0,
                    help="skip episodes whose recorded length covers less than this "
                         "fraction of the annotation's task_duration (0 = keep all). "
                         "e.g. 0.5 drops episode 544 (148 frames vs 8790 annotated)")
    args = ap.parse_args()
    out = args.out or f"{args.dataset}/meta/subtask_segments.parquet"
    seg_key = "primitive_annotation" if args.use_primitive else "skill_annotation"
    dur_key = "frame_duration"

    episodes = load_episode_meta(args.dataset)
    print(f"dataset episodes: {len(episodes)}")

    rows_ep, rows_start, rows_end, rows_txt = [], [], [], []
    n_uncovered = 0
    n_ep_done = 0
    n_missing_files = 0
    n_skipped_coverage = 0
    vocab = set()

    for e, length, ann_path in episodes:
        # annotation_path is relative to the dataset root; allow overriding the
        # annotations root while keeping the task-XXXX/episode_*.json tail.
        if args.annotations:
            path = os.path.join(args.annotations, *ann_path.split("/")[1:])
        else:
            path = os.path.join(args.dataset, ann_path)
        if not os.path.exists(path):
            n_missing_files += 1
            continue
        j = json.load(open(path))
        if args.min_annotated_coverage > 0:
            dur = (j.get("meta_data") or {}).get("task_duration") or 0
            if dur > 0 and length / dur < args.min_annotated_coverage:
                n_skipped_coverage += 1
                continue
        segs = j.get(seg_key) or []
        # build a per-local-frame label array of size `length`
        labels = [None] * length
        for s in segs:
            fd = s.get(dur_key)
            if not fd:
                continue
            # frame_duration is usually [start, end]; ~0.18% of segments use a
            # nested [[s1,e1],[s2,e2],...] form for a non-contiguous skill.
            intervals = fd if isinstance(fd[0], list) else [fd]
            txt = _subtask_text(s)
            vocab.add(txt)
            for iv in intervals:
                a, b = int(iv[0]), int(iv[1])
                for f in range(max(0, a), min(length, b)):
                    labels[f] = txt
        # fill gaps + trailing tail with the last known label (settle frames)
        last = None
        for f in range(length):
            if labels[f] is None:
                labels[f] = last if last is not None else ""
                n_uncovered += 1
            else:
                last = labels[f]
        # run-length-encode the per-frame labels into contiguous segments
        run_start = 0
        for f in range(1, length + 1):
            if f == length or labels[f] != labels[run_start]:
                rows_ep.append(e)
                rows_start.append(run_start)
                rows_end.append(f)
                rows_txt.append(labels[run_start])
                run_start = f
        n_ep_done += 1

    tbl = pa.table({
        "episode_index": pa.array(rows_ep, pa.int64()),
        "start": pa.array(rows_start, pa.int64()),
        "end": pa.array(rows_end, pa.int64()),
        "subtask": pa.array(rows_txt, pa.string()),
    })
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pq.write_table(tbl, out)

    total_frames = sum(r[1] for r in episodes)
    print(f"\nwrote {out}")
    print(f"  episodes covered: {n_ep_done}/{len(episodes)}   missing annotation files: {n_missing_files}"
          f"   skipped(low coverage): {n_skipped_coverage}")
    print(f"  segments: {len(rows_ep):,}   (avg {len(rows_ep)/max(n_ep_done,1):.1f}/episode)")
    print(f"  frames: {total_frames:,}   uncovered/tail-filled: {n_uncovered:,} "
          f"({n_uncovered/max(total_frames,1)*100:.1f}%)")
    print(f"  distinct sub-task strings: {len(vocab)}")
    print("  sample sub-tasks:")
    for v in list(sorted(vocab))[:10]:
        print(f"    - {v!r}")


if __name__ == "__main__":
    main()
