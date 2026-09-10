#!/usr/bin/env python
"""Compute GR00T relative-action statistics in parallel.

``gr00t.data.stats`` walks all 20000 episodes serially for every RELATIVE action
key (~4.3 h per key, ~13 h total here) and only writes the cache once every key
is done, so an interrupted run loses everything. The per-episode work is
independent, so this fans it out across processes and writes the same
``meta/relative_stats.json`` (same statistics, same fingerprints) after each key,
making the result identical but resumable.

Usage:
  ./scripts/gen_groot_rel_stats.py --dataset-path /var/tmp/behavior-1k/2026-challenge-demos \
      --modality-config examples/b1k/r1pro.py --embodiment-tag NEW_EMBODIMENT [--jobs 64]
  # verify against the serial implementation on a few episodes:
  ... --verify-episodes 40
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

_LOADER = None


def _init_worker(dataset_path: str, embodiment_tag: str, action_key: str, modality_config: str):
    global _LOADER
    _load_modality_config(modality_config)
    from gr00t.data.stats import RelativeActionLoader

    _LOADER = RelativeActionLoader(dataset_path, _tag(embodiment_tag), action_key)


def _work(episode_ids: list[int]) -> np.ndarray | None:
    out = []
    for eid in episode_ids:
        out.extend(_LOADER.load_relative_actions(eid))
    if not out:
        return None
    return np.stack(out).astype(np.float32)


def _load_modality_config(path: str) -> None:
    p = Path(path)
    # Forked workers inherit the parent's registry; re-executing the module trips
    # register_modality_config's "already registered" assertion.
    if p.stem in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[p.stem] = mod
    spec.loader.exec_module(mod)


def _stats(arr: np.ndarray) -> dict:
    return {
        "max": np.max(arr, axis=0),
        "min": np.min(arr, axis=0),
        "q01": np.quantile(arr, 0.01, axis=0),
        "q99": np.quantile(arr, 0.99, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }


def _relative_keys(embodiment_tag: str) -> list[str]:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ActionRepresentation

    cfg = MODALITY_CONFIGS[_tag(embodiment_tag).value]["action"]
    return sorted(
        key
        for key, ac in zip(cfg.modality_keys, cfg.action_configs)
        if ac.rep == ActionRepresentation.RELATIVE
    )


def _tag(embodiment_tag: str):
    """Accept either the enum name (NEW_EMBODIMENT) or its value (new_embodiment)."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    try:
        return EmbodimentTag(embodiment_tag)
    except ValueError:
        return EmbodimentTag[embodiment_tag]


def _gather(dataset_path: str, embodiment_tag: str, key: str, modality_config: str,
            jobs: int, limit: int | None) -> np.ndarray:
    from gr00t.data.stats import RelativeActionLoader

    probe = RelativeActionLoader(dataset_path, _tag(embodiment_tag), key)
    total = len(probe) if limit is None else min(limit, len(probe))
    del probe

    ids = list(range(total))
    # Chunk so each task is big enough to amortise IPC but small enough to balance.
    chunk = max(1, total // (jobs * 4))
    batches = [ids[i : i + chunk] for i in range(0, total, chunk)]

    parts: list[np.ndarray] = []
    done = 0
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=jobs,
        initializer=_init_worker,
        initargs=(dataset_path, embodiment_tag, key, modality_config),
    ) as pool:
        for res in pool.imap_unordered(_work, batches):
            if res is not None:
                parts.append(res)
            done += 1
            if done % 20 == 0 or done == len(batches):
                el = time.time() - t0
                print(
                    f"  {key}: {done}/{len(batches)} batches  {el:.0f}s"
                    f"  eta {el / done * (len(batches) - done):.0f}s",
                    flush=True,
                )
    return np.concatenate(parts, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--modality-config", required=True)
    ap.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--verify-episodes", type=int, default=0,
                    help="compare against the serial implementation on N episodes and exit")
    args = ap.parse_args()

    _load_modality_config(args.modality_config)
    from gr00t.data.stats import (
        LE_ROBOT_REL_STATS_FILENAME,
        STATS_FINGERPRINTS_KEY,
        _compute_relative_action_fingerprint,
        _dump_stats_cache_atomic,
        _load_stats_cache,
        calculate_stats_for_key,
    )
    from gr00t.data.utils import to_json_serializable

    keys = _relative_keys(args.embodiment_tag)
    print(f"relative action keys: {keys}")

    if args.verify_episodes:
        key = keys[0]
        print(f"verifying '{key}' on {args.verify_episodes} episodes")
        serial = calculate_stats_for_key(
            args.dataset_path, _tag(args.embodiment_tag), key,
            max_episodes=args.verify_episodes,
        )
        arr = _gather(args.dataset_path, args.embodiment_tag, key, args.modality_config,
                      args.jobs, args.verify_episodes)
        par = _stats(arr)
        ok = True
        for stat in serial:
            same = np.allclose(serial[stat], par[stat], rtol=0, atol=0)
            close = np.allclose(serial[stat], par[stat], rtol=1e-6, atol=1e-6)
            print(f"  {stat:5s} exact={same} close={close}")
            ok = ok and close
        print("VERIFY PASS" if ok else "VERIFY FAIL")
        sys.exit(0 if ok else 1)

    stats_path = Path(args.dataset_path) / LE_ROBOT_REL_STATS_FILENAME
    stats = _load_stats_cache(stats_path)
    fingerprints = stats.setdefault(STATS_FINGERPRINTS_KEY, {})

    for key in keys:
        expected = _compute_relative_action_fingerprint(_tag(args.embodiment_tag), key)
        if key in stats and fingerprints.get(key) == expected:
            print(f"{key}: cached, skipping")
            continue
        print(f"{key}: computing with {args.jobs} processes", flush=True)
        t0 = time.time()
        arr = _gather(args.dataset_path, args.embodiment_tag, key, args.modality_config,
                      args.jobs, None)
        stats[key] = _stats(arr)
        fingerprints[key] = expected
        del arr
        # Persist after every key so an interruption keeps finished work.
        _dump_stats_cache_atomic(stats_path, to_json_serializable(dict(stats)))
        print(f"{key}: done in {time.time() - t0:.0f}s -> {stats_path}", flush=True)

    print("all relative stats written")


if __name__ == "__main__":
    main()
