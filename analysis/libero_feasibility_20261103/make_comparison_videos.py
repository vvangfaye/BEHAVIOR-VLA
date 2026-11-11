#!/usr/bin/env python3
"""Stack per-policy LIBERO eval videos into one comparison video per trial (2026-11-05).

Input: <out>/<arm>/seed_*/<n>.mp4 written by RLinf's RecordVideo (one video per env worker = one trial, tasks tiled left
to right) and results/video_<arm>/console.log ("[libero eval] task_id=T, trial_id=K, success=B" per episode).
Output: <out>/comparison_set<W>.mp4, one per env worker W (11/5 fix: the 6 episodes are dealt round-robin to 2
workers, so a worker's tiles are e.g. task1/trial0, task7/trial0, task5/trial1, read from the rank-tagged eval log):
a header with task/trial per column, one row per policy (same initial states), a success/fail bar under every tile.
Usage: make_comparison_videos.py <out_dir> <arm> [<arm> ...]
"""
import glob
import os
import re
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out, arms = sys.argv[1], sys.argv[2:]
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"sft": "SFT (pi0.5)", "ppo_seed1": "PPO seed 1", "ppo_seed3": "PPO seed 3", "subgoal_seed3": "PPO+subgoal seed 3"}
PAT = re.compile(r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)")
RANK = re.compile(r"EnvGroup\(rank=(\d+)\)")
try:
    FONT = ImageFont.load_default(size=18)
except TypeError:  # Pillow < 10.1
    FONT = ImageFont.load_default()


def outcomes(arm):
    """Per env worker (EnvGroup rank), the (task, trial, success) of its episodes in log order = its tile order."""
    res = {}
    path = os.path.join(HERE, "results", f"video_{arm}", "console.log")
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            m = PAT.search(line)
            r = RANK.search(line)
            if m and r:
                res.setdefault(int(r.group(1)), []).append((int(m.group(1)), int(m.group(2)), m.group(3) == "True"))
    return res


def videos(arm):
    """MP4s of one arm ordered by env-worker seed (seed_<w> = EnvGroup rank w)."""
    dirs = sorted(glob.glob(os.path.join(out, arm, "seed_*")), key=lambda d: int(d.rsplit("_", 1)[1]))
    return [sorted(glob.glob(os.path.join(d, "*.mp4")), key=lambda p: int(os.path.basename(p)[:-4]))[0]
            for d in dirs if glob.glob(os.path.join(d, "*.mp4"))]


def text_img(lines, h, w, bg=(24, 24, 24), fg=(255, 255, 255)):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    y = 8
    for i, t in enumerate(lines):
        d.text((10, y), t, fill=fg if i == 0 else (200, 200, 200), font=FONT)
        y += 24
    return np.asarray(img)


SHORT = {0: "soup + sauce -> basket", 1: "cheese + butter -> basket", 2: "stove on + moka pot", 3: "bowl -> drawer, close",
         4: "two mugs -> two plates", 5: "book -> back of caddy", 6: "mug -> plate, pudding right",
         7: "soup + cheese -> basket", 8: "both moka pots -> stove", 9: "mug -> microwave, close"}
LW, BAR = 230, 24
per_arm = {a: (videos(a), outcomes(a)) for a in arms}
for w in (0, 1):
    rows, oc_rows, names, fps, tiles = [], [], [], 10, None
    for a in arms:
        vids, oc = per_arm[a]
        if len(vids) <= w or w not in oc:
            print(f"{a}: no video/outcomes for worker {w}")
            continue
        t_k = [(t, k) for t, k, _ in oc[w]]
        if tiles is None:
            tiles = t_k
        elif t_k != tiles:
            print(f"WARNING {a}: worker {w} tiles {t_k} differ from {tiles}")
        r = imageio.get_reader(vids[w])
        fps = r.get_meta_data().get("fps", fps)
        rows.append([f for f in r])
        r.close()
        oc_rows.append([s for _, _, s in oc[w]])
        names.append(NAMES.get(a, a))
    if not rows:
        continue
    n = min(len(r) for r in rows)
    H, W = rows[0][0].shape[0], max(r[0].shape[1] for r in rows)
    tw = W // len(tiles)
    header = np.concatenate([text_img(["policy"], 56, LW)] +
                            [text_img([f"task {t} / trial {k}", SHORT.get(t, "")], 56, tw) for t, k in tiles], axis=1)
    header = np.pad(header, ((0, 0), (0, LW + W - header.shape[1]), (0, 0)))
    labels = [text_img([nm, f"{sum(o)}/{len(o)} success"], H, LW) for nm, o in zip(names, oc_rows)]
    bars = [[text_img(["success" if s else "fail"], BAR, tw, bg=(30, 120, 50) if s else (150, 40, 40)) for s in o]
            for o in oc_rows]
    path = os.path.join(out, f"comparison_set{w}.mp4")
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    for k in range(n):
        frame_rows = [header]
        for i, r in enumerate(rows):
            f = r[k].copy()
            if f.shape[1] < W:
                f = np.pad(f, ((0, 0), (0, W - f.shape[1]), (0, 0)))
            for j, b in enumerate(bars[i]):
                f[H - BAR:H, j * tw:j * tw + b.shape[1]] = b[:, : min(b.shape[1], W - j * tw)]
            frame_rows.append(np.concatenate([labels[i], f], axis=1))
        frame = np.concatenate(frame_rows, axis=0)
        h2, w2 = frame.shape[0] - frame.shape[0] % 2, frame.shape[1] - frame.shape[1] % 2
        writer.append_data(frame[:h2, :w2])
    writer.close()
    print(f"wrote {path}: {len(rows)} policies x tiles {tiles} (task, trial), {n} frames @ {fps} fps")
