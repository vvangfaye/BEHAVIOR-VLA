"""Single-/multi-node PyTorch training and held-out calibration of the small head.

This trains GoalKeeper, not the multi-billion-parameter Comet backbone. Keep base
VLA fine-tuning in the official 2026 OpenPI pipeline to preserve its tested recipe.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from .data import load_manifest, load_record, collate
from .model import GoalTransitionModel, ModelConfig, supervision_loss
from .calibration import ReliabilityCalibrator
from .artifacts import load_checkpoint


class Records(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return load_record(self.rows[index])


def _device_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model, batch):
    return model(batch["visual"], batch["proprio"], batch["queries"], batch["actions"])


def fit_normalization(rows):
    stats = {}
    # Only training groups contribute. These are HEAD statistics; they never
    # replace the Comet backbone's supplied norm_stats.json.
    for field in ("proprio", "actions"):
        total = square = None
        count = 0
        for row in rows:
            value = load_record(row)[field]
            x = value.numpy().reshape(-1, value.shape[-1]).astype(np.float64)
            total = x.sum(0) if total is None else total + x.sum(0)
            square = (x * x).sum(0) if square is None else square + (x * x).sum(0)
            count += len(x)
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean * mean, 1e-6))
        stats[field] = {"mean": mean.tolist(), "std": std.tolist()}
    return stats


def normalize(batch, stats):
    out = dict(batch)
    for field in ("proprio", "actions"):
        x = batch[field]
        mean = x.new_tensor(stats[field]["mean"])
        std = x.new_tensor(stats[field]["std"])
        out[field] = (x - mean) / std
    return out


def train(manifest, output, *, epochs=20, batch_size=16, learning_rate=3e-4,
          hidden_dim=128, seed=7, device="auto", workers=0):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "auto": device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"): torch.cuda.set_device(torch.device(device))
    if distributed:
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = load_manifest(manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("Need independent train and dev initial-state groups")
    if epochs < 1 or batch_size < 1: raise ValueError("Positive epochs and batch_size required")
    example = load_record(train_rows[0])
    config = ModelConfig(visual_dim=example["visual"].shape[-1], text_dim=example["queries"].shape[-1],
                         cameras=example["visual"].shape[1], proprio_dim=example["proprio"].shape[-1],
                         action_dim=example["actions"].shape[-1], horizon=example["actions"].shape[1],
                         hidden_dim=hidden_dim)
    model = GoalTransitionModel(config).to(device)
    # Compute once and broadcast; do not scan the full dataset on all 32 ranks.
    stats_box = [fit_normalization(train_rows) if rank == 0 else None]
    if distributed: dist.broadcast_object_list(stats_box, src=0)
    stats = stats_box[0]
    sampler = DistributedSampler(Records(train_rows), shuffle=True, seed=seed) if distributed else None
    loader = DataLoader(Records(train_rows), batch_size=batch_size, sampler=sampler,
                        shuffle=sampler is None, collate_fn=collate, num_workers=workers)
    dev_loader = DataLoader(Records(dev_rows), batch_size=batch_size, collate_fn=collate, num_workers=workers)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.startswith("cuda") else None) if distributed else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=learning_rate, weight_decay=1e-4)
    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    best, history = math.inf, []
    started = time.time()
    for epoch in range(epochs):
        if sampler: sampler.set_epoch(epoch)
        wrapped.train(); train_sum = count = 0
        for raw in loader:
            batch = normalize(_device_batch(raw, device), stats)
            optimizer.zero_grad(set_to_none=True)
            losses = supervision_loss(_forward(wrapped, batch), batch)
            if not torch.isfinite(losses["loss"]): raise FloatingPointError("Nonfinite training loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.)
            optimizer.step()
            train_sum += float(losses["loss"].detach()) * len(batch["visual"])
            count += len(batch["visual"])
        aggregate = torch.tensor([train_sum, count], device=device)
        if distributed: dist.all_reduce(aggregate)
        if rank == 0:
            model.eval(); dev_sum = dev_count = 0
            with torch.inference_mode():
                for raw in dev_loader:
                    batch = normalize(_device_batch(raw, device), stats)
                    loss = supervision_loss(_forward(model, batch), batch)["loss"]
                    dev_sum += float(loss) * len(batch["visual"]); dev_count += len(batch["visual"])
            record = {"epoch": epoch + 1, "train_loss": float(aggregate[0] / aggregate[1]),
                      "dev_loss": dev_sum / dev_count, "elapsed_seconds": time.time() - started}
            history.append(record); print(json.dumps(record), flush=True)
            if record["dev_loss"] < best:
                best = record["dev_loss"]
                artifact = {"format": "goalkeeper-v1", "config": asdict(config),
                            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "normalization": stats, "feature_id": rows[0]["feature_id"],
                            "action_spec": rows[0]["action_spec"], "seed": seed,
                            "synthetic": any(r["source"] == "synthetic_test" for r in rows),
                            "train_groups": sorted({r["group_id"] for r in train_rows}),
                            "dev_groups": sorted({r["group_id"] for r in dev_rows}),
                            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}
                torch.save(artifact, outdir / "best.pt")
            (outdir / "training.json").write_text(json.dumps(history, indent=2))
        if distributed: dist.barrier()
    if distributed: dist.destroy_process_group()
    return history


def calibrate(manifest, checkpoint, output, *, device="cpu", min_count=30):
    rows = [r for r in load_manifest(manifest) if r["split"] == "calibration"]
    if not rows: raise ValueError("No independent calibration groups")
    model, artifact = load_checkpoint(checkpoint, device)
    used = set(artifact["train_groups"]) | set(artifact["dev_groups"])
    if used & {r["group_id"] for r in rows}: raise ValueError("Calibration group leakage")
    if any(r["feature_id"] != artifact["feature_id"] or r["action_spec"] != artifact["action_spec"] for r in rows):
        raise ValueError("Calibration feature/action specification mismatch")
    predictions = {k: [] for k in ["current", "future", "visibility"]}
    targets = {k: [] for k in predictions}
    groups = {k: [] for k in predictions}
    with torch.inference_mode():
        for row in rows:
            raw = collate([load_record(row)]); batch = normalize(_device_batch(raw, device), artifact["normalization"])
            out = _forward(model, batch)
            for name, logit, target, mask in [
                ("current", out["current_logits"][:, -1], batch["truth"][:, -1], batch["visibility"][:, -1] >= .5),
                ("future", out["future_logits"], batch["future_truth"], torch.ones_like(batch["future_truth"], dtype=torch.bool)),
                ("visibility", out["visibility_logits"][:, -1], batch["visibility"][:, -1], batch["visibility"][:, -1] >= 0),
            ]:
                predictions[name].extend(logit.sigmoid()[mask].cpu().tolist())
                targets[name].extend(target[mask].cpu().tolist())
                groups[name].extend([row["group_id"]] * int(mask.sum()))
    result = {"format": "goalkeeper-calibration-v1", "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "groups": sorted({r["group_id"] for r in rows}), "interval_semantics": "group_balanced_bin_reliability_not_state_conditional_guarantee"}
    for name in predictions:
        result[name] = ReliabilityCalibrator(min_count=min_count).fit(predictions[name], targets[name], groups[name]).to_dict()
        p, y = np.asarray(predictions[name]), np.asarray(targets[name])
        result[f"{name}_brier"] = float(np.mean((p - y) ** 2)) if len(y) else None
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
