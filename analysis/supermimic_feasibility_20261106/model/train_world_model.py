#!/usr/bin/env python3
"""Small offline CPU surrogate probe for b1k_action_conditioned_transition_v1.

This does not simulate the environment or prove counterfactual gradients. The
no_action control deliberately has identical observation inputs to the full
model. Split membership is supplied by the exporter and never randomized.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mean_scale(x):
    return x.mean(0).astype(np.float32), np.maximum(x.std(0), 1e-3).astype(np.float32)


def binary_metrics(y, score):
    """Threshold metrics, grouped-tie AP and rank AUROC; null if undefined."""
    y, score = np.asarray(y, bool).ravel(), np.asarray(score, float).ravel()
    if not len(y):
        return {"n": 0, "auroc": None, "average_precision": None}
    pred = score >= .5
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    order = np.argsort(-score, kind="stable")
    sorted_y, sorted_s = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_s)), len(y)-1]
    positives = int(y.sum())
    cumulative = np.cumsum(sorted_y)[ends]
    ap = float(np.sum(np.diff(np.r_[0, cumulative]) / max(positives, 1) * cumulative / (ends + 1))) if positives else None
    negatives = len(y) - positives
    auc = None
    if positives and negatives:
        ascending = np.argsort(score, kind="stable")
        ranks = np.empty(len(y), float)
        start = 0
        while start < len(y):
            end = start + 1
            while end < len(y) and score[ascending[end]] == score[ascending[start]]:
                end += 1
            ranks[ascending[start:end]] = (start + 1 + end) / 2
            start = end
        auc = float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    return {"n": len(y), "positives": positives, "prevalence": positives / len(y),
            "precision_at_0_5": precision, "recall_at_0_5": recall,
            "f1_at_0_5": 2*precision*recall/max(precision+recall, 1e-12),
            "accuracy_at_0_5": float((pred == y).mean()), "auroc": auc, "average_precision": ap}


def load_data(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    required = {"obs_state", "next_obs_state", "next_obs_valid", "execution_actions", "action_mask",
        "visual_features", "visual_valid", "next_visual_features", "next_visual_valid", "goals", "goal_valid",
        "next_goals", "next_goal_valid", "reward", "split", "episode_index", "chunk_index", "instance"}
    if required - d.keys():
        raise ValueError(f"Missing exporter fields: {required-d.keys()}")
    n = len(d["split"])
    if d["obs_state"].shape != (n, 23) or d["execution_actions"].shape != (n, 16, 23):
        raise ValueError("Expected N x 23 proprio and N x 16 x 23 executed command array")
    if not np.isfinite(d["obs_state"]).all() or not np.isfinite(d["reward"]).all():
        raise ValueError("Nonfinite current state/reward")
    if "task_reward" in d and (d["task_reward"].shape != (n,) or not np.isfinite(d["task_reward"]).all()):
        raise ValueError("Invalid task_reward shape or nonfinite task reward")
    if not np.isfinite(d["execution_actions"]).all():
        raise ValueError("Nonfinite action")
    if not all((d["split"] == k).any() for k in range(3)):
        raise ValueError("All train/validation/test splits must be populated")
    for group in np.unique(d["instance"]):
        if len(np.unique(d["split"][d["instance"] == group])) != 1:
            raise ValueError("An instance occurs in multiple splits")
    for episode in np.unique(d["episode_index"]):
        ids = np.flatnonzero(d["episode_index"] == episode)
        if len(np.unique(d["split"][ids])) != 1 or len(np.unique(d["chunk_index"][ids])) != len(ids):
            raise ValueError("Episode crosses split or repeats a chunk")
    for valid, field in [("next_obs_valid", "next_obs_state"), ("visual_valid", "visual_features"),
                          ("next_visual_valid", "next_visual_features")]:
        if not np.isfinite(d[field][d[valid]]).all():
            raise ValueError(f"Valid {field} contains NaN")
    metadata_path=path.with_suffix(".json")
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    d["_goal_names"]=metadata.get("goal_names",[f"goal_{i}" for i in range(d["goals"].shape[1])])
    if len(d["_goal_names"]) != d["goals"].shape[1]:
        raise ValueError("Metadata goal_names disagree with goal columns")
    d["_full_chunk"]=d["action_mask"].all(axis=1)
    if "executed_steps" in d and not np.array_equal(d["_full_chunk"],d["executed_steps"]==16):
        raise ValueError("action_mask differs from executed_steps")
    if not all(((d["split"]==k)&d["_full_chunk"]).any() for k in range(3)):
        raise ValueError("All splits need full 16-step chunks")
    return d


class Transform:
    def __init__(self, d, visual_dim=32):
        train = (d["split"] == 0) & d["action_mask"].all(axis=1)
        self.state_mean, self.state_scale = mean_scale(d["obs_state"][train])
        actions = d["execution_actions"] * d["action_mask"][..., None]
        self.action_mean, self.action_scale = mean_scale(actions[train].reshape(train.sum(), -1))
        self.reward_mean, self.reward_scale = mean_scale(d["reward"][train, None])
        v = d["visual_features"][train & d["visual_valid"]]
        if len(v) < 2 or visual_dim == 0:
            self.visual_mean = np.zeros(d["visual_features"].shape[1], np.float32)
            self.components = np.zeros((len(self.visual_mean), 0), np.float32)
            self.visual_scale = np.empty(0, np.float32)
        else:
            self.visual_mean = v.mean(0).astype(np.float32)
            centered = torch.from_numpy(v - self.visual_mean)
            # Only training observations enter this eigendecomposition.
            covariance = centered.T @ centered / max(len(v) - 1, 1)
            _, vectors = torch.linalg.eigh(covariance)
            dim = min(visual_dim, len(v)-1, v.shape[1])
            self.components = vectors[:, -dim:].flip(1).numpy().copy()
            projected = (v - self.visual_mean) @ self.components
            self.visual_scale = np.maximum(projected.std(0), 1e-3).astype(np.float32)

    @property
    def visual_dim(self):
        return self.components.shape[1]

    def project(self, x, valid):
        filled = np.where(valid[:, None], x, self.visual_mean)
        return ((filled - self.visual_mean) @ self.components / self.visual_scale).astype(np.float32)

    def decode(self, x):
        return (x * self.visual_scale) @ self.components.T + self.visual_mean

    def save(self):
        return {k: torch.from_numpy(v.copy()) for k, v in vars(self).items()}


def tensors(d, transform):
    t = transform
    next_state = np.where(d["next_obs_valid"][:, None], d["next_obs_state"], d["obs_state"])
    actions = ((d["execution_actions"] * d["action_mask"][..., None]).reshape(len(next_state), -1) - t.action_mean) / t.action_scale
    actions *= np.repeat(d["action_mask"], 23, axis=1)
    visual = t.project(d["visual_features"], d["visual_valid"])
    next_visual = t.project(d["next_visual_features"], d["next_visual_valid"])
    raw = {"state": (d["obs_state"]-t.state_mean)/t.state_scale,
        "state_delta": (next_state-d["obs_state"])/t.state_scale,
        "visual": visual, "visual_delta": next_visual-visual,
        "visual_valid": d["visual_valid"].astype(np.float32)[:, None],
        "actions": actions, "action_mask": d["action_mask"].astype(np.float32),
        "goals": d["goals"], "goal_valid": d["goal_valid"].astype(np.float32),
        "next_goals": d["next_goals"], "next_goal_valid": d["next_goal_valid"].astype(np.float32),
        "state_valid": d["next_obs_valid"].astype(np.float32),
        "next_visual_valid": (d["next_visual_valid"] & d["visual_valid"]).astype(np.float32),
        "reward": ((d["reward"][:, None]-t.reward_mean)/t.reward_scale)}
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in raw.items()}


class WorldModel(nn.Module):
    def __init__(self, goals, visual_dim, use_action=True, use_visual=True, hidden=192):
        super().__init__()
        self.goals, self.visual_dim = goals, visual_dim
        self.use_action, self.use_visual = use_action, use_visual
        # Full and no_action retain identical parameter shapes/initial weights.
        inputs = 23 + goals*2 + (visual_dim+1 if use_visual else 0) + 16*23+16
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
            nn.SiLU(), nn.Linear(hidden, 23+visual_dim+goals+1))

    def forward(self, batch):
        x = [batch["state"], batch["goals"], batch["goal_valid"]]
        if self.use_visual:
            x += [batch["visual"], batch["visual_valid"]]
        x += [batch["actions"] if self.use_action else torch.zeros_like(batch["actions"]),
              batch["action_mask"] if self.use_action else torch.zeros_like(batch["action_mask"])]
        out = self.net(torch.cat(x, -1))
        state, visual, goals, reward = torch.split(out, [23, self.visual_dim, self.goals, 1], -1)
        # The goal prediction starts around the persistent state, but can change.
        base_logit = torch.logit(batch["goals"].clamp(.05, .95))
        return {"state_delta": state, "visual_delta": visual,
                "goal_logits": goals + base_logit, "reward": reward}


def loss_terms(model, batch):
    pred = model(batch)
    def masked_mse(delta, target, mask):
        if delta.shape[1] == 0:
            return delta.sum()*0
        per_row = (delta-target).square().mean(-1)
        return (per_row*mask).sum()/mask.sum().clamp_min(1)
    state = masked_mse(pred["state_delta"], batch["state_delta"], batch["state_valid"])
    visual = masked_mse(pred["visual_delta"], batch["visual_delta"], batch["next_visual_valid"])
    # Rare event weighting is identical for the action and no-action controls.
    event = (batch["next_goals"] != batch["goals"]).float()
    weights = batch["next_goal_valid"] * (1+4*event)
    goal = (F.binary_cross_entropy_with_logits(pred["goal_logits"], batch["next_goals"], reduction="none")*weights).sum()/weights.sum().clamp_min(1)
    reward = F.mse_loss(pred["reward"], batch["reward"])
    return {"state": state, "visual": visual, "goal": goal, "reward": reward}


def subset(table, ids):
    return {k: v[ids] for k, v in table.items()}


def fit(d, table, name, seed, args):
    torch.manual_seed(seed)
    model = WorldModel(d["goals"].shape[1], table["visual"].shape[1], name != "no_action", name != "no_visual", args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    full=d["action_mask"].all(axis=1)
    train, validation = np.flatnonzero((d["split"] == 0)&full), np.flatnonzero((d["split"] == 1)&full)
    rng=np.random.default_rng(seed)
    best, best_state, history = float("inf"), None, []
    began = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        train_sum, seen = 0., 0
        # Only shuffle within the already isolated training instances.
        order=rng.permutation(train)
        for start in range(0, len(train), args.batch_size):
            ids = order[start:start+args.batch_size]
            terms = loss_terms(model, subset(table, ids))
            loss = sum(terms.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            train_sum += float(loss.detach())*len(ids)
            seen += len(ids)
        model.eval()
        with torch.no_grad():
            val = sum(float(v) for v in loss_terms(model, subset(table, validation)).values())
        history.append({"epoch": epoch+1, "train_loss": train_sum/seen, "validation_loss": val})
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"seed": seed, "seconds": time.perf_counter()-began,
        "selected_epoch": min(history, key=lambda x:x["validation_loss"])["epoch"], "history": history}


def predict(models, table, batch_size=512):
    if not models:
        return {"state": table["state"].numpy(), "visual": table["visual"].numpy(),
            "goals": table["goals"].numpy(), "reward": np.zeros((len(table["state"]), 1), np.float32)}
    result = {k: [] for k in ["state", "visual", "goals", "reward"]}
    for start in range(0, len(table["state"]), batch_size):
        part = {k: v[start:start+batch_size] for k,v in table.items()}
        outputs = []
        for model in models:
            with torch.no_grad():
                p = model(part)
                outputs.append({"state": part["state"]+p["state_delta"], "visual": part["visual"]+p["visual_delta"],
                    "goals": torch.sigmoid(p["goal_logits"]), "reward": p["reward"]})
        for key in result:
            result[key].append(torch.stack([o[key] for o in outputs]).mean(0).numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def errors(pred, d, ids, t, start_ids=None):
    actual_start = ids if start_ids is None else start_ids
    state_valid = d["next_obs_valid"][ids]
    state_pred = pred["state"]*t.state_scale + t.state_mean
    delta = state_pred[state_valid]-d["next_obs_state"][ids][state_valid]
    normalized = delta/t.state_scale
    vv = d["next_visual_valid"][ids] & d["visual_valid"][actual_start]
    vp = t.decode(pred["visual"])
    if pred.get("raw_visual_persistence", False):
        # Give persistence the stronger, literal raw-image-descriptor baseline;
        # the latent metric below still uses the common train-only projection.
        vp = np.where(d["visual_valid"][actual_start,None],
                      d["visual_features"][actual_start], t.visual_mean)
    visual_delta = vp[vv]-d["next_visual_features"][ids][vv]
    true_visual = t.project(d["next_visual_features"][ids], d["next_visual_valid"][ids])
    visual_latent = pred["visual"][vv]-true_visual[vv]
    gv = d["next_goal_valid"][ids] & d["goal_valid"][actual_start]
    goals, current = d["next_goals"][ids], d["goals"][actual_start]
    event = goals != current
    event_score = np.where(current > .5, 1-pred["goals"], pred["goals"])
    goal_errors = ((pred["goals"] >= .5) != (goals >= .5))
    reward = pred["reward"].ravel()*float(t.reward_scale[0])+float(t.reward_mean[0])
    # Persistence's zero reward is expressed in raw units, independent of centering.
    if pred.get("raw_zero_reward", False):
        reward = np.zeros(len(ids))
    terminal = d.get("done",np.zeros(len(d["split"]),bool))[ids].astype(bool)
    nonzero=np.abs(d["reward"][ids])>1e-8
    task_events=np.abs(d["task_reward"][ids])>1e-8 if "task_reward" in d else None
    def rmse(x):
        return float(np.sqrt(np.mean(x*x))) if x.size else None
    per_goal={}
    for g,name in enumerate(d.get("_goal_names",[f"goal_{i}" for i in range(goals.shape[1])])):
        valid=gv[:,g]
        changed=event[:,g]&valid
        per_goal[name]={"next_bit":binary_metrics(goals[valid,g],pred["goals"][valid,g]),
            "change_event":binary_metrics(event[valid,g],event_score[valid,g]),
            "changed_entries":int(changed.sum()),
            "changed_next_bit_accuracy":float(1-goal_errors[changed,g].mean()) if changed.any() else None}
    return {"rows": len(ids), "state_rows": int(state_valid.sum()),
        "state_rmse": rmse(delta), "state_normalized_rmse": rmse(normalized),
        "visual_rows": int(vv.sum()), "visual_raw_576_rmse": rmse(visual_delta),
        "visual_projected_normalized_rmse": rmse(visual_latent),
        "reward_mse": float(np.mean((reward-d["reward"][ids])**2)),
        "reward_nonzero_rows":int(nonzero.sum()),
        "reward_nonzero_mse":float(np.mean((reward[nonzero]-d["reward"][ids][nonzero])**2)) if nonzero.any() else None,
        "reward_zero_mse":float(np.mean(reward[~nonzero]**2)) if (~nonzero).any() else None,
        "task_reward_events_available":task_events is not None,
        "task_reward_nonzero_rows":int(task_events.sum()) if task_events is not None else None,
        "total_reward_mse_on_task_reward_events":float(np.mean((reward[task_events]-d["reward"][ids][task_events])**2)) if task_events is not None and task_events.any() else None,
        "task_reward_event_metric_note":"Predicts total reward including shaping; event subset selected only by recorded nonzero task_reward. Missing task_reward is unavailable, never inferred from total reward.",
        "terminal_reward_rows":int(terminal.sum()),
        "terminal_reward_mse":float(np.mean((reward[terminal]-d["reward"][ids][terminal])**2)) if terminal.any() else None,
        "goal_accuracy": float(1-goal_errors[gv].mean()) if gv.any() else None,
        "changed_goal_entries": int((event & gv).sum()),
        "changed_goal_next_bit_accuracy": float(1-goal_errors[event & gv].mean()) if (event & gv).any() else None,
        "goal_change_event": binary_metrics(event[gv], event_score[gv]), "per_goal":per_goal}


def windows(d, split, horizon):
    lookup = {(int(ep),int(chunk)):i for i,(ep,chunk) in enumerate(zip(d["episode_index"],d["chunk_index"]))}
    result = []
    for i in np.flatnonzero(d["split"] == split):
        row = [lookup.get((int(d["episode_index"][i]), int(d["chunk_index"][i])+h)) for h in range(horizon)]
        if all(j is not None and d["split"][j] == split and d["next_obs_valid"][j]
               and d["action_mask"][j].all() for j in row):
            result.append(row)
    return np.asarray(result, dtype=int).reshape(-1,horizon)


def window_scope(d,split,horizon):
    lookup={(int(ep),int(chunk)):i for i,(ep,chunk) in enumerate(zip(d["episode_index"],d["chunk_index"]))}
    counts={"candidate_starts":0,"eligible":0,"missing_future_chunk":0,"missing_next_observation":0,"partial_chunk":0}
    for i in np.flatnonzero(d["split"]==split):
        counts["candidate_starts"]+=1
        rows=[lookup.get((int(d["episode_index"][i]),int(d["chunk_index"][i])+h)) for h in range(horizon)]
        if any(j is None or d["split"][j]!=split for j in rows):
            counts["missing_future_chunk"]+=1
        elif any(not d["action_mask"][j].all() for j in rows):
            counts["partial_chunk"]+=1
        elif any(not d["next_obs_valid"][j] for j in rows):
            counts["missing_next_observation"]+=1
        else:
            counts["eligible"]+=1
    counts["note"]="Mutually exclusive exclusion counts in the order above; open-loop drift excludes missing terminal next observations. Full terminal reward is evaluated in one_step only."
    return counts


def evaluate(models, d, table, t, persistence_reward="zero"):
    result = {}
    for split,name in enumerate(["train","validation","test"]):
        ids = np.flatnonzero((d["split"] == split)&d["action_mask"].all(axis=1))
        one = predict(models, subset(table, ids))
        if not models:
            one["raw_zero_reward"] = persistence_reward=="zero"
            one["raw_visual_persistence"] = True
        report = {"one_step": errors(one,d,ids,t), "factual_action_open_loop": {}}
        for horizon in (1,2,4):
            seq = windows(d,split,horizon)
            if not len(seq):
                report["factual_action_open_loop"][str(horizon)] = {"rows":0,"scope":window_scope(d,split,horizon)}
                continue
            batch = subset(table, seq[:,0])
            for h in range(horizon):
                # Replay observed future actions; predicted states/visual/goals
                # remain open-loop. Never cross an episode or split boundary.
                batch["actions"] = table["actions"][seq[:,h]]
                batch["action_mask"] = table["action_mask"][seq[:,h]]
                pred = predict(models,batch)
                batch = {**batch, "state":torch.from_numpy(pred["state"]),
                    "visual":torch.from_numpy(pred["visual"]), "goals":torch.from_numpy(pred["goals"])}
            if not models:
                pred["raw_zero_reward"] = persistence_reward=="zero"
                pred["raw_visual_persistence"] = True
            report["factual_action_open_loop"][str(horizon)] = {**errors(pred,d,seq[:,-1],t,seq[:,0]),
                "scope":window_scope(d,split,horizon)}
        result[name] = report
    return result


def diagnostics(models, table, d):
    ids = np.flatnonzero((d["split"] == 2)&d["action_mask"].all(axis=1))[:256]
    batch = subset(table,ids)
    for model in models:
        model.eval()
    for _ in range(3):
        predict(models,batch)
    start = time.perf_counter()
    for _ in range(20):
        predict(models,batch)
    elapsed = time.perf_counter()-start
    result = {"device":"cpu", "cpu_threads":torch.get_num_threads(),
        "batch_rows":len(ids), "ensemble_members":len(models),
        "predicted_transitions_per_second":len(ids)*20/elapsed,
        "note":"CPU batched surrogate inference only; excludes VLA, rendering and physics and is not a GPU speedup measurement."}
    gradients=[]
    for model in models:
        action=batch["actions"].detach().clone().requires_grad_()
        pred=model({**batch,"actions":action})
        grad=torch.autograd.grad(pred["reward"].sum(),action,allow_unused=True)[0]
        if grad is not None:
            gradients.append(grad.detach().numpy())
    result["predicted_reward_action_gradient"] = {"members":len(gradients),
        "finite":bool(all(np.isfinite(g).all() for g in gradients)),
        "mean_absolute_normalized_action_derivative":float(np.mean(np.abs(gradients))) if gradients else None,
        "ensemble_mean_coordinate_std":float(np.std(gradients,axis=0).mean()) if len(gradients)>1 else None,
        "note":"Gradient of this fitted reward head in normalized-action coordinates only; not a simulator derivative, Q gradient, policy gradient or counterfactual validation."}
    return result


def run(args):
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    d = load_data(args.data)
    transform = Transform(d,args.visual_dim)
    table = tensors(d,transform)
    args.output.mkdir(parents=True,exist_ok=True)
    report={"schema":"b1k_small_world_model_probe_v1", "device":"cpu", "threads":args.threads,
        "data_sha256":hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows":len(d["split"]), "splits":{name:int((d["split"]==i).sum()) for i,name in enumerate(["train","validation","test"])},
        "main_experiment_rows":{name:int(((d["split"]==i)&d["_full_chunk"]).sum()) for i,name in enumerate(["train","validation","test"])},
        "partial_chunks_excluded":{name:int(((d["split"]==i)&~d["_full_chunk"]).sum()) for i,name in enumerate(["train","validation","test"])},
        "goal_names":d["_goal_names"],
        "action_mask_policy":"Main train/validation/test and preprocessing use full 16-step chunks only; partial lengths are post-action outcome information and are excluded. Full model receives only a constant all-one mask.",
        "visual_projection_dim":transform.visual_dim,
        "preprocessing_fit_split":"full-chunk train only; no next-observation or validation/test PCA fit",
        "model_selection":"lowest validation combined loss; test never used for selection",
        "goal_loss":"masked BCE weighted 5x on changed goal entries, identical for full/no_action",
        "goal_score_interpretation":"Weighted BCE sigmoid scores are uncalibrated event/goal scores, not real task-success probabilities.",
        "action_control":"Full and no_action share architecture, parameter count and same-seed initialization; no_action zeroes normalized action and mask inputs.",
        "models":{}, "limitations":["Offline observational action-conditioned prediction is not causal or counterfactual identification.",
            "Open-loop evaluation replays factual logged actions, without resetting to future states; does not test unobserved actions.",
            "23D proprio plus pooled RGB is partially observed, not full physical simulator state.",
            "A negative result is limited to this data, representation, model capacity and training budget.",
            "No new environment interaction, policy optimization or true success-rate measurement is performed."]}
    report["reward_distribution"]={}
    report["task_reward_distribution"]={"available":"task_reward" in d,
        "note":"Recorded task_reward only; total reward can be nonzero from shaping without any task event."}
    for split,name in enumerate(["train","validation","test"]):
        rewards=d["reward"][(d["split"]==split)&d["_full_chunk"]]
        report["reward_distribution"][name]={"rows":len(rewards),
            "zero_tolerance":1e-8,"zero_count":int((np.abs(rewards)<=1e-8).sum()),
            "zero_fraction":float((np.abs(rewards)<=1e-8).mean()),
            "positive_count":int((rewards>1e-8).sum()),"negative_count":int((rewards < -1e-8).sum()),
            "mean":float(rewards.mean()),"std":float(rewards.std()),
            "min":float(rewards.min()),"max":float(rewards.max())}
        if "task_reward" in d:
            task=d["task_reward"][(d["split"]==split)&d["_full_chunk"]]
            report["task_reward_distribution"][name]={"rows":len(task),"zero_tolerance":1e-8,
                "zero_count":int((np.abs(task)<=1e-8).sum()),
                "positive_count":int((task>1e-8).sum()),"negative_count":int((task < -1e-8).sum()),
                "zero_fraction":float((np.abs(task)<=1e-8).mean()),
                "positive_fraction":float((task>1e-8).mean()),"negative_fraction":float((task < -1e-8).mean()),
                "mean":float(task.mean()),"std":float(task.std())}
    report["models"]["persistence"]={"evaluation":evaluate([],d,table,transform),
        "description":"Copies current proprio, raw current visual descriptor and goal bits; predicts raw reward zero. Projected visual error uses the same train-only PCA as learned models."}
    report["models"]["train_mean_reward"]={"evaluation":evaluate([],d,table,transform,persistence_reward="train_mean"),
        "description":"Same persistence state/visual/goals, predicts the full-chunk training split mean reward for every row; never uses validation/test reward statistics.",
        "constant_reward":float(transform.reward_mean[0])}
    for name in ["full","no_action"] + (["no_visual"] if args.include_no_visual else []):
        models, histories=[],[]
        for member in range(args.ensemble):
            model, history=fit(d,table,name,args.seed+member,args)
            models.append(model); histories.append(history)
        report["models"][name]={"training":histories,"evaluation":evaluate(models,d,table,transform),
            "diagnostics":diagnostics(models,table,d)}
        checkpoint={"schema":report["schema"],"name":name,"model_states":[m.state_dict() for m in models],
            "transform":transform.save(),"config":{"goals":d["goals"].shape[1],"visual_dim":transform.visual_dim,
                "use_action":name!="no_action","use_visual":name!="no_visual","hidden":args.hidden},
            "data_sha256":report["data_sha256"]}
        torch.save(checkpoint,args.output/f"{name}.pt")
        print(json.dumps({"model":name,"test":report["models"][name]["evaluation"]["test"]["one_step"],
                          "training_seconds":sum(h["seconds"] for h in histories)}),flush=True)
    (args.output/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--hidden",type=int,default=192)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--visual-dim",type=int,default=32)
    p.add_argument("--ensemble",type=int,choices=[1,2,3],default=1)
    p.add_argument("--include-no-visual",action="store_true")
    return p


if __name__=="__main__":
    arguments=parser().parse_args()
    if arguments.epochs<1 or arguments.threads<1:
        raise SystemExit("epochs and threads must be positive")
    run(arguments)
