"""Shared, object-query-conditioned observation and action-outcome model.

Inputs are visual tokens, public goal text embeddings, proprioception and proposed
actions. Oracle labels are accepted ONLY by the training loss, never by forward.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    visual_dim: int = 768
    text_dim: int = 768
    proprio_dim: int = 61
    action_dim: int = 23
    hidden_dim: int = 128
    heads: int = 4
    cameras: int = 3
    horizon: int = 16

    def __post_init__(self):
        if any(v <= 0 for v in asdict(self).values()) or self.hidden_dim % self.heads:
            raise ValueError("Positive dimensions and hidden_dim divisible by heads required")


class GoalTransitionModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.visual = nn.Linear(config.visual_dim, d)
        self.query = nn.Linear(config.text_dim, d)
        self.proprio = nn.Sequential(nn.Linear(config.proprio_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.camera = nn.Embedding(config.cameras, d)
        self.attention = nn.MultiheadAttention(d, config.heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.memory = nn.GRUCell(d, d)
        self.current_head = nn.Linear(d, 1)
        self.visibility_head = nn.Linear(d, 1)
        self.action_encoder = nn.GRU(config.action_dim, d, batch_first=True)
        self.future_head = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 1))

    def observe(self, visual, proprio, queries, hidden=None):
        """visual [B,T,C,P,Dv], proprio [B,T,S], queries [B,N,Dg]."""
        if visual.ndim != 5 or proprio.ndim != 3 or queries.ndim != 3:
            raise ValueError("Invalid observation tensor ranks")
        b, steps, cameras, patches, _ = visual.shape
        if cameras != self.config.cameras or proprio.shape[:2] != (b, steps) or queries.shape[0] != b:
            raise ValueError("Observation dimensions do not match model configuration")
        n, d = queries.shape[1], self.config.hidden_dim
        query = self.query(queries)
        hidden = visual.new_zeros((b, n, d)) if hidden is None else hidden
        if hidden.shape != (b, n, d):
            raise ValueError("Episode hidden state has wrong shape")
        current, visibility = [], []
        camera = self.camera.weight[None, :, None, :]
        for t in range(steps):
            keys = (self.visual(visual[:, t]) + camera).reshape(b, cameras * patches, d)
            q = query + self.proprio(proprio[:, t])[:, None]
            observed = self.attention(q, keys, keys, need_weights=False)[0]
            x = self.norm(observed + q)
            hidden = self.memory(x.reshape(b * n, d), hidden.reshape(b * n, d)).reshape(b, n, d)
            current.append(self.current_head(hidden).squeeze(-1))
            visibility.append(self.visibility_head(hidden).squeeze(-1))
        return {"current_logits": torch.stack(current, 1),
                "visibility_logits": torch.stack(visibility, 1), "hidden": hidden,
                "query_embedding": query}

    def predict_futures(self, hidden, queries, actions):
        """actions [B,K,H,A]; outcomes are at the END of the executed H steps."""
        if actions.ndim != 4 or actions.shape[2:] != (self.config.horizon, self.config.action_dim):
            raise ValueError("Candidate action prefix differs from trained horizon/action space")
        b, k, h, a = actions.shape
        n = hidden.shape[1]
        _, action_hidden = self.action_encoder(actions.reshape(b * k, h, a))
        action_hidden = action_hidden[-1].reshape(b, k, 1, -1).expand(-1, -1, n, -1)
        state = hidden[:, None].expand(-1, k, -1, -1)
        text = queries[:, None].expand(-1, k, -1, -1)
        return self.future_head(torch.cat((state, text, action_hidden), -1)).squeeze(-1)

    def forward(self, visual, proprio, queries, actions, hidden=None):
        out = self.observe(visual, proprio, queries, hidden)
        out["future_logits"] = self.predict_futures(out["hidden"], out["query_embedding"], actions)
        return out


def clause_progress(prob, clauses):
    """prob [B,K,N], clauses [B,C,N] with {-1,0,+1}; padded clauses excluded."""
    signs = clauses[:, None]
    present = signs != 0
    literal = torch.where(signs > 0, prob[:, :, None], 1 - prob[:, :, None])
    denom = present.sum(-1).clamp_min(1)
    score = (literal * present).sum(-1) / denom
    score = score.masked_fill(~present.any(-1), -1)
    return score.max(-1).values


def supervision_loss(out, labels, ranking_weight=0.2, preservation_penalty=0.5):
    """Offline labels may include truth. They are not features.

    Pairwise ranking is active only when *observed branch outcomes* are supplied
    for K>1 candidates from the SAME initial state. Never fabricate counterfactual
    labels by attaching one rollout's outcome to another action sequence.
    """
    atom_mask = labels["atom_mask"].bool()
    visibility = labels["visibility"]
    current = labels["truth"]
    future = labels["future_truth"]
    valid_current = atom_mask[:, None] & (visibility >= 0.5)
    valid_visibility = atom_mask[:, None] & (visibility >= 0)
    candidate_mask = labels["candidate_mask"].bool()
    valid_future = atom_mask[:, None] & candidate_mask[:, :, None]

    def bce(pred, target, mask):
        values = F.binary_cross_entropy_with_logits(pred, target.clamp(0, 1), reduction="none")
        return (values * mask).sum() / mask.sum().clamp_min(1)

    now_loss = bce(out["current_logits"], current, valid_current)
    visibility_loss = bce(out["visibility_logits"], visibility, valid_visibility)
    future_loss = bce(out["future_logits"], future, valid_future)
    predicted = out["future_logits"].sigmoid()
    clauses = labels["clauses"]
    # Select one target alternative using the current ground truth during training.
    current_literal = torch.where(clauses > 0, current[:, -1, None], 1 - current[:, -1, None])
    present = clauses != 0
    clause_scores = (current_literal * present).sum(-1) / present.sum(-1).clamp_min(1)
    clause_scores = clause_scores.masked_fill(~present.any(-1), -1)
    best = clause_scores.argmax(-1)
    selected = clauses[torch.arange(clauses.shape[0], device=clauses.device), best]
    protected = (selected != 0) & (torch.where(selected > 0, current[:, -1], 1 - current[:, -1]) > .5)
    true_lit = torch.where(selected[:, None] > 0, future, 1 - future)
    pred_lit = torch.where(selected[:, None] > 0, predicted, 1 - predicted)
    true_risk = ((1 - true_lit) * protected[:, None]).sum(-1).clamp_max(1)
    pred_risk = ((1 - pred_lit) * protected[:, None]).sum(-1).clamp_max(1)
    true_utility = clause_progress(future, clauses) - preservation_penalty * true_risk
    pred_utility = clause_progress(predicted, clauses) - preservation_penalty * pred_risk
    delta = true_utility[:, :, None] - true_utility[:, None, :]
    valid_pairs = (delta > .05) & candidate_mask[:, :, None] & candidate_mask[:, None, :]
    pair_loss = F.softplus(-(pred_utility[:, :, None] - pred_utility[:, None, :]) * 5)
    ranking_loss = (pair_loss * valid_pairs).sum() / valid_pairs.sum().clamp_min(1)
    total = now_loss + future_loss + .2 * visibility_loss + ranking_weight * ranking_loss
    return {"loss": total, "current": now_loss, "future": future_loss,
            "visibility": visibility_loss, "ranking": ranking_loss}
