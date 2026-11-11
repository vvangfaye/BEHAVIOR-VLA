"""Training-only, variable-length terminal-outcome density head.

Inputs are observation/history features and clause embeddings, not ground-truth
current or future simulator predicates. This is an implementation prototype,
not a claim that autoregressive density estimation is itself novel.
"""
import torch
from torch import nn
from torch.nn import functional as F


class OutcomeHead(nn.Module):
    def __init__(self, context_dim, clause_dim, width=64, layers=2, heads=4):
        super().__init__()
        self.context = nn.Linear(context_dim, width)
        self.clause = nn.Linear(clause_dim, width)
        # 0/1 outcome tokens and a start token. Shift prevents target leakage.
        self.previous = nn.Embedding(3, width)
        layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout=0., batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, 1)

    def forward(self, context, clauses, outcomes, valid):
        b, m = outcomes.shape
        if clauses.shape[:2] != (b, m) or valid.shape != (b, m):
            raise ValueError('incompatible clause shapes')
        if not torch.all(valid[:, 0]) or torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError('valid clauses must form a nonempty prefix')
        previous = torch.full_like(outcomes, 2, dtype=torch.long)
        previous[:, 1:] = outcomes[:, :-1].long()
        x = self.context(context)[:, None] + self.clause(clauses) + self.previous(previous)
        causal = torch.ones(m, m, device=x.device, dtype=torch.bool).triu(1)
        z = self.decoder(x, mask=causal, src_key_padding_mask=~valid)
        return self.output(z).squeeze(-1)

    def nll(self, context, clauses, outcomes, valid):
        logits = self(context, clauses, outcomes, valid)
        terms = F.binary_cross_entropy_with_logits(logits, outcomes.float(), reduction='none')
        # Per-episode averaging avoids overweighting tasks with more clauses.
        return ((terms * valid).sum(-1) / valid.sum(-1)).mean()

    def success_log_probability(self, context, clauses, valid):
        # All-one prefix is a query to the density, not a supplied true outcome.
        ones = torch.ones_like(valid, dtype=context.dtype)
        logits = self(context, clauses, ones, valid)
        return (F.logsigmoid(logits) * valid).sum(-1)


def verify():
    import itertools
    torch.set_num_threads(2)
    torch.manual_seed(29)
    model = OutcomeHead(5, 7, width=16, layers=1, heads=2).eval()
    x, clauses = torch.randn(1, 5), torch.randn(1, 3, 7)
    y = torch.tensor(list(itertools.product([0., 1.], repeat=3)))
    valid = torch.ones_like(y, dtype=torch.bool)
    logits = model(x.expand(8, -1), clauses.expand(8, -1, -1), y, valid)
    logp = -F.binary_cross_entropy_with_logits(logits, y, reduction='none').sum(-1)
    mass = logp.exp().sum()
    assert torch.allclose(mass, torch.tensor(1.), atol=1e-6)
    assert torch.allclose(model.success_log_probability(x, clauses, valid[:1]), logp[-1:], atol=1e-6)
    # First output cannot read any outcomes; second only reads first bit.
    assert (logits[:, 0] - logits[0, 0]).abs().max() < 1e-6
    assert torch.allclose(logits[0, :2], logits[1, :2], atol=1e-6)
    loss = model.nll(x.expand(8, -1), clauses.expand(8, -1, -1), y, valid)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # Future padded clauses do not affect valid predictions.
    short_valid = torch.tensor([[True, True, False]])
    a = model(x, clauses, y[:1], short_valid)[:, :2]
    changed = clauses.clone(); changed[:, 2] += 1000
    b = model(x, changed, y[:1], short_valid)[:, :2]
    assert torch.allclose(a, b, atol=1e-6)
    return {'enumerated_mass': float(mass.detach()), 'success_probability': float(logp[-1].exp().detach()),
            'finite_backward': True, 'causality': True, 'padding_invariance': True,
            'scope': 'head correctness only; no trained visual critic or RL benefit'}


if __name__ == '__main__':
    import json
    from pathlib import Path
    result = verify()
    Path(__file__).with_name('head_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
