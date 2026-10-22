"""Functional low-rank action-expert adaptation, with frozen base weights."""
import torch
from torch import nn
from torch.nn import functional as F


class LowRankLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device,
                                          dtype=torch.float32) / base.in_features ** .5)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                         dtype=torch.float32))
        self.scale = 1 / rank

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))


def install_action_lora(network, rank=4):
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    installed = []
    for layer_id, layer in enumerate(network.llm.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            collection = getattr(layer.attn, name)
            if collection[1] is not None:
                collection[1] = LowRankLinear(collection[1], rank)
                installed.append(f"llm.layers.{layer_id}.attn.{name}.1")
    if not installed:
        raise ValueError("No action-expert attention projections found")
    return installed
