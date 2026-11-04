#!/usr/bin/env python3
"""Write a copy of an RLinf full_weights.pt without the value head (the eval-mode policy has none, and RLinf's eval
loads the checkpoint with a strict load_state_dict). Usage: strip_value_head.py <in.pt> <out.pt>"""
import sys

import torch

sd = torch.load(sys.argv[1], map_location="cpu")
drop = [k for k in sd if k.startswith("value_head.")]
for k in drop:
    del sd[k]
torch.save(sd, sys.argv[2])
print(f"dropped {len(drop)} value_head tensors, kept {len(sd)}")
