"""Serve the original or fine-tuned VLA in the native B1K evaluation loop."""
import argparse
from pathlib import Path
import sys

import torch

from vla_runtime import load_base
from vla_support_adapter import SupportActionAdapter
from vla_native_policy import TorchRawPolicy, SeededNativeWrapper, acknowledged_server_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--trained-weights', type=Path)
    p.add_argument('--method', choices=['standard', 'conditioned', 'projected'], required=True)
    p.add_argument('--native-code', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--server-token', required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if not 1000 <= args.port < 65536:
        raise ValueError('Use a nonprivileged server port')
    torch.set_num_threads(4)
    base = load_base(args.base_checkpoint, args.tokenizer, device='cuda', precision='bf16')
    adapter = SupportActionAdapter(base).cuda().eval()
    if args.trained_weights:
        saved = torch.load(args.trained_weights, map_location='cpu', weights_only=False)
        if not saved['complete'] or saved['method'] != args.method:
            raise ValueError('Weights are incomplete or belong to a different method')
        expected = {n for n, p in adapter.named_parameters() if p.requires_grad}
        if set(saved['weights']) != expected:
            raise ValueError('Trainable state schema differs')
        result = adapter.load_state_dict(saved['weights'], strict=False)
        assert not result.unexpected_keys and not (set(result.missing_keys) & expected)
        del saved
    elif args.method != 'standard':
        raise ValueError('Untrained evaluation is permitted only for the original standard baseline')
    sys.path.insert(0, str(args.native_code / 'openpi/src'))
    sys.path.insert(0, str(args.native_code / 'openpi/packages/openpi-client/src'))
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.serving import websocket_b1k_server
    from openpi.configs.tasks import TASK_REGISTRY
    task_bucket, task = args.task.split('/', 1)
    raw = TorchRawPolicy(adapter, args.method, seed=args.seed)
    wrapped = B1KPolicyWrapper(raw, 'b1k/R1Pro', TASK_REGISTRY[task_bucket][task],
                               'receding_horizon', action_horizon=16, max_len=32)
    wrapped = SeededNativeWrapper(wrapped, raw)
    server = acknowledged_server_class(websocket_b1k_server)(
        wrapped, host='127.0.0.1', port=args.port, server_token=args.server_token,
        metadata={**raw.metadata, 'experimental_method': args.method,
                  'trained': args.trained_weights is not None})
    server.serve_forever()


if __name__ == '__main__':
    main()
