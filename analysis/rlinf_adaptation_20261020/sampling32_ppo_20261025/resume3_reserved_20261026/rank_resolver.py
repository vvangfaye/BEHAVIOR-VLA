"""Resolve one process per single-GPU worker without initializing DDP."""
import re


def resolve_rank(environment):
    candidates = []
    for key in ('SENSECORE_PYTORCH_NODE_RANK', 'NODE_RANK', 'GROUP_RANK'):
        if key in environment:
            candidates.append(int(environment[key]))
    match = re.search(r'worker-(\d+)$', environment.get('HOSTNAME', ''))
    if match:
        candidates.append(int(match.group(1)))
    if not candidates:
        raise RuntimeError('Cannot identify the two-worker rank; stop before writing results')
    if len(set(candidates)) != 1 or candidates[0] not in (0, 1):
        raise RuntimeError('Inconsistent two-worker rank; stop before writing results')
    return candidates[0]
