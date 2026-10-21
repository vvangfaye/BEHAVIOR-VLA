"""Read three fixed-time observations from the already selected 90 archives."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
READER = ROOT / 'analysis/training_boundary_audit_20261019/failure_receipts_20261020/sparse_native_frames.py'
FRAMES = [0, 1500, 3000]


def digest(b):
    return hashlib.sha256(b).hexdigest()


def main():
    source = json.loads((HERE/'read_receipt.json').read_text())
    plan = {'source_sha256': digest((HERE/'read_receipt.json').read_bytes()),
            'reader_sha256': digest(READER.read_bytes()), 'frames': FRAMES,
            'observation_time': 'fixed control steps, not relative to terminal time',
            'bucket': '260010165', 'max_archive_bytes_read': 16*1024**2,
            'max_total_bytes_read': 512*1024**2,
            'files': source['files'], 'jobs': 0, 'uploads': 0}
    pp = HERE/'frame_read_plan.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == plan
    else:
        pp.write_text(json.dumps(plan, indent=2)+'\n')
    spec = importlib.util.spec_from_file_location('bounded_native_reader', READER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    budget = module.RangeBudget(per_archive=plan['max_archive_bytes_read'], total=plan['max_total_bytes_read'])
    client = module.make_client()
    dest = HERE/'frames'; dest.mkdir(exist_ok=True)
    results = []
    for f in source['files']:
        rp = HERE/f['file']; raw = rp.read_bytes()
        assert digest(raw) == f['sha256']
        r = json.loads(raw)
        assert min(FRAMES) >= 0 and max(FRAMES) < r['steps']
        assert r['archive']['object_key'].startswith(module.PREFIX)
        target = dest/(rp.stem+'.npz'); provenance = target.with_suffix('.provenance.json')
        if target.exists() and provenance.exists():
            saved = json.loads(provenance.read_text())
            assert saved['npz_sha256'] == digest(target.read_bytes())
            assert saved['frames'] == FRAMES and saved['receipt_sha256'] == f['sha256']
            results.append(saved)
            continue
        arrays, meta = module.read_episode(client, plan['bucket'], r, FRAMES, budget=budget)
        np.savez_compressed(target, **arrays)
        meta.update({'npz': str(target.relative_to(HERE)), 'npz_sha256': digest(target.read_bytes()),
                     'receipt_sha256': f['sha256'], 'task': r['task'], 'instance': r['instance_id']})
        provenance.write_text(json.dumps(meta, indent=2)+'\n')
        results.append(meta)
        print(json.dumps({'completed': len(results), 'total': len(source['files']),
                          'downloaded_this_run': budget.claimed}), flush=True)
    out = {'plan_sha256': digest(pp.read_bytes()), 'episodes': results,
           'reserved_bytes_this_run': budget.claimed, 'whole_archives_hash_verified': False}
    (HERE/'frame_read_receipt.json').write_text(json.dumps(out, indent=2)+'\n')
    client.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error_type': type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
