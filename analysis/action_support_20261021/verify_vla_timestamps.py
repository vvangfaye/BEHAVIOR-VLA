"""Check every frozen image anchor against source parquet timestamps."""
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/training_boundary_audit_20261019'))
from prepare_probe_cache import ObjectReader, TransferBudget, open_source_bucket


def main():
    plan = json.loads((HERE / 'vla_pilot/data_plan.json').read_text())
    budget = TransferBudget(16 * 1024**2, 600)
    source = open_source_bucket(Path.home() / '.ossutilconfig', plan['source_bucket'])
    records = {r['relative_path']: r for r in plan['objects']}
    checked = []
    for relative in sorted({s['data_relative_path'] for s in plan['samples']}):
        samples = [s for s in plan['samples'] if s['data_relative_path'] == relative]
        episodes = sorted({s['episode_index'] for s in samples})
        with ObjectReader(source, records[relative], budget) as stream:
            data = pq.read_table(stream, columns=['episode_index', 'frame_index', 'timestamp'],
                                 filters=[('episode_index', 'in', episodes)],
                                 pre_buffer=False, use_threads=False).to_pydict()
        lookup = {(int(e), int(f)): float(t) for e, f, t in zip(data['episode_index'], data['frame_index'], data['timestamp'])}
        assert len(lookup) == len(data['timestamp'])
        errors = [abs(lookup[(s['episode_index'], s['frame_index'])] - s['frame_index'] / plan['fps']) for s in samples]
        assert max(errors) < plan['image_tolerance_seconds']
        checked.append({'relative_path': relative, 'anchors': len(samples), 'max_seconds_error': max(errors)})
    result = {'passed': True, 'checked_anchors': sum(x['anchors'] for x in checked),
              'files': checked, 'reserved_source_bytes': budget.used_bytes,
              'source_requests': budget.requests,
              'scope': 'Independent source timestamp-to-frame check; image decoder itself validates nearest PTS'}
    (HERE / 'vla_pilot/timestamp_alignment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
