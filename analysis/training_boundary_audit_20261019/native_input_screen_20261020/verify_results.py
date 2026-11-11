"""Independently bind exported numeric actions to hashes and comparison values."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_actions(name):
    doc = json.loads((HERE / name).read_text())
    result = {}
    for row in doc['samples']:
        value = row['actions']
        a = np.asarray(value['values'], dtype=value['dtype'])
        assert list(a.shape) == value['shape'] == [32, 23]
        assert np.isfinite(a).all()
        digest = hashlib.sha256()
        digest.update(f'array:{a.dtype.str}:{a.shape}\0'.encode())
        digest.update(np.ascontiguousarray(a).tobytes())
        assert digest.hexdigest() == row['actions_sha256']
        result[(row['id'], row['frame'], row['mode'])] = a
    assert len(result) == 12
    return result


def main():
    base = read_actions('baseline1688977.json')
    candidate = read_actions('expert1000.json')
    comparison = json.loads((HERE / 'comparison.json').read_text())
    assert base.keys() == candidate.keys()
    errors = []
    for row in comparison['rows']:
        for mode, metric in row['cross_checkpoint'].items():
            key = (row['id'], row['frame'], mode)
            delta = candidate[key] - base[key]
            for n, field in ((16, 'prefix16_rms_per_coordinate'),
                             (32, 'horizon32_rms_per_coordinate')):
                expected = np.linalg.norm(delta[:n], axis=0) / np.sqrt(n)
                errors.append(float(np.max(np.abs(expected - metric[field]))))
            errors.append(abs(float(np.max(np.abs(delta))) - metric['max_absolute_difference']))
    assert max(errors) < 1e-12
    result = {'status': 'passed', 'numeric_action_hashes_checked': 24,
              'max_comparison_discrepancy': max(errors),
              'scope': 'Numeric content and cross-checkpoint arithmetic only; no action correctness or physical outcome inference.'}
    with (HERE / 'numeric_verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
