"""Verify the scope and internal consistency of a finished native check."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
path = HERE/'native_cpu_01/result.json'
result = json.loads(path.read_text())
assert result['complete'] and result['stage'] == 'complete'
assert not any(result[k] for k in ('training', 'simulation', 'cloud', 'calibration'))
assert result['euler_steps'] == 10 and result['executed_shape'] == [16, 23]
assert result['maximum_sampler_difference'] == 0
assert result['intended_trainable_parameters'] == 430098464
assert 0 <= result['right_inverse_error'] < 1e-5
assert 0 <= result['null_projection_error'] < 1e-5
assert result['control_nonzero_gradient_elements'] > 0
for key in ('base_gradient_norm', 'control_gradient_norm', 'control_gradient_seconds',
            'base_gradient_seconds', 'elapsed_seconds', 'peak_rss_gib'):
    assert math.isfinite(result[key]) and result[key] > 0, key
assert math.isfinite(result['gradient_cosine']) and abs(result['gradient_cosine']) <= 1+1e-8
for name, expected in result['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
for key, index in [('base_gradient_norm', 0), ('control_gradient_norm', 1)]:
    assert abs(result[key]**2-result['gram'][index][index]) <= result[key]**2*1e-10
assert abs(result['gram'][0][1]-result['gram'][1][0]) <= 1e-10
verification = dict(passed=True, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    scope='Native conditional-null gradient execution, algebra residuals and source consistency only; no learning/variance claim.',
                    runtime_seconds=result['elapsed_seconds'],
                    total_gradient_stage_ratio=(result['base_gradient_seconds']+result['control_gradient_seconds'])/result['base_gradient_seconds'])
(HERE/'native_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
print(json.dumps(verification, indent=2))
