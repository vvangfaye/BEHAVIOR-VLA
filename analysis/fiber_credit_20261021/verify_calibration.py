"""Audit a completed, frozen native calibration pilot and its claim limits."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((args.directory/'plan.json').read_text())
    result = json.loads((args.directory/'result.json').read_text())
    updates = json.loads((args.directory/'training.json').read_text())
    frozen = json.loads((args.directory/'frozen_calibrator.json').read_text())
    assert result['complete'] and result['stage'] == 'complete'
    assert plan['actor_frozen'] and not plan['simulation'] and not plan['cloud']
    train, test = plan['selection']['train'], plan['selection']['test']
    assert len(train) == len(test) == 12
    assert not ({s['episode'] for s in train} & {s['episode'] for s in test})
    assert len(updates) == plan['epochs']*math.ceil(len(train)/plan['batch_size'])
    assert frozen == result['frozen_calibrator']
    assert frozen['phi'] == updates[-1]['phi']
    for relative, expected in plan['source_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative
    for sample in test:
        geometry = json.loads((args.directory/'jobs'/f'prepare-{sample["id"]}.json').read_text())
        assert geometry['geometry_only'] and not any(k in geometry for k in ('HH', 'HC', 'CC'))
    amplitude = frozen['phi'][0]*frozen['control_scale']
    totals = [0., 0.]
    for sample, row in zip(test, result['heldout_results']):
        assert row['sample'] == sample['id'] and row['episode'] == sample['episode']
        hh, hc, cc = (row[k] for k in ('HH', 'HC', 'CC'))
        assert all(math.isfinite(v) for v in (hh, hc, cc)) and hh > 0 and cc >= 0
        corrected = hh+2*amplitude*hc+amplitude**2*cc
        assert math.isclose(corrected, row['corrected_second_moment'], rel_tol=1e-10)
        totals[0] += hh
        totals[1] += corrected
    ratio = totals[1]/totals[0]
    assert math.isclose(ratio, result['reward_free_second_moment_ratio'], rel_tol=1e-12)
    passed_gate = ratio <= plan['evaluation']['advance_if_point_ratio_at_most'] and result['episode_bootstrap_95_interval'][1] < 1
    assert result['advance_to_actor_learning_pilot'] == passed_gate
    output = dict(verified=True, passed_gate=passed_gate, ratio=ratio,
                  source_and_split_integrity=True,
                  claim='Small held-out native gradient second-moment pilot only; no closed-loop learning evidence.')
    (args.directory/'verification.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
