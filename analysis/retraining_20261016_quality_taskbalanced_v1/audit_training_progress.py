"""Read-only checks of actual logged source counts and learning-rate clocks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'formal_stage/bundle'


def expected_sources():
    sys.path.insert(0, str(BUNDLE / 'openpi/src'))
    from openpi.training.subtask_mixture import MixturePlan
    path = ROOT / 'expected_source_prefixes.json'
    audit = json.loads((ROOT / 'metadata_sampling_validation.json').read_text())
    if path.exists():
        value = json.loads(path.read_text())
        if value['mixture_fingerprint'] != audit['mixture_fingerprint']:
            raise ValueError('Expected counts belong to another sampler')
        return value
    plan = MixturePlan.load(BUNDLE / 'inputs/compiled/mixture.json')
    if plan.fingerprint != audit['mixture_fingerprint']:
        raise ValueError('Local sampling plan differs from accepted smoke')
    full, prefixes = 0, [0]
    sampler = plan._quality_sampler
    for index in range(plan.num_samples):
        task, ordinal = sampler.task_at(index)
        full += int(ordinal in sampler.slots.get(task, {}))
        if (index + 1) % plan.global_batch_size == 0:
            prefixes.append(full)
    if len(prefixes) != 5001 or prefixes[-1] != 48829:
        raise ValueError('Expected finite source budget changed')
    value = {'mixture_fingerprint': plan.fingerprint, 'global_batch': 192,
             'full_prefix_counts': prefixes, 'meaning': 'metadata expectation, not actual training evidence'}
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    return value


def expected_rate(contract, step):
    s = contract['schedule']
    if step < s['warmup_steps']:
        return contract['initial_lr'] + (s['peak_lr'] - contract['initial_lr']) * step / s['warmup_steps']
    phase = min(max((step - s['warmup_steps']) / (s['decay_steps'] - s['warmup_steps']), 0), 1)
    return s['decay_lr'] + .5 * (s['peak_lr'] - s['decay_lr']) * (1 + math.cos(math.pi * phase))


def check_record(record, expected, contract):
    n = record['completed_updates']
    if type(n) is not int or not 0 < n <= 5000 or (n > 200 and n % 10):
        raise ValueError('Unexpected actual logging step')
    if record['mixture_fingerprint'] != expected['mixture_fingerprint']:
        raise ValueError('Actual training uses another sampling plan')
    metrics = record['metrics']
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values()):
        raise ValueError('Nonfinite or nonnumeric actual metric')
    full = expected['full_prefix_counts'][n]
    totals = {'replay': n * 192 - full, 'full_episode': full, 'recovered': 0, 'collected': 0}
    for source, count in totals.items():
        if metrics['source_draws_total/' + source] != count:
            raise ValueError('Actual cumulative samples differ for ' + source)
    # After warmup the logger averages groups of ten optimizer updates.
    start = n - 1 if n <= 200 else n - 10
    rate = mean(expected_rate(contract, i) for i in range(start, n))
    if not math.isclose(metrics['learning_rate'], rate, rel_tol=2e-5, abs_tol=1e-12):
        raise ValueError('Actual applied LR differs from the frozen curve')
    if metrics['learning_rate'] > contract['peak_limit'] * (1 + 2e-6):
        raise ValueError('Applied LR exceeds authorized peak')
    if metrics['lr_clock_matches_step'] != 1 or metrics['nonfinite_count'] != 0:
        raise ValueError('Actual optimizer clock or finite-metric guard failed')
    if metrics['completed_optimizer_updates'] != n:
        raise ValueError('Logged completed-update clock differs')
    if metrics['optimizer_internal_step'] != mean(range(start, n)):
        raise ValueError('Internal optimizer clock differs from logged update group')
    return {'completed_updates': n, 'actual_source_totals': totals,
            'actual_learning_rate': metrics['learning_rate'], 'expected_logged_learning_rate': rate}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation', type=Path, default=ROOT / 'latest_runtime_observation.json')
    args = p.parse_args()
    expected = expected_sources()
    contract = json.loads((BUNDLE / 'inputs/compiled/learning_rate_contract.json').read_text())
    snapshot = json.loads(args.observation.read_text())
    formal = snapshot.get('stages', {}).get('formal', {})
    records = []
    for key, value in formal.get('metadata', {}).items():
        if not key.endswith('training_metrics.jsonl_tail'):
            continue
        lines = value.splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                if i in (0, len(lines) - 1):
                    continue
                raise
            records.append(check_record(record, expected, contract))
    result = {'observed_at': snapshot['observed_at'], 'read_only': True,
              'state': 'actual_records_verified' if records else 'waiting_for_actual_update_records',
              'actual_record_count_checked': len(records), 'records': records,
              'source_observation_sha256': hashlib.sha256(args.observation.read_bytes()).hexdigest(),
              'model_improvement_proven': False}
    (ROOT / 'latest_progress_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
