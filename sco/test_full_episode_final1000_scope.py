import copy
import hashlib
import json
from pathlib import Path

import pytest
from full_episode_scope import validate_final1000, pilot_gate_waived

ROOT = Path(__file__).resolve().parents[1] / 'eval_results/sco_full_episode_collection_20261014_final1000_round3/round-3'


def plan():
    return json.loads((ROOT / 'plan.json').read_text())


def test_exact_1000_scope_and_no_seed_reuse():
    value = validate_final1000(plan())
    assert sum(len(s['instance_ids']) for s in value['shards']) == 1000
    old = ROOT.parents[1] / 'sco_full_episode_collection_20261012_three_runs'
    old_seeds = set()
    for round_id in (1, 2):
        old_seeds.update(json.loads((old / f'round-{round_id}/plan.json').read_text())['policy_seeds'].values())
    assert not old_seeds.intersection(value['policy_seeds'].values())


@pytest.mark.parametrize('change', ['instance311', 'duplicate', 'wrong_count', 'wrong_round', 'same_seed'])
def test_scope_expansion_or_duplicate_rejected(change):
    value = copy.deepcopy(plan())
    if change == 'instance311': value['shards'][1]['instance_ids'][0] = 311
    if change == 'duplicate': value['shards'][1]['instance_ids'][1] = value['shards'][1]['instance_ids'][0]
    if change == 'wrong_count': value['expected_episodes'] = 2000
    if change == 'wrong_round': value['sampling_round'] = 2
    if change == 'same_seed':
        keys = list(value['policy_seeds'])
        value['policy_seeds'][keys[1]] = value['policy_seeds'][keys[0]]
    with pytest.raises(AssertionError): validate_final1000(value)


def test_frozen_runtime_seed_protocol_accepts_new_round():
    import runpy
    helper = runpy.run_path(str(ROOT / 'overlay_sources/scripts/full_episode_repeat_seeding.py'))
    value = plan()
    proof = helper['seed_spec'](value, 'turning_on_radio', 301)
    assert proof['sampling_round'] == 3
    assert proof['policy_seed'] == value['policy_seeds']['turning_on_radio:301']


def test_waiver_is_bound_to_new_authorization_and_is_not_acceptance(tmp_path):
    root = tmp_path / 'round-3'
    root.mkdir()
    value = plan()
    (root / 'plan.json').write_text(json.dumps(value))
    auth = dict(kind='native_full_episode_independent_final1000', roots=[str(root)],
                fingerprints=[value['fingerprint']], deadline_epoch=123)
    raw = json.dumps(auth).encode()
    (tmp_path / 'authorization.json').write_bytes(raw)
    waiver = dict(kind='user_waived_pilot_production_gate', root=str(root),
                  plan_fingerprint=value['fingerprint'], deadline_epoch=123,
                  total_episodes=1000, max_pending_tasks=5,
                  authorization_sha256=hashlib.sha256(raw).hexdigest(),
                  allow_production_without_pilot_acceptance=True,
                  artifact_acceptance_complete=False)
    path = tmp_path / 'pilot_acceptance_waiver.json'
    path.write_text(json.dumps(waiver))
    assert pilot_gate_waived(root, value)
    assert not (root / 'pilot_accepted.json').exists()
    waiver['plan_fingerprint'] = 'wrong'
    path.write_text(json.dumps(waiver))
    with pytest.raises(AssertionError): pilot_gate_waived(root, value)


def test_without_explicit_waiver_gate_is_required(tmp_path):
    assert pilot_gate_waived(tmp_path / 'round-3') is False
