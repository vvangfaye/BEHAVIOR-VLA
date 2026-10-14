import json
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / 'scripts')]
import full_episode_collection as collection
from full_episode_repeats import policy_seeds, replace_once, revised_sources, submission_context

PARENT = Path(__file__).resolve().parents[1] / 'eval_results/sco_full_episode_collection_20261012_v2'


def test_all_4000_additional_seeds_are_unique_and_reproducible():
    plan = json.loads((PARENT / 'plan.json').read_text())
    seeds = policy_seeds(plan)
    assert seeds == policy_seeds(plan)
    assert len(seeds[1]) == len(seeds[2]) == 2000
    assert len(set(seeds[1].values()) | set(seeds[2].values())) == 4000
    assert all(type(v) is int and 0 <= v < 2**32 for d in seeds.values() for v in d.values())


def test_additional_source_is_based_on_frozen_parent_and_does_not_modify_it():
    before = {p: p.read_bytes() for p in (PARENT / 'plan.json', PARENT / 'overlay.tar.gz')}
    files = revised_sources(PARENT)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert b'serve_full_episode_repeat.py' in files['scripts/eval_b1k.sh']
    assert b'begin_episode(evaluator, manifest' in files['scripts/eval_b1k_queue.py']
    assert b'evaluator.policy.proof()' in files['scripts/eval_b1k_queue.py']
    assert b'validate_result_seed(result, metadata)' in files['scripts/full_episode_trajectory.py']
    assert b"policy_seeds=payload['policy_seeds']" in files['sco/run_full_episode_collection.py']


def test_changed_source_anchor_fails_closed():
    with pytest.raises(ValueError): replace_once('missing', 'expected', 'new')
    with pytest.raises(ValueError): replace_once('xx', 'x', 'new')


def test_capacity_charges_jobs_from_other_rounds_without_charging_own_preparation_twice(tmp_path, monkeypatch):
    roots = [tmp_path / str(i) for i in range(3)]
    for root in roots: (root / 'submissions').mkdir(parents=True)
    for root, row in [(roots[0], {'name': 'old-after-observation', 'uid': 'a'}),
                       (roots[1], {'name': 'own-new-intent'}),
                       (roots[2], {'name': 'other-pending'})]:
        (root / 'submissions/0001.json').write_text(json.dumps(row))
    calls = []
    old_repo = collection.REPO
    original = lambda observation, ledger, cli, env: calls.append(ledger)
    monkeypatch.setattr(collection, 'require_capacity', original)
    with submission_context(roots[1], roots):
        collection.require_capacity({}, {}, 'cli', {})
    assert calls == [{'old-after-observation': {'name': 'old-after-observation', 'uid': 'a'},
                      'other-pending': {'name': 'other-pending'}}]
    assert collection.require_capacity is original and collection.REPO == old_repo


def test_controller_submits_seeded_pilot_before_releasing_round_production(tmp_path, monkeypatch):
    import watch_full_episode_repeats as controller
    roots = [tmp_path / str(i) for i in range(3)]
    for root in roots: (root / 'submissions').mkdir(parents=True)
    (roots[0] / 'pilot_accepted.json').write_text('{}')
    (roots[0] / 'submissions/0000.json').write_text(json.dumps(dict(index=0, uid='p0', name='pilot0')))
    monkeypatch.setattr(controller, 'read_plan', lambda root: dict(fingerprint=str(root.name),
        shards=[dict(index=0), dict(index=1), dict(index=2)]))
    monkeypatch.setattr(controller, 'validate_seed_gate', lambda root: None)
    auth = dict(roots=roots, fingerprints=['0', '1', '2'])
    state, candidate = controller.choose(auth, 'cli', None)
    assert state == 'candidate_ready' and candidate == (roots[1], 0)
    for i in (1, 2):
        (roots[i] / 'submissions/0000.json').write_text(json.dumps(dict(index=0, uid=f'p{i}', name=f'pilot{i}')))
    monkeypatch.setattr(controller, 'describe', lambda cli, env, name:
        dict(uid='p' + name[-1], state='RUNNING'))
    state, candidate = controller.choose(auth, 'cli', None)
    assert candidate == (roots[0], 1)
    (roots[1] / 'submissions/0001.json').write_text(json.dumps(dict(index=1, name='unknown')))
    with pytest.raises(ValueError, match='Unknown creation'):
        controller.choose(auth, 'cli', None)


def test_query_retry_refuses_uncertain_creation_in_any_authorized_root(tmp_path):
    import watch_full_episode_repeats as controller
    roots = [tmp_path / 'first', tmp_path / 'second']
    for root in roots:
        (root / 'submissions').mkdir(parents=True)
        (root / 'submissions/0000.json').write_text(json.dumps(dict(uid='known',state='submitted')))
    auth = dict(roots=roots)
    assert controller.query_failure_can_retry(auth)
    p = roots[1] / 'submissions/0001.json'
    p.write_text(json.dumps(dict(name='copy-result-unknown',state='submitting')))
    assert not controller.query_failure_can_retry(auth)
    p.write_text(json.dumps(dict(name='copy-result-unknown',state='submitted',uid='exact-uid-reconciled')))
    assert controller.query_failure_can_retry(auth)
