import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import watch_full_episode_collection as watch


@pytest.fixture
def root(tmp_path):
    (tmp_path / 'submissions').mkdir()
    (tmp_path / 'plan.json').write_text(json.dumps(dict(shards=[dict(index=0), dict(index=1)])))
    (tmp_path / 'submissions/0000.json').write_text(json.dumps(dict(name='pilot', uid='exact')))
    return tmp_path


@pytest.mark.parametrize('state,expected', [('RUNNING', 'waiting_for_pilot'),
    ('FAILED', 'pilot_failed_no_production'), ('CANCELLED', 'pilot_failed_no_production')])
def test_no_production_before_real_pilot_acceptance(root, monkeypatch, state, expected):
    monkeypatch.setattr(watch, 'describe', lambda *a: dict(uid='exact', state=state))
    monkeypatch.setattr(watch, 'accept_pilot', lambda *a: pytest.fail('Must not accept an incomplete pilot'))
    monkeypatch.setattr(watch, 'submit', lambda *a: pytest.fail('Must not expand without accepted pilot'))
    assert watch.step(root, None, None, None) == expected


def test_changed_uid_stops(root, monkeypatch):
    monkeypatch.setattr(watch, 'describe', lambda *a: dict(uid='different', state='SUCCEEDED'))
    with pytest.raises(ValueError, match='UID changed'):
        watch.step(root, None, None, None)


def test_waits_for_fresh_capacity_after_codec_acceptance(root, monkeypatch):
    (root / 'pilot_accepted.json').write_text('{}')
    monkeypatch.setattr(watch, 'validate_gate', lambda *a: None)
    def no_capacity(*args):
        raise watch.CapacityUnavailable('Expired')
    monkeypatch.setattr(watch, 'submit', no_capacity)
    assert watch.step(root, None, None, None) == 'waiting_for_fresh_idle_capacity'


def test_uncertain_creation_is_not_retried(root, monkeypatch):
    (root / 'pilot_accepted.json').write_text('{}')
    (root / 'submissions/0001.json').write_text(json.dumps(dict(state='submitting')))
    monkeypatch.setattr(watch, 'validate_gate', lambda *a: None)
    monkeypatch.setattr(watch, 'submit', lambda *a: pytest.fail('Must reconcile first'))
    assert watch.step(root, None, None, None) == 'needs_exact_submission_reconciliation'
