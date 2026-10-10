import json
from pathlib import Path

import pytest

from public_test_schedule import pending_order


def plan():
    return json.loads((Path(__file__).resolve().parents[1] /
        'eval_results/sco_test2000_1688977_20261010/production_plan.json').read_text())


def test_task_rounds_preserve_every_frozen_case_and_exclude_all_receipts():
    frozen = plan()
    before = json.dumps(frozen, sort_keys=True)
    order = pending_order(frozen, set())
    assert len(order) == len(set(order)) == 400
    assert len({frozen['shards'][i]['task'] for i in order[:100]}) == 100
    assert order[:3] == [0, 4, 8]
    remaining = pending_order(frozen, {0, 4, 8, 399})
    assert remaining[0] == 12 and set(remaining) == set(range(400)) - {0, 4, 8, 399}
    assert json.dumps(frozen, sort_keys=True) == before


def test_missing_or_duplicate_frozen_episode_rejected():
    frozen = plan()
    frozen['shards'][1]['instance_ids'][0] = 302
    with pytest.raises(AssertionError):
        pending_order(frozen, set())
