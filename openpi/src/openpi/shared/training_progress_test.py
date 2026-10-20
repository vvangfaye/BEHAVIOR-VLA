import pytest

from openpi.shared.training_progress import checkpoint_milestone, checkpoint_to_save


def test_indexed_saves_are_completed_updates_without_off_by_one():
    saved = []
    for iteration in range(5000):
        step = checkpoint_to_save(iteration=iteration, completed_updates=iteration + 1,
            start_step=0, end_step=5000, interval=1000, indexed=True)
        if step is not None:
            saved.append(step)
    assert saved == [1000, 2000, 3000, 4000, 5000]


def test_resumed_and_disposable_final_boundaries():
    assert checkpoint_to_save(iteration=1000, completed_updates=1001,
        start_step=1000, end_step=5000, interval=1000, indexed=True) is None
    assert checkpoint_to_save(iteration=1, completed_updates=2,
        start_step=0, end_step=2, interval=1000, indexed=True) == 2


def test_legacy_numbering_is_preserved_and_clock_mismatch_rejected():
    assert checkpoint_to_save(iteration=1000, completed_updates=1001,
        start_step=0, end_step=5000, interval=1000, indexed=False) == 1000
    with pytest.raises(ValueError, match="disagree"):
        checkpoint_to_save(iteration=999, completed_updates=999,
            start_step=0, end_step=5000, interval=1000, indexed=True)
    assert [i for i in range(0, 20001, 1000) if checkpoint_milestone(i)] == [1000, 5000, 10000, 20000]
