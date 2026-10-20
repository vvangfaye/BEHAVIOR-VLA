"""Checkpoint numbering and immutable provenance for indexed fine-tuning."""
from __future__ import annotations


def checkpoint_to_save(*, iteration, completed_updates, start_step, end_step, interval, indexed):
    if min(iteration, completed_updates, start_step) < 0 or interval <= 0:
        raise ValueError("Invalid training progress")
    final = iteration == end_step - 1
    if indexed:
        if completed_updates != iteration + 1:
            raise ValueError("Optimizer state and indexed training loop disagree")
        return completed_updates if final or (completed_updates > start_step and completed_updates % interval == 0) else None
    return iteration if final or (iteration > start_step and iteration % interval == 0) else None


def checkpoint_milestone(step):
    return step in (1000, 5000, 10000, 20000)
