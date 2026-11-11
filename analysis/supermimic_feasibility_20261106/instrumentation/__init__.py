"""Opt-in instrumentation for a future, separately launched simulator pilot."""

from .recorder import ControlOutcome, StepRecorder, TimingRecorder, snapshot_entities

__all__ = ["ControlOutcome", "StepRecorder", "TimingRecorder", "snapshot_entities"]
