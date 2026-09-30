"""Held-out reliability bins, deliberately unknown for insufficient evidence.

These are population/bin reliability intervals under an IID sampling assumption;
they do not certify a conditional bound at an individual or shifted robot state.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .belief import Estimate, UNKNOWN


@dataclass
class ReliabilityCalibrator:
    bins: int = 10
    delta: float = 0.05
    min_count: int = 30
    counts: list[int] | None = None
    positives: list[float] | None = None

    def __post_init__(self):
        if self.bins < 1 or not 0 < self.delta < 1 or self.min_count < 1:
            raise ValueError("Invalid calibration configuration")
        if self.counts is None:
            self.counts, self.positives = [0] * self.bins, [0] * self.bins
        if len(self.counts) != self.bins or self.positives is None or len(self.positives) != self.bins:
            raise ValueError("Invalid calibration state")
        if any(type(n) is not int or not isinstance(y, (int, float)) or not 0 <= y <= n
               for n, y in zip(self.counts, self.positives)):
            raise ValueError("Invalid calibration counts")

    def fit(self, probabilities, targets, groups=None):
        p, y = np.asarray(probabilities).ravel(), np.asarray(targets).ravel()
        if p.shape != y.shape or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Invalid calibration probabilities")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Calibration targets must be binary")
        groups = list(range(len(p))) if groups is None else list(groups)
        if len(groups) != len(p): raise ValueError("Calibration group count mismatch")
        grouped = {}
        for prob, target, group in zip(p, y, groups):
            i = min(self.bins - 1, int(prob * self.bins))
            total, count = grouped.get((i, group), (0., 0))
            grouped[(i, group)] = (total + float(target), count + 1)
        self.counts, self.positives = [0] * self.bins, [0.] * self.bins
        for (i, _), (total, count) in grouped.items():
            # Correlated frames/atoms from one initial state are ONE sampling
            # unit. This estimates group-balanced bin reliability, not pointwise
            # state confidence. Hoeffding applies only if groups are independent.
            self.counts[i] += 1
            self.positives[i] += total / count
        return self

    def estimate(self, probability: float) -> Estimate:
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")
        i = min(self.bins - 1, int(probability * self.bins))
        n = self.counts[i]
        if n < self.min_count:
            return UNKNOWN
        mean = self.positives[i] / n
        radius = math.sqrt(math.log(2 * self.bins / self.delta) / (2 * n))
        return Estimate(mean, max(0., mean - radius), min(1., mean + radius))

    def to_dict(self):
        return {"bins": self.bins, "delta": self.delta, "min_count": self.min_count,
                "counts": self.counts, "positives": self.positives}
