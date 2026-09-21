"""Bootstrap confidence intervals for the per-case rates this attack reports.

The paper reports every Table 1 / Table 2 cell as ``mean +/- half-CI`` from a
"95% bootstrap confidence interval" (arXiv:2605.15338 v2). This reproduces that:
a percentile bootstrap over the list of per-case 0/1 outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RateCI:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    half_width: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "half_width": round(self.half_width, 4),
            "pretty": f"{100 * self.mean:.1f} +/- {100 * self.half_width:.1f}",
        }


def bootstrap_ci(
    values,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RateCI:
    """Percentile bootstrap CI for the mean of a list of per-case outcomes
    (0/1 for a rate, but any real values work).

    Deterministic given ``seed`` so re-running a report is reproducible.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return RateCI(0, 0.0, 0.0, 0.0, 0.0)
    if arr.size == 1:
        m = float(arr[0])
        return RateCI(1, m, m, m, 0.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resampled_means = arr[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(resampled_means, [100 * tail, 100 * (1.0 - tail)])
    mean = float(arr.mean())
    return RateCI(arr.size, mean, float(lo), float(hi), float((hi - lo) / 2.0))


def rate_with_ci(values, **kwargs) -> dict:
    """Convenience: bootstrap_ci(...).as_dict()."""
    return bootstrap_ci(values, **kwargs).as_dict()
