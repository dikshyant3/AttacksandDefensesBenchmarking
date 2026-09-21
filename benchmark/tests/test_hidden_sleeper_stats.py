"""Deterministic tests for the bootstrap-CI helper."""

from __future__ import annotations

from benchmark.attacks.hidden_sleeper.stats import bootstrap_ci, rate_with_ci


def test_bootstrap_ci_all_ones_is_a_point_at_one():
    ci = bootstrap_ci([1, 1, 1, 1, 1])
    assert ci.n == 5
    assert ci.mean == 1.0
    assert ci.ci_low == 1.0 and ci.ci_high == 1.0
    assert ci.half_width == 0.0


def test_bootstrap_ci_all_zeros_is_a_point_at_zero():
    ci = bootstrap_ci([0] * 8)
    assert ci.mean == 0.0 and ci.half_width == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_wider_for_small_n():
    half = [1, 0] * 10  # mean 0.5, n=20
    ci_small = bootstrap_ci([1, 0, 1, 0])  # n=4
    ci_big = bootstrap_ci(half)  # n=20
    assert abs(ci_small.mean - 0.5) < 1e-9
    assert ci_small.ci_low <= ci_small.mean <= ci_small.ci_high
    assert ci_big.ci_low <= 0.5 <= ci_big.ci_high
    # smaller n -> wider interval
    assert ci_small.half_width > ci_big.half_width


def test_bootstrap_ci_is_deterministic_given_seed():
    v = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    assert bootstrap_ci(v, seed=42).as_dict() == bootstrap_ci(v, seed=42).as_dict()


def test_bootstrap_ci_empty_is_all_zero():
    ci = bootstrap_ci([])
    assert ci.n == 0 and ci.mean == 0.0


def test_rate_with_ci_shape():
    d = rate_with_ci([1, 0, 1])
    assert set(d) == {"n", "mean", "ci_low", "ci_high", "half_width", "pretty"}
    assert d["n"] == 3
    assert "+/-" in d["pretty"]
