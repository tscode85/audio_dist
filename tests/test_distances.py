# file: tests/test_distances.py
"""Tests for distance metrics: identical distributions -> ~0, shifted -> larger."""
from __future__ import annotations

import numpy as np
import pytest

from acoustic_gap.config import DistanceConfig
from acoustic_gap.distances import (
    compute_distances,
    frechet_distance,
    mmd_rbf,
    per_dimension_wasserstein,
    sliced_wasserstein,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_frechet_zero_for_same_distribution(rng):
    x = rng.standard_normal((500, 8))
    y = rng.standard_normal((500, 8))
    same = frechet_distance(x, x)
    close = frechet_distance(x, y)
    shifted = frechet_distance(x, y + 5.0)
    assert same < 1e-6
    assert shifted > close
    assert shifted > 20  # mean shift of 5 across 8 dims dominates


def test_mmd_monotonic_with_shift(rng):
    x = rng.standard_normal((300, 5))
    y = rng.standard_normal((300, 5))
    small = mmd_rbf(x, y, rng=rng)
    large = mmd_rbf(x, y + 2.0, rng=rng)
    assert large > small
    assert small < 0.05  # near zero for same distribution


def test_mmd_nonnegative_and_self_zero(rng):
    x = rng.standard_normal((200, 4))
    assert mmd_rbf(x, x, rng=rng) < 1e-6


def test_per_dimension_wasserstein_shift(rng):
    x = rng.standard_normal((400, 6))
    y = rng.standard_normal((400, 6))
    base = per_dimension_wasserstein(x, y)
    shifted = per_dimension_wasserstein(x, y + 3.0)
    assert shifted > base
    assert shifted == pytest.approx(3.0, abs=0.3)


def test_sliced_wasserstein_shift(rng):
    x = rng.standard_normal((400, 6))
    y = rng.standard_normal((400, 6))
    base = sliced_wasserstein(x, y, n_projections=64, rng=rng)
    shifted = sliced_wasserstein(x, y + 3.0, n_projections=64, rng=rng)
    assert shifted > base


def test_compute_distances_keys(rng):
    x = rng.standard_normal((200, 5))
    y = rng.standard_normal((200, 5)) + 1.0
    cfg = DistanceConfig(metrics=["fad", "mmd", "wasserstein"],
                         wasserstein_backend="sliced")
    out = compute_distances(x, y, cfg)
    assert set(["fad", "mmd", "wasserstein_1d", "wasserstein_mv", "wasserstein"]) <= set(out)
    assert all(np.isfinite(v) for v in out.values())


def test_pot_backend_if_available(rng):
    pytest.importorskip("ot")
    from acoustic_gap.distances import wasserstein_pot

    x = rng.standard_normal((150, 4))
    y = rng.standard_normal((150, 4)) + 2.0
    d_same = wasserstein_pot(x, x, rng=rng)
    d_shift = wasserstein_pot(x, y, rng=rng)
    assert d_shift > d_same
