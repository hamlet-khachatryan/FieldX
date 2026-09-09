"""Sigma-A weighting (Read 1986).

Per resolution shell the model gets a scale D and a residual variance; each reflection
then gets a figure of merit m that down-weights what the model explains poorly. The
properties tested here are the ones that make the weights meaningful: m must live in
[0, 1], rise towards 1 as the model improves, fall towards 0 when the model is unrelated
to the data, and use the restricted-phase form for centric reflections.
"""

import numpy as np
import pytest

from crystal_field.analysis.sigmaa import (
    _figure_of_merit,
    apply_sigmaa,
    estimate_sigmaa,
    map_coefficients,
    resolution_bins,
)


def _dataset(n=2000, noise=0.05, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    d = np.linspace(1.5, 20.0, n)
    fcalc = np.abs(rng.gamma(2.0, 10.0, n))
    fobs = scale * np.abs(fcalc + noise * fcalc.mean() * rng.standard_normal(n))
    centric = rng.random(n) < 0.1
    epsilon = np.ones(n)
    return fobs, fcalc, d, centric, epsilon


# --- figure of merit ----------------------------------------------------------------


def test_figure_of_merit_is_bounded():
    x = np.concatenate([[0.0], np.logspace(-6, 5, 200)])
    for centric in (False, True):
        m = _figure_of_merit(x, np.full(x.shape, centric))
        assert np.all(np.isfinite(m))
        assert np.all((m >= 0.0) & (m <= 1.0))


def test_figure_of_merit_is_monotonic_and_saturates():
    x = np.logspace(-3, 4, 100)
    for centric in (False, True):
        m = _figure_of_merit(x, np.full(x.shape, centric))
        assert np.all(np.diff(m) >= -1e-12)
        assert m[0] < 0.05 and m[-1] > 0.99


def test_centric_and_acentric_use_different_forms():
    x = np.array([0.5, 1.0, 2.0])
    acentric = _figure_of_merit(x, np.zeros(3, dtype=bool))
    centric = _figure_of_merit(x, np.ones(3, dtype=bool))
    np.testing.assert_allclose(centric, np.tanh(x))
    assert not np.allclose(acentric, centric)


def test_large_arguments_do_not_overflow():
    """I0 and I1 both overflow well before this; the ratio must not."""
    m = _figure_of_merit(np.array([1e3, 1e5, 1e6]), np.zeros(3, dtype=bool))
    assert np.all(np.isfinite(m))
    np.testing.assert_allclose(m, 1.0, atol=1e-3)


# --- binning and estimation ---------------------------------------------------------


def test_bins_partition_every_reflection():
    d = np.linspace(1.2, 30.0, 997)
    labels = resolution_bins(d, 13)
    assert set(np.unique(labels)) == set(range(13))
    counts = np.bincount(labels)
    assert counts.sum() == len(d)
    assert counts.max() - counts.min() <= 1


def test_bin_count_is_capped_by_the_reflection_count():
    fobs, fcalc, d, centric, epsilon = _dataset(n=120)
    result = estimate_sigmaa(fobs, fcalc, d, centric, epsilon, n_bins=50)
    assert 1 <= len(result["shells"]) <= 3, "50 shells over 120 reflections would be meaningless"


def test_the_scale_recovers_a_known_factor():
    fobs, fcalc, d, centric, epsilon = _dataset(noise=1e-6, scale=2.5)
    result = estimate_sigmaa(fobs, fcalc, d, centric, epsilon)
    assert np.mean(result["D"]) == pytest.approx(2.5, rel=0.02)


def test_a_good_model_gets_a_high_figure_of_merit():
    fobs, fcalc, d, centric, epsilon = _dataset(noise=0.01)
    assert np.mean(estimate_sigmaa(fobs, fcalc, d, centric, epsilon)["m"]) > 0.9


def test_an_unrelated_model_gets_a_low_figure_of_merit():
    """The property that makes the weighting worth doing."""
    fobs, _, d, centric, epsilon = _dataset()
    unrelated = np.random.default_rng(99).gamma(2.0, 10.0, len(fobs))
    good = np.mean(estimate_sigmaa(fobs, fobs.copy(), d, centric, epsilon, n_bins=10)["m"])
    bad = np.mean(estimate_sigmaa(fobs, unrelated, d, centric, epsilon, n_bins=10)["m"])
    assert bad < good
    assert bad < 0.8


def test_shell_parameters_are_finite_and_positive():
    fobs, fcalc, d, centric, epsilon = _dataset()
    for shell in estimate_sigmaa(fobs, fcalc, d, centric, epsilon)["shells"]:
        assert shell["n"] > 0
        assert shell["D"] > 0
        assert shell["sigma_delta_squared"] > 0
        assert shell["d_max"] >= shell["d_min"]


def test_a_perfectly_fitted_shell_does_not_divide_by_zero():
    """Residual variance is floored, so an exact match cannot produce inf."""
    fcalc = np.abs(np.random.default_rng(1).gamma(2.0, 10.0, 500))
    d = np.linspace(1.5, 20.0, 500)
    result = estimate_sigmaa(fcalc, fcalc, d, np.zeros(500, dtype=bool), np.ones(500))
    assert np.all(np.isfinite(result["m"]))
    assert np.all(np.asarray([s["sigma_delta_squared"] for s in result["shells"]]) > 0)


# --- applying shells to another reflection set ---------------------------------------


def test_shells_estimated_on_one_set_apply_to_another():
    fobs, fcalc, d, centric, epsilon = _dataset(n=1500)
    subset = np.arange(len(fobs)) % 3 != 0
    estimate = estimate_sigmaa(fobs[subset], fcalc[subset], d[subset], centric[subset], epsilon[subset])
    applied = apply_sigmaa(estimate["shells"], fobs, fcalc, d, centric, epsilon)
    assert applied["m"].shape == fobs.shape
    assert np.all((applied["m"] >= 0) & (applied["m"] <= 1))
    assert np.all(applied["D"] > 0)


# --- coefficients ---------------------------------------------------------------------


def test_coefficients_keep_the_calculated_phase():
    rng = np.random.default_rng(3)
    fcalc = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    fobs = np.abs(fcalc) * 1.1
    weights = {"m": np.full(50, 0.9), "D": np.ones(50)}
    for kind in ("2mFo-DFc", "mFo-DFc"):
        coefficients = map_coefficients(fobs, fcalc, weights, np.zeros(50, dtype=bool), kind)
        aligned = np.abs(np.angle(coefficients) - np.angle(fcalc)) % (2 * np.pi)
        assert np.all((aligned < 1e-6) | (np.abs(aligned - np.pi) < 1e-6))


def test_centric_reflections_are_not_double_counted():
    """A centric phase is restricted, so 2mFo would count the observation twice."""
    fcalc = np.array([3.0 + 0j, 3.0 + 0j])
    fobs = np.array([4.0, 4.0])
    weights = {"m": np.array([1.0, 1.0]), "D": np.array([1.0, 1.0])}
    centric = np.array([True, False])
    coefficients = map_coefficients(fobs, fcalc, weights, centric, "2mFo-DFc")
    assert np.abs(coefficients[0]) == pytest.approx(4.0)  # m*Fo
    assert np.abs(coefficients[1]) == pytest.approx(2 * 4.0 - 3.0)  # 2m*Fo - D*Fc


def test_difference_coefficients_vanish_for_a_perfect_model():
    fcalc = np.array([5.0 + 0j])
    weights = {"m": np.array([1.0]), "D": np.array([1.0])}
    coefficients = map_coefficients(np.array([5.0]), fcalc, weights, np.zeros(1, dtype=bool), "mFo-DFc")
    assert np.abs(coefficients[0]) == pytest.approx(0.0, abs=1e-12)


def test_an_unknown_map_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown map kind"):
        map_coefficients(
            np.ones(1),
            np.ones(1, dtype=complex),
            {"m": np.ones(1), "D": np.ones(1)},
            np.zeros(1, dtype=bool),
            "3Fo-2Fc",
        )
