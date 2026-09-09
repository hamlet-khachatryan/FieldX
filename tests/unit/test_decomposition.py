"""Decomposition of the inferred correction onto the atomic tangent space."""

import gemmi
import numpy as np
import pytest
from conftest import write_tiny_model

from crystal_field.analysis.decomposition import solve_normal_equations, split_symmetric, symmetrize_grid


def _problem(n_rows=200, n_columns=12, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((n_rows, n_columns))
    truth = rng.standard_normal(n_columns)
    target = basis @ truth
    return basis, truth, target


def test_exact_recovery_when_the_target_lies_in_the_span():
    basis, truth, target = _problem()
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    np.testing.assert_allclose(result["amplitudes"], truth, rtol=1e-8, atol=1e-8)
    assert result["explained_fraction"] == pytest.approx(1.0, abs=1e-10)
    assert result["rank"] == basis.shape[1]


def test_a_target_orthogonal_to_the_basis_explains_nothing():
    basis, _, _ = _problem()
    rng = np.random.default_rng(7)
    orthogonal = rng.standard_normal(basis.shape[0])
    orthogonal -= basis @ np.linalg.lstsq(basis, orthogonal, rcond=None)[0]
    result = solve_normal_equations(basis.T @ basis, basis.T @ orthogonal, float(orthogonal @ orthogonal))
    assert result["explained_fraction"] == pytest.approx(0.0, abs=1e-8)


def test_explained_fraction_is_bounded():
    basis, _, _ = _problem()
    rng = np.random.default_rng(3)
    for _ in range(5):
        target = rng.standard_normal(basis.shape[0])
        result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
        assert -1e-9 <= result["explained_fraction"] <= 1.0 + 1e-9


def test_ridge_shrinks_the_amplitudes():
    basis, _, target = _problem()
    plain = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    ridged = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target), ridge=10.0)
    assert np.linalg.norm(ridged["amplitudes"]) < np.linalg.norm(plain["amplitudes"])
    assert ridged["explained_fraction"] < plain["explained_fraction"]


def test_rank_deficiency_is_reported_not_hidden():
    """A duplicated column makes the basis rank deficient; that is a fact, not an error."""
    basis, _, _ = _problem(n_columns=6)
    basis = np.hstack([basis, basis[:, :1]])
    target = basis @ np.ones(basis.shape[1])
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    assert result["rank"] < basis.shape[1]
    assert np.isfinite(result["condition_number"])
    assert result["explained_fraction"] == pytest.approx(1.0, abs=1e-8)


def test_a_zero_target_does_not_divide_by_zero():
    basis, _, _ = _problem()
    result = solve_normal_equations(basis.T @ basis, np.zeros(basis.shape[1]), 0.0)
    assert np.isfinite(result["explained_fraction"])
    assert result["explained_fraction"] == 0.0


def test_ridge_does_not_leak_into_the_reported_residual():
    """The reported residual and explained_fraction must track the DATA misfit, not the
    ridge-regularized objective. A ridge penalty inflates ||target - Phi a||^2 by
    ridge * ||a||^2 if the ridged gram is reused for the residual computation instead of
    the original one -- that bug is invisible to test_ridge_shrinks_the_amplitudes because
    it only checks direction, and inflating the residual pushes the fraction in the same
    direction that ridge is expected to push it anyway.
    """
    basis, _, target = _problem(n_rows=50, n_columns=8, seed=1)
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target), ridge=1.0)

    data_residual = float(np.sum((target - basis @ result["amplitudes"]) ** 2))
    np.testing.assert_allclose(result["residual_norm_squared"], data_residual, rtol=1e-8, atol=1e-10)

    expected_fraction = 1.0 - data_residual / float(target @ target)
    np.testing.assert_allclose(result["explained_fraction"], expected_fraction, rtol=1e-8, atol=1e-10)


def test_symmetrizing_an_already_symmetric_grid_changes_nothing(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    from crystal_field.crystallography.density import model_density_on_grid

    density, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, (24, 30, 36), 2.6, 1e-6)
    again = symmetrize_grid(density.astype(np.float64), structure.cell, spacegroup)
    np.testing.assert_allclose(again, density, rtol=1e-5, atol=1e-7)


def test_symmetrizing_is_idempotent(tmp_path):
    rng = np.random.default_rng(0)
    cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    field = rng.standard_normal((24, 30, 36))
    once = symmetrize_grid(field, cell, spacegroup)
    twice = symmetrize_grid(once, cell, spacegroup)
    np.testing.assert_allclose(twice, once, rtol=1e-5, atol=1e-7)


def test_the_antisymmetric_fraction_is_reported():
    """A random field in a symmetric group is mostly antisymmetric; that must be visible."""
    rng = np.random.default_rng(1)
    cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    field = rng.standard_normal((24, 30, 36))
    symmetric, antisymmetric_fraction = split_symmetric(field, cell, spacegroup)
    assert 0.0 < antisymmetric_fraction < 1.0
    assert np.linalg.norm(symmetric) < np.linalg.norm(field)


def test_a_symmetric_field_has_no_antisymmetric_part(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    from crystal_field.crystallography.density import model_density_on_grid

    density, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, (24, 30, 36), 2.6, 1e-6)
    _, fraction = split_symmetric(density.astype(np.float64), structure.cell, spacegroup)
    assert fraction < 1e-3


def test_p1_symmetrization_is_the_identity():
    rng = np.random.default_rng(2)
    field = rng.standard_normal((16, 16, 16))
    result = symmetrize_grid(field, gemmi.UnitCell(20, 20, 20, 90, 90, 90), gemmi.SpaceGroup("P 1"))
    np.testing.assert_allclose(result, field, rtol=1e-6, atol=1e-8)
