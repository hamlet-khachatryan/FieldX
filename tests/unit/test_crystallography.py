"""Crystallographic primitives (plan section 20C)."""

import gemmi
import numpy as np
import pytest

from crystal_field.crystallography.density import (
    direct_structure_factors,
    model_density_on_grid,
    model_electron_count,
)
from crystal_field.crystallography.geometry import (
    cell_tuple,
    direct_metric_from_parameters,
    reciprocal_metric_from_parameters,
)
from crystal_field.crystallography.symmetry import symmetry_arrays

SPACE_GROUPS = ["P 1", "P 21 21 21", "P 41", "C 2", "P 63", "I 4", "R 3 :H", "F 4 3 2"]


def test_orthogonal_reciprocal_metric():
    g = reciprocal_metric_from_parameters(10, 20, 40, 90, 90, 90)
    np.testing.assert_allclose(g, np.diag([1 / 100, 1 / 400, 1 / 1600]), atol=1e-12)


@pytest.mark.parametrize(
    "cell",
    [(10, 20, 40, 90, 90, 90), (30, 30, 45, 90, 90, 120), (23.1, 27.4, 31.9, 78.2, 84.5, 69.7)],
)
def test_metric_tensors_are_mutual_inverses(cell):
    direct = direct_metric_from_parameters(*cell)
    reciprocal = reciprocal_metric_from_parameters(*cell)
    np.testing.assert_allclose(direct @ reciprocal, np.eye(3), atol=1e-10)


@pytest.mark.parametrize(
    "cell",
    [(10, 20, 40, 90, 90, 90), (30, 30, 45, 90, 90, 120), (23.1, 27.4, 31.9, 78.2, 84.5, 69.7)],
)
@pytest.mark.parametrize("hkl", [(1, 0, 0), (2, -3, 1), (0, 4, -5)])
def test_reciprocal_metric_reproduces_gemmi_resolution(cell, hkl):
    """h^T G* h is 1/d^2 for the same Miller index, which is what the prior consumes."""
    g = reciprocal_metric_from_parameters(*cell)
    h = np.asarray(hkl, dtype=np.float64)
    assert float(h @ g @ h) == pytest.approx(gemmi.UnitCell(*cell).calculate_1_d2(hkl), rel=1e-10)


def test_cell_tuple_round_trips_through_gemmi():
    values = (23.1, 27.4, 31.9, 78.2, 84.5, 69.7)
    assert cell_tuple(gemmi.UnitCell(*values)) == pytest.approx(values)


@pytest.mark.parametrize("name", SPACE_GROUPS)
def test_symmetry_arrays_are_integral_and_group_sized(name):
    sg = gemmi.SpaceGroup(name)
    rotations, translations = symmetry_arrays(sg)
    assert rotations.shape == (len(sg.operations()), 3, 3)
    assert translations.shape == (len(sg.operations()), 3)
    assert rotations.dtype == np.int32
    # Crystallographic rotations are orthogonal in fractional coordinates.
    for r in rotations:
        assert abs(round(float(np.linalg.det(r)))) == 1
    assert np.all((translations >= 0.0) & (translations < 1.0))


@pytest.mark.parametrize("name", SPACE_GROUPS)
def test_first_symmetry_operator_is_the_identity(name):
    rotations, translations = symmetry_arrays(gemmi.SpaceGroup(name))
    np.testing.assert_array_equal(rotations[0], np.eye(3, dtype=np.int32))
    np.testing.assert_allclose(translations[0], np.zeros(3), atol=1e-12)


def test_systematic_absences_are_detected():
    """P 21 21 21 extinguishes odd h00, 0k0 and 00l."""
    operations = gemmi.SpaceGroup("P 21 21 21").operations()
    hkls = np.array([[1, 0, 0], [2, 0, 0], [0, 3, 0], [0, 4, 0], [0, 0, 5], [1, 1, 1]], dtype=np.int32)
    absent = operations.systematic_absences(hkls)
    assert list(absent) == [True, False, True, False, True, False]


def test_model_density_honours_the_declared_grid(tmp_path):
    """put_model_density_on_grid() silently re-derives the grid size; we must not.

    rho0, the solvent mask and the Nyquist guard all have to live on one grid, or the
    same (h,k,l) gathers a different Fourier bin from each.
    """
    from conftest import write_tiny_model

    st = gemmi.read_structure(str(write_tiny_model(tmp_path / "tiny.pdb")))
    st.setup_entities()
    sg = gemmi.SpaceGroup("P 1")
    for shape in [(24, 30, 36), (40, 48, 56), (20, 24, 28)]:
        rho, _ = model_density_on_grid(st[0], st.cell, sg, shape, 2.6, 1e-6)
        assert rho.shape == shape
        assert float(rho.max()) > 0.0


def test_integrated_density_equals_the_model_electron_count(tmp_path):
    from conftest import write_tiny_model

    st = gemmi.read_structure(str(write_tiny_model(tmp_path / "tiny.pdb")))
    st.setup_entities()
    sg = gemmi.SpaceGroup("P 1")
    shape = (40, 48, 56)
    rho, _ = model_density_on_grid(st[0], st.cell, sg, shape, 1.5, 1e-7)
    integrated = float(rho.sum()) * st.cell.volume / rho.size
    assert integrated == pytest.approx(model_electron_count(st[0], sg), rel=0.01)


def test_fft_of_rho0_matches_direct_summation(tmp_path):
    """An FFT-free reference for the starting density."""
    from conftest import write_tiny_model

    st = gemmi.read_structure(str(write_tiny_model(tmp_path / "tiny.pdb")))
    st.setup_entities()
    sg = gemmi.SpaceGroup("P 1")
    shape = (40, 48, 56)
    rho, _ = model_density_on_grid(st[0], st.cell, sg, shape, 2.0, 1e-7)

    hkls = np.array([[1, 0, 0], [2, 1, 0], [0, 2, 3], [3, -2, 1], [-1, 4, 2]], dtype=np.int32)
    direct = direct_structure_factors(st[0], st.cell, sg, hkls, 2.0)
    fgrid = np.fft.fftn(rho.astype(np.float64)) * (st.cell.volume / rho.size)
    idx = np.mod(hkls, np.asarray(shape))
    gridded = np.conj(fgrid[idx[:, 0], idx[:, 1], idx[:, 2]])
    np.testing.assert_allclose(np.abs(gridded), np.abs(direct), rtol=0.05)


def test_grid_shape_follows_cell_dmin_and_sampling(tiny_dataset):
    """The computational grid is derived, never assumed."""
    from crystal_field.crystallography.io import choose_grid_shape, load_reflections

    cfg = tiny_dataset["cfg"]
    ds = load_reflections(cfg)
    coarse = choose_grid_shape(
        ds, cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"samples_per_dmin": 2.0})})
    )
    fine = choose_grid_shape(ds, cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"samples_per_dmin": 4.0})}))
    assert all(f > c for f, c in zip(fine, coarse, strict=True))
    # Real-space sampling must be at least d_min / samples_per_dmin along every axis.
    for length, n in zip(cell_tuple(ds.cell)[:3], coarse, strict=True):
        assert length / n <= cfg.resolution.d_min_angstrom / 2.0 + 1e-9


def test_explicit_grid_shape_is_honoured(tiny_dataset):
    from crystal_field.crystallography.io import choose_grid_shape, load_reflections

    cfg = tiny_dataset["cfg"]
    cfg = cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"shape": (32, 32, 32)})})
    assert choose_grid_shape(load_reflections(cfg), cfg) == (32, 32, 32)


def test_nyquist_guard_rejects_a_grid_too_small_for_the_reflections(tiny_dataset):
    """A grid that cannot hold the retained HKLs aliases them; prepare must refuse."""
    from crystal_field.crystallography.io import prepare_reflections

    cfg = tiny_dataset["cfg"]
    cfg = cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"shape": (8, 8, 8)})})
    with pytest.raises(ValueError, match="too small for retained HKLs"):
        prepare_reflections(cfg)
