"""The Phi operator: per-atom density derivatives.

Density is additive over atoms, so the derivative with respect to one atom's parameter
involves only that atom -- a single-atom structure gives the exact column. Gemmi's own
density cutoff then makes each column naturally sparse, which is what keeps the basis
affordable without any box arithmetic.
"""

import gemmi
import numpy as np
import pytest
from conftest import write_tiny_model

from crystal_field.analysis.tangent import (
    PARAMETER_STEPS,
    build_tangent_basis,
    selected_atoms,
    single_atom_model,
    tangent_column,
)
from crystal_field.crystallography.density import model_density_on_grid

SHAPE = (24, 30, 36)
D_MIN = 2.6
CUTOFF = 1e-6


@pytest.fixture
def model(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    return structure


def _atoms(model):
    return [cra.atom for cra in model[0].all()]


def test_density_is_additive_over_atoms(model):
    """The premise the whole operator rests on."""
    spacegroup = gemmi.SpaceGroup("P 1")
    full, _ = model_density_on_grid(model[0], model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    total = np.zeros(SHAPE, dtype=np.float64)
    for atom in _atoms(model):
        one, _ = model_density_on_grid(single_atom_model(atom), model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
        total += one
    assert np.abs(total - full).max() / full.max() < 1e-5


@pytest.mark.parametrize("kind", ["x", "y", "z", "b_iso", "occupancy"])
def test_column_is_finite_nonzero_and_sparse(model, kind):
    spacegroup = gemmi.SpaceGroup("P 1")
    column = tangent_column(_atoms(model)[1], kind, model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    assert column.shape == SHAPE
    assert np.all(np.isfinite(column))
    assert np.linalg.norm(column) > 0
    occupied = np.count_nonzero(np.abs(column) > 1e-12)
    assert occupied < 0.5 * column.size, "an atom's derivative must not fill the cell"


def test_coordinate_column_matches_a_real_displacement(model):
    """The derivative must predict what actually happens when the atom moves."""
    spacegroup = gemmi.SpaceGroup("P 1")
    atom = _atoms(model)[1]
    column = tangent_column(atom, "x", model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)

    delta = 0.01
    before, _ = model_density_on_grid(single_atom_model(atom), model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    original = atom.pos.x
    atom.pos.x = original + delta
    after, _ = model_density_on_grid(single_atom_model(atom), model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    atom.pos.x = original

    actual = after.astype(np.float64) - before.astype(np.float64)
    predicted = delta * column
    assert np.linalg.norm(actual - predicted) / np.linalg.norm(actual) < 0.05


def test_the_column_leaves_the_atom_unchanged(model):
    """A derivative that mutates the model would corrupt every later column."""
    atom = _atoms(model)[0]
    before = (atom.pos.x, atom.pos.y, atom.pos.z, atom.b_iso, atom.occ)
    tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    tangent_column(atom, "b_iso", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    assert (atom.pos.x, atom.pos.y, atom.pos.z, atom.b_iso, atom.occ) == before


def test_columns_include_symmetry_copies(model):
    """rho0 is symmetrize_sum-ed, so a column must cover every copy."""
    atom = _atoms(model)[1]
    p1 = tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    p212121 = tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 21 21 21"), SHAPE, D_MIN, CUTOFF)
    assert np.count_nonzero(np.abs(p212121) > 1e-12) > 2 * np.count_nonzero(np.abs(p1) > 1e-12)


def test_an_unknown_parameter_is_rejected(model):
    with pytest.raises(ValueError, match="Unknown tangent parameter"):
        tangent_column(_atoms(model)[0], "charge", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)


def test_parameter_steps_are_declared_for_every_kind():
    assert set(PARAMETER_STEPS) == {"x", "y", "z", "b_iso", "occupancy"}
    assert all(step > 0 for step in PARAMETER_STEPS.values())


def _basis(model, name, spacegroup="P 1"):
    return build_tangent_basis(model[0], model.cell, gemmi.SpaceGroup(spacegroup), SHAPE, D_MIN, CUTOFF, basis=name)


def test_selected_atoms_excludes_hydrogens_and_zero_occupancy(model):
    atoms = selected_atoms(model[0])
    assert len(atoms) == 10
    assert all(not atom.element.is_hydrogen for _, atom in atoms)
    assert all(atom.occ > 0 for _, atom in atoms)


@pytest.mark.parametrize(("name", "per_atom"), [("coordinates", 3), ("coordinates_b", 4), ("full", 5)])
def test_column_count_follows_the_basis(model, name, per_atom):
    basis = _basis(model, name)
    assert basis.n_columns == 10 * per_atom
    assert basis.matrix.shape == (int(np.prod(SHAPE)), basis.n_columns)
    assert len(basis.labels) == basis.n_columns
    assert basis.basis == name


def test_labels_identify_atom_and_parameter(model):
    basis = _basis(model, "coordinates")
    kinds = {kind for _, kind in basis.labels}
    assert kinds == {"x", "y", "z"}
    assert {index for index, _ in basis.labels} == set(range(10))


def test_the_matrix_is_sparse(model):
    basis = _basis(model, "coordinates")
    density = basis.matrix.nnz / (basis.matrix.shape[0] * basis.matrix.shape[1])
    assert density < 0.5, f"columns should be local, got {density:.2%} filled"


def test_columns_match_tangent_column_exactly(model):
    """Assembly must not alter what Task 2 produces."""
    basis = _basis(model, "coordinates")
    index, kind = basis.labels[4]
    atom = selected_atoms(model[0])[index][1]
    expected = tangent_column(atom, kind, model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    stored = np.asarray(basis.matrix[:, 4].todense()).ravel().reshape(SHAPE)
    np.testing.assert_allclose(stored, expected, atol=1e-10)


def test_norm_capture_is_reported(model):
    """The sparsity guard: stored columns must retain essentially all their norm."""
    basis = _basis(model, "coordinates")
    assert basis.min_norm_fraction > 0.999


def test_truncation_below_tolerance_is_refused(model):
    with pytest.raises(ValueError, match="box_radius_angstrom"):
        build_tangent_basis(
            model[0],
            model.cell,
            gemmi.SpaceGroup("P 1"),
            SHAPE,
            D_MIN,
            CUTOFF,
            basis="coordinates",
            truncate_radius=0.05,
        )


def test_residue_rigid_gives_six_columns_per_multi_atom_group(model):
    """Single-atom groups have no meaningful rotation, so they contribute translation only."""
    basis = _basis(model, "residue_rigid")
    kinds = [kind for _, kind in basis.labels]
    assert set(kinds) <= {"t_x", "t_y", "t_z", "r_x", "r_y", "r_z"}
    # ALA(5 atoms) + GLY(4) + MET(1): two multi-atom groups x 6, one single-atom x 3.
    assert basis.n_columns == 2 * 6 + 1 * 3


def test_an_unknown_basis_is_rejected(model):
    with pytest.raises(ValueError, match="Unknown basis"):
        _basis(model, "everything")


def test_a_model_with_no_usable_atoms_is_rejected(tmp_path):
    empty = gemmi.Structure()
    empty.cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    empty.add_model(gemmi.Model("1"))
    with pytest.raises(ValueError, match="No atoms"):
        build_tangent_basis(empty[0], empty.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
