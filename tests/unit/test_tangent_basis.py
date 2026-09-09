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


def test_rigid_translation_matches_the_sum_of_atom_derivatives(model):
    """A rigid t_x column must equal the sum of the group's per-atom d(rho)/dx columns.

    Catches a sign flip or dropped atom in the translation branch of `_rigid_column`.
    """
    basis = _basis(model, "residue_rigid")
    # ALA 1 is the first (multi-atom, 5-atom) residue_rigid group -- group index 0.
    group_atoms = [atom for _, atom in selected_atoms(model[0])][:5]
    label_index = basis.labels.index((0, "t_x"))
    stored = np.asarray(basis.matrix[:, label_index].todense()).ravel().reshape(SHAPE)
    expected = np.zeros(SHAPE, dtype=np.float64)
    for atom in group_atoms:
        expected += tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    np.testing.assert_allclose(stored, expected, atol=1e-10)


def test_rigid_rotation_weights_sum_to_zero_net_translation(model):
    """The rotation-weight identity sum_i (axis x (r_i - centroid)) == 0 holds for any centroid.

    This is a mathematical identity of the mean, true for `cross(axis, d)` and equally for
    `cross(d, axis)` -- it does NOT pin the cross-product operand order (a bug there would
    still pass). It is kept only as a cheap, independent check that the centroid used here
    is in fact the mean position; the operand order and overall rotation math are covered
    separately by `test_rigid_rotation_column_matches_the_explicit_cross_product`.
    """
    # ALA 1 is the first (multi-atom, 5-atom) residue_rigid group.
    group_atoms = [atom for _, atom in selected_atoms(model[0])][:5]
    centroid = np.mean([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in group_atoms], axis=0)
    for axis_index in range(3):
        axis = np.zeros(3)
        axis[axis_index] = 1.0
        total = np.zeros(3)
        for atom in group_atoms:
            position = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
            total += np.cross(axis, position - centroid)
        np.testing.assert_allclose(total, np.zeros(3), atol=1e-10)


def test_rigid_rotation_column_matches_the_explicit_cross_product(model):
    """The stored r_z column must match the production `build_tangent_basis` output, checked
    against an expected value written out WITHOUT `np.cross` -- so the test cannot inherit
    a swapped-operand bug from the code it is meant to catch.

    For rotation about z, `cross(z_hat, d)` where `d = r_i - centroid` is `(-d_y, d_x, 0)`
    by definition of the cross product with a unit z axis. Writing that out literally means
    a reversed operand order in `_rigid_column` (giving `(d_y, -d_x, 0)`, the exact negation)
    would make this expected value diverge from the stored column and the test would fail.
    """
    basis = _basis(model, "residue_rigid")
    # ALA 1 is the first (multi-atom, 5-atom) residue_rigid group -- group index 0.
    group_atoms = [atom for _, atom in selected_atoms(model[0])][:5]
    centroid = np.mean([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in group_atoms], axis=0)

    label_index = basis.labels.index((0, "r_z"))
    stored = np.asarray(basis.matrix[:, label_index].todense()).ravel().reshape(SHAPE)

    expected = np.zeros(SHAPE, dtype=np.float64)
    for atom in group_atoms:
        d_x = atom.pos.x - centroid[0]
        d_y = atom.pos.y - centroid[1]
        # cross(z_hat, d) = (-d_y, d_x, 0): written out literally, not via np.cross.
        weight_x, weight_y = -d_y, d_x
        if weight_x != 0.0:
            expected += weight_x * tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
        if weight_y != 0.0:
            expected += weight_y * tangent_column(atom, "y", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    np.testing.assert_allclose(stored, expected, atol=1e-10)


def test_an_unknown_basis_is_rejected(model):
    with pytest.raises(ValueError, match="Unknown basis"):
        _basis(model, "everything")


def test_a_model_with_no_usable_atoms_is_rejected(tmp_path):
    empty = gemmi.Structure()
    empty.cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    empty.add_model(gemmi.Model("1"))
    with pytest.raises(ValueError, match="No atoms"):
        build_tangent_basis(empty[0], empty.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
