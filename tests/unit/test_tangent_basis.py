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

from crystal_field.analysis.tangent import PARAMETER_STEPS, single_atom_model, tangent_column
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
