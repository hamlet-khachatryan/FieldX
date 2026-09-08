"""Locks the Gemmi/JAX Fourier sign convention that the fft-check gate asserts.

These two invariants are coupled: fft_structure_factor_grid() uses jnp.fft.fftn, i.e.
exp(-2 pi i h.r), while Gemmi uses exp(+2 pi i h.r), and symmetry_projected_fcalc()'s
exp(-2 pi i h.t) phase is only correct for the unconjugated grid. Changing either one
without the other silently corrupts F_calc, so both are pinned here rather than being
checked only by a cluster job.
"""

import gemmi
import numpy as np
import pytest

from crystal_field.crystallography.symmetry import symmetry_arrays
from crystal_field.validation.fft_check import EXPECTED_GEMMI_RELATION, TOLERANCE

jnp = pytest.importorskip("jax.numpy")

from crystal_field.forward.diffraction import (  # noqa: E402  (needs the jax import guard above)
    fft_structure_factor_grid,
    gather_hkl,
    symmetry_projected_fcalc,
)

HKLS = np.array([[1, 2, 3], [5, -2, 4], [0, 1, 0], [7, 7, 7], [-3, 5, -6]], dtype=np.int32)


def _gemmi_structure_factors(rho, cell, sg, hkls):
    grid = gemmi.FloatGrid(np.ascontiguousarray(rho, dtype=np.float32))
    grid.spacegroup = sg
    grid.set_unit_cell(cell)
    gf = gemmi.transform_map_to_f_phi(grid)
    return np.array([gf.get_value(int(h), int(k), int(m)) for h, k, m in hkls], dtype=np.complex128)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@pytest.mark.parametrize("shape", [(16, 16, 16), (30, 24, 20)])
def test_gemmi_equals_conjugate_of_fftn_grid(shape):
    cell = gemmi.UnitCell(20, 20, 20, 90, 90, 90)
    sg = gemmi.SpaceGroup("P 1")
    rho = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    expected = _gemmi_structure_factors(rho, cell, sg, HKLS)

    grid = fft_structure_factor_grid(jnp.asarray(rho), cell.volume)
    mine = np.asarray(gather_hkl(grid, jnp.asarray(HKLS, dtype=jnp.int32)))

    assert EXPECTED_GEMMI_RELATION == "conjugate"
    assert _rel(np.conj(mine), expected) < TOLERANCE
    # The direct comparison must NOT pass, or the gate's assertion is vacuous.
    assert _rel(mine, expected) > 0.1


def test_symmetry_projection_sign_pairs_with_the_fftn_convention():
    """On a P4_1-symmetric density the projection must reproduce F(h) itself.

    P2_12_12_1 cannot discriminate the variants (diagonal +/-1 rotations, half-integer
    translations), so this uses P4_1, whose rotation is non-symmetric with t = (0, 0, 1/4).
    """
    cell = gemmi.UnitCell(20, 20, 30, 90, 90, 90)
    sg = gemmi.SpaceGroup("P 41")
    rng = np.random.default_rng(7)

    grid = gemmi.FloatGrid(rng.standard_normal((24, 24, 32)).astype(np.float32))
    grid.spacegroup = sg
    grid.set_unit_cell(cell)
    grid.symmetrize_sum()
    rho = jnp.asarray(np.array(grid.array, dtype=np.float32, copy=True))

    fgrid = fft_structure_factor_grid(rho, cell.volume)
    hkls = jnp.asarray(HKLS, dtype=jnp.int32)
    reference = np.asarray(gather_hkl(fgrid, hkls))

    rotations, translations = symmetry_arrays(sg)
    as_implemented = np.asarray(symmetry_projected_fcalc(fgrid, hkls, rotations, translations))
    assert _rel(as_implemented, reference) < 1e-3

    # Flipping exactly one of the two must be plainly wrong, not marginally so. This is
    # what pins the phase sign against the rotation handedness.
    for label, rots, trans in [
        ("transposed_rotation", np.transpose(rotations, (0, 2, 1)), translations),
        ("flipped_phase", rotations, -translations),
    ]:
        wrong = np.asarray(symmetry_projected_fcalc(fgrid, hkls, rots, trans))
        assert _rel(wrong, reference) > 0.1, label

    # Flipping BOTH is not an error: it is the same projection expressed in the conjugate
    # convention. The rotations are orthogonal in fractional coordinates, so R^T = R^-1 and
    # summing over (R^T, +t) permutes the sum over the group. This is precisely why
    # conjugating fft_structure_factor_grid() would require flipping the phase sign too.
    equivalent = np.asarray(symmetry_projected_fcalc(fgrid, hkls, np.transpose(rotations, (0, 2, 1)), -translations))
    assert _rel(equivalent, reference) < 1e-3
