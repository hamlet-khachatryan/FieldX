"""Crystallographic symmetry operators for the reciprocal-space projection.

`symmetry_projected_fcalc` averages F over the space group, projecting an arbitrary
density onto the symmetry-invariant subspace. That average factorises:

    (1/|G|) sum_{s,j} exp(-2 pi i h.(t_s + c_j)) F(h R_s)
        = [ (1/n_sym) sum_s exp(-2 pi i h.t_s) F(h R_s) ]
          x [ (1/n_cen) sum_j exp(-2 pi i h.c_j) ]

because Gemmi composes a centred group as (R_s, t_s + c_j). The second factor is exactly
1 for a reflection allowed by the centring and exactly 0 for a systematically absent one
-- it *is* the absence indicator. `prepare_reflections` already removes absent
reflections, so on the retained set the centring operators contribute a factor of one and
can be dropped.

That is worth doing: F m -3 m has 192 operators but only 48 rotational ones, so the
projection loop shrinks fourfold on F-centred groups. The saving is exact, not an
approximation -- but it depends on the absence filter, so `include_centring=True` is kept
available and the equivalence is asserted in the tests.
"""

from __future__ import annotations

import gemmi
import numpy as np

# Gemmi stores rotations and translations as integers scaled by 24, which represents
# every crystallographic screw and centring translation exactly (1/2, 1/3, 1/4, 1/6).
DEN = 24


def _as_arrays(ops):
    rotations, translations = [], []
    for op in ops:
        raw = np.asarray(op.rot, dtype=np.int64)
        rotation = raw // DEN
        if not np.array_equal(raw, rotation * DEN):
            raise ValueError(f"Unexpected non-integral crystallographic rotation: {op.triplet()}")
        rotations.append(rotation.astype(np.int32))
        translations.append(np.asarray(op.tran, dtype=np.float64) / DEN)
    return np.stack(rotations), np.stack(translations)


def symmetry_arrays(spacegroup: gemmi.SpaceGroup, include_centring: bool = False):
    """Real-space integer rotations R and fractional translations t.

    By default only the rotational operators are returned, which is what the projection
    needs once systematically absent reflections have been removed. Pass
    `include_centring=True` for the full group, e.g. to verify that equivalence.
    """
    ops = spacegroup.operations()
    rotations, translations = _as_arrays(ops if include_centring else ops.sym_ops)
    # Translations are taken modulo one so a lattice shift never changes the phase.
    return rotations, np.mod(translations, 1.0)


def centring_translations(spacegroup: gemmi.SpaceGroup) -> np.ndarray:
    """Fractional centring vectors, including the identity."""
    return np.stack([np.asarray(c, dtype=np.float64) / DEN for c in spacegroup.operations().cen_ops])


def group_order(spacegroup: gemmi.SpaceGroup) -> int:
    """Total number of symmetry copies in the unit cell (rotational x centring)."""
    return len(spacegroup.operations())


def validate_grid_shape(shape, cell: gemmi.UnitCell, spacegroup: gemmi.SpaceGroup):
    """Reject a grid Gemmi cannot symmetrise, at configuration time rather than mid-run.

    Tetragonal, trigonal, hexagonal and cubic groups require equal sampling along
    symmetry-related axes; Gemmi otherwise raises "Grid must have the same size in
    symmetry-related directions" from inside the density stage.
    """
    shape = tuple(int(n) for n in shape)
    grid = gemmi.FloatGrid()
    grid.spacegroup = spacegroup
    grid.set_unit_cell(cell)
    try:
        grid.set_size(*shape)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"grid.shape {shape} is incompatible with space group {spacegroup.xhm()}: {exc}. "
            "Symmetry-related axes must be sampled equally; leave grid.shape null to let it be "
            "derived from the cell, d_min and grid.samples_per_dmin."
        ) from exc
    return (grid.nu, grid.nv, grid.nw)
