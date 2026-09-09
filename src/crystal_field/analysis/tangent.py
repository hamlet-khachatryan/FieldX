"""The atomic tangent-space operator Phi.

A column of Phi is the derivative of the unit-cell density with respect to one parameter
of one atom. Density is additive over atoms, so that derivative involves only the atom in
question: a structure containing just that atom yields the exact column, at a fraction of
the cost of rebuilding the whole model. Gemmi's density cutoff then truncates each column
to a neighbourhood of the atom and its symmetry copies, so the columns are naturally
sparse -- roughly 7% of the grid -- with no box arithmetic of our own.

This module knows nothing about fitted runs, targets or reporting. It answers only "what
is Phi".
"""

from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np
from scipy.sparse import csc_matrix, hstack

from crystal_field.crystallography.density import model_density_on_grid

# Central-difference steps, matching the existing atomic_benchmark configuration so the
# two diagnostics perturb atoms identically.
PARAMETER_STEPS = {
    "x": 0.02,
    "y": 0.02,
    "z": 0.02,
    "b_iso": 0.5,
    "occupancy": 0.02,
}


def single_atom_model(atom: gemmi.Atom) -> gemmi.Model:
    """A Model containing one atom, for computing that atom's density alone."""
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "UNK"
    residue.seqid = gemmi.SeqId(1, " ")
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    return model


def _read(atom, kind):
    if kind == "x":
        return atom.pos.x
    if kind == "y":
        return atom.pos.y
    if kind == "z":
        return atom.pos.z
    if kind == "b_iso":
        return atom.b_iso
    if kind == "occupancy":
        return atom.occ
    raise ValueError(f"Unknown tangent parameter: {kind}")


def _write(atom, kind, value):
    if kind == "x":
        atom.pos.x = value
    elif kind == "y":
        atom.pos.y = value
    elif kind == "z":
        atom.pos.z = value
    elif kind == "b_iso":
        atom.b_iso = value
    elif kind == "occupancy":
        atom.occ = value
    else:
        raise ValueError(f"Unknown tangent parameter: {kind}")


def tangent_column(atom, kind, cell, spacegroup, shape, d_min, cutoff, scattering="xray"):
    """d(rho)/d(kind) for one atom, on the full grid, including symmetry copies.

    The atom is restored to its original value before returning: a derivative that
    mutated the model would corrupt every column computed after it.
    """
    step = PARAMETER_STEPS.get(kind)
    if step is None:
        raise ValueError(f"Unknown tangent parameter: {kind}")

    original = _read(atom, kind)
    try:
        _write(atom, kind, original + step)
        plus, _ = model_density_on_grid(single_atom_model(atom), cell, spacegroup, shape, d_min, cutoff, scattering)
        _write(atom, kind, original - step)
        minus, _ = model_density_on_grid(single_atom_model(atom), cell, spacegroup, shape, d_min, cutoff, scattering)
    finally:
        _write(atom, kind, original)

    return (plus.astype(np.float64) - minus.astype(np.float64)) / (2.0 * step)


PARAMETER_SETS = {
    "coordinates": ("x", "y", "z"),
    "coordinates_b": ("x", "y", "z", "b_iso"),
    "full": ("x", "y", "z", "b_iso", "occupancy"),
}

RIGID_KINDS = ("t_x", "t_y", "t_z", "r_x", "r_y", "r_z")
TRANSLATION_KINDS = ("t_x", "t_y", "t_z")

# A stored column must keep essentially all of its norm, or the sparsity that makes the
# basis affordable is quietly discarding signal.
MIN_NORM_FRACTION = 0.999


@dataclass(frozen=True)
class TangentBasis:
    matrix: csc_matrix
    labels: list
    grid_shape: tuple
    basis: str
    n_columns: int
    min_norm_fraction: float


def selected_atoms(model):
    """Non-hydrogen atoms with positive occupancy, matching atomic_benchmark's selection."""
    return [
        (index, cra.atom)
        for index, cra in enumerate(cra for cra in model.all() if not cra.atom.element.is_hydrogen and cra.atom.occ > 0)
    ]


def _residue_groups(model):
    groups = {}
    for cra in model.all():
        if cra.atom.element.is_hydrogen or cra.atom.occ <= 0:
            continue
        groups.setdefault((cra.chain.name, cra.residue.seqid.num), []).append(cra.atom)
    return list(groups.values())


def _truncate(column, centre_pos, cell, shape, radius):
    """Zero everything beyond `radius` of `centre_pos`, for an explicit truncation test.

    `centre_pos` is a Position (Cartesian Angstrom) -- an atom's own position for the
    per-atom bases, or a rigid group's centroid for `residue_rigid`, so the truncation box
    is always centred on the same point the column's derivative was taken about.

    Only reachable when box_radius_angstrom is set. The default (None) keeps whatever
    Gemmi's density cutoff produced, which is already sparse.
    """
    if radius is None:
        return column
    fractional = cell.fractionalize(centre_pos)
    centre = (fractional.x, fractional.y, fractional.z)
    lengths = (cell.a, cell.b, cell.c)
    # Offset along each axis in Angstrom, wrapped into [-L/2, L/2) for periodicity.
    offsets = []
    for axis, (n, middle, length) in enumerate(zip(shape, centre, lengths, strict=True)):
        delta = ((np.arange(n) / n - middle + 0.5) % 1.0 - 0.5) * length
        offsets.append(delta.reshape(tuple(-1 if i == axis else 1 for i in range(3))))
    distance2 = offsets[0] ** 2 + offsets[1] ** 2 + offsets[2] ** 2
    return np.where(distance2 <= radius**2, column, 0.0)


def build_tangent_basis(
    model, cell, spacegroup, shape, d_min, cutoff, basis="coordinates", scattering="xray", truncate_radius=None
):
    """Assemble Phi as a sparse (n_voxels x n_columns) matrix."""
    shape = tuple(int(n) for n in shape)
    n_voxels = int(np.prod(shape))
    columns, labels, norm_fractions = [], [], []

    def add(atom, kind, label, vector=None, centre=None):
        dense = (
            tangent_column(atom, kind, cell, spacegroup, shape, d_min, cutoff, scattering) if vector is None else vector
        )
        full_norm = float(np.linalg.norm(dense))
        kept = _truncate(dense, centre if centre is not None else atom.pos, cell, shape, truncate_radius)
        kept_norm = float(np.linalg.norm(kept))
        norm_fractions.append(kept_norm / full_norm if full_norm > 0 else 1.0)
        columns.append(csc_matrix(kept.reshape(n_voxels, 1)))
        labels.append(label)

    if basis in PARAMETER_SETS:
        atoms = selected_atoms(model)
        if not atoms:
            raise ValueError("No atoms survive the selection (non-hydrogen, occupancy > 0)")
        for index, atom in atoms:
            for kind in PARAMETER_SETS[basis]:
                add(atom, kind, (index, kind))
    elif basis == "residue_rigid":
        groups = _residue_groups(model)
        if not groups:
            raise ValueError("No atoms survive the selection (non-hydrogen, occupancy > 0)")
        for index, atoms in enumerate(groups):
            kinds = RIGID_KINDS if len(atoms) > 1 else TRANSLATION_KINDS
            centroid = _group_centroid(atoms)
            for kind in kinds:
                add(
                    atoms[0],
                    _rigid_source_kind(kind),
                    (index, kind),
                    vector=_rigid_column(kind, atoms, centroid, cell, spacegroup, shape, d_min, cutoff, scattering),
                    centre=centroid,
                )
    else:
        raise ValueError(f"Unknown basis: {basis}")

    minimum = float(min(norm_fractions))
    if minimum < MIN_NORM_FRACTION:
        raise ValueError(
            f"Truncation keeps only {minimum:.4%} of a column's norm, below "
            f"{MIN_NORM_FRACTION:.1%}. Raise box_radius_angstrom or leave it null."
        )

    return TangentBasis(
        matrix=csc_matrix(hstack(columns)),
        labels=labels,
        grid_shape=shape,
        basis=basis,
        n_columns=len(labels),
        min_norm_fraction=minimum,
    )


def _rigid_source_kind(kind):
    """Rigid columns are assembled from per-atom columns; this names the driving parameter."""
    return {"t_x": "x", "t_y": "y", "t_z": "z", "r_x": "x", "r_y": "y", "r_z": "z"}[kind]


def _group_centroid(atoms):
    """The centroid Position (Cartesian Angstrom) of a rigid group's atoms.

    Shared by `_rigid_column`'s rotation math and the truncation centre in `build_tangent_basis`,
    so both agree on where the group is "centred".
    """
    coords = np.mean([[a.pos.x, a.pos.y, a.pos.z] for a in atoms], axis=0)
    return gemmi.Position(*coords)


def _rigid_column(kind, atoms, centroid, cell, spacegroup, shape, d_min, cutoff, scattering):
    """A group translation or rotation, as the sum of its atoms' coordinate derivatives.

    A rigid translation along an axis moves every atom identically. A rotation about an
    axis through the group centroid moves atom i by (axis x (r_i - centroid)), so its
    density derivative is that displacement contracted with the atom's coordinate
    derivatives.
    """
    centroid_array = np.array([centroid.x, centroid.y, centroid.z])
    axis_index = {"t_x": 0, "t_y": 1, "t_z": 2, "r_x": 0, "r_y": 1, "r_z": 2}[kind]
    total = np.zeros(tuple(int(n) for n in shape), dtype=np.float64)
    for atom in atoms:
        if kind.startswith("t_"):
            weights = np.zeros(3)
            weights[axis_index] = 1.0
        else:
            axis = np.zeros(3)
            axis[axis_index] = 1.0
            weights = np.cross(axis, np.array([atom.pos.x, atom.pos.y, atom.pos.z]) - centroid_array)
        for component, name in zip(weights, ("x", "y", "z"), strict=True):
            if component == 0.0:
                continue
            total += component * tangent_column(atom, name, cell, spacegroup, shape, d_min, cutoff, scattering)
    return total
