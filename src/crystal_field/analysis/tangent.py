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

import gemmi
import numpy as np

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
