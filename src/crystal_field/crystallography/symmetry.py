from __future__ import annotations

import gemmi
import numpy as np

DEN = 24


def symmetry_arrays(spacegroup: gemmi.SpaceGroup):
    """Return real-space integer rotations R and fractional translations t."""
    rotations, translations = [], []
    for op in spacegroup.operations():
        raw = np.asarray(op.rot, dtype=np.int32)
        r = raw // DEN
        if not np.array_equal(raw, r * DEN):
            raise ValueError(f"Unexpected non-integral crystallographic rotation: {op.triplet()}")
        rotations.append(r)
        translations.append(np.asarray(op.tran, dtype=np.float64) / DEN)
    return np.stack(rotations), np.stack(translations)
