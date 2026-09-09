"""Decomposing a fitted correction onto the atomic tangent space.

This answers the manuscript's central claim directly: could ordinary refinement -- moving
atoms, changing B factors or occupancies -- have produced the density the field inferred?
The part it cannot explain is the object the project exists to find.

An explained fraction is never reported alone. With thousands of free parameters the
basis fits a great deal of anything, so every result is accompanied by what the same
basis explains of matched random fields.
"""

from __future__ import annotations

import numpy as np


def solve_normal_equations(gram, rhs, target_norm_squared, ridge: float = 0.0) -> dict:
    """Least squares from precomputed normal equations.

    `gram` is Phi^T Phi, `rhs` is Phi^T target. Working from the normal equations keeps
    memory at O(n_columns^2) rather than O(n_voxels x n_columns), which is what makes a
    few thousand columns affordable. lstsq is SVD-based, so a rank-deficient basis is
    resolved rather than producing a spurious solution.

    `condition_number` is the condition number of the normal-equations matrix (gram)
    itself, i.e. approximately the square of the basis Phi's own condition number --
    this function only ever sees gram, not Phi.
    """
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    data_gram = gram
    if ridge > 0.0:
        gram = gram + ridge * np.eye(gram.shape[0])

    amplitudes, _, rank, singular = np.linalg.lstsq(gram, rhs, rcond=None)

    # ||target - Phi a||^2 = ||target||^2 - 2 a.rhs + a.(Phi^T Phi a), expanded so the full
    # residual vector never has to be formed. This must use the ORIGINAL (unridged) gram --
    # the data residual, not the regularized objective the ridge term nudges the solve towards.
    residual = float(target_norm_squared) - 2.0 * float(amplitudes @ rhs) + float(amplitudes @ (data_gram @ amplitudes))
    residual = max(residual, 0.0)
    explained = 0.0 if target_norm_squared <= 0 else 1.0 - residual / float(target_norm_squared)

    positive = singular[singular > 0]
    condition = float(positive.max() / positive.min()) if positive.size else float("inf")
    return {
        "amplitudes": amplitudes,
        "explained_fraction": float(np.clip(explained, 0.0, 1.0)),
        "rank": int(rank),
        "condition_number": condition,
        "residual_norm_squared": residual,
    }
