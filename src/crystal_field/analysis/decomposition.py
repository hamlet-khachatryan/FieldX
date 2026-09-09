"""Decomposing a fitted correction onto the atomic tangent space.

This answers the manuscript's central claim directly: could ordinary refinement -- moving
atoms, changing B factors or occupancies -- have produced the density the field inferred?
The part it cannot explain is the object the project exists to find.

An explained fraction is never reported alone. With thousands of free parameters the
basis fits a great deal of anything, so every result is accompanied by what the same
basis explains of matched random fields.
"""

from __future__ import annotations

import gemmi
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


def symmetrize_grid(array, cell, spacegroup):
    """Average a grid over the space group.

    Only the symmetric component of the correction reaches F_calc through
    symmetry_projected_fcalc, and every Phi column is symmetric by construction, so the
    antisymmetric part is orthogonal to the basis and would otherwise register as
    permanently unexplained density that no basis could reach.
    """
    grid = gemmi.FloatGrid(np.ascontiguousarray(array, dtype=np.float32))
    grid.set_unit_cell(cell)
    grid.spacegroup = spacegroup
    grid.symmetrize_avg()
    return np.array(grid.array, dtype=np.float64, copy=True)


def split_symmetric(array, cell, spacegroup):
    """Return the symmetric component and the norm fraction carried by the rest."""
    array = np.asarray(array, dtype=np.float64)
    symmetric = symmetrize_grid(array, cell, spacegroup)
    total = float(np.linalg.norm(array))
    antisymmetric = float(np.linalg.norm(array - symmetric))
    return symmetric, (antisymmetric / total if total > 0 else 0.0)


def correction_from_fit(cfg, arrays):
    """Recover u = L z on the grid from the fitted latent field."""
    import jax
    import jax.numpy as jnp

    from crystal_field.model.prior import apply_transfer, build_transfer

    path = cfg.run.output_dir / "fit" / "z_map.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `fieldrefine fit-map CONFIG` first")

    transfer = build_transfer(
        tuple(arrays.rho0.shape),
        arrays.reciprocal_metric,
        arrays.unit_cell_volume,
        arrays.d_min_angstrom,
        cfg.prior,
        arrays.rho0.dtype,
    )
    z = jnp.asarray(np.load(path), dtype=arrays.rho0.dtype)
    return np.asarray(jax.device_get(apply_transfer(z, transfer, arrays.unit_cell_volume)), dtype=np.float64)


def decompose_target(basis, target, ridge: float = 0.0) -> dict:
    """Least-squares projection of a grid-shaped target onto the basis."""
    flat = np.asarray(target, dtype=np.float64).reshape(-1)
    if flat.size != basis.matrix.shape[0]:
        raise ValueError(f"Target has {flat.size} voxels but the basis expects {basis.matrix.shape[0]}")

    gram = np.asarray((basis.matrix.T @ basis.matrix).todense(), dtype=np.float64)
    rhs = np.asarray(basis.matrix.T @ flat, dtype=np.float64).ravel()
    result = solve_normal_equations(gram, rhs, float(flat @ flat), ridge=ridge)

    explained = np.asarray(basis.matrix @ result["amplitudes"]).reshape(basis.grid_shape)
    result["explained"] = explained
    result["unexplained"] = np.asarray(target, dtype=np.float64) - explained
    return result


def capacity_control(
    basis, reference_norm, transfer, unit_cell_volume, cell, spacegroup, seed, n_trials, ridge: float = 0.0
) -> dict:
    """What the same basis explains of matched random fields.

    An explained fraction on its own says nothing: with thousands of free parameters the
    basis fits a great deal of anything. Each trial draws a field through the *same*
    prior operator the fit used, scales it to the correction's norm, symmetrizes it, and
    decomposes it. If the control scores 0.70, a real score of 0.75 is not a finding.
    """
    fractions = []
    for trial in range(int(n_trials)):
        rng = np.random.default_rng(int(seed) + 1000 + trial)
        draw = rng.standard_normal(basis.grid_shape)
        if transfer is not None:
            import jax
            import jax.numpy as jnp

            from crystal_field.model.prior import apply_transfer

            draw = np.asarray(
                jax.device_get(apply_transfer(jnp.asarray(draw, dtype=transfer.dtype), transfer, unit_cell_volume)),
                dtype=np.float64,
            )
        draw = symmetrize_grid(draw, cell, spacegroup)
        norm = float(np.linalg.norm(draw))
        if norm > 0:
            draw *= float(reference_norm) / norm
        fractions.append(decompose_target(basis, draw, ridge=ridge)["explained_fraction"])

    return {
        "fractions": [float(f) for f in fractions],
        "mean": float(np.mean(fractions)),
        "sd": float(np.std(fractions)),
        "n_trials": int(n_trials),
    }


def _data_supported_target(cfg, arrays, basis, correction):
    """Delta_F on the work reflections, decomposed against the columns' structure factors.

    Only about one band-limited frequency in seven has a measured reflection behind it
    for a typical dataset; the rest of the correction is prior interpolation. This target
    isolates the part the data actually supports.
    """
    import jax
    import jax.numpy as jnp

    from crystal_field.forward.diffraction import fft_structure_factor_grid, symmetry_projected_fcalc

    work = np.asarray(jax.device_get(arrays.split)) != 2
    hkls = jnp.asarray(np.asarray(jax.device_get(arrays.hkls))[work], dtype=jnp.int32)

    def structure_factors(grid):
        transformed = fft_structure_factor_grid(jnp.asarray(grid, dtype=arrays.rho0.dtype), arrays.unit_cell_volume)
        return np.asarray(
            jax.device_get(
                symmetry_projected_fcalc(transformed, hkls, arrays.symmetry_rotations, arrays.symmetry_translations)
            ),
            dtype=np.complex128,
        )

    delta_f = structure_factors(correction)

    # One FFT per column. The stacked matrix is (n_reflections x n_columns) complex --
    # about 190 MB at 1UBQ's 6029 reflections and 1980 columns, which is affordable; if a
    # larger dataset makes it not, accumulate gram and rhs inside the loop instead.
    columns = [
        structure_factors(np.asarray(basis.matrix[:, index].todense()).reshape(basis.grid_shape))
        for index in range(basis.n_columns)
    ]
    stacked = np.stack(columns, axis=1)
    # a is real, so the least-squares normal equations take the real part.
    gram = np.real(stacked.conj().T @ stacked)
    rhs = np.real(stacked.conj().T @ delta_f)

    result = solve_normal_equations(gram, rhs, float(np.real(delta_f.conj() @ delta_f)), ridge=cfg.decomposition.ridge)
    return {
        "explained_fraction": result["explained_fraction"],
        "rank": result["rank"],
        "condition_number": result["condition_number"],
        "n_reflections": int(work.sum()),
        "target_norm": float(np.linalg.norm(delta_f)),
    }


def run_decomposition(cfg, basis: str | None = None, n_trials: int | None = None) -> dict:
    """Decompose a fitted correction onto the atomic tangent space and write the report."""
    import json

    from crystal_field.analysis.tangent import build_tangent_basis
    from crystal_field.crystallography.io import _model_for_metadata
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.maps import _write
    from crystal_field.model.prior import build_transfer

    if not cfg.decomposition.enabled:
        # v3 is the special case Phi = 0 (spec section 9.4): the correction is reported
        # whole and nothing is claimed to be explained.
        return {
            "enabled": False,
            "basis": {"name": None, "n_columns": 0},
            "targets": {"full_correction": {"explained_fraction": 0.0}},
            "note": "decomposition.enabled is false; the whole correction is unexplained by definition.",
        }

    metadata = json.loads((cfg.run.data_dir / "metadata.json").read_text())
    structure, cell, spacegroup = _model_for_metadata(cfg, metadata)
    shape = tuple(int(n) for n in metadata["grid_shape"])

    arrays = load_problem_arrays(cfg)
    correction = correction_from_fit(cfg, arrays)
    symmetric, antisymmetric_fraction = split_symmetric(correction, cell, spacegroup)

    basis_name = basis or cfg.decomposition.basis
    tangent = build_tangent_basis(
        structure[0],
        cell,
        spacegroup,
        shape,
        cfg.resolution.d_min_angstrom,
        cfg.baseline.density_cutoff,
        basis=basis_name,
        scattering=cfg.input.scattering,
        truncate_radius=cfg.decomposition.box_radius_angstrom,
    )

    full = decompose_target(tangent, symmetric, ridge=cfg.decomposition.ridge)
    data_supported = _data_supported_target(cfg, arrays, tangent, symmetric)

    transfer = build_transfer(
        tuple(arrays.rho0.shape),
        arrays.reciprocal_metric,
        arrays.unit_cell_volume,
        arrays.d_min_angstrom,
        cfg.prior,
        arrays.rho0.dtype,
    )
    control = capacity_control(
        tangent,
        float(np.linalg.norm(symmetric)),
        transfer,
        arrays.unit_cell_volume,
        cell,
        spacegroup,
        cfg.run.seed,
        n_trials or cfg.decomposition.n_capacity_trials,
        ridge=cfg.decomposition.ridge,
    )

    out = cfg.run.output_dir / "decomposition"
    out.mkdir(parents=True, exist_ok=True)
    if cfg.decomposition.write_maps:
        _write(full["explained"], out / "explained.ccp4", cell, spacegroup)
        _write(full["unexplained"], out / "unexplained.ccp4", cell, spacegroup)

    by_kind = {}
    for (_, kind), amplitude in zip(tangent.labels, full["amplitudes"], strict=True):
        by_kind.setdefault(kind, []).append(float(amplitude))

    report = {
        "enabled": True,
        "basis": {
            "name": basis_name,
            "n_columns": tangent.n_columns,
            "rank": full["rank"],
            "condition_number": full["condition_number"],
            "min_norm_fraction": tangent.min_norm_fraction,
        },
        "starting_density": cfg.baseline.starting_density,
        "antisymmetric_fraction": antisymmetric_fraction,
        "targets": {
            "full_correction": {
                "explained_fraction": full["explained_fraction"],
                "target_norm": float(np.linalg.norm(symmetric)),
                "n_voxels": int(symmetric.size),
            },
            "data_supported": data_supported,
        },
        "amplitude_rms_by_parameter": {
            kind: float(np.sqrt(np.mean(np.square(values)))) for kind, values in by_kind.items()
        },
        "capacity_control": control,
        "verdict": {
            "explained_above_control": full["explained_fraction"] - control["mean"],
            "note": (
                "An explained fraction is only meaningful against its control. A real "
                "score close to the control's mean means the basis is fitting capacity, "
                "not structure."
            ),
        },
        "note": (
            "Read-only diagnostic. The target is symmetrize(u): only the symmetric part of "
            "the correction reaches F_calc, and every Phi column is symmetric."
        ),
    }
    (out / "decomposition.json").write_text(json.dumps(report, indent=2))
    return report
