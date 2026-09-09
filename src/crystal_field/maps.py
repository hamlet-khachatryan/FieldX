"""CCP4 maps for visual inspection.

Two kinds of map are produced and they answer different questions.

*Calculated* maps -- the starting density, the inferred correction and their sum -- show
what the model is. They are smooth by construction: rho0 comes from ideal atomic form
factors, and the correction is band-limited and spatially correlated by design.

*Experimental-coefficient* maps -- 2Fo-Fc and Fo-Fc, computed from the measured
amplitudes with model phases -- show what the data say. These are the ones to compare
against a conventional 2Fo-Fc map, because they carry measurement noise and series
termination that no calculated density has. Comparing rho_refined to a deposited 2Fo-Fc
map is not a like-for-like comparison.

Every map is built from WORK reflections only. The free set is excluded even here: model
interpretation is one of the things it must stay out of.
"""

from __future__ import annotations

import json

import gemmi
import numpy as np

WORK = 2  # split id of the free set; work is everything else


def _write(array, path, cell, spacegroup):
    import reciprocalspaceship as rs

    rs.io.write_ccp4_map(np.ascontiguousarray(array, dtype=np.float32), str(path), cell, spacegroup)


def _reduce_to_asu(hkls, spacegroup):
    """Indices of one representative per symmetry-unique reflection, inside the ASU.

    ComplexAsuData expands what it is given by symmetry, so it must receive the
    asymmetric unit and nothing else. A reflection file that stores a full sphere -- both
    Friedel mates, as unmerged-then-merged data often does -- would otherwise be counted
    twice per equivalent and the map comes out wrong: a pure Fc map built that way
    correlates only ~0.12 with the density it came from.
    """
    asu = gemmi.ReciprocalAsu(spacegroup)
    operations = spacegroup.operations()
    keep, seen = [], set()
    for index, hkl in enumerate(hkls.tolist()):
        if not asu.is_in(hkl):
            continue
        key = tuple(hkl)
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)

    # Some symmetry-unique reflections may have no representative inside the ASU once the
    # free set is removed -- deposited free flags are not always Friedel-paired. That
    # leaves ordinary gaps in the map rather than an error, but it is worth reporting.
    unique = {tuple(asu.to_asu(hkl, operations)[0]) for hkl in hkls.tolist()}
    return np.asarray(keep, dtype=np.int64), len(unique) - len(seen)


def _asu_indices(hkls, spacegroup):
    return _reduce_to_asu(hkls, spacegroup)[0]


def _map_from_coefficients(hkls, coefficients, cell, spacegroup, shape):
    """Real-space map from complex Fourier coefficients on the measured reflections."""
    keep = _asu_indices(hkls, spacegroup)
    asu = gemmi.ComplexAsuData(
        cell,
        spacegroup,
        np.ascontiguousarray(hkls[keep], dtype=np.int32),
        np.ascontiguousarray(np.asarray(coefficients)[keep], dtype=np.complex64),
    )
    return np.array(asu.transform_f_phi_to_map(exact_size=tuple(int(n) for n in shape)), copy=True)


def _stats(array):
    array = np.asarray(array, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "rms": float(np.sqrt(np.mean(array**2))),
    }


def _sigmaa_reflection_set(cfg, sigmaa_from, split):
    """Which reflections may be used to estimate sigma-A, and whether that is allowed."""
    if sigmaa_from == "work":
        return split != WORK, "work reflections (biased: the model was fitted to them)"
    if sigmaa_from != "free":
        raise ValueError(f"sigma_a_from must be 'work' or 'free', not {sigmaa_from!r}")

    from crystal_field.analysis.model_selection import read_free_set_ledger

    if not read_free_set_ledger(cfg):
        raise RuntimeError(
            "sigma_a_from='free' requires the one-shot free-set evaluation to have happened "
            "already: estimating map weights on the free set before then would spend the "
            "held-out reflections on interpretation. Run `fieldrefine evaluate-free` first, "
            "or use the default sigma_a_from='work'."
        )
    return split == WORK, "free reflections (unbiased; the free set was already spent)"


def export_maps(cfg, sigma_a_from: str = "work", n_bins: int = 20) -> dict:
    """Write the map set for a fitted model into <output_dir>/maps/."""
    import jax
    import jax.numpy as jnp

    from crystal_field.inference.problem import build_complex_fcalc, build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    data_dir = cfg.run.data_dir
    metadata = json.loads((data_dir / "metadata.json").read_text())
    cell = gemmi.UnitCell(*metadata["cell"])
    spacegroup = gemmi.SpaceGroup(metadata["spacegroup"])
    shape = tuple(int(n) for n in metadata["grid_shape"])

    arrays = load_problem_arrays(cfg)
    *_, metrics, _ = build_functions(arrays, cfg)
    complex_fcalc, correction = build_complex_fcalc(arrays, cfg)

    z_path = cfg.run.output_dir / "fit" / "z_map.npy"
    if not z_path.exists():
        raise FileNotFoundError(f"{z_path} is missing; run `fieldrefine fit-map CONFIG` first")
    z = jnp.asarray(np.load(z_path), dtype=arrays.rho0.dtype)
    zero = jnp.zeros_like(z)

    rho0 = np.asarray(jax.device_get(arrays.rho0), dtype=np.float32)
    field = np.asarray(jax.device_get(correction(z)), dtype=np.float32)
    refined = rho0 + field

    work = np.asarray(jax.device_get(arrays.split)) != WORK
    hkls = np.asarray(jax.device_get(arrays.hkls))[work]
    fobs = np.asarray(jax.device_get(arrays.observation), dtype=np.float64)[work]

    asu_indices, missing_representatives = _reduce_to_asu(hkls, spacegroup)

    # Sigma-A weighting needs resolution, centricity and the epsilon factor per reflection.
    from crystal_field.analysis.sigmaa import apply_sigmaa, estimate_sigmaa
    from crystal_field.analysis.sigmaa import map_coefficients as weighted_coefficients

    reflection_file = np.load(data_dir / "reflections.npz")
    d_all = np.asarray(reflection_file["dHKL"], dtype=np.float64)
    all_hkls = np.asarray(reflection_file["hkls"], dtype=np.int32)
    operations = spacegroup.operations()
    centric_all = np.asarray(operations.centric_flag_array(all_hkls), dtype=bool)
    epsilon_all = np.asarray(operations.epsilon_factor_array(all_hkls), dtype=np.float64)

    split_all = np.asarray(jax.device_get(arrays.split))
    estimator_mask, sigmaa_provenance = _sigmaa_reflection_set(cfg, sigma_a_from, split_all)

    out = cfg.run.output_dir / "maps"
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    def coefficient_map(z_value, name, weight):
        fcalc = np.asarray(jax.device_get(complex_fcalc(z_value)), dtype=np.complex128)[work]
        # The forward model is built on jnp.fft.fftn, i.e. exp(-2 pi i h.r), while Gemmi's
        # transform_f_phi_to_map expects the crystallographic exp(+2 pi i h.r). Handing
        # our coefficients over unconjugated negates every phase; the resulting map
        # correlates ~0.12 with the density it came from instead of ~0.87.
        fcalc = np.conj(fcalc)
        scale = float(metrics(z_value)["scale"])
        amplitude = scale * np.abs(fcalc)
        phase = np.exp(1j * np.angle(fcalc))
        # Unweighted: sigma-A weights are normally estimated from the free set, which is
        # exactly what must not happen here. These are plain (2Fo - Fc) and (Fo - Fc).
        coefficients = (weight * fobs - amplitude) * phase
        grid = _map_from_coefficients(hkls, coefficients, cell, spacegroup, shape)
        _write(grid, out / f"{name}.ccp4", cell, spacegroup)
        written[name] = _stats(grid)
        return grid

    _write(rho0, out / "rho0.ccp4", cell, spacegroup)
    written["rho0"] = _stats(rho0)
    _write(field, out / "field.ccp4", cell, spacegroup)
    written["field"] = _stats(field)
    _write(refined, out / "refined.ccp4", cell, spacegroup)
    written["refined"] = _stats(refined)

    coefficient_map(z, "2FoFc_field", 2.0)
    coefficient_map(zero, "2FoFc_atomic", 2.0)
    coefficient_map(z, "FoFc_field", 1.0)
    coefficient_map(zero, "FoFc_atomic", 1.0)

    def weighted_map(z_value, label):
        """Sigma-A weighted 2mFo-DFc and mFo-DFc for one model."""
        fcalc = np.conj(np.asarray(jax.device_get(complex_fcalc(z_value)), dtype=np.complex128))
        scale = float(metrics(z_value)["scale"])
        scaled_fcalc = scale * fcalc
        observed = np.asarray(jax.device_get(arrays.observation), dtype=np.float64)

        estimate = estimate_sigmaa(
            observed[estimator_mask],
            np.abs(scaled_fcalc)[estimator_mask],
            d_all[estimator_mask],
            centric_all[estimator_mask],
            epsilon_all[estimator_mask],
            n_bins=n_bins,
        )
        applied = apply_sigmaa(estimate["shells"], observed, np.abs(scaled_fcalc), d_all, centric_all, epsilon_all)

        for kind, name in (("2mFo-DFc", f"2mFoDFc_{label}"), ("mFo-DFc", f"mFoDFc_{label}")):
            coefficients = weighted_coefficients(
                observed[work],
                scaled_fcalc[work],
                {"m": applied["m"][work], "D": applied["D"][work]},
                centric_all[work],
                kind,
            )
            grid = _map_from_coefficients(hkls, coefficients, cell, spacegroup, shape)
            _write(grid, out / f"{name}.ccp4", cell, spacegroup)
            written[name] = _stats(grid)
        return estimate, applied

    field_weights, field_applied = weighted_map(z, "field")
    weighted_map(zero, "atomic")

    manifest = {
        "directory": str(out),
        "grid_shape": list(shape),
        "spacegroup": metadata["spacegroup"],
        "cell": metadata["cell"],
        "n_reflections_used": int(work.sum()),
        "n_reflections_in_asu": len(asu_indices),
        "n_unique_without_asu_representative": missing_representatives,
        "reflections_used": "work only (train + tune); the free set is excluded from map calculation",
        "weighting": "unweighted 2Fo-Fc and Fo-Fc; sigma-A weights would need the free set",
        "phase_convention": "coefficients conjugated into Gemmi's exp(+2 pi i h.r) convention",
        "sigma_a": {
            "estimated_on": sigmaa_provenance,
            "source": sigma_a_from,
            "n_reflections": int(estimator_mask.sum()),
            "n_bins": len(field_weights["shells"]),
            "mean_figure_of_merit": float(np.mean(field_applied["m"][work])),
            "shells": field_weights["shells"],
        },
        "maps": {
            "rho0.ccp4": "starting atomistic density (calculated)",
            "field.ccp4": "the inferred correction L z on its own -- what FieldX added",
            "refined.ccp4": "rho0 + L z (calculated)",
            "2FoFc_field.ccp4": "experimental amplitudes with FieldX phases",
            "2FoFc_atomic.ccp4": "experimental amplitudes with atomic-model phases, for comparison",
            "FoFc_field.ccp4": "difference map with FieldX phases",
            "FoFc_atomic.ccp4": "difference map with atomic-model phases, for comparison",
            "2mFoDFc_field.ccp4": "sigma-A weighted 2mFo-DFc with FieldX phases",
            "2mFoDFc_atomic.ccp4": "sigma-A weighted 2mFo-DFc with atomic-model phases",
            "mFoDFc_field.ccp4": "sigma-A weighted difference map with FieldX phases",
            "mFoDFc_atomic.ccp4": "sigma-A weighted difference map with atomic-model phases",
        },
        "statistics": written,
        "note": (
            "Compare 2FoFc_field against 2FoFc_atomic, not against refined.ccp4: the "
            "coefficient maps carry experimental noise, the calculated ones do not."
        ),
    }
    (out / "maps.json").write_text(json.dumps(manifest, indent=2))
    return manifest
