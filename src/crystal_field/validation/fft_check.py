import json

import gemmi
import numpy as np

# Gemmi's transform_map_to_f_phi() uses the crystallographic sign convention,
# F(h) = sum_r rho(r) exp(+2 pi i h.r), while fft_structure_factor_grid() is built on
# jnp.fft.fftn, i.e. exp(-2 pi i h.r). The two therefore agree up to a conjugation:
#
#     gemmi.transform_map_to_f_phi(rho) == conj(fftn(rho)) * (V / N)
#
# verified exactly (relative error ~1e-7) across several cells and grid shapes. The
# forward model is self-consistent in the fftn convention -- observables go through
# |F|, which conjugation leaves unchanged, and symmetry_projected_fcalc()'s exp(-2 pi i h.t)
# phase is only correct for the unconjugated grid -- so the gate must assert this
# relationship rather than demand a direct match. Requiring pass_direct made the gate
# unpassable (measured relative error ~1.5 against a 5e-4 tolerance), which blocked the
# whole afterok SLURM DAG at the numerics stage.
EXPECTED_GEMMI_RELATION = "conjugate"
TOLERANCE = 5e-4


def run_fft_check(cfg, arrays):
    import jax
    import jax.numpy as jnp

    from crystal_field.forward.diffraction import fft_structure_factor_grid, gather_hkl

    metadata = json.loads((cfg.run.data_dir / "metadata.json").read_text())
    rho = np.asarray(jax.device_get(arrays.rho0), dtype=np.float32)
    g = gemmi.FloatGrid(rho)
    g.spacegroup = gemmi.SpaceGroup(metadata["spacegroup"])
    g.set_unit_cell(gemmi.UnitCell(*metadata["cell"]))
    gf = gemmi.transform_map_to_f_phi(g)
    hkls = np.asarray(jax.device_get(arrays.hkls))
    step = max(1, len(hkls) // 1024)
    sample = hkls[::step][:1024]
    gemmi_f = np.array([gf.get_value(int(h), int(k), int(m)) for h, k, m in sample], dtype=np.complex64)
    jgrid = fft_structure_factor_grid(arrays.rho0, arrays.unit_cell_volume)
    jax_f = np.asarray(jax.device_get(gather_hkl(jgrid, jnp.asarray(sample, dtype=jnp.int32))))
    denom = max(np.linalg.norm(gemmi_f), 1e-30)
    rel_direct = float(np.linalg.norm(jax_f - gemmi_f) / denom)
    rel_conjugate = float(np.linalg.norm(np.conj(jax_f) - gemmi_f) / denom)
    errors = {"direct": rel_direct, "conjugate": rel_conjugate}
    matched = sorted([name for name, rel in errors.items() if rel < TOLERANCE])
    result = {
        "n_checked": len(sample),
        "expected_relation": EXPECTED_GEMMI_RELATION,
        "tolerance": TOLERANCE,
        "relative_l2_error_direct": rel_direct,
        "relative_l2_error_conjugated": rel_conjugate,
        "matched_relations": matched,
        "ambiguous": bool(len(matched) > 1),
        "pass_expected_relation": bool(EXPECTED_GEMMI_RELATION in matched),
        # Retained for artifact compatibility; the gate is pass_expected_relation.
        "pass_direct": bool(rel_direct < TOLERANCE),
        "pass_conjugated": bool(rel_conjugate < TOLERANCE),
    }
    (cfg.run.output_dir / "fft_check.json").parent.mkdir(parents=True, exist_ok=True)
    (cfg.run.output_dir / "fft_check.json").write_text(json.dumps(result, indent=2))
    if not result["pass_expected_relation"]:
        raise RuntimeError(
            f"JAX FFT does not match Gemmi under the expected {EXPECTED_GEMMI_RELATION!r} relation "
            f"(relative error direct={rel_direct:.3e}, conjugate={rel_conjugate:.3e}, "
            f"tolerance={TOLERANCE:.1e}). Fitting is blocked. If the Gemmi convention has changed, "
            "EXPECTED_GEMMI_RELATION and the phase sign in symmetry_projected_fcalc() must be "
            "updated together."
        )
    return result
