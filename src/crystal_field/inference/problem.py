from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemArrays:
    rho0: object
    solvent_mask: object
    hkls: object
    observation: object
    sigma: object
    split: object
    reciprocal_metric: object
    symmetry_rotations: object
    symmetry_translations: object
    overall_scale: object
    solvent_scale: object
    unit_cell_volume: float
    d_min_angstrom: float
    observation_kind: str


def build_functions(arrays, cfg):
    import jax.numpy as jnp

    from crystal_field.analysis.metrics import amplitude_r_factor, observable_to_amplitude
    from crystal_field.forward.diffraction import (
        fft_structure_factor_grid,
        observable_from_fcalc,
        symmetry_projected_fcalc,
    )
    from crystal_field.model.prior import apply_transfer, build_transfer, latent_penalty

    transfer = build_transfer(
        tuple(arrays.rho0.shape),
        arrays.reciprocal_metric,
        arrays.unit_cell_volume,
        arrays.d_min_angstrom,
        cfg.prior,
        arrays.rho0.dtype,
    )
    solvent_grid = fft_structure_factor_grid(arrays.solvent_mask, arrays.unit_cell_volume)
    solvent_f = symmetry_projected_fcalc(
        solvent_grid,
        arrays.hkls,
        arrays.symmetry_rotations,
        arrays.symmetry_translations,
    )

    def density_from_z(z):
        return arrays.rho0 + apply_transfer(z, transfer, arrays.unit_cell_volume)

    def complex_fcalc(z):
        fgrid = fft_structure_factor_grid(density_from_z(z), arrays.unit_cell_volume)
        fcryst = symmetry_projected_fcalc(
            fgrid,
            arrays.hkls,
            arrays.symmetry_rotations,
            arrays.symmetry_translations,
        )
        if cfg.baseline.bulk_solvent.enabled:
            fcryst = fcryst + arrays.solvent_scale * solvent_f
        return arrays.overall_scale * fcryst

    def predict_all(z):
        return observable_from_fcalc(complex_fcalc(z), arrays.observation_kind)

    sigma_eff = jnp.maximum(arrays.sigma, cfg.likelihood.sigma_floor)

    def scope_mask(scope):
        if scope == "train":
            return arrays.split == 0
        if scope == "work":
            return arrays.split != 2
        raise ValueError(scope)

    fit_mask = scope_mask(cfg.optimizer.fit_scope)

    def scale_for(pred, mask):
        if cfg.likelihood.global_scale == "fixed":
            return jnp.asarray(cfg.likelihood.fixed_scale, dtype=pred.dtype)
        w = mask.astype(pred.dtype) / (sigma_eff * sigma_eff)
        num = jnp.sum(w * pred * arrays.observation)
        den = jnp.sum(w * pred * pred)
        return jnp.maximum(num / jnp.maximum(den, jnp.finfo(pred.dtype).tiny), 0.0)

    def full_residuals(z):
        pred = predict_all(z)
        scale = scale_for(pred, fit_mask)
        return (scale * pred - arrays.observation) / sigma_eff, scale, pred

    def data_loss(r, mask):
        m = mask.astype(r.dtype)
        if cfg.likelihood.loss == "gaussian":
            return 0.5 * jnp.sum(m * r * r)
        nu = jnp.asarray(cfg.likelihood.student_t_df, dtype=r.dtype)
        return 0.5 * (nu + 1.0) * jnp.sum(m * jnp.log1p((r * r) / nu))

    def residuals_for_split(z, split_id):
        r, scale, _ = full_residuals(z)
        mask = (arrays.split == split_id).astype(r.dtype)
        return r * mask, scale

    def objective(z):
        r, _, _ = full_residuals(z)
        return data_loss(r, fit_mask) + latent_penalty(z, cfg.prior)

    def split_metrics(r, pred, scale, mask, prefix):
        m = mask.astype(r.dtype)
        n = jnp.sum(m)
        chi2 = jnp.sum(m * r * r)
        fobs, fcalc = observable_to_amplitude(arrays.observation, pred, arrays.observation_kind, scale)
        return {
            f"chi2_{prefix}": chi2,
            f"rmsz_{prefix}": jnp.sqrt(chi2 / jnp.maximum(n, 1.0)),
            f"r_{prefix}": amplitude_r_factor(fobs, fcalc, mask),
            f"n_{prefix}": n,
        }

    def metrics(z):
        r, scale, pred = full_residuals(z)
        result = {"scale": scale, "prior_penalty": latent_penalty(z, cfg.prior), "objective": objective(z)}
        result.update(split_metrics(r, pred, scale, arrays.split == 0, "train"))
        result.update(split_metrics(r, pred, scale, arrays.split == 1, "tune"))
        result.update(split_metrics(r, pred, scale, arrays.split != 2, "work"))
        return result

    def free_metrics(z):
        r, scale, pred = full_residuals(z)
        result = {"scale_from_work": scale}
        result.update(split_metrics(r, pred, scale, arrays.split == 2, "free"))
        return result

    return transfer, density_from_z, predict_all, residuals_for_split, objective, metrics, free_metrics
