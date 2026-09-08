import json

import numpy as np


def run_information_spectrum(cfg, arrays, residuals_for_split):
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse.linalg import lobpcg_standard

    z0 = jnp.zeros_like(arrays.rho0)

    def residual_work(z):
        r0 = residuals_for_split(z, 0)[0]
        r1 = residuals_for_split(z, 1)[0]
        return r0 + r1

    _, jvp = jax.linearize(residual_work, z0)
    jtv = jax.linear_transpose(jvp, z0)

    def action_vec(x):
        xf = x.reshape(z0.shape)
        return jtv(jvp(xf))[0].reshape(-1)

    def action(x):
        if x.ndim == 1:
            return action_vec(x)
        return jax.lax.map(action_vec, x.T).T

    # Fail before allocating a 3k-wide basis of full latent fields, not hours in.
    from crystal_field.hpc import check_information_budget

    budget = check_information_budget(cfg)
    n, k = z0.size, cfg.information.n_modes
    x0 = jax.random.normal(jax.random.PRNGKey(cfg.run.seed), (n, k), dtype=z0.dtype)
    theta, vecs, iters = lobpcg_standard(action, x0, m=cfg.information.max_iterations, tol=cfg.information.tolerance)
    theta_np = np.asarray(jax.device_get(theta))
    vecs_np = np.asarray(jax.device_get(vecs), dtype=np.float32)
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.run.output_dir / "info_spectrum.npz",
        eigenvalues=theta_np,
        eigenvectors_latent=vecs_np,
        grid_shape=np.asarray(z0.shape, dtype=np.int32),
    )
    summary = {
        "n_modes": int(k),
        "estimated_gpu_GiB": budget["estimated_gpu_GiB_information"],
        "lobpcg_iterations": int(iters),
        "eigenvalues": theta_np.tolist(),
        "d_eff_lower_bound_from_top_modes": float(np.sum(theta_np / (1 + theta_np))),
        "information_lower_bound_from_top_modes_nats": float(0.5 * np.sum(np.log1p(theta_np))),
        "note": "Lower bounds from leading work-set modes only; free reflections are excluded.",
    }
    (cfg.run.output_dir / "info_spectrum.json").write_text(json.dumps(summary, indent=2))
    return summary
