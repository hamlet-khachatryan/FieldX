import json


def run_derivative_check(cfg, arrays, residuals_for_split, objective):
    import jax
    import jax.numpy as jnp

    z0 = jnp.zeros_like(arrays.rho0)
    key1, key2 = jax.random.split(jax.random.PRNGKey(cfg.run.seed + 17))
    v = jax.random.normal(key1, z0.shape, dtype=z0.dtype)
    v = v / jnp.linalg.norm(v.reshape(-1))

    value, grad = jax.value_and_grad(objective)(z0)
    directional_ad = jnp.vdot(grad.reshape(-1), v.reshape(-1)).real
    eps = jnp.asarray(2e-3 if z0.dtype == jnp.float32 else 1e-5, dtype=z0.dtype)
    directional_fd = (objective(z0 + eps * v) - objective(z0 - eps * v)) / (2 * eps)
    fd_rel = jnp.abs(directional_fd - directional_ad) / jnp.maximum(
        1.0, jnp.abs(directional_fd), jnp.abs(directional_ad)
    )

    def residual_train(z):
        return residuals_for_split(z, 0)[0]

    r0, pullback = jax.vjp(residual_train, z0)
    _, jv = jax.jvp(residual_train, (z0,), (v,))
    w = jax.random.normal(key2, r0.shape, dtype=r0.dtype)
    jtw = pullback(w)[0]
    lhs = jnp.vdot(jv.reshape(-1), w.reshape(-1)).real
    rhs = jnp.vdot(v.reshape(-1), jtw.reshape(-1)).real
    adj_rel = jnp.abs(lhs - rhs) / jnp.maximum(1.0, jnp.abs(lhs), jnp.abs(rhs))

    result = {
        "objective_at_zero": float(value),
        "directional_derivative_ad": float(directional_ad),
        "directional_derivative_fd": float(directional_fd),
        "directional_relative_error": float(fd_rel),
        "adjoint_lhs": float(lhs),
        "adjoint_rhs": float(rhs),
        "adjoint_relative_error": float(adj_rel),
    }
    fd_tol = 2e-2 if z0.dtype == jnp.float32 else 1e-5
    adj_tol = 2e-4 if z0.dtype == jnp.float32 else 1e-10
    result["pass"] = bool(float(fd_rel) < fd_tol and float(adj_rel) < adj_tol)
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run.output_dir / "derivative_check.json").write_text(json.dumps(result, indent=2))
    if not result["pass"]:
        raise RuntimeError("Derivative/adjoint check failed; fitting is blocked.")
    return result
