"""Derivative and adjoint gates.

Nothing is fitted until finite differences agree with autodiff and the adjoint
identity <Jv, w> = <v, J^T w> holds. Both J and J^T are applied matrix-free.
"""

import json


def _relative(a, b):
    import jax.numpy as jnp

    # jnp.maximum is strictly binary; a three-argument call raises TypeError, which
    # previously made this gate crash rather than report.
    scale = jnp.maximum(jnp.maximum(jnp.abs(a), jnp.abs(b)), 1.0)
    return jnp.abs(a - b) / scale


def run_derivative_check(cfg, arrays, residuals_for_split, objective):
    import jax
    import jax.numpy as jnp

    z0 = jnp.zeros_like(arrays.rho0)
    key1, key2 = jax.random.split(jax.random.PRNGKey(cfg.run.seed + 17))
    v = jax.random.normal(key1, z0.shape, dtype=z0.dtype)
    v = v / jnp.linalg.norm(v.reshape(-1))

    value, grad = jax.value_and_grad(objective)(z0)
    directional_ad = jnp.vdot(grad.reshape(-1), v.reshape(-1)).real

    # The objective is a chi-squared sum over thousands of reflections, so it is orders
    # of magnitude larger than its directional derivative. A fixed small step makes the
    # central difference pure cancellation noise in float32. The step below balances
    # roundoff against truncation: eps ~ cbrt(u |f| / |f'|), which is the standard
    # optimum for a central difference of a function evaluated to relative accuracy u.
    unit_roundoff = float(jnp.finfo(z0.dtype).eps)
    eps = float(
        max(
            1e-4,
            (unit_roundoff * max(abs(float(value)), 1.0) / max(abs(float(directional_ad)), 1.0)) ** (1.0 / 3.0),
        )
    )
    eps = jnp.asarray(eps, dtype=z0.dtype)
    directional_fd = (objective(z0 + eps * v) - objective(z0 - eps * v)) / (2 * eps)
    fd_rel = _relative(directional_fd, directional_ad)

    def residual_train(z):
        return residuals_for_split(z, 0)[0]

    r0, pullback = jax.vjp(residual_train, z0)
    _, jv = jax.jvp(residual_train, (z0,), (v,))
    w = jax.random.normal(key2, r0.shape, dtype=r0.dtype)
    jtw = pullback(w)[0]
    lhs = jnp.vdot(jv.reshape(-1), w.reshape(-1)).real
    rhs = jnp.vdot(v.reshape(-1), jtw.reshape(-1)).real
    adj_rel = _relative(lhs, rhs)

    result = {
        "objective_at_zero": float(value),
        "finite_difference_step": float(eps),
        "directional_derivative_ad": float(directional_ad),
        "directional_derivative_fd": float(directional_fd),
        "directional_relative_error": float(fd_rel),
        "adjoint_lhs": float(lhs),
        "adjoint_rhs": float(rhs),
        "adjoint_relative_error": float(adj_rel),
        "precision": "float64" if z0.dtype == jnp.float64 else "float32",
    }
    fd_tol = 2e-2 if z0.dtype == jnp.float32 else 1e-5
    adj_tol = 2e-4 if z0.dtype == jnp.float32 else 1e-10
    result["finite_difference_tolerance"] = fd_tol
    result["adjoint_tolerance"] = adj_tol
    result["pass"] = bool(float(fd_rel) < fd_tol and float(adj_rel) < adj_tol)
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run.output_dir / "derivative_check.json").write_text(json.dumps(result, indent=2))
    if not result["pass"]:
        raise RuntimeError(
            f"Derivative/adjoint check failed (finite-difference {float(fd_rel):.3e} vs {fd_tol:.1e}, "
            f"adjoint {float(adj_rel):.3e} vs {adj_tol:.1e}); fitting is blocked."
        )
    return result
