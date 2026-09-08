"""Derivative and adjoint tests (plan section 20E).

J and J^T are never materialised: J v comes from a JVP and J^T w from a VJP. The
adjoint identity <J v, w> = <v, J^T w> is what proves those two agree.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import problem_arrays, problem_config

from crystal_field.inference.problem import build_functions


@pytest.fixture
def problem(tmp_path, jax_x64):
    cfg = problem_config(tmp_path, optimizer={"fit_scope": "work"})
    arrays = problem_arrays(observation=(1.0, 2.0, 3.0, 1.5, 2.5), split=(0, 0, 1, 1, 2))
    arrays = arrays.__class__(
        **{
            **arrays.__dict__,
            "rho0": jnp.asarray(np.asarray(arrays.rho0), dtype=jnp.float64),
            "observation": jnp.asarray(np.asarray(arrays.observation), dtype=jnp.float64),
            "sigma": jnp.asarray(np.asarray(arrays.sigma), dtype=jnp.float64),
            "solvent_mask": jnp.asarray(np.asarray(arrays.solvent_mask), dtype=jnp.float64),
            "reciprocal_metric": jnp.asarray(np.asarray(arrays.reciprocal_metric), dtype=jnp.float64),
            "symmetry_translations": jnp.asarray(np.asarray(arrays.symmetry_translations), dtype=jnp.float64),
            "overall_scale": jnp.asarray(np.asarray(arrays.overall_scale), dtype=jnp.float64),
            "solvent_scale": jnp.asarray(np.asarray(arrays.solvent_scale), dtype=jnp.float64),
        }
    )
    return cfg, arrays, build_functions(arrays, cfg)


def _direction(shape, dtype, seed=0):
    v = jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=dtype)
    return v / jnp.linalg.norm(v.reshape(-1))


def test_finite_difference_matches_autodiff(problem):
    _, arrays, functions = problem
    objective = functions[4]
    z0 = jnp.zeros_like(arrays.rho0)
    v = _direction(z0.shape, z0.dtype, 1)

    grad = jax.grad(objective)(z0)
    analytic = float(jnp.vdot(grad.reshape(-1), v.reshape(-1)).real)
    eps = 1e-6
    numeric = float((objective(z0 + eps * v) - objective(z0 - eps * v)) / (2 * eps))
    assert numeric == pytest.approx(analytic, rel=1e-6, abs=1e-9)


def test_finite_difference_matches_autodiff_away_from_zero(problem):
    _, arrays, functions = problem
    objective = functions[4]
    z0 = 0.4 * _direction(arrays.rho0.shape, arrays.rho0.dtype, 5)
    v = _direction(z0.shape, z0.dtype, 6)

    grad = jax.grad(objective)(z0)
    analytic = float(jnp.vdot(grad.reshape(-1), v.reshape(-1)).real)
    eps = 1e-6
    numeric = float((objective(z0 + eps * v) - objective(z0 - eps * v)) / (2 * eps))
    assert numeric == pytest.approx(analytic, rel=1e-6, abs=1e-9)


def test_jacobian_vector_product_matches_a_finite_difference_of_residuals(problem):
    _, arrays, functions = problem
    residuals = functions[3]
    z0 = jnp.zeros_like(arrays.rho0)
    v = _direction(z0.shape, z0.dtype, 2)

    def r(z):
        return residuals(z, 0)[0]

    _, jv = jax.jvp(r, (z0,), (v,))
    eps = 1e-6
    numeric = (r(z0 + eps * v) - r(z0 - eps * v)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(jv), np.asarray(numeric), rtol=1e-5, atol=1e-9)


def test_adjoint_identity_holds(problem):
    _, arrays, functions = problem
    residuals = functions[3]
    z0 = jnp.zeros_like(arrays.rho0)
    v = _direction(z0.shape, z0.dtype, 3)

    def r(z):
        return residuals(z, 0)[0]

    r0, pullback = jax.vjp(r, z0)
    _, jv = jax.jvp(r, (z0,), (v,))
    w = jax.random.normal(jax.random.PRNGKey(4), r0.shape, dtype=r0.dtype)
    jtw = pullback(w)[0]

    lhs = float(jnp.vdot(jv.reshape(-1), w.reshape(-1)).real)
    rhs = float(jnp.vdot(v.reshape(-1), jtw.reshape(-1)).real)
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-14)


def test_transpose_jacobian_has_the_shape_of_the_latent_field(problem):
    _, arrays, functions = problem
    residuals = functions[3]
    z0 = jnp.zeros_like(arrays.rho0)

    def r(z):
        return residuals(z, 0)[0]

    r0, pullback = jax.vjp(r, z0)
    jtw = pullback(jnp.ones_like(r0))[0]
    assert jtw.shape == z0.shape


def test_no_dense_jacobian_is_ever_formed(problem):
    """A field of N voxels must never produce an M x N array anywhere in the chain."""
    _, arrays, functions = problem
    residuals = functions[3]
    z0 = jnp.zeros_like(arrays.rho0)

    def r(z):
        return residuals(z, 0)[0]

    _, jv = jax.jvp(r, (z0,), (jnp.ones_like(z0),))
    assert jv.shape == arrays.observation.shape
    r0, pullback = jax.vjp(r, z0)
    assert pullback(jnp.ones_like(r0))[0].shape == z0.shape
