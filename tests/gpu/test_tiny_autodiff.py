import jax
import jax.numpy as jnp
import pytest

from crystal_field.config import PriorConfig
from crystal_field.model.prior import apply_transfer, build_transfer

pytestmark = pytest.mark.gpu


def test_tiny_prior_is_differentiable():
    prior = PriorConfig(kernel="matern", tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    g = jnp.eye(3) / 100.0
    t = build_transfer((8, 8, 8), g, 1000.0, 2.0, prior, jnp.float32)

    def f(z):
        u = apply_transfer(z, t, 1000.0)
        return jnp.sum(u * u)

    z = jnp.zeros((8, 8, 8), dtype=jnp.float32)
    grad = jax.grad(f)(z)
    assert grad.shape == z.shape
