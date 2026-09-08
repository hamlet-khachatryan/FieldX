"""Accelerator gate, run through slurm/40_gpu_tests.sbatch.

These verify that the accelerator is actually present and that the field operator is
differentiable on it. No production fitting happens here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crystal_field.config import PriorConfig
from crystal_field.model.prior import apply_transfer, build_transfer

pytestmark = pytest.mark.gpu

G = jnp.eye(3) / 100.0


def test_an_accelerator_is_present():
    assert jax.default_backend() != "cpu", f"expected a GPU backend, got {jax.devices()}"


def test_tiny_prior_is_differentiable():
    prior = PriorConfig(kernel="matern", tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    transfer = build_transfer((8, 8, 8), G, 1000.0, 2.0, prior, jnp.float32)

    def f(z):
        return jnp.sum(apply_transfer(z, transfer, 1000.0) ** 2)

    z = jnp.zeros((8, 8, 8), dtype=jnp.float32)
    assert jax.grad(f)(z).shape == z.shape


def test_fft_round_trips_on_the_accelerator():
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.standard_normal((16, 16, 16)), dtype=jnp.float32)
    recovered = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(rho)))
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(rho), atol=1e-4)


def test_jit_compiles_and_reuses_the_executable():
    prior = PriorConfig(kernel="matern", tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    transfer = build_transfer((16, 16, 16), G, 1000.0, 2.0, prior, jnp.float32)
    fn = jax.jit(lambda z: jnp.sum(apply_transfer(z, transfer, 1000.0) ** 2))
    z = jnp.zeros((16, 16, 16), dtype=jnp.float32)
    assert float(fn(z)) == pytest.approx(float(fn(z)))
