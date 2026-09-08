import jax.numpy as jnp
import numpy as np
import pytest

from crystal_field.config import PriorConfig
from crystal_field.model.prior import build_transfer, latent_penalty

G = jnp.eye(3) / 100.0
SHAPE = (8, 8, 8)


@pytest.mark.parametrize("kernel", ["matern", "squared_exponential", "bandlimited_white"])
def test_transfer_finite_and_bandlimited(kernel):
    prior = PriorConfig(kernel=kernel, tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    t = np.asarray(build_transfer(SHAPE, G, 1000.0, 2.0, prior, jnp.float32))
    assert np.isfinite(t).all()
    assert t[0, 0, 0] == pytest.approx(0.0)
    assert np.max(t) > 0


def test_multiscale_transfer():
    prior = PriorConfig.model_validate({
        "kernel": "multiscale_matern",
        "tau_density": 0.05,
        "components": [
            {"correlation_length_angstrom": 0.5, "alpha": 2.0, "weight": 0.7},
            {"correlation_length_angstrom": 2.0, "alpha": 3.0, "weight": 0.3},
        ],
    })
    t = np.asarray(build_transfer(SHAPE, G, 1000.0, 2.0, prior, jnp.float32))
    assert np.isfinite(t).all()
    assert np.max(t) > 0


@pytest.mark.parametrize("latent", ["gaussian", "student_t", "laplace", "cauchy"])
def test_latent_penalty_nonnegative(latent):
    prior = PriorConfig(latent_distribution=latent)
    z = jnp.asarray([-2.0, 0.0, 1.0])
    assert float(latent_penalty(z, prior)) >= 0.0


def test_gaussian_prior_transfer_has_requested_rms_in_expectation():
    prior = PriorConfig(kernel="matern", tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    transfer = np.asarray(build_transfer(SHAPE, G, 1000.0, 2.0, prior, jnp.float32))
    expected_variance = np.sum(transfer * transfer) / 1000.0
    assert expected_variance == pytest.approx(prior.tau_density**2, rel=2e-5)
