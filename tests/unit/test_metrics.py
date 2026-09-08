import jax.numpy as jnp
import pytest

from crystal_field.analysis.metrics import amplitude_r_factor


def test_r_factor_zero_for_exact_prediction():
    f = jnp.asarray([1.0, 2.0, 3.0])
    mask = jnp.asarray([True, True, True])
    assert float(amplitude_r_factor(f, f, mask)) == pytest.approx(0.0)


def test_r_factor_known_value():
    obs = jnp.asarray([1.0, 2.0])
    calc = jnp.asarray([0.5, 2.5])
    mask = jnp.asarray([True, True])
    assert float(amplitude_r_factor(obs, calc, mask)) == pytest.approx(1.0 / 3.0)
