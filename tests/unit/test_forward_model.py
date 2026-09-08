"""Forward-model tests against analytic densities (plan section 20D).

tests/unit/test_fft_convention.py pins the Gemmi/JAX sign convention. These tests check
the transform itself against closed-form answers, so a convention change cannot hide
behind a self-consistent pair of errors.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from crystal_field.forward.diffraction import (
    fft_structure_factor_grid,
    gather_hkl,
    observable_from_fcalc,
    solvent_attenuation,
    symmetry_projected_fcalc,
)

CELL = (20.0, 24.0, 28.0)
VOLUME = float(np.prod(CELL))
SHAPE = (20, 24, 28)


def test_constant_density_scatters_only_into_the_origin():
    rho = jnp.full(SHAPE, 0.25, dtype=jnp.float32)
    grid = fft_structure_factor_grid(rho, VOLUME)
    hkls = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, -3, 1]], dtype=jnp.int32)
    f = np.asarray(gather_hkl(grid, hkls))
    assert f[0].real == pytest.approx(0.25 * VOLUME, rel=1e-5)
    np.testing.assert_allclose(np.abs(f[1:]), 0.0, atol=1e-3)


def test_single_voxel_density_has_flat_amplitude():
    """A delta function transforms to constant modulus, whatever the Miller index."""
    rho = np.zeros(SHAPE, dtype=np.float32)
    rho[3, 5, 7] = 1.0
    grid = fft_structure_factor_grid(jnp.asarray(rho), VOLUME)
    hkls = jnp.asarray([[0, 0, 0], [1, 2, 3], [-4, 5, -6], [7, 0, 2]], dtype=jnp.int32)
    amplitude = np.abs(np.asarray(gather_hkl(grid, hkls)))
    np.testing.assert_allclose(amplitude, VOLUME / rho.size, rtol=1e-5)


def test_single_voxel_phase_follows_its_position():
    """F(h) = (V/N) exp(-2 pi i h.r) for a delta at fractional position r."""
    rho = np.zeros(SHAPE, dtype=np.float64)
    index = (3, 5, 7)
    rho[index] = 1.0
    fractional = np.asarray(index, dtype=np.float64) / np.asarray(SHAPE, dtype=np.float64)
    grid = np.fft.fftn(rho) * (VOLUME / rho.size)
    for hkl in [(1, 2, 3), (-4, 5, -6), (7, 0, 2)]:
        expected = (VOLUME / rho.size) * np.exp(-2j * math.pi * np.dot(hkl, fractional))
        got = grid[tuple(np.mod(hkl, SHAPE))]
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_periodic_gaussian_matches_its_analytic_transform():
    """A Gaussian blob of width sigma transforms to exp(-2 pi^2 sigma^2 |s|^2)."""
    shape = (32, 32, 32)
    length = 20.0
    sigma = 1.6
    volume = length**3
    axis = (np.arange(shape[0]) - shape[0] // 2) * (length / shape[0])
    r2 = axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2
    rho = np.exp(-0.5 * r2 / sigma**2) / (sigma**3 * (2 * math.pi) ** 1.5)
    rho = np.fft.ifftshift(rho)

    grid = np.fft.fftn(rho) * (volume / rho.size)
    for hkl in [(0, 0, 0), (1, 0, 0), (2, 3, 1), (5, -4, 2)]:
        s2 = sum((h / length) ** 2 for h in hkl)
        expected = math.exp(-2 * math.pi**2 * sigma**2 * s2)
        got = abs(grid[tuple(np.mod(hkl, shape))])
        assert got == pytest.approx(expected, rel=2e-3, abs=1e-6)


def test_gather_wraps_negative_and_out_of_range_indices():
    grid = jnp.asarray(np.arange(np.prod(SHAPE), dtype=np.complex64).reshape(SHAPE))
    hkls = jnp.asarray([[-1, -1, -1], [SHAPE[0] - 1, SHAPE[1] - 1, SHAPE[2] - 1]], dtype=jnp.int32)
    values = np.asarray(gather_hkl(grid, hkls))
    assert values[0] == values[1]


def test_p1_symmetry_projection_is_the_identity():
    rng = np.random.default_rng(0)
    grid = jnp.asarray(rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE), dtype=jnp.complex64)
    hkls = jnp.asarray([[1, 2, 3], [-4, 5, -6]], dtype=jnp.int32)
    rotations = jnp.asarray(np.eye(3, dtype=np.int32)[None, ...])
    translations = jnp.zeros((1, 3), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(symmetry_projected_fcalc(grid, hkls, rotations, translations)),
        np.asarray(gather_hkl(grid, hkls)),
        rtol=1e-6,
    )


@pytest.mark.parametrize(("kind", "expected"), [("amplitude", 5.0), ("intensity", 25.0)])
def test_observable_from_fcalc(kind, expected):
    fcalc = jnp.asarray([3.0 + 4.0j])
    assert float(observable_from_fcalc(fcalc, kind)[0]) == pytest.approx(expected)


def test_unknown_observable_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown observation kind"):
        observable_from_fcalc(jnp.asarray([1.0 + 0j]), "flux")


def test_solvent_attenuation_decays_with_resolution():
    g = jnp.asarray(np.diag([1 / 400.0, 1 / 576.0, 1 / 784.0]), dtype=jnp.float32)
    hkls = jnp.asarray([[0, 0, 0], [1, 0, 0], [8, 0, 0]], dtype=jnp.int32)
    attenuation = np.asarray(solvent_attenuation(hkls, g, 46.0, jnp.float32))
    assert attenuation[0] == pytest.approx(1.0)
    assert attenuation[0] > attenuation[1] > attenuation[2] > 0.0
