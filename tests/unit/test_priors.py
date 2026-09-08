"""Prior operator tests (plan section 20B).

The decisive property is that the prior is defined in physical reciprocal-space units:
refining the FFT grid must not change the correlation length, the field variance or the
band limit. Everything else follows from that.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from crystal_field.config import PriorConfig
from crystal_field.model.prior import apply_transfer, build_transfer, latent_penalty, spectral_geometry

CELL = (20.0, 24.0, 28.0)
G = np.diag([1.0 / CELL[0] ** 2, 1.0 / CELL[1] ** 2, 1.0 / CELL[2] ** 2])
VOLUME = float(np.prod(CELL))
D_MIN = 2.5
# A grid only reproduces the physical prior if it contains the whole band-limited
# sphere: |h| <= a/d_min in every direction. At d_min = 5 A both 16^3 and 32^3 do.
D_MIN_FULLY_SAMPLED = 5.0
KERNELS = ["matern", "squared_exponential", "bandlimited_white", "multiscale_matern"]


def _prior(kernel, **kwargs):
    if kernel == "multiscale_matern":
        kwargs.setdefault(
            "components",
            [
                {"correlation_length_angstrom": 0.5, "alpha": 2.0, "weight": 0.7},
                {"correlation_length_angstrom": 2.0, "alpha": 3.0, "weight": 0.3},
            ],
        )
    return PriorConfig.model_validate({"kernel": kernel, **kwargs})


@pytest.mark.parametrize("kernel", KERNELS)
def test_transfer_is_finite_positive_and_mean_free(kernel):
    prior = _prior(kernel, tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    transfer = np.asarray(build_transfer((12, 12, 12), G, VOLUME, D_MIN, prior, jnp.float32))
    assert np.isfinite(transfer).all()
    assert transfer.min() >= 0.0
    assert transfer.max() > 0.0
    assert transfer[0, 0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("kernel", KERNELS)
def test_remove_mean_false_keeps_the_dc_component(kernel):
    prior = _prior(kernel, tau_density=0.05, correlation_length_angstrom=1.0, remove_mean=False)
    transfer = np.asarray(build_transfer((12, 12, 12), G, VOLUME, D_MIN, prior, jnp.float32))
    assert transfer[0, 0, 0] > 0.0


@pytest.mark.parametrize("kernel", KERNELS)
def test_prior_variance_matches_tau_density(kernel):
    """RMS(L z) = tau for z ~ N(0, I), by construction of the normalisation."""
    prior = _prior(kernel, tau_density=0.037, correlation_length_angstrom=1.0, alpha=2.5)
    transfer = np.asarray(build_transfer((16, 16, 16), G, VOLUME, D_MIN, prior, jnp.float32))
    assert float(np.sum(transfer * transfer) / VOLUME) == pytest.approx(prior.tau_density**2, rel=2e-5)


@pytest.mark.parametrize("kernel", ["matern", "squared_exponential", "multiscale_matern"])
def test_prior_is_grid_independent(jax_x64, kernel):
    """Refining the grid must not change the physical prior.

    The prior lives in physical reciprocal-space units, so the transfer evaluated at a
    given Miller index must be the same on a coarse and on a fine grid, and the total
    field variance must agree. If either drifted, changing samples_per_dmin would
    silently change the correlation length of the experiment.
    """
    prior = _prior(kernel, tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    coarse = np.asarray(build_transfer((16, 16, 16), G, VOLUME, D_MIN_FULLY_SAMPLED, prior, jnp.float64))
    fine = np.asarray(build_transfer((32, 32, 32), G, VOLUME, D_MIN_FULLY_SAMPLED, prior, jnp.float64))

    for h, k, m in [(1, 0, 0), (0, 2, 1), (3, -1, 2), (-2, 2, -3)]:
        assert coarse[h % 16, k % 16, m % 16] == pytest.approx(fine[h % 32, k % 32, m % 32], rel=1e-9)

    assert np.sum(coarse**2) / VOLUME == pytest.approx(np.sum(fine**2) / VOLUME, rel=1e-9)


def test_a_grid_too_small_for_the_band_limit_changes_the_prior(jax_x64):
    """Why the Nyquist guard in prepare_reflections() exists.

    At d_min = 2.5 A this cell needs |l| <= 11, which a 16^3 grid cannot hold. The
    truncated grid normalises over a clipped sphere and so represents a different
    physical prior. prepare_reflections() refuses such a grid outright.
    """
    prior = _prior("matern", tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    truncated = np.asarray(build_transfer((16, 16, 16), G, VOLUME, D_MIN, prior, jnp.float64))
    adequate = np.asarray(build_transfer((32, 32, 32), G, VOLUME, D_MIN, prior, jnp.float64))
    assert abs(truncated[1, 0, 0] - adequate[1, 0, 0]) / adequate[1, 0, 0] > 1e-3


def test_band_limit_is_physical_not_voxel_based(jax_x64):
    """Everything beyond 1/d_min is exactly zero, on any grid."""
    prior = _prior("bandlimited_white", tau_density=0.05)
    shape = (32, 32, 32)
    transfer = np.asarray(build_transfer(shape, G, VOLUME, D_MIN, prior, jnp.float64))
    s2, _ = spectral_geometry(shape, G, jnp.float64)
    outside = np.asarray(s2) > (1.0 / D_MIN) ** 2
    assert outside.any(), "the test grid must actually reach beyond the band limit"
    assert np.all(transfer[outside] == 0.0)
    assert np.any(transfer[~outside] > 0.0)


def test_correlation_length_orders_the_spectrum(jax_x64):
    """A longer correlation length concentrates power at low resolution."""
    shape = (24, 24, 24)
    s2, _ = spectral_geometry(shape, G, jnp.float64)
    s2 = np.asarray(s2)
    low = (s2 > 0) & (s2 < (0.25 / D_MIN) ** 2)
    high = s2 > (0.75 / D_MIN) ** 2

    def ratio(ell):
        prior = _prior("matern", tau_density=0.05, correlation_length_angstrom=ell, alpha=2.5)
        t = np.asarray(build_transfer(shape, G, VOLUME, D_MIN, prior, jnp.float64)) ** 2
        return t[low].sum() / max(t[high].sum(), 1e-300)

    assert ratio(2.0) > ratio(0.5)


def test_multiscale_mixes_its_components(jax_x64):
    """The multiscale power spectrum is the weight-normalised sum of component spectra."""
    from crystal_field.model.prior import _matern_transfer
    from crystal_field.model.prior import spectral_geometry as geometry

    shape = (16, 16, 16)
    components = [
        {"correlation_length_angstrom": 0.5, "alpha": 2.0, "weight": 0.7},
        {"correlation_length_angstrom": 2.0, "alpha": 3.0, "weight": 0.3},
    ]
    prior = _prior("multiscale_matern", tau_density=1.0, components=components, remove_mean=False)
    transfer = np.asarray(build_transfer(shape, G, VOLUME, D_MIN, prior, jnp.float64))

    _, q2 = geometry(shape, G, jnp.float64)
    total_weight = sum(c["weight"] for c in components)
    power = np.zeros(shape)
    for c in components:
        raw = np.asarray(_matern_transfer(q2, c["correlation_length_angstrom"], c["alpha"]))
        power += (c["weight"] / total_weight) * raw**2
    expected = np.sqrt(power)
    s2, _ = geometry(shape, G, jnp.float64)
    expected = np.where(np.asarray(s2) <= (1.0 / D_MIN) ** 2, expected, 0.0)
    expected = expected / np.sqrt(np.sum(expected**2) / VOLUME) * prior.tau_density
    np.testing.assert_allclose(transfer, expected, rtol=1e-9, atol=1e-12)


def test_single_component_multiscale_equals_plain_matern(jax_x64):
    shape = (16, 16, 16)
    single = _prior(
        "multiscale_matern",
        tau_density=0.05,
        components=[{"correlation_length_angstrom": 0.9, "alpha": 2.5, "weight": 1.0}],
    )
    plain = _prior("matern", tau_density=0.05, correlation_length_angstrom=0.9, alpha=2.5)
    np.testing.assert_allclose(
        np.asarray(build_transfer(shape, G, VOLUME, D_MIN, single, jnp.float64)),
        np.asarray(build_transfer(shape, G, VOLUME, D_MIN, plain, jnp.float64)),
        rtol=1e-9,
        atol=1e-12,
    )


def test_apply_transfer_produces_the_configured_rms(jax_x64):
    prior = _prior("matern", tau_density=0.05, correlation_length_angstrom=1.0, alpha=2.5)
    shape = (24, 24, 24)
    transfer = build_transfer(shape, G, VOLUME, D_MIN, prior, jnp.float64)
    rng = np.random.default_rng(3)
    rms = np.mean(
        [
            float(jnp.sqrt(jnp.mean(apply_transfer(jnp.asarray(rng.standard_normal(shape)), transfer, VOLUME) ** 2)))
            for _ in range(8)
        ]
    )
    assert rms == pytest.approx(prior.tau_density, rel=0.15)


def test_isolated_voxel_spikes_cost_more_than_coherent_features(jax_x64):
    """The point of the correlated prior, stated as a test.

    A single-voxel excursion and a smooth multi-voxel blob of the same density RMS must
    not cost the same: the spike carries power far outside the prior's spectrum and so
    needs a much larger latent field to express.
    """
    shape = (24, 24, 24)
    prior = _prior("matern", tau_density=0.05, correlation_length_angstrom=1.5, alpha=2.5)
    transfer = np.asarray(build_transfer(shape, G, VOLUME, D_MIN, prior, jnp.float64))
    supported = transfer > 0

    def latent_rms(delta):
        uhat = np.fft.fftn(delta, norm="ortho")
        zhat = np.zeros_like(uhat)
        zhat[supported] = uhat[supported] / (transfer[supported] * np.sqrt(delta.size / VOLUME))
        return float(np.sqrt(np.mean(np.abs(zhat) ** 2)))

    spike = np.zeros(shape)
    spike[12, 12, 12] = 1.0

    grid = np.indices(shape) - 12
    blob = np.exp(-0.5 * (grid**2).sum(axis=0) / 2.0**2)

    spike = spike / np.sqrt(np.mean(spike**2))
    blob = blob / np.sqrt(np.mean(blob**2))
    assert latent_rms(spike) > 3 * latent_rms(blob)


@pytest.mark.parametrize("latent", ["gaussian", "student_t", "laplace", "cauchy"])
def test_latent_penalty_is_nonnegative_and_minimal_at_zero(latent):
    prior = PriorConfig(latent_distribution=latent)
    z = jnp.asarray([-2.0, 0.0, 1.0])
    assert float(latent_penalty(z, prior)) >= 0.0
    assert float(latent_penalty(jnp.zeros(3), prior)) == pytest.approx(0.0, abs=1e-6)
    assert float(latent_penalty(z, prior)) > float(latent_penalty(jnp.zeros(3), prior))


@pytest.mark.parametrize("latent", ["student_t", "laplace", "cauchy"])
def test_heavy_tailed_latents_penalise_large_values_less_than_gaussian(latent):
    z = jnp.asarray([6.0])
    gaussian = float(latent_penalty(z, PriorConfig(latent_distribution="gaussian")))
    assert float(latent_penalty(z, PriorConfig(latent_distribution=latent))) < gaussian


def test_unknown_latent_distribution_is_rejected():
    prior = PriorConfig()
    object.__setattr__(prior, "latent_distribution", "nonsense")
    with pytest.raises(ValueError, match="Unknown latent prior"):
        latent_penalty(jnp.zeros(3), prior)
