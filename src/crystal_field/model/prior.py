from __future__ import annotations

import math


def _frequency_axis(n, dtype):
    import jax.numpy as jnp

    return jnp.fft.fftfreq(n, d=1.0 / n).astype(dtype)


def spectral_geometry(shape, reciprocal_metric, dtype):
    import jax.numpy as jnp

    h = _frequency_axis(shape[0], dtype)[:, None, None]
    k = _frequency_axis(shape[1], dtype)[None, :, None]
    ell = _frequency_axis(shape[2], dtype)[None, None, :]
    g = jnp.asarray(reciprocal_metric, dtype=dtype)
    s2 = (
        g[0, 0] * h * h
        + g[1, 1] * k * k
        + g[2, 2] * ell * ell
        + 2 * g[0, 1] * h * k
        + 2 * g[0, 2] * h * ell
        + 2 * g[1, 2] * k * ell
    )
    q2 = (2 * math.pi) ** 2 * s2
    return s2, q2


def _matern_transfer(q2, ell, alpha):
    kappa = 1.0 / ell
    return (kappa * kappa + q2) ** (-0.5 * alpha)


def _squared_exponential_transfer(q2, ell):
    # Square-root spectral density of a Gaussian covariance, up to normalization.
    import jax.numpy as jnp

    return jnp.exp(-0.25 * ell * ell * q2)


def build_transfer(shape, reciprocal_metric, unit_cell_volume, d_min_angstrom, prior, dtype):
    import jax.numpy as jnp

    s2, q2 = spectral_geometry(shape, reciprocal_metric, dtype)
    if prior.kernel == "matern":
        base = _matern_transfer(q2, prior.correlation_length_angstrom, prior.alpha)
    elif prior.kernel == "squared_exponential":
        base = _squared_exponential_transfer(q2, prior.correlation_length_angstrom)
    elif prior.kernel == "bandlimited_white":
        base = jnp.ones_like(q2)
    elif prior.kernel == "multiscale_matern":
        # Sum covariances, then take square root of the resulting spectral power.
        power = jnp.zeros_like(q2)
        total_w = sum(float(c.weight) for c in prior.components)
        for c in prior.components:
            t = _matern_transfer(q2, c.correlation_length_angstrom, c.alpha)
            power = power + (float(c.weight) / total_w) * t * t
        base = jnp.sqrt(jnp.maximum(power, 0.0))
    else:
        raise ValueError(f"Unknown prior kernel: {prior.kernel}")

    base = jnp.where(s2 <= (1.0 / d_min_angstrom) ** 2, base, 0.0)
    if prior.remove_mean:
        base = base.at[0, 0, 0].set(0.0)
    # With orthonormal FFT and scale sqrt(N/V), this normalization makes RMS(u)=tau for z~N(0,I).
    norm = jnp.sqrt(jnp.sum(base * base) / unit_cell_volume)
    return prior.tau_density * base / jnp.maximum(norm, jnp.finfo(dtype).tiny)


def apply_transfer(z, transfer, unit_cell_volume):
    import jax.numpy as jnp

    zhat = jnp.fft.fftn(z, norm="ortho")
    uhat = transfer * zhat
    scale = jnp.sqrt(jnp.asarray(z.size / unit_cell_volume, dtype=z.dtype))
    return jnp.real(jnp.fft.ifftn(uhat, norm="ortho")) * scale


def latent_penalty(z, prior):
    import jax.numpy as jnp

    if prior.latent_distribution == "gaussian":
        return 0.5 * jnp.sum(z * z)
    if prior.latent_distribution == "student_t":
        nu = jnp.asarray(prior.student_t_df, dtype=z.dtype)
        return 0.5 * (nu + 1.0) * jnp.sum(jnp.log1p((z * z) / nu))
    if prior.latent_distribution == "laplace":
        eps = jnp.asarray(prior.laplace_softening, dtype=z.dtype)
        return jnp.sum(jnp.sqrt(z * z + eps * eps) - eps)
    if prior.latent_distribution == "cauchy":
        scale = jnp.asarray(prior.cauchy_scale, dtype=z.dtype)
        return jnp.sum(jnp.log1p((z / scale) ** 2))
    raise ValueError(f"Unknown latent prior: {prior.latent_distribution}")
