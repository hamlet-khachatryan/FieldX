from __future__ import annotations


def amplitude_r_factor(fobs, fcalc_amp, mask):
    import jax.numpy as jnp

    m = mask.astype(fobs.dtype)
    num = jnp.sum(m * jnp.abs(fobs - fcalc_amp))
    den = jnp.sum(m * jnp.abs(fobs))
    return num / jnp.maximum(den, jnp.finfo(fobs.dtype).tiny)


def observable_to_amplitude(obs, pred, kind, scale):
    import jax.numpy as jnp

    if kind == "amplitude":
        return obs, scale * pred
    # For intensity likelihoods, report an amplitude-style R using sqrt(max(I,0)).
    return jnp.sqrt(jnp.maximum(obs, 0.0)), jnp.sqrt(jnp.maximum(scale * pred, 0.0))
