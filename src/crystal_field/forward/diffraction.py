from __future__ import annotations

import math


# Convention: exp(-2 pi i h.r), i.e. jnp.fft.fftn. Gemmi uses exp(+2 pi i h.r), so
# gemmi.transform_map_to_f_phi() equals conj() of this grid; run_fft_check() asserts that
# relationship. Observables go through |F|, which conjugation leaves unchanged. Do not
# conjugate here without also flipping the phase sign in symmetry_projected_fcalc().
def fft_structure_factor_grid(rho, unit_cell_volume):
    import jax.numpy as jnp
    return jnp.fft.fftn(rho) * (unit_cell_volume / rho.size)


def gather_hkl(grid, hkls):
    import jax.numpy as jnp
    idx = jnp.mod(hkls, jnp.asarray(grid.shape, dtype=hkls.dtype))
    return grid[idx[:, 0], idx[:, 1], idx[:, 2]]


def symmetry_projected_fcalc(fgrid, hkls, rotations, translations):
    import jax
    import jax.numpy as jnp
    hkls = jnp.asarray(hkls, dtype=jnp.int32)
    rotations = jnp.asarray(rotations, dtype=jnp.int32)
    translations = jnp.asarray(translations, dtype=fgrid.real.dtype)
    accum0 = jnp.zeros((hkls.shape[0],), dtype=fgrid.dtype)

    def body(i, accum):
        hg = hkls @ rotations[i]
        # Sign pairs with the unconjugated fftn convention in fft_structure_factor_grid().
        phase = -2.0 * math.pi * (hkls.astype(fgrid.real.dtype) @ translations[i])
        return accum + jnp.exp(1j * phase) * gather_hkl(fgrid, hg)

    return jax.lax.fori_loop(0, rotations.shape[0], body, accum0) / rotations.shape[0]


def observable_from_fcalc(fcalc, kind):
    import jax.numpy as jnp
    amp = jnp.abs(fcalc)
    if kind == "amplitude":
        return amp
    if kind == "intensity":
        return amp * amp
    raise ValueError(f"Unknown observation kind: {kind}")


def solvent_attenuation(hkls, reciprocal_metric, b_sol, dtype):
    import jax.numpy as jnp
    h = hkls.astype(dtype)
    g = reciprocal_metric.astype(dtype)
    s2 = jnp.einsum("ni,ij,nj->n", h, g, h)
    return jnp.exp(-0.25 * jnp.asarray(b_sol, dtype=dtype) * s2)
