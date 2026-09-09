"""Sigma-A weighting for 2mFo-DFc and mFo-DFc maps.

Following Read (1986). Per resolution shell the model is characterised by a scale D and a
residual variance sigma_delta^2, from which each reflection gets a figure of merit m. The
weighted coefficients down-weight reflections the model explains poorly, which is why
2mFo-DFc is the standard map for rebuilding rather than plain 2Fo-Fc.

WHERE THE PARAMETERS COME FROM MATTERS. Conventionally D and sigma_delta are estimated on
the free set, precisely because a model fitted to the work set explains it optimistically
and the weights come out too confident. FieldX cannot do that before the model lock
without spending the held-out set on map interpretation, so:

  - `work` (default) estimates on the work reflections. Safe at any time, and biased
    towards m too large. This is stated in the output rather than hidden.
  - `free` estimates on the free set, and is permitted only once the one-shot evaluation
    has already happened -- at which point the set is spent and using it costs nothing
    further.
"""

from __future__ import annotations

import numpy as np

MIN_REFLECTIONS_PER_BIN = 50


def _figure_of_merit(x, centric):
    """m = I1(X)/I0(X) for acentric reflections, tanh(X) for centric ones."""
    from scipy.special import i0e, i1e

    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1e6)
    # i0e/i1e are exponentially scaled, so their ratio is stable for large arguments
    # where I0 and I1 both overflow.
    acentric = i1e(x) / np.maximum(i0e(x), np.finfo(np.float64).tiny)
    return np.where(centric, np.tanh(x), acentric)


def resolution_bins(d_spacings, n_bins):
    """Equal-count bins in 1/d^2, the shells sigma-A is conventionally estimated in."""
    inverse = 1.0 / np.maximum(np.asarray(d_spacings, dtype=np.float64), 1e-12) ** 2
    order = np.argsort(inverse)
    labels = np.empty(len(inverse), dtype=np.int64)
    for bin_index, chunk in enumerate(np.array_split(order, n_bins)):
        labels[chunk] = bin_index
    return labels


def estimate_sigmaa(fobs, fcalc_amplitude, d_spacings, centric, epsilon, n_bins=20):
    """Per-shell scale D and residual variance, plus the per-reflection figure of merit.

    D is the phase-free least-squares scale <|Fo||Fc|>/<|Fc|^2>; the residual variance is
    <|Fo|^2> - D^2<|Fc|^2>, floored at a small positive value so a shell the model fits
    almost perfectly cannot produce a divide-by-zero.
    """
    fobs = np.asarray(fobs, dtype=np.float64)
    fcalc_amplitude = np.asarray(fcalc_amplitude, dtype=np.float64)
    centric = np.asarray(centric, dtype=bool)
    epsilon = np.asarray(epsilon, dtype=np.float64)

    n_bins = max(1, min(int(n_bins), max(1, len(fobs) // MIN_REFLECTIONS_PER_BIN)))
    labels = resolution_bins(d_spacings, n_bins)

    d_per_reflection = np.ones_like(fobs)
    sigma_per_reflection = np.ones_like(fobs)
    shells = []
    for bin_index in range(n_bins):
        mask = labels == bin_index
        o, c = fobs[mask], fcalc_amplitude[mask]
        mean_oc, mean_cc, mean_oo = float(np.mean(o * c)), float(np.mean(c * c)), float(np.mean(o * o))
        scale = mean_oc / max(mean_cc, np.finfo(np.float64).tiny)
        residual = max(mean_oo - scale * scale * mean_cc, 1e-6 * max(mean_oo, 1.0))
        d_per_reflection[mask] = scale
        sigma_per_reflection[mask] = residual
        shells.append(
            {
                "bin": bin_index,
                "n": int(mask.sum()),
                "d_max": float(np.max(np.asarray(d_spacings)[mask])),
                "d_min": float(np.min(np.asarray(d_spacings)[mask])),
                "D": scale,
                "sigma_delta_squared": residual,
            }
        )

    dfc = d_per_reflection * fcalc_amplitude
    argument = np.where(centric, 1.0, 2.0) * fobs * dfc / np.maximum(epsilon * sigma_per_reflection, 1e-30)
    m = np.clip(_figure_of_merit(argument, centric), 0.0, 1.0)
    return {"m": m, "D": d_per_reflection, "sigma_delta_squared": sigma_per_reflection, "shells": shells}


def apply_sigmaa(shells, fobs, fcalc_amplitude, d_spacings, centric, epsilon):
    """Apply per-shell parameters to a (possibly different) set of reflections.

    Sigma-A is estimated on one set and used to weight another -- estimated on work,
    applied to every reflection that goes into the map -- so the shell parameters are
    looked up by resolution rather than by index.
    """
    d_spacings = np.asarray(d_spacings, dtype=np.float64)
    boundaries = np.array([shell["d_min"] for shell in shells], dtype=np.float64)
    # Shells are ordered from low to high resolution, i.e. by decreasing d_min.
    index = np.clip(np.searchsorted(-boundaries, -d_spacings, side="left"), 0, len(shells) - 1)
    scale = np.array([shells[i]["D"] for i in index])
    residual = np.array([shells[i]["sigma_delta_squared"] for i in index])

    centric = np.asarray(centric, dtype=bool)
    argument = (
        np.where(centric, 1.0, 2.0)
        * np.asarray(fobs, dtype=np.float64)
        * scale
        * np.asarray(fcalc_amplitude, dtype=np.float64)
    ) / np.maximum(np.asarray(epsilon, dtype=np.float64) * residual, 1e-30)
    return {"m": np.clip(_figure_of_merit(argument, centric), 0.0, 1.0), "D": scale}


def map_coefficients(fobs, fcalc, weights, centric, kind):
    """Sigma-A weighted coefficients.

    `fcalc` must already be in Gemmi's exp(+2 pi i h.r) convention. For centric
    reflections the phase is restricted, so the 2mFo-DFc coefficient is m|Fo| rather than
    2m|Fo| - D|Fc|; doubling would count the observation twice.
    """
    m, scale = weights["m"], weights["D"]
    amplitude = np.abs(fcalc)
    phase = np.exp(1j * np.angle(fcalc))
    if kind == "2mFo-DFc":
        magnitude = np.where(centric, m * fobs, 2.0 * m * fobs - scale * amplitude)
    elif kind == "mFo-DFc":
        magnitude = m * fobs - scale * amplitude
    else:
        raise ValueError(f"Unknown map kind: {kind}")
    return magnitude * phase
