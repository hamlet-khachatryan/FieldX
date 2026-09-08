"""Atomic representability benchmark.

The v3 prior must not be so restrictive that it loses the capabilities of ordinary
atomic refinement. Perturbing an atom's coordinate, occupancy, isotropic B or
anisotropic ADP produces a density change that lies in the tangent space of the
atomic model; each such change is mapped back to latent coordinates so that priors
charging an absurd latent cost for elementary refinement directions are rejected --
even if they score well on broad density changes.

This is a diagnostic, not a restriction: a meaningful v3 density feature may lie
well outside this tangent space.
"""

from __future__ import annotations

import json

import gemmi
import numpy as np

from crystal_field.crystallography.density import model_density_on_grid


def _density(st, cfg, cell, sg, shape):
    rho, _ = model_density_on_grid(
        st[0], cell, sg, shape, cfg.resolution.d_min_angstrom, cfg.baseline.density_cutoff, cfg.input.scattering
    )
    return rho.astype(np.float64)


def _selected_atoms(model, n_atoms):
    atoms = [cra.atom for cra in model.all() if not cra.atom.element.is_hydrogen and cra.atom.occ > 0]
    if not atoms:
        raise ValueError("No non-hydrogen atoms available for the atomic benchmark")
    step = max(len(atoms) // n_atoms, 1)
    return atoms[::step][:n_atoms]


def _prior_cost(delta, transfer, volume):
    """Latent cost and band-limited fraction of a density perturbation."""
    uhat = np.fft.fftn(delta, norm="ortho")
    scale = np.sqrt(delta.size / volume)
    supported = np.asarray(transfer) > np.finfo(np.float32).tiny
    zhat = np.zeros_like(uhat, dtype=np.complex128)
    zhat[supported] = uhat[supported] / (np.asarray(transfer)[supported] * scale)
    projected = np.fft.ifftn(np.where(supported, uhat, 0.0), norm="ortho").real
    delta_norm = np.linalg.norm(delta.ravel())
    return {
        "bandlimited_fraction": float(np.linalg.norm(projected.ravel()) / max(delta_norm, 1e-30)),
        "latent_rms": float(np.sqrt(np.mean(np.abs(zhat) ** 2))),
        "delta_rms": float(np.sqrt(np.mean(delta * delta))),
    }


def run_atomic_benchmark(cfg):
    import jax

    from crystal_field.crystallography.io import _model_for_metadata
    from crystal_field.model.prior import build_transfer

    data_dir = cfg.run.data_dir
    metadata = json.loads((data_dir / "metadata.json").read_text())
    refl = np.load(data_dir / "reflections.npz")
    st, cell, sg = _model_for_metadata(cfg, metadata)
    shape = tuple(int(x) for x in metadata["grid_shape"])
    base = np.load(data_dir / "rho0.npy").astype(np.float64)

    dtype = jax.numpy.float64 if cfg.run.enable_x64 else jax.numpy.float32
    transfer = np.asarray(
        jax.device_get(
            build_transfer(
                shape,
                refl["reciprocal_metric"],
                float(metadata["unit_cell_volume"]),
                cfg.resolution.d_min_angstrom,
                cfg.prior,
                dtype,
            )
        ),
        dtype=np.float64,
    )
    volume = float(metadata["unit_cell_volume"])

    def cost(kind, index):
        return {"atom": index, "kind": kind, **_prior_cost(_density(st, cfg, cell, sg, shape) - base, transfer, volume)}

    rows = []
    for index, atom in enumerate(_selected_atoms(st[0], cfg.atomic_benchmark.n_atoms)):
        x0 = atom.pos.x
        atom.pos.x = x0 + cfg.atomic_benchmark.coordinate_delta_angstrom
        rows.append(cost("x", index))
        atom.pos.x = x0

        b0 = atom.b_iso
        atom.b_iso = b0 + cfg.atomic_benchmark.b_delta_angstrom2
        rows.append(cost("b_iso", index))
        atom.b_iso = b0

        q0 = atom.occ
        atom.occ = max(0.0, min(1.0, q0 - cfg.atomic_benchmark.occupancy_delta))
        rows.append(cost("occupancy", index))
        atom.occ = q0

        if atom.aniso.nonzero():
            u0 = gemmi.SMat33f(
                atom.aniso.u11, atom.aniso.u22, atom.aniso.u33, atom.aniso.u12, atom.aniso.u13, atom.aniso.u23
            )
            # An ADP-like perturbation: extend the ellipsoid along u11 only, which is
            # not reachable by an isotropic B change.
            du = cfg.atomic_benchmark.b_delta_angstrom2 / (8.0 * np.pi**2)
            atom.aniso = gemmi.SMat33f(u0.u11 + du, u0.u22, u0.u33, u0.u12, u0.u13, u0.u23)
            rows.append(cost("adp_aniso", index))
            atom.aniso = u0

    kinds = sorted({row["kind"] for row in rows})
    result = {
        "prior_kernel": cfg.prior.kernel,
        "latent_distribution": cfg.prior.latent_distribution,
        "correlation_length_angstrom": cfg.prior.correlation_length_angstrom,
        "perturbation_kinds": kinds,
        "n_atoms": len({row["atom"] for row in rows}),
        "rows": rows,
        "summary": {
            "max_latent_rms": max(row["latent_rms"] for row in rows),
            "min_bandlimited_fraction": min(row["bandlimited_fraction"] for row in rows),
            "per_kind_max_latent_rms": {
                kind: max(row["latent_rms"] for row in rows if row["kind"] == kind) for kind in kinds
            },
        },
    }
    out = cfg.run.output_dir / "atomic_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
