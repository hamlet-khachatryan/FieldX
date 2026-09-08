from __future__ import annotations

import json

import gemmi
import numpy as np


def _density(st, cfg, metadata):
    calc = gemmi.DensityCalculatorX()
    calc.d_min = cfg.resolution.d_min_angstrom
    calc.cutoff = cfg.baseline.density_cutoff
    calc.grid.spacegroup = gemmi.SpaceGroup(metadata["spacegroup"])
    calc.grid.set_unit_cell(gemmi.UnitCell(*metadata["cell"]))
    calc.grid.set_size(*metadata["grid_shape"])
    calc.put_model_density_on_grid(st[0])
    return np.asarray(calc.grid.array, dtype=np.float32).copy()


def _selected_atoms(model, n_atoms):
    atoms = [cra.atom for cra in model.all() if not cra.atom.element.is_hydrogen() and cra.atom.occ > 0]
    if not atoms:
        raise ValueError("No non-hydrogen atoms available for the atomic benchmark")
    step = max(len(atoms) // n_atoms, 1)
    return atoms[::step][:n_atoms]


def _prior_cost(delta, transfer, volume):
    uhat = np.fft.fftn(delta, norm="ortho")
    transfer = np.asarray(transfer)
    scale = np.sqrt(delta.size / volume)
    supported = transfer > np.finfo(np.float32).tiny
    zhat = np.zeros_like(uhat, dtype=np.complex128)
    zhat[supported] = uhat[supported] / (transfer[supported] * scale)
    projected = np.fft.ifftn(np.where(supported, uhat, 0.0), norm="ortho").real
    delta_norm = np.linalg.norm(delta.ravel())
    return {
        "bandlimited_fraction": float(np.linalg.norm(projected.ravel()) / max(delta_norm, 1e-30)),
        "latent_rms": float(np.sqrt(np.mean(np.abs(zhat) ** 2))),
        "delta_rms": float(np.sqrt(np.mean(delta * delta))),
    }


def run_atomic_benchmark(cfg):
    import jax

    from crystal_field.model.prior import build_transfer

    data_dir = cfg.run.data_dir
    metadata = json.loads((data_dir / "metadata.json").read_text())
    refl = np.load(data_dir / "reflections.npz")
    st = gemmi.read_structure(str(cfg.input.model))
    st.setup_entities()
    base = np.load(data_dir / "rho0.npy")
    dtype = jax.numpy.float64 if cfg.run.enable_x64 else jax.numpy.float32
    transfer = build_transfer(
        tuple(base.shape),
        refl["reciprocal_metric"],
        float(metadata["unit_cell_volume"]),
        cfg.resolution.d_min_angstrom,
        cfg.prior,
        dtype,
    )
    transfer = np.asarray(jax.device_get(transfer), dtype=np.float64)

    rows = []
    for index, atom in enumerate(_selected_atoms(st[0], cfg.atomic_benchmark.n_atoms)):
        x0 = atom.pos.x
        atom.pos.x = x0 + cfg.atomic_benchmark.coordinate_delta_angstrom
        rows.append({"atom": index, "kind": "x", **_prior_cost(_density(st, cfg, metadata) - base, transfer, metadata["unit_cell_volume"])})
        atom.pos.x = x0

        b0 = atom.b_iso
        atom.b_iso = b0 + cfg.atomic_benchmark.b_delta_angstrom2
        rows.append({"atom": index, "kind": "b", **_prior_cost(_density(st, cfg, metadata) - base, transfer, metadata["unit_cell_volume"])})
        atom.b_iso = b0

        q0 = atom.occ
        atom.occ = max(0.0, min(1.0, q0 - cfg.atomic_benchmark.occupancy_delta))
        rows.append({"atom": index, "kind": "q", **_prior_cost(_density(st, cfg, metadata) - base, transfer, metadata["unit_cell_volume"])})
        atom.occ = q0

    result = {
        "prior_kernel": cfg.prior.kernel,
        "latent_distribution": cfg.prior.latent_distribution,
        "n_atoms": len(rows) // 3,
        "rows": rows,
        "summary": {
            "max_latent_rms": max(row["latent_rms"] for row in rows),
            "min_bandlimited_fraction": min(row["bandlimited_fraction"] for row in rows),
        },
    }
    out = cfg.run.output_dir / "atomic_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
