import json

import numpy as np


def load_problem_arrays(cfg):
    import jax.numpy as jnp

    from crystal_field.inference.problem import ProblemArrays

    data_dir = cfg.run.data_dir
    metadata = json.loads((data_dir / "metadata.json").read_text())
    data = np.load(data_dir / "reflections.npz")
    rho0 = np.load(data_dir / "rho0.npy", mmap_mode="r")
    mask_path = data_dir / "solvent_mask.npy"
    solvent_mask = np.load(mask_path, mmap_mode="r") if mask_path.exists() else np.zeros(rho0.shape, dtype=np.float32)

    # gather_hkl() reduces indices modulo each grid's own shape, so a rho0 and a solvent
    # mask on different grids read different Fourier bins for the same (h,k,l) and quietly
    # corrupt F_calc at high resolution. The Nyquist guard in prepare_reflections() is also
    # written against the declared shape, so all three must agree.
    declared = tuple(int(x) for x in metadata["grid_shape"])
    if tuple(rho0.shape) != declared or tuple(solvent_mask.shape) != declared:
        raise RuntimeError(
            f"Grid mismatch: rho0={tuple(rho0.shape)}, solvent_mask={tuple(solvent_mask.shape)}, "
            f"metadata grid_shape={declared}. Re-run the density and solvent-mask stages."
        )

    if cfg.baseline.scaling.enabled:
        scaling_path = data_dir / f"scaling_{cfg.optimizer.fit_scope}.npz"
        if not scaling_path.exists():
            raise FileNotFoundError(
                f"Missing {scaling_path}. Run `cfi fit-scaling CONFIG` for fit_scope={cfg.optimizer.fit_scope}."
            )
        scaling = np.load(scaling_path)
        overall_scale = scaling["overall_scale"]
        solvent_scale = scaling["solvent_scale"]
    else:
        n = len(data["hkls"])
        overall_scale = np.ones(n, dtype=np.float32)
        solvent_scale = np.full(n, cfg.baseline.bulk_solvent.k_sol, dtype=np.float32)

    dtype = jnp.float64 if cfg.run.enable_x64 else jnp.float32
    return ProblemArrays(
        rho0=jnp.asarray(rho0, dtype=dtype),
        solvent_mask=jnp.asarray(solvent_mask, dtype=dtype),
        hkls=jnp.asarray(data["hkls"], dtype=jnp.int32),
        observation=jnp.asarray(data["observation"], dtype=dtype),
        sigma=jnp.asarray(data["sigma"], dtype=dtype),
        split=jnp.asarray(data["split"], dtype=jnp.int8),
        reciprocal_metric=jnp.asarray(data["reciprocal_metric"], dtype=dtype),
        symmetry_rotations=jnp.asarray(data["symmetry_rotations"], dtype=jnp.int32),
        symmetry_translations=jnp.asarray(data["symmetry_translations"], dtype=dtype),
        overall_scale=jnp.asarray(overall_scale, dtype=dtype),
        solvent_scale=jnp.asarray(solvent_scale, dtype=dtype),
        unit_cell_volume=float(metadata["unit_cell_volume"]),
        d_min_angstrom=float(cfg.resolution.d_min_angstrom),
        observation_kind=cfg.input.observation_kind,
    )
