"""GPU memory planning.

v3 carries a full-grid latent field, so device memory is dominated by a small number
of full 3-D fields plus the optimizer or eigensolver basis. Nothing here materializes
a Jacobian or an N_vox x N_vox matrix; the estimates below describe the matrix-free
implementation that is actually run.
"""

from __future__ import annotations

import json
from pathlib import Path

import gemmi

GIB = 1024**3


def grid_shape_for(cfg) -> tuple[tuple[int, int, int], str]:
    """Grid shape from prepared metadata when it exists, otherwise from the model cell."""
    metadata_path = cfg.run.data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        return tuple(int(x) for x in metadata["grid_shape"]), "prepared metadata"
    if cfg.grid.shape is not None:
        return tuple(int(x) for x in cfg.grid.shape), "grid.shape"
    model_path = Path(cfg.input.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Cannot estimate memory: neither {metadata_path} nor input.model ({model_path}) is available"
        )
    st = gemmi.read_structure(str(model_path))
    grid = gemmi.FloatGrid()
    grid.set_unit_cell(st.cell)
    grid.spacegroup = gemmi.SpaceGroup(st.spacegroup_hm) if st.spacegroup_hm else gemmi.SpaceGroup("P 1")
    grid.set_size_from_spacing(cfg.resolution.d_min_angstrom / cfg.grid.samples_per_dmin, gemmi.GridSizeRounding.Up)
    return (grid.nu, grid.nv, grid.nw), "derived from input.model cell and d_min"


def estimate_memory(cfg) -> dict:
    shape, source = grid_shape_for(cfg)
    n_voxels = shape[0] * shape[1] * shape[2]
    real_bytes = 8 if cfg.run.enable_x64 else 4
    field = n_voxels * real_bytes
    complex_field = n_voxels * 2 * real_bytes

    # rho0, the solvent mask, the transfer, the current z and a few temporaries.
    resident = 6 * field + 3 * complex_field
    adam = resident + 3 * field
    lbfgs = resident + 2 * cfg.optimizer.lbfgs_memory * field
    # LOBPCG keeps X, R and P, each n x k, plus a Jv/J^T w working pair.
    information = resident + 3 * cfg.information.n_modes * field + 2 * field

    # XLA workspaces, cuFFT plans and allocator fragmentation are not free.
    overhead = 1.35
    result = {
        "grid_shape": list(shape),
        "grid_shape_source": source,
        "n_voxels": n_voxels,
        "precision": "float64" if cfg.run.enable_x64 else "float32",
        "single_real_field_GiB": field / GIB,
        "single_complex_field_GiB": complex_field / GIB,
        "resident_GiB": resident / GIB,
        "overhead_factor": overhead,
        "estimated_gpu_GiB_adam": adam * overhead / GIB,
        "estimated_gpu_GiB_lbfgs": lbfgs * overhead / GIB,
        "estimated_gpu_GiB_information": information * overhead / GIB,
        "information_n_modes": cfg.information.n_modes,
        "information_memory_budget_GiB": cfg.information.memory_budget_gib,
        "lbfgs_memory": cfg.optimizer.lbfgs_memory,
        "note": "Planning estimates. XLA/cuFFT workspaces and allocator fragmentation vary by GPU and JAX version.",
    }
    result["information_within_budget"] = bool(
        result["estimated_gpu_GiB_information"] <= cfg.information.memory_budget_gib
    )
    return result


def check_information_budget(cfg) -> dict:
    """Refuse an obviously oversized information-spectrum job before it allocates."""
    estimate = estimate_memory(cfg)
    n_voxels = estimate["n_voxels"]
    k = cfg.information.n_modes
    if not 0 < 5 * k < n_voxels:
        raise ValueError(
            f"JAX LOBPCG requires 0 < 5*n_modes < n_voxels; got n_modes={k}, n_voxels={n_voxels}. "
            "Reduce information.n_modes or use a finer grid."
        )
    if not estimate["information_within_budget"]:
        raise MemoryError(
            f"Information spectrum needs about {estimate['estimated_gpu_GiB_information']:.1f} GiB on a "
            f"{estimate['grid_shape']} grid with n_modes={k}, above the configured "
            f"information.memory_budget_gib={cfg.information.memory_budget_gib}. Reduce information.n_modes, "
            "raise the budget for a larger GPU, or coarsen the grid."
        )
    return estimate
