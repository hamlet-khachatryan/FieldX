import json


def estimate_memory(cfg):
    metadata = json.loads((cfg.run.data_dir / "metadata.json").read_text())
    shape = tuple(int(x) for x in metadata["grid_shape"])
    n = shape[0] * shape[1] * shape[2]
    real_bytes = 8 if cfg.run.enable_x64 else 4
    complex_bytes = 2 * real_bytes
    m = cfg.optimizer.lbfgs_memory
    k = cfg.information.n_modes
    base = n * real_bytes
    fft_grid = n * complex_bytes
    adam = 8 * base + 3 * fft_grid
    lbfgs = (10 + 2 * m) * base + 3 * fft_grid
    info = (10 + 5 * k) * base + 3 * fft_grid
    gib = 1024 ** 3
    return {
        "grid_shape": shape,
        "n_voxels": n,
        "precision": "float64" if cfg.run.enable_x64 else "float32",
        "single_real_field_GiB": base / gib,
        "single_complex_field_GiB": fft_grid / gib,
        "rough_gpu_GiB_adam": adam / gib,
        "rough_gpu_GiB_lbfgs": lbfgs / gib,
        "rough_gpu_GiB_information": info / gib,
        "warning": "Conservative planning estimates; XLA/cuFFT workspaces and fragmentation vary by GPU/JAX version.",
    }
