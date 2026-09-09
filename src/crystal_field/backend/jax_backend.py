from __future__ import annotations

import contextlib
import os


def configure_jax(enable_x64: bool, compilation_cache_dir=None):
    import jax

    jax.config.update("jax_enable_x64", bool(enable_x64))
    if compilation_cache_dir is not None:
        with contextlib.suppress(Exception):
            jax.config.update("jax_compilation_cache_dir", str(compilation_cache_dir))
    return jax


def ptxas_report():
    """Which ptxas XLA will use, and which release it is.

    A ptxas older than the GPU's compute capability aborts compilation with
    "UNIMPLEMENTED: ... ptxas too old", so it is worth recording before any fit rather
    than discovering it from a traceback halfway through the DAG. XLA prefers
    --xla_gpu_cuda_data_dir / CUDA_DIR over the copy bundled in the pip CUDA wheels;
    slurm/dls/cuda.sh sets the former to the toolkit that `module load cuda` provides.
    """
    import os
    import re
    import shutil
    import subprocess

    flags = os.environ.get("XLA_FLAGS", "")
    match = re.search(r"--xla_gpu_cuda_data_dir=(\S+)", flags)
    cuda_dir = match.group(1) if match else (os.environ.get("CUDA_DIR") or os.environ.get("CUDA_HOME"))

    candidate = None
    if cuda_dir:
        bundled = os.path.join(cuda_dir, "bin", "ptxas")
        candidate = bundled if os.path.exists(bundled) else None
    candidate = candidate or shutil.which("ptxas")

    release = None
    if candidate:
        with contextlib.suppress(Exception):
            out = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=15).stdout
            found = re.search(r"release ([0-9.]+)", out)
            release = found.group(1) if found else None
    return {
        "xla_flags": flags or None,
        "xla_gpu_cuda_data_dir": cuda_dir,
        "ptxas": candidate,
        "ptxas_release": release,
        "ptxas_is_bundled_wheel": bool(candidate and "site-packages" in candidate),
    }


def device_report():
    import jax

    ds = jax.devices()
    report = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in ds],
        "device_count": len(ds),
        "x64_enabled": bool(jax.config.x64_enabled),
    }
    if report["backend"] != "cpu":
        report.update(ptxas_report())
        with contextlib.suppress(Exception):
            report["compute_capability"] = [getattr(d, "compute_capability", None) for d in ds]
        report["memory"] = _memory_report(ds)
        report["preallocation"] = {
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        }
    return report


def _memory_report(devices):
    """Per-device memory, so an out-of-memory failure can be read rather than guessed.

    A tiny problem that dies with CUDA_ERROR_OUT_OF_MEMORY while loading a kernel is
    almost always preallocation starving the driver, not the data: compare bytes_limit
    against the total the card reports.
    """
    GIB = 1024**3
    stats = []
    for device in devices:
        entry = {"device": str(device)}
        try:
            raw = device.memory_stats() or {}
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "bytes_reservable_limit"):
                if key in raw:
                    entry[f"{key}_GiB"] = raw[key] / GIB
        except Exception as exc:  # a diagnostic must never break the report
            entry["error"] = str(exc)
        stats.append(entry)
    return stats
