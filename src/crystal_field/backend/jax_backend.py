from __future__ import annotations

import contextlib


def configure_jax(enable_x64: bool, compilation_cache_dir=None):
    import jax
    jax.config.update("jax_enable_x64", bool(enable_x64))
    if compilation_cache_dir is not None:
        with contextlib.suppress(Exception):
            jax.config.update("jax_compilation_cache_dir", str(compilation_cache_dir))
    return jax


def device_report():
    import jax
    ds = jax.devices()
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in ds],
        "device_count": len(ds),
        "x64_enabled": bool(jax.config.x64_enabled),
    }
