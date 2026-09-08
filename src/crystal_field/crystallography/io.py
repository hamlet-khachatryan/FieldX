from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np

from crystal_field.crystallography.geometry import cell_tuple, reciprocal_metric_from_parameters
from crystal_field.crystallography.symmetry import symmetry_arrays


def _require_rs():
    try:
        import reciprocalspaceship as rs
    except ImportError as exc:
        raise RuntimeError(
            "reciprocalspaceship is required for reflection-table I/O. Install the base environment first."
        ) from exc
    return rs


def load_reflections(cfg):
    rs = _require_rs()
    p = Path(cfg.input.reflections)
    suffixes = "".join(p.suffixes).lower()
    ds = rs.read_mtz(str(p)) if suffixes.endswith(".mtz") or suffixes.endswith(".mtz.gz") else rs.read_cif(str(p))
    if not bool(ds.merged):
        raise ValueError("v1 expects merged reflection data")
    return ds


def _canonical_hash_key(hkl, pair_friedel):
    h = tuple(int(x) for x in hkl)
    if pair_friedel:
        h = min(h, tuple(-x for x in h))
    return f"{h[0]},{h[1]},{h[2]}"


def _u01_for_key(key, seed):
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def choose_grid_shape(ds, cfg):
    grid = gemmi.FloatGrid()
    grid.spacegroup = ds.spacegroup
    grid.set_unit_cell(ds.cell)
    if cfg.grid.shape is not None:
        grid.set_size(*tuple(int(x) for x in cfg.grid.shape))
    else:
        spacing = cfg.resolution.d_min_angstrom / cfg.grid.samples_per_dmin
        grid.set_size_from_spacing(spacing, gemmi.GridSizeRounding.Up)
    return (grid.nu, grid.nv, grid.nw)


def _suggest_columns(ds):
    columns = [str(c) for c in ds.columns]
    upper = {c.upper(): c for c in columns}
    amp_names = ["FP", "F", "FOBS", "F-OBS", "FMEAN"]
    sig_names = ["SIGFP", "SIGF", "SIGFOBS", "SIGF-OBS", "SIGFMEAN"]
    free_names = ["FREE", "FREER_FLAG", "FREERFLAG", "RFREE", "R-FREE-FLAGS"]

    def picks(names):
        return [upper[n] for n in names if n in upper]

    return {
        "observation_candidates": picks(amp_names),
        "sigma_candidates": picks(sig_names),
        "free_candidates": picks(free_names),
    }


def inspect_dataset(cfg):
    ds = load_reflections(cfg)
    ds.compute_dHKL(inplace=True)
    free_counts = None
    key = cfg.input.columns.free
    if key and key in ds:
        vals, counts = np.unique(np.asarray(ds[key]), return_counts=True)
        free_counts = {str(v): int(c) for v, c in zip(vals.tolist(), counts.tolist(), strict=False)}
    return {
        "path": str(cfg.input.reflections),
        "n_reflections": len(ds),
        "merged": bool(ds.merged),
        "spacegroup": ds.spacegroup.xhm(),
        "cell": cell_tuple(ds.cell),
        "d_min_in_file": float(np.nanmin(np.asarray(ds["dHKL"]))),
        "d_max_in_file": float(np.nanmax(np.asarray(ds["dHKL"]))),
        "columns": list(map(str, ds.columns)),
        "column_suggestions": _suggest_columns(ds),
        "free_counts": free_counts,
        "suggested_grid_shape": choose_grid_shape(ds, cfg),
    }


def prepare_reflections(cfg):
    ds = load_reflections(cfg)
    obs_key, sig_key = cfg.input.columns.observation, cfg.input.columns.sigma
    for key in (obs_key, sig_key):
        if key not in ds:
            raise KeyError(f"Required reflection column {key!r} not found; run `cfi inspect CONFIG` first")
    ds.compute_dHKL(inplace=True)
    d = np.asarray(ds["dHKL"], dtype=np.float64)
    obs = np.asarray(ds[obs_key], dtype=np.float64)
    sig = np.asarray(ds[sig_key], dtype=np.float64)
    hkls = np.asarray(ds.hkls, dtype=np.int32)

    keep = np.isfinite(obs) & np.isfinite(sig) & (sig > 0) & (d >= cfg.resolution.d_min_angstrom)
    if cfg.resolution.d_max_angstrom is not None:
        keep &= d <= cfg.resolution.d_max_angstrom
    keep &= ~ds.spacegroup.operations().systematic_absences(hkls)
    hkls, obs, sig, d = hkls[keep], obs[keep], sig[keep], d[keep]

    split = np.full(len(obs), -1, dtype=np.int8)
    pair = cfg.split.pair_friedel_when_nonanomalous and not cfg.split.anomalous
    u = np.array([_u01_for_key(_canonical_hash_key(h, pair), cfg.run.seed) for h in hkls])

    if cfg.split.strategy == "existing_free_then_hash":
        free_key = cfg.input.columns.free
        if free_key not in ds:
            raise KeyError(f"Free-flag column {free_key!r} not found")
        free = np.asarray(ds[free_key])[keep]
        free_mask = free == cfg.input.free_test_value
        free_fraction = float(np.mean(free_mask))
        if not (cfg.input.expected_free_fraction_min <= free_fraction <= cfg.input.expected_free_fraction_max):
            raise ValueError(
                f"Selected free_test_value={cfg.input.free_test_value!r} gives free fraction {free_fraction:.3%}, "
                f"outside configured range [{cfg.input.expected_free_fraction_min:.1%}, "
                f"{cfg.input.expected_free_fraction_max:.1%}]. Verify the free-flag convention before proceeding."
            )
        split[free_mask] = 2
        work = ~free_mask
        split[work & (u < cfg.split.tune_fraction_of_work)] = 1
        split[work & (u >= cfg.split.tune_fraction_of_work)] = 0
    else:
        f = cfg.split.final_fraction_if_hash
        split[u < f] = 2
        work = split < 0
        u_work = np.array([_u01_for_key(_canonical_hash_key(h, pair), cfg.run.seed + 1) for h in hkls])
        split[work & (u_work < cfg.split.tune_fraction_of_work)] = 1
        split[work & (u_work >= cfg.split.tune_fraction_of_work)] = 0

    if np.any(split < 0):
        raise RuntimeError("Internal split assignment failure")
    counts = {sid: int(np.sum(split == sid)) for sid in (0, 1, 2)}
    if any(v == 0 for v in counts.values()):
        raise ValueError(
            "Train/tune/free split must contain at least one reflection in every set; "
            f"got train={counts[0]}, tune={counts[1]}, free={counts[2]}."
        )

    shape = choose_grid_shape(ds, cfg)
    half = np.asarray(shape, dtype=np.float64) / 2.0
    if np.any(np.abs(hkls.astype(np.float64)) >= half[None, :]):
        offending = hkls[np.any(np.abs(hkls.astype(np.float64)) >= half[None, :], axis=1)][:5]
        raise ValueError(
            f"FFT grid {shape} is too small for retained HKLs; examples: {offending.tolist()}. "
            "Increase grid.samples_per_dmin or provide a larger explicit grid.shape."
        )

    cell = cell_tuple(ds.cell)
    rotations, translations = symmetry_arrays(ds.spacegroup)
    out = cfg.run.data_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "reflections.npz",
        hkls=hkls,
        observation=obs.astype(np.float32),
        sigma=sig.astype(np.float32),
        dHKL=d.astype(np.float32),
        split=split,
        reciprocal_metric=reciprocal_metric_from_parameters(*cell),
        symmetry_rotations=rotations,
        symmetry_translations=translations,
    )
    metadata = {
        "spacegroup": ds.spacegroup.xhm(),
        "cell": cell,
        "grid_shape": shape,
        "unit_cell_volume": float(ds.cell.volume),
        "observation_kind": cfg.input.observation_kind,
        "observation_column": obs_key,
        "sigma_column": sig_key,
        "free_column": cfg.input.columns.free,
        "free_test_value": cfg.input.free_test_value,
        "n_total_kept": len(obs),
        "n_train": counts[0],
        "n_tune": counts[1],
        "n_free": counts[2],
        "free_fraction": counts[2] / len(obs),
        "d_min": float(np.min(d)),
        "d_max": float(np.max(d)),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "splits.json").write_text(
        json.dumps({k: metadata[k] for k in ("n_train", "n_tune", "n_free", "free_fraction")}, indent=2)
    )
    return metadata


def _assert_model_matches_metadata(st, metadata):
    cell = gemmi.UnitCell(*metadata["cell"])
    sg = gemmi.SpaceGroup(metadata["spacegroup"])
    model_cell = cell_tuple(st.cell)
    if not np.allclose(np.asarray(model_cell), np.asarray(metadata["cell"]), rtol=1e-5, atol=1e-3):
        raise ValueError(f"Model/reflection unit-cell mismatch: model={model_cell}, reflections={tuple(metadata['cell'])}")
    if st.spacegroup_hm:
        model_sg = gemmi.SpaceGroup(st.spacegroup_hm)
        if model_sg.xhm() != sg.xhm():
            raise ValueError(f"Model/reflection space-group mismatch: model={model_sg.xhm()}, reflections={sg.xhm()}")
    else:
        st.spacegroup_hm = sg.xhm()
    return cell, sg


def make_model_density(cfg):
    out = cfg.run.data_dir
    metadata = json.loads((out / "metadata.json").read_text())
    st = gemmi.read_structure(str(cfg.input.model))
    st.setup_entities()
    if len(st) < 1:
        raise ValueError("Coordinate model contains no models")
    cell, sg = _assert_model_matches_metadata(st, metadata)

    cls = {
        "xray": gemmi.DensityCalculatorX,
        "electron": gemmi.DensityCalculatorE,
        "neutron": gemmi.DensityCalculatorN,
    }[cfg.input.scattering]
    dc = cls()
    dc.d_min = cfg.resolution.d_min_angstrom
    dc.cutoff = cfg.baseline.density_cutoff
    dc.grid.spacegroup = sg
    dc.grid.set_unit_cell(cell)
    shape = tuple(int(x) for x in metadata["grid_shape"])
    if cfg.baseline.refmac_compatible_density_blur and hasattr(dc, "set_refmac_compatible_blur"):
        dc.set_refmac_compatible_blur(st[0], allow_negative=False)
    # put_model_density_on_grid() calls initialize_grid() internally, which re-derives the
    # grid size from d_min and dc.rate and silently discards any earlier set_size(). That
    # put rho0 on a different grid than the declared shape and the solvent mask, so the
    # same (h,k,l) gathered a different Fourier bin from each. Expand gemmi's own sequence
    # here instead and override the size in the middle of it; set_size() zero-fills, so
    # this is the identical computation carried out on the declared grid.
    dc.initialize_grid()
    dc.grid.set_size(*shape)
    dc.add_model_density_to_grid(st[0])
    dc.grid.symmetrize_sum()
    rho = np.array(dc.grid.array, dtype=np.float32, copy=True, order="C")
    if rho.shape != shape:
        raise RuntimeError(
            f"rho0 was built on grid {rho.shape} but metadata.json declares {shape}. "
            "The density, the solvent mask and the Nyquist guard must share one grid; "
            "re-run `cfi prepare CONFIG` and do not proceed with a mismatched grid."
        )
    np.save(out / "rho0.npy", rho)

    rs = _require_rs()
    rs.io.write_ccp4_map(rho, str(out / "rho0.ccp4"), cell, sg)
    stats = {
        "shape": list(rho.shape),
        "declared_grid_shape": list(shape),
        "dtype": str(rho.dtype),
        "mean": float(rho.mean()),
        "std": float(rho.std()),
        "min": float(rho.min()),
        "max": float(rho.max()),
        "gemmi_blur_Bextra": float(getattr(dc, "blur", 0.0)),
        "gemmi_cutoff": float(dc.cutoff),
    }
    (out / "rho0_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def make_solvent_mask(cfg):
    out = cfg.run.data_dir
    metadata = json.loads((out / "metadata.json").read_text())
    st = gemmi.read_structure(str(cfg.input.model))
    cell, sg = _assert_model_matches_metadata(st, metadata)
    grid = gemmi.FloatGrid()
    grid.spacegroup = sg
    grid.set_unit_cell(cell)
    shape = tuple(int(x) for x in metadata["grid_shape"])
    grid.set_size(*shape)

    radii = {
        "refmac": gemmi.AtomicRadiiSet.Refmac,
        "cctbx": gemmi.AtomicRadiiSet.Cctbx,
        "van_der_waals": gemmi.AtomicRadiiSet.VanDerWaals,
    }[cfg.baseline.bulk_solvent.radii]
    masker = gemmi.SolventMasker(radii)
    masker.put_mask_on_float_grid(grid, st[0])
    mask = np.array(grid.array, dtype=np.float32, copy=True, order="C")
    if mask.shape != shape:
        raise RuntimeError(f"Solvent mask grid {mask.shape} does not match declared grid {shape}")
    rho0_path = out / "rho0.npy"
    if rho0_path.exists():
        rho0_shape = tuple(int(x) for x in np.load(rho0_path, mmap_mode="r").shape)
        if rho0_shape != mask.shape:
            raise RuntimeError(
                f"Solvent mask grid {mask.shape} does not match rho0 grid {rho0_shape}. "
                "Both must share one grid or the bulk-solvent term is gathered from the wrong "
                "Fourier coefficients; re-run `cfi make-rho0 CONFIG` and `cfi make-solvent-mask CONFIG`."
            )
    np.save(out / "solvent_mask.npy", mask)
    stats = {
        "shape": list(mask.shape),
        "declared_grid_shape": list(shape),
        "solvent_fraction": float(mask.mean()),
        "radii": cfg.baseline.bulk_solvent.radii,
        "rprobe": float(masker.rprobe),
        "rshrink": float(masker.rshrink),
    }
    (out / "solvent_mask_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def inspect_h5(path: str | Path):
    import h5py
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    result = {"path": str(path), "datasets": {}, "groups": {}}
    with h5py.File(path, "r") as h5:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                result["datasets"][name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                }
            elif isinstance(obj, h5py.Group):
                result["groups"][name] = {k: str(v) for k, v in obj.attrs.items()}
        h5.visititems(visitor)
        result["root_attrs"] = {k: str(v) for k, v in h5.attrs.items()}
    return result
