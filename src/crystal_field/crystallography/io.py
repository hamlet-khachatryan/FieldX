"""Reflection I/O, the train/tune/free split, and starting-density construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np

from crystal_field.crystallography.density import (
    direct_structure_factors,
    model_density_on_grid,
    model_electron_count,
)
from crystal_field.crystallography.geometry import cell_tuple, reciprocal_metric_from_parameters
from crystal_field.crystallography.pdb import AMPLITUDE_NAMES, FREE_NAMES, SIGMA_NAMES
from crystal_field.crystallography.symmetry import symmetry_arrays

TRAIN, TUNE, FREE = 0, 1, 2


def _require_rs():
    try:
        import reciprocalspaceship as rs
    except ImportError as exc:
        raise RuntimeError("reciprocalspaceship is required for reflection-table I/O") from exc
    return rs


def load_reflections(cfg):
    rs = _require_rs()
    path = Path(cfg.input.reflections)
    if not path.is_file():
        raise FileNotFoundError(f"input.reflections does not exist: {path}")
    suffixes = "".join(path.suffixes).lower()
    ds = rs.read_mtz(str(path)) if suffixes.endswith((".mtz", ".mtz.gz")) else rs.read_cif(str(path))
    if not bool(ds.merged):
        raise ValueError("FieldX v3 expects merged reflection data")
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
    """Derive the FFT grid from the cell, d_min and the sampling target.

    The physical cell and the computational grid are distinct: the grid follows from
    d_min / samples_per_dmin and is rounded up to an FFT-friendly size by Gemmi.
    """
    grid = gemmi.FloatGrid()
    grid.spacegroup = ds.spacegroup
    grid.set_unit_cell(ds.cell)
    if cfg.grid.shape is not None:
        grid.set_size(*(int(x) for x in cfg.grid.shape))
    else:
        grid.set_size_from_spacing(cfg.resolution.d_min_angstrom / cfg.grid.samples_per_dmin, gemmi.GridSizeRounding.Up)
    return (grid.nu, grid.nv, grid.nw)


def _suggest_columns(ds):
    upper = {str(c).upper(): str(c) for c in ds.columns}
    return {
        "observation_candidates": [upper[n] for n in AMPLITUDE_NAMES if n in upper],
        "sigma_candidates": [upper[n] for n in SIGMA_NAMES if n in upper],
        "free_candidates": [upper[n] for n in FREE_NAMES if n in upper],
    }


def _model_reflection_consistency(cfg, ds):
    """Compare the coordinate model's cell/space group with the reflection file's."""
    model_path = Path(cfg.input.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"input.model does not exist: {model_path}")
    st = gemmi.read_structure(str(model_path))
    model_cell = cell_tuple(st.cell)
    refl_cell = cell_tuple(ds.cell)
    consistent_cell = bool(np.allclose(model_cell, refl_cell, rtol=1e-4, atol=1e-2))
    model_sg = gemmi.SpaceGroup(st.spacegroup_hm).xhm() if st.spacegroup_hm else None
    return {
        "model_cell": model_cell,
        "model_spacegroup": model_sg,
        "cell_consistent": consistent_cell,
        "spacegroup_consistent": model_sg is None or model_sg == ds.spacegroup.xhm(),
        "n_atoms": sum(1 for _ in st[0].all()),
    }


def inspect_dataset(cfg):
    ds = load_reflections(cfg)
    ds.compute_dHKL(inplace=True)
    free_counts = None
    key = cfg.input.columns.free
    if key and key in ds:
        values = np.asarray(ds[key])
        if np.issubdtype(values.dtype, np.number):
            unique, counts = np.unique(values, return_counts=True)
            free_counts = {str(v): int(c) for v, c in zip(unique.tolist(), counts.tolist(), strict=True)}
    return {
        "path": str(cfg.input.reflections),
        "n_reflections": len(ds),
        "merged": bool(ds.merged),
        "spacegroup": ds.spacegroup.xhm(),
        "cell": cell_tuple(ds.cell),
        "d_min_in_file": float(np.nanmin(np.asarray(ds["dHKL"]))),
        "d_max_in_file": float(np.nanmax(np.asarray(ds["dHKL"]))),
        "d_min_configured": cfg.resolution.d_min_angstrom,
        "columns": [str(c) for c in ds.columns],
        "column_suggestions": _suggest_columns(ds),
        "free_column": key,
        "free_test_value": cfg.input.free_test_value,
        "free_counts": free_counts,
        "split_strategy": cfg.split.strategy,
        "suggested_grid_shape": choose_grid_shape(ds, cfg),
        "model": _model_reflection_consistency(cfg, ds),
    }


def prepare_reflections(cfg):
    """Filter reflections and assign the immutable train / tune / free split.

    D_work = D_train u D_tune is everything the optimizer, the prior selection and the
    nuisance calibration may ever see. D_free is written once here and then never read
    until a model lock exists.
    """
    ds = load_reflections(cfg)
    obs_key, sig_key = cfg.input.columns.observation, cfg.input.columns.sigma
    for key in (obs_key, sig_key):
        if key not in ds:
            raise KeyError(
                f"Required reflection column {key!r} not found in {list(map(str, ds.columns))}; "
                "run `fieldrefine inspect CONFIG` first"
            )
    ds.compute_dHKL(inplace=True)
    d = np.asarray(ds["dHKL"], dtype=np.float64)
    obs = np.asarray(ds[obs_key], dtype=np.float64)
    sig = np.asarray(ds[sig_key], dtype=np.float64)
    hkls = np.asarray(ds.hkls, dtype=np.int32)

    keep = np.isfinite(obs) & np.isfinite(sig) & (sig > 0) & (d >= cfg.resolution.d_min_angstrom)
    if cfg.resolution.d_max_angstrom is not None:
        keep &= d <= cfg.resolution.d_max_angstrom
    keep &= ~ds.spacegroup.operations().systematic_absences(hkls)
    if not np.any(keep):
        raise ValueError("No reflections survive the resolution/sigma/absence filters")
    hkls, obs, sig, d = hkls[keep], obs[keep], sig[keep], d[keep]

    split = np.full(len(obs), -1, dtype=np.int8)
    pair = cfg.split.pair_friedel_when_nonanomalous and not cfg.split.anomalous
    u = np.array([_u01_for_key(_canonical_hash_key(h, pair), cfg.run.seed) for h in hkls])

    if cfg.split.strategy == "existing_free_then_hash":
        free_key = cfg.input.columns.free
        if free_key not in ds:
            raise KeyError(f"Free-flag column {free_key!r} not found")
        free_mask = np.asarray(ds[free_key])[keep] == cfg.input.free_test_value
        free_fraction = float(np.mean(free_mask))
        if not (cfg.input.expected_free_fraction_min <= free_fraction <= cfg.input.expected_free_fraction_max):
            raise ValueError(
                f"free_test_value={cfg.input.free_test_value!r} gives free fraction {free_fraction:.3%}, outside "
                f"[{cfg.input.expected_free_fraction_min:.1%}, {cfg.input.expected_free_fraction_max:.1%}]. "
                "Verify the free-flag convention before proceeding."
            )
        split[free_mask] = FREE
        work = ~free_mask
        split[work & (u < cfg.split.tune_fraction_of_work)] = TUNE
        split[work & (u >= cfg.split.tune_fraction_of_work)] = TRAIN
    elif cfg.split.strategy == "none":
        # No held-out set exists. Everything is work, partitioned into train and tune so
        # prior selection and early stopping still have something to rank on. The only
        # target is R_work.
        split[u < cfg.split.tune_fraction_of_work] = TUNE
        split[u >= cfg.split.tune_fraction_of_work] = TRAIN
    else:
        split[u < cfg.split.final_fraction_if_hash] = FREE
        work = split < 0
        u_work = np.array([_u01_for_key(_canonical_hash_key(h, pair), cfg.run.seed + 1) for h in hkls])
        split[work & (u_work < cfg.split.tune_fraction_of_work)] = TUNE
        split[work & (u_work >= cfg.split.tune_fraction_of_work)] = TRAIN

    if np.any(split < 0):
        raise RuntimeError("Internal split assignment failure")
    counts = {sid: int(np.sum(split == sid)) for sid in (TRAIN, TUNE, FREE)}
    required = [TRAIN, TUNE, FREE] if cfg.has_free_set else [TRAIN, TUNE]
    if any(counts[sid] == 0 for sid in required):
        detail = f"got train={counts[TRAIN]}, tune={counts[TUNE]}, free={counts[FREE]}."
        hint = (
            ""
            if not cfg.has_free_set
            else " If this dataset genuinely cannot support a holdout, set split.strategy=none; "
            "the run then targets R_work only and reports that the primary criterion "
            "could not be evaluated."
        )
        raise ValueError(f"Every required split must contain at least one reflection; {detail}{hint}")
    if not cfg.has_free_set and counts[FREE]:
        raise RuntimeError("split.strategy=none must not assign any reflection to the free set")

    shape = choose_grid_shape(ds, cfg)
    half = np.asarray(shape, dtype=np.float64) / 2.0
    too_large = np.any(np.abs(hkls.astype(np.float64)) >= half[None, :], axis=1)
    if np.any(too_large):
        raise ValueError(
            f"FFT grid {shape} is too small for retained HKLs; examples: {hkls[too_large][:5].tolist()}. "
            "Increase grid.samples_per_dmin or set an explicit grid.shape."
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
        "split_strategy": cfg.split.strategy,
        "has_free_set": cfg.has_free_set,
        "n_total_kept": len(obs),
        "n_train": counts[TRAIN],
        "n_tune": counts[TUNE],
        "n_free": counts[FREE],
        "free_fraction": counts[FREE] / len(obs),
        "d_min": float(np.min(d)),
        "d_max": float(np.max(d)),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "splits.json").write_text(
        json.dumps({k: metadata[k] for k in ("n_train", "n_tune", "n_free", "free_fraction")}, indent=2)
    )
    return metadata


def _model_for_metadata(cfg, metadata):
    st = gemmi.read_structure(str(cfg.input.model))
    st.setup_entities()
    if len(st) < 1:
        raise ValueError("Coordinate model contains no models")
    cell = gemmi.UnitCell(*metadata["cell"])
    sg = gemmi.SpaceGroup(metadata["spacegroup"])
    model_cell = cell_tuple(st.cell)
    if not np.allclose(np.asarray(model_cell), np.asarray(metadata["cell"]), rtol=1e-5, atol=1e-3):
        raise ValueError(
            f"Model/reflection unit-cell mismatch: model={model_cell}, reflections={tuple(metadata['cell'])}"
        )
    if st.spacegroup_hm:
        model_sg = gemmi.SpaceGroup(st.spacegroup_hm)
        if model_sg.xhm() != sg.xhm():
            raise ValueError(f"Model/reflection space-group mismatch: model={model_sg.xhm()}, reflections={sg.xhm()}")
    else:
        st.spacegroup_hm = sg.xhm()
    return st, cell, sg


def make_model_density(cfg):
    out = cfg.run.data_dir
    metadata = json.loads((out / "metadata.json").read_text())
    st, cell, sg = _model_for_metadata(cfg, metadata)
    shape = tuple(int(x) for x in metadata["grid_shape"])
    rho, dc = model_density_on_grid(
        st[0], cell, sg, shape, cfg.resolution.d_min_angstrom, cfg.baseline.density_cutoff, cfg.input.scattering
    )
    np.save(out / "rho0.npy", rho)
    _require_rs().io.write_ccp4_map(rho, str(out / "rho0.ccp4"), cell, sg)
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
        "scattering": cfg.input.scattering,
    }
    (out / "rho0_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def check_model_density(cfg, n_sample=256):
    """Verify the generated rho0 really is the crystallographic model's density.

    Two independent checks, neither of which shares code with the construction path:
    the integrated density must equal the model's electron count, and the FFT of rho0
    must reproduce structure factors obtained by direct summation over atoms.
    """
    out = cfg.run.data_dir
    metadata = json.loads((out / "metadata.json").read_text())
    st, cell, sg = _model_for_metadata(cfg, metadata)
    rho = np.load(out / "rho0.npy")
    volume = float(metadata["unit_cell_volume"])

    integrated = float(rho.sum()) * volume / rho.size
    expected = model_electron_count(st[0], sg)
    electron_error = abs(integrated - expected) / max(expected, 1.0)

    refl = np.load(out / "reflections.npz")
    hkls = np.asarray(refl["hkls"], dtype=np.int32)
    step = max(1, len(hkls) // n_sample)
    sample = hkls[::step][:n_sample]
    direct = direct_structure_factors(st[0], cell, sg, sample, cfg.resolution.d_min_angstrom, cfg.input.scattering)
    # jnp.fft.fftn uses exp(-2 pi i h.r); Gemmi's crystallographic convention is
    # exp(+2 pi i h.r), so the grid transform is the conjugate of the direct sum.
    fgrid = np.fft.fftn(rho.astype(np.float64)) * (volume / rho.size)
    idx = np.mod(sample, np.asarray(rho.shape))
    gridded = np.conj(fgrid[idx[:, 0], idx[:, 1], idx[:, 2]])
    amplitude_error = float(
        np.linalg.norm(np.abs(gridded) - np.abs(direct)) / max(np.linalg.norm(np.abs(direct)), 1e-30)
    )

    result = {
        "n_structure_factors_checked": len(sample),
        "integrated_electrons": integrated,
        "model_electrons": expected,
        "electron_count_relative_error": electron_error,
        "amplitude_relative_l2_error": amplitude_error,
        "electron_count_tolerance": 0.02,
        "amplitude_tolerance": 0.05,
    }
    result["pass"] = bool(electron_error < 0.02 and amplitude_error < 0.05)
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run.output_dir / "rho0_check.json").write_text(json.dumps(result, indent=2))
    if not result["pass"]:
        raise RuntimeError(
            "The starting density does not match the crystallographic model "
            f"(electron-count error {electron_error:.3%}, amplitude error {amplitude_error:.3%}). "
            "Refusing to fit a correction to a density that is not rho0."
        )
    return result


def make_solvent_mask(cfg):
    out = cfg.run.data_dir
    metadata = json.loads((out / "metadata.json").read_text())
    st, cell, sg = _model_for_metadata(cfg, metadata)
    shape = tuple(int(x) for x in metadata["grid_shape"])
    grid = gemmi.FloatGrid()
    grid.spacegroup = sg
    grid.set_unit_cell(cell)
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
                f"Solvent mask grid {mask.shape} does not match rho0 grid {rho0_shape}. Both must share one "
                "grid or the bulk-solvent term is gathered from the wrong Fourier coefficients."
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
    """Describe a diffuse-scattering HDF5 file. v3 never uses its contents."""
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
