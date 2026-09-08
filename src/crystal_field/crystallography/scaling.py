from __future__ import annotations

import json

import gemmi
import numpy as np


def _grid_from_array(array: np.ndarray, cell: gemmi.UnitCell, sg: gemmi.SpaceGroup) -> gemmi.FloatGrid:
    grid = gemmi.FloatGrid(np.asarray(array, dtype=np.float32))
    grid.set_unit_cell(cell)
    grid.spacegroup = sg
    return grid


def _gather_grid_values(grid: gemmi.ReciprocalComplexGrid, hkls: np.ndarray) -> np.ndarray:
    return np.asarray([grid.get_value(int(h), int(k), int(l)) for h, k, l in hkls], dtype=np.complex64)


def fit_baseline_scaling(cfg) -> dict:
    if not cfg.baseline.scaling.enabled:
        return {"enabled": False}
    if cfg.input.observation_kind != "amplitude":
        raise ValueError("Gemmi scaling requires amplitude observations")

    data_dir = cfg.run.data_dir
    metadata = json.loads((data_dir / "metadata.json").read_text())
    refl = np.load(data_dir / "reflections.npz")
    hkls = np.asarray(refl["hkls"], dtype=np.int32)
    obs = np.asarray(refl["observation"], dtype=np.float32)
    sig = np.asarray(refl["sigma"], dtype=np.float32)
    split = np.asarray(refl["split"], dtype=np.int8)
    d = np.asarray(refl["dHKL"], dtype=np.float64)

    if cfg.optimizer.fit_scope == "train":
        fit_mask = split == 0
    elif cfg.optimizer.fit_scope == "work":
        fit_mask = split != 2
    else:
        raise ValueError(cfg.optimizer.fit_scope)

    rho = np.load(data_dir / "rho0.npy", mmap_mode="r")
    solvent = np.load(data_dir / "solvent_mask.npy", mmap_mode="r")
    cell = gemmi.UnitCell(*metadata["cell"])
    sg = gemmi.SpaceGroup(metadata["spacegroup"])

    f_cryst_grid = gemmi.transform_map_to_f_phi(_grid_from_array(rho, cell, sg))
    f_mask_grid = gemmi.transform_map_to_f_phi(_grid_from_array(solvent, cell, sg))
    f_cryst = _gather_grid_values(f_cryst_grid, hkls[fit_mask])
    f_mask = _gather_grid_values(f_mask_grid, hkls[fit_mask])

    calc = gemmi.ComplexAsuData(cell, sg, hkls[fit_mask], f_cryst)
    mask = gemmi.ComplexAsuData(cell, sg, hkls[fit_mask], f_mask)
    obs_sigma = np.ascontiguousarray(np.column_stack([obs[fit_mask], sig[fit_mask]]), dtype=np.float32)
    observed = gemmi.ValueSigmaAsuData(cell, sg, hkls[fit_mask], obs_sigma)

    scaling = gemmi.Scaling(cell, sg)
    scaling.use_solvent = bool(cfg.baseline.bulk_solvent.enabled)
    scaling.k_sol = float(cfg.baseline.bulk_solvent.k_sol)
    scaling.b_sol = float(cfg.baseline.bulk_solvent.b_sol_angstrom2)
    scaling.prepare_points(calc, observed, mask if scaling.use_solvent else None)
    if cfg.baseline.scaling.fit_isotropic_b_first:
        scaling.fit_isotropic_b_approximately()
    wssr = float(scaling.fit_parameters())

    overall = np.asarray(scaling.get_overall_scale_factor(hkls), dtype=np.float32)
    stol2 = 0.25 / np.maximum(d * d, np.finfo(np.float64).tiny)
    solvent_scale = np.asarray([scaling.get_solvent_scale(float(x)) for x in stol2], dtype=np.float32)
    if not scaling.use_solvent:
        solvent_scale.fill(0.0)

    path = data_dir / f"scaling_{cfg.optimizer.fit_scope}.npz"
    np.savez_compressed(path, overall_scale=overall, solvent_scale=solvent_scale)
    result = {
        "enabled": True,
        "fit_scope": cfg.optimizer.fit_scope,
        "n_fit": int(np.sum(fit_mask)),
        "wssr": wssr,
        "rmse": float(np.sqrt(wssr / max(np.sum(fit_mask), 1))),
        "r_factor": float(scaling.calculate_r_factor()),
        "k_overall": float(scaling.k_overall),
        "k_sol": float(scaling.k_sol),
        "b_sol_angstrom2": float(scaling.b_sol),
        "b_overall_voigt": [float(x) for x in scaling.b_overall.elements_voigt()],
        "scale_file": str(path),
    }
    out = cfg.run.output_dir / f"scaling_{cfg.optimizer.fit_scope}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
