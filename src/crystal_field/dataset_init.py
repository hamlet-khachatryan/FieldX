"""Turn a PDB identifier into a ready-to-run FieldX v3 dataset configuration.

`fieldrefine init-pdb <PDBID>` downloads coordinates and deposited structure factors,
detects the crystallographic metadata and reflection columns, and writes both the
dataset configuration and its dataset-specific prior grid. The core pipeline has no
per-entry special cases; 1UBQ and 6O2H exist only as committed example configs.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from crystal_field.config import AppConfig, load_config
from crystal_field.crystallography.pdb import discover_reflection_columns, download_pdb_entry, normalize_pdb_id

# The v3 pilot grid: three Matern correlation lengths plus a squared-exponential
# control, all with Gaussian latent statistics. Heavy-tailed latents and multiscale
# mixtures are deliberately excluded from a first experiment -- the pilot tests the
# existing v3 formulation, not extra prior flexibility.
V3_PILOT_PRIOR_GRID = {
    "candidates": [
        {
            "name": "matern_fine",
            "prior": {
                "kernel": "matern",
                "tau_density": 0.025,
                "correlation_length_angstrom": 0.50,
                "alpha": 2.0,
                "latent_distribution": "gaussian",
            },
        },
        {
            "name": "matern_baseline",
            "prior": {
                "kernel": "matern",
                "tau_density": 0.040,
                "correlation_length_angstrom": 0.75,
                "alpha": 2.5,
                "latent_distribution": "gaussian",
            },
        },
        {
            "name": "matern_broad",
            "prior": {
                "kernel": "matern",
                "tau_density": 0.050,
                "correlation_length_angstrom": 1.50,
                "alpha": 2.5,
                "latent_distribution": "gaussian",
            },
        },
        {
            "name": "squared_exponential",
            "prior": {
                "kernel": "squared_exponential",
                "tau_density": 0.040,
                "correlation_length_angstrom": 0.80,
                "latent_distribution": "gaussian",
            },
        },
    ]
}


def default_data_root() -> Path:
    return Path(os.environ.get("FIELDX_DATA_ROOT", "data"))


def default_runs_root() -> Path:
    return Path(os.environ.get("FIELDX_RUNS_ROOT", "runs"))


def build_config_payload(entry: dict, refl: dict, runs_root: Path, no_free_set: bool = False) -> dict:
    """Compose a physically sensible v3 configuration from discovered metadata."""
    has_free = refl["free"] is not None and refl["free_test_value"] is not None
    if no_free_set:
        strategy = "none"
    elif has_free:
        strategy = "existing_free_then_hash"
    else:
        strategy = "hash"
    return {
        "run": {
            "output_dir": str((Path(runs_root) / entry["pdb_id"]).resolve()),
            "seed": 20260907,
            "enable_x64": False,
        },
        "input": {
            "reflections": str(Path(entry["reflections"]).resolve()),
            "model": str(Path(entry["model"]).resolve()),
            "scattering": "xray",
            "observation_kind": "amplitude",
            "columns": {
                "observation": refl["observation"],
                "sigma": refl["sigma"],
                "free": refl["free"],
            },
            "mtz_dataset": (refl.get("mtz_selection") or {}).get("dataset_id"),
            "free_test_value": refl["free_test_value"] if has_free and not no_free_set else None,
            "expected_free_fraction_min": 0.02,
            "expected_free_fraction_max": 0.15,
        },
        "split": {
            # No usable deposited free flag means a deterministic hash holdout. The
            # resulting statistic is a genuine held-out test statistic but is NOT the
            # deposited crystallographic R-free, and must be reported as such.
            # `none` holds nothing out at all and targets R_work only.
            "strategy": strategy,
            "tune_fraction_of_work": 0.10,
            "final_fraction_if_hash": 0.10,
            "pair_friedel_when_nonanomalous": True,
            "anomalous": False,
        },
        "resolution": {
            "d_min_angstrom": float(refl["d_min"]),
            "d_max_angstrom": float(refl["d_max"]),
        },
        # samples_per_dmin = 3.0 is the v3 convention; the grid itself is derived from
        # the cell and d_min by Gemmi, never assumed.
        "grid": {"shape": None, "samples_per_dmin": 3.0},
        "prior": {
            "kernel": "matern",
            "tau_density": 0.040,
            "correlation_length_angstrom": 0.75,
            "alpha": 2.5,
            "components": [],
            "remove_mean": True,
            "latent_distribution": "gaussian",
            "student_t_df": 4.0,
            "laplace_softening": 1.0e-3,
            "cauchy_scale": 1.0,
        },
        "baseline": {
            "density_cutoff": 1.0e-6,
            "bulk_solvent": {"enabled": True, "radii": "refmac", "k_sol": 0.35, "b_sol_angstrom2": 46.0},
            "scaling": {"enabled": True, "fit_isotropic_b_first": True},
        },
        "likelihood": {
            "global_scale": "profile",
            "fixed_scale": 1.0,
            "sigma_floor": 1.0e-6,
            "loss": "gaussian",
            "student_t_df": 6.0,
        },
        # A first run stays small: measure compilation time, time per iteration, GPU
        # memory and convergence before enlarging anything.
        "optimizer": {
            "method": "lbfgs",
            "fit_scope": "train",
            "max_iterations": 100,
            "tolerance": 5.0e-5,
            "lbfgs_memory": 8,
            "adam_learning_rate": 3.0e-3,
            "checkpoint_every": 10,
            "validation_patience": 8,
            "validation_min_delta": 1.0e-5,
        },
        "information": {"n_modes": 8, "max_iterations": 80, "tolerance": 1.0e-4, "memory_budget_gib": 40.0},
        "atomic_benchmark": {
            "enabled": True,
            "n_atoms": 8,
            "coordinate_delta_angstrom": 0.02,
            "b_delta_angstrom2": 0.5,
            "occupancy_delta": 0.02,
        },
    }


def init_pdb_datasets(
    pdb_ids,
    data_root: Path | None = None,
    runs_root: Path | None = None,
    force: bool = False,
    no_free_set: bool = False,
    fail_fast: bool = False,
) -> dict:
    """Initialise several PDB entries in one call.

    Every identifier is validated and de-duplicated *before* anything is downloaded, so a
    typo in the last argument does not cost four downloads first. Entries are then
    processed one at a time, each isolated from the others: one entry failing -- most
    often because it deposits no structure factors -- does not stop the others, and every
    failure is collected and reported at the end. Pass `fail_fast` to stop at the first
    one instead.

    Downloads are sequential on purpose -- these are public RCSB endpoints, not a service
    to parallelise against.
    """
    requested = list(pdb_ids)
    if not requested:
        raise ValueError("Give at least one PDB identifier")

    # Guard 1: validate everything up front.
    invalid, normalized = [], []
    for raw in requested:
        try:
            normalized.append(normalize_pdb_id(raw))
        except ValueError as exc:
            invalid.append(f"{raw!r}: {exc}")
    if invalid:
        raise ValueError("Invalid PDB identifier(s) -- nothing was downloaded:\n  - " + "\n  - ".join(invalid))

    # Guard 2: a repeated entry is a mistake, and would race on the same output paths.
    duplicates = sorted({pdb_id for pdb_id in normalized if normalized.count(pdb_id) > 1})
    if duplicates:
        raise ValueError(
            f"Repeated PDB identifier(s) {[d.upper() for d in duplicates]}; they would write to the same paths"
        )

    succeeded, failed, skipped = [], [], []
    for index, pdb_id in enumerate(normalized):
        try:
            succeeded.append(
                init_pdb_dataset(
                    pdb_id,
                    data_root=data_root,
                    runs_root=runs_root,
                    force=force,
                    no_free_set=no_free_set,
                )
            )
        except Exception as exc:
            failed.append({"pdb_id": pdb_id.upper(), "error_type": type(exc).__name__, "error": str(exc)})
            if fail_fast:
                skipped = [p.upper() for p in normalized[index + 1 :]]
                break

    return {
        "requested": [p.upper() for p in normalized],
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "n_skipped": len(skipped),
        "succeeded": succeeded,
        "failed": failed,
        "skipped_after_failure": skipped,
        "ok": not failed,
    }


def init_pdb_dataset(
    pdb_id: str,
    data_root: Path | None = None,
    runs_root: Path | None = None,
    config_out: Path | None = None,
    force: bool = False,
    no_free_set: bool = False,
) -> dict:
    pdb_id = normalize_pdb_id(pdb_id)
    data_root = Path(data_root) if data_root is not None else default_data_root()
    runs_root = Path(runs_root) if runs_root is not None else default_runs_root()
    config_path = Path(config_out) if config_out is not None else Path("configs") / pdb_id / "default.yaml"
    grid_path = config_path.parent / "prior_grid.yaml"

    entry = download_pdb_entry(pdb_id, data_root)
    refl = discover_reflection_columns(entry["reflections"])

    if config_path.exists() and not force:
        cfg = load_config(config_path)
        written = []
    else:
        payload = build_config_payload(entry, refl, runs_root, no_free_set=no_free_set)
        AppConfig.model_validate(payload)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        cfg = load_config(config_path)
        written = [str(config_path)]

    if not grid_path.exists() or force:
        grid_path.write_text(yaml.safe_dump(V3_PILOT_PRIOR_GRID, sort_keys=False), encoding="utf-8")
        written.append(str(grid_path))

    return {
        "pdb_id": entry["pdb_id"],
        "config": str(config_path),
        "prior_grid": str(grid_path),
        "written": written,
        "kept_existing_config": config_path.exists() and not written,
        "data_dir": str((data_root / pdb_id).resolve()),
        "output_dir": str(cfg.run.output_dir),
        "entry": entry,
        "reflections": refl,
        "free_set": {
            "existing_free_then_hash": "deposited free flags",
            "hash": "deterministic hash holdout (NOT the deposited R-free set)",
            "none": "no held-out set; target is R_work only and R_free is not defined",
        }[cfg.split.strategy],
        "target": "R_work and R_free" if cfg.has_free_set else "R_work only",
    }
