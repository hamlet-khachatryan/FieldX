"""FieldX command line.

Every command takes a configuration path. Login-node commands (inspect, config-check,
init-pdb, expand-priors, estimate-memory, select-prior, freeze-model) do no accelerator
work; everything else is intended to run inside a SLURM GPU job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich import print

from crystal_field.config import load_config

app = typer.Typer(
    no_args_is_help=True,
    help="FieldX: correlated density-field refinement of crystallographic electron density (v3)",
)


DATA_ROOT_OPTION = typer.Option(None, help="Where datasets live; defaults to $FIELDX_DATA_ROOT or ./data")
RUNS_ROOT_OPTION = typer.Option(None, help="Where runs live; defaults to $FIELDX_RUNS_ROOT or ./runs")
CONFIG_OUT_OPTION = typer.Option(None, help="Config path; defaults to configs/<pdbid>/default.yaml")
FORCE_OPTION = typer.Option(False, "--force", help="Regenerate an existing config and prior grid")
SIGMA_A_OPTION = typer.Option(
    "work",
    "--sigma-a-from",
    help="Reflections used to estimate sigma-A weights: 'work' (safe, slightly biased) or "
    "'free' (unbiased, permitted only after the one-shot free evaluation has happened).",
)
SIGMA_A_BINS_OPTION = typer.Option(20, "--sigma-a-bins", help="Resolution shells for sigma-A estimation.")
PDB_IDS_ARGUMENT = typer.Argument(..., metavar="PDBID...", help="One or more four-character PDB identifiers")
FAIL_FAST_OPTION = typer.Option(
    False,
    "--fail-fast",
    help="Stop at the first failed entry. By default the batch continues and every failure is reported.",
)
NO_FREE_SET_OPTION = typer.Option(
    False,
    "--no-free-set",
    help="Hold nothing out: split.strategy=none, target R_work only. R_free is then undefined "
    "and the primary criterion cannot be evaluated.",
)
DECOMPOSE_BASIS_OPTION = typer.Option(
    None, "--basis", help="Override decomposition.basis: coordinates, coordinates_b, full or residue_rigid."
)
DECOMPOSE_TRIALS_OPTION = typer.Option(
    None,
    "--n-trials",
    # The same `ge=1` the configuration enforces. Without it `--n-trials 0` falls through
    # the `or` in run_decomposition to the configured default, and a negative value makes
    # the control a mean over no trials -- NaN, which json.dumps writes as bare `NaN`.
    min=1,
    help="Override decomposition.n_capacity_trials (at least 1).",
)


def _cfg(path):
    return load_config(path)


def _configure(cfg):
    from crystal_field.backend.jax_backend import configure_jax

    configure_jax(cfg.run.enable_x64, cfg.run.compilation_cache_dir)


@app.command("config-check")
def config_check(config: Path):
    """Validate a configuration and its inputs without touching an accelerator."""
    from crystal_field.config import check_config

    print(json.dumps(check_config(config), indent=2))


@app.command("init-pdb")
def init_pdb(
    pdb_ids: list[str] = PDB_IDS_ARGUMENT,
    data_root: Path | None = DATA_ROOT_OPTION,
    runs_root: Path | None = RUNS_ROOT_OPTION,
    config_out: Path | None = CONFIG_OUT_OPTION,
    force: bool = FORCE_OPTION,
    no_free_set: bool = NO_FREE_SET_OPTION,
    fail_fast: bool = FAIL_FAST_OPTION,
):
    """Download one or more PDB entries and write their configs and prior grids.

    Identifiers are all validated before anything is downloaded. One entry failing does
    not stop the others: the batch continues and every failure is reported at the end,
    with a non-zero exit code. Pass --fail-fast to stop at the first one instead.
    """
    from crystal_field.dataset_init import init_pdb_dataset, init_pdb_datasets

    if config_out is not None:
        if len(pdb_ids) != 1:
            raise typer.BadParameter(
                f"--config-out names a single file but {len(pdb_ids)} identifiers were given; "
                "omit it to write configs/<pdbid>/default.yaml for each."
            )
        print(json.dumps(init_pdb_dataset(pdb_ids[0], data_root, runs_root, config_out, force, no_free_set), indent=2))
        return

    report = init_pdb_datasets(pdb_ids, data_root, runs_root, force, no_free_set, fail_fast)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command()
def inspect(config: Path):
    """Report reflection metadata, column suggestions and the derived FFT grid."""
    from crystal_field.crystallography.io import inspect_dataset

    print(json.dumps(inspect_dataset(_cfg(config)), indent=2))


@app.command("inspect-h5")
def inspect_h5(config: Path):
    """Describe the associated diffuse-scattering file. v3 never uses its contents."""
    cfg = _cfg(config)
    if cfg.input.diffuse_h5 is None:
        raise typer.BadParameter("input.diffuse_h5 is not configured")
    from crystal_field.crystallography.io import inspect_h5 as inspect_h5_file

    print(json.dumps(inspect_h5_file(cfg.input.diffuse_h5), indent=2))


@app.command()
def prepare(config: Path):
    """Filter reflections and write the immutable train/tune/free split."""
    from crystal_field.crystallography.io import prepare_reflections

    print(json.dumps(prepare_reflections(_cfg(config)), indent=2))


@app.command("make-rho0")
def make_rho0(config: Path):
    """Build the starting atomistic density on the declared grid."""
    from crystal_field.crystallography.io import make_model_density

    print(json.dumps(make_model_density(_cfg(config)), indent=2))


@app.command("check-rho0")
def check_rho0(config: Path):
    """Verify rho0 against the crystallographic model (electron count and direct-sum F)."""
    from crystal_field.crystallography.io import check_model_density

    print(json.dumps(check_model_density(_cfg(config)), indent=2))


@app.command("make-solvent-mask")
def make_solvent_mask(config: Path):
    from crystal_field.crystallography.io import make_solvent_mask as build_mask

    print(json.dumps(build_mask(_cfg(config)), indent=2))


@app.command("fit-scaling")
def fit_scaling(config: Path):
    """Fit the Gemmi overall/bulk-solvent nuisance scaling on the current fit scope."""
    from crystal_field.crystallography.scaling import fit_baseline_scaling

    print(json.dumps(fit_baseline_scaling(_cfg(config)), indent=2))


@app.command("device-report")
def device_report(config: Path):
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.backend.jax_backend import device_report as report

    print(json.dumps(report(), indent=2))


@app.command("estimate-memory")
def estimate_memory_cmd(config: Path):
    """Estimate GPU memory for the field, FFT buffers, optimizer state and LOBPCG basis."""
    from crystal_field.hpc import estimate_memory

    print(json.dumps(estimate_memory(_cfg(config)), indent=2))


@app.command("baseline-metrics")
def baseline_metrics(config: Path):
    """Metrics of the unmodified atomic baseline, i.e. the field model at z = 0."""
    cfg = _cfg(config)
    _configure(cfg)
    import jax
    import jax.numpy as jnp

    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    arrays = load_problem_arrays(cfg)
    *_, metrics, _ = build_functions(arrays, cfg)
    result = {k: float(v) for k, v in jax.jit(metrics)(jnp.zeros_like(arrays.rho0)).items()}
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run.output_dir / "baseline_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


@app.command("fft-check")
def fft_check(config: Path):
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.validation.fft_check import run_fft_check

    print(json.dumps(run_fft_check(cfg, load_problem_arrays(cfg)), indent=2))


@app.command("derivative-check")
def derivative_check(config: Path):
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.validation.derivatives import run_derivative_check

    arrays = load_problem_arrays(cfg)
    *_, residuals, objective, _, _ = build_functions(arrays, cfg)
    print(json.dumps(run_derivative_check(cfg, arrays, residuals, objective), indent=2))


@app.command("atomic-benchmark")
def atomic_benchmark(config: Path):
    """Latent cost of elementary atomic refinement directions under the current prior."""
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.crystallography.atomic_benchmark import run_atomic_benchmark

    print(json.dumps(run_atomic_benchmark(cfg), indent=2))


@app.command("information-spectrum")
def information_spectrum(config: Path):
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.inference.information import run_information_spectrum
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    arrays = load_problem_arrays(cfg)
    _, _, _, residuals, _, _, _ = build_functions(arrays, cfg)
    print(json.dumps(run_information_spectrum(cfg, arrays, residuals), indent=2))


@app.command("fit-map")
def fit_map(config: Path):
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.output import write_fit_map

    arrays = load_problem_arrays(cfg)
    _, density, _, _, objective, metrics, _ = build_functions(arrays, cfg)
    result = run_map_fit(cfg, arrays, objective, metrics, density)
    write_fit_map(cfg)
    print(json.dumps(result, indent=2))


@app.command("export-maps")
def export_maps_cmd(config: Path, sigma_a_from: str = SIGMA_A_OPTION, n_bins: int = SIGMA_A_BINS_OPTION):
    """Write the CCP4 map set: starting, inferred, refined, 2Fo-Fc and 2mFo-DFc."""
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.maps import export_maps

    print(json.dumps(export_maps(cfg, sigma_a_from=sigma_a_from, n_bins=n_bins), indent=2))


@app.command("decompose")
def decompose_cmd(
    config: Path,
    basis: str | None = DECOMPOSE_BASIS_OPTION,
    n_trials: int | None = DECOMPOSE_TRIALS_OPTION,
):
    """Decompose the inferred correction onto the atomic tangent space (read-only)."""
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.analysis.decomposition import run_decomposition

    print(json.dumps(run_decomposition(cfg, basis=basis, n_trials=n_trials), indent=2))


@app.command("expand-priors")
def expand_priors(base_config: Path, prior_grid: Path, output_dir: Path):
    """Expand a prior grid into one candidate configuration per candidate."""
    from crystal_field.analysis.model_selection import expand_prior_grid

    print(json.dumps(expand_prior_grid(base_config, prior_grid, output_dir), indent=2))


@app.command("select-prior")
def select_prior(base_config: Path, manifest: Path, selected_config: Path):
    """Rank candidates on the tune subset only and emit the work-scope refit config."""
    from crystal_field.analysis.model_selection import select_prior as do_select

    print(json.dumps(do_select(base_config, manifest, selected_config), indent=2))


@app.command("freeze-model")
def freeze_model(
    config: Path,
    allow_after_free_evaluation: bool = typer.Option(
        False,
        "--allow-after-free-evaluation",
        help="Re-freeze a revised model even though the free set has already been read. "
        "This forfeits the held-out status of the free set; the repeat is recorded.",
    ),
):
    """Hash the final model into MODEL_LOCK.json. Nothing may change after this point."""
    from crystal_field.analysis.model_selection import freeze_model as do_freeze

    print(json.dumps(do_freeze(config, allow_after_free_evaluation=allow_after_free_evaluation), indent=2))


@app.command("evaluate-free")
def evaluate_free(
    config: Path,
    allow_repeat_free_evaluation: bool = typer.Option(
        False,
        "--allow-repeat-free-evaluation",
        help="Read the free set again after it has already been evaluated. The one-shot "
        "guarantee is void; the repeat is written to a separate file and recorded.",
    ),
):
    """One-shot free-set evaluation. Nothing is optimized here."""
    cfg = _cfg(config)
    from crystal_field.analysis.model_selection import (
        free_set_ledger_path,
        read_free_set_ledger,
        record_free_evaluation,
        verify_lock,
    )

    if not cfg.has_free_set:
        # Nothing is held out, so there is nothing to consume and nothing to verify a
        # lock against. Report that plainly and stop; this is not an error condition.
        report = {
            "has_free_set": False,
            "split_strategy": cfg.split.strategy,
            "target": "R_work only",
            "primary_criterion_met": None,
            "note": (
                "This configuration declares no held-out set, so R_free does not exist and the "
                "primary criterion (R_work AND R_free both improving) cannot be evaluated. A lower "
                "R_work alone is not evidence of recovered density: a correlated field of this "
                "capacity can reduce R_work without predicting anything. Read "
                "final/fit/metrics.json for the work-set result."
            ),
        }
        cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.run.output_dir / "FREE_EVALUATION_SKIPPED.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    lock = verify_lock(config, allow_consumed=allow_repeat_free_evaluation)
    if cfg.optimizer.fit_scope != "work":
        raise typer.BadParameter("Free evaluation requires optimizer.fit_scope=work")
    ledger = read_free_set_ledger(cfg)
    evaluation_index = len(ledger) + 1
    out = cfg.run.output_dir / (
        "FREE_EVALUATION.json" if evaluation_index == 1 else f"FREE_EVALUATION_repeat_{evaluation_index}.json"
    )
    if out.exists() and not allow_repeat_free_evaluation:
        raise typer.BadParameter(
            f"{out} already exists but {free_set_ledger_path(cfg)} records no evaluation. "
            "Refusing to overwrite a free-set result; move the existing file aside deliberately "
            "or pass --allow-repeat-free-evaluation."
        )
    _configure(cfg)
    import jax
    import jax.numpy as jnp

    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    arrays = load_problem_arrays(cfg)
    if not int(np.sum(np.asarray(arrays.split) == 2)):
        raise typer.BadParameter(
            "The prepared split contains no free reflections even though "
            f"split.strategy={cfg.split.strategy!r} declares one. Re-run `fieldrefine prepare`, "
            "or set split.strategy=none to target R_work only."
        )
    *_, metrics, free_metrics = build_functions(arrays, cfg)
    z = jnp.asarray(np.load(cfg.run.output_dir / "fit" / "z_map.npy"), dtype=arrays.rho0.dtype)
    z0 = jnp.zeros_like(z)
    free_fn, work_fn = jax.jit(free_metrics), jax.jit(metrics)

    # z = 0 is exactly the atomic baseline under the same nuisance scaling, so the two
    # models are compared on identical reflections with identical calibration.
    r_work_atomic = float(work_fn(z0)["r_work"])
    r_work_field = float(work_fn(z)["r_work"])
    r_free_atomic = float(free_fn(z0)["r_free"])
    r_free_field = float(free_fn(z)["r_free"])
    gap_atomic = r_free_atomic - r_work_atomic
    gap_field = r_free_field - r_work_field

    result = {
        "r_work_atomic": r_work_atomic,
        "r_work_field": r_work_field,
        "r_free_atomic": r_free_atomic,
        "r_free_field": r_free_field,
        "gap_atomic": gap_atomic,
        "gap_field": gap_field,
        "delta_r_work": r_work_field - r_work_atomic,
        "delta_r_free": r_free_field - r_free_atomic,
        "delta_gap": gap_field - gap_atomic,
        "primary_criterion_met": bool(r_work_field < r_work_atomic and r_free_field < r_free_atomic),
        "free_set_kind": (
            "deposited free flags"
            if cfg.split.strategy == "existing_free_then_hash"
            else "deterministic hash holdout, NOT the deposited R-free set"
        ),
        "n_free": float(free_fn(z)["n_free"]),
        "n_work": float(work_fn(z)["n_work"]),
        "lock_config_sha256": lock["config_sha256"],
        "evaluation_index": evaluation_index,
        "one_shot": evaluation_index == 1,
    }
    out.write_text(json.dumps(result, indent=2))
    entry = record_free_evaluation(
        cfg, lock, out, {k: result[k] for k in ("delta_r_work", "delta_r_free", "delta_gap")}
    )
    print(json.dumps({**result, "ledger_entry": entry}, indent=2))


if __name__ == "__main__":
    app()
