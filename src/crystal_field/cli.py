from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich import print

from crystal_field.config import load_config

app = typer.Typer(no_args_is_help=True, help="Crystal Field Inference: GPU-first crystallographic density-field refinement")


def _cfg(path):
    return load_config(path)


def _configure(cfg):
    from crystal_field.backend.jax_backend import configure_jax
    configure_jax(cfg.run.enable_x64, cfg.run.compilation_cache_dir)


@app.command("config-check")
def config_check(config: Path):
    print(_cfg(config).model_dump(mode="json"))



@app.command("init-pdb")
def init_pdb(pdb_id: str, data_root: Path = Path("data"), config_out: Path | None = None):
    from crystal_field.crystallography.pdb import discover_reflection_columns, download_pdb_entry
    info = download_pdb_entry(pdb_id, data_root)
    refl = discover_reflection_columns(info["reflections"])
    from crystal_field.config import write_pdb_template
    out = config_out or Path("configs") / pdb_id.lower() / "default.yaml"
    write_pdb_template(out, info, refl)
    print(json.dumps({"config": str(out), "entry": info, "reflections": refl}, indent=2))

@app.command()
def inspect(config: Path):
    from crystal_field.crystallography.io import inspect_dataset
    print(json.dumps(inspect_dataset(_cfg(config)), indent=2))


@app.command("inspect-h5")
def inspect_h5(config: Path):
    cfg = _cfg(config)
    if cfg.input.diffuse_h5 is None:
        raise typer.BadParameter("input.diffuse_h5 is not configured")
    from crystal_field.crystallography.io import inspect_h5 as inspect_h5_file
    print(json.dumps(inspect_h5_file(cfg.input.diffuse_h5), indent=2))


@app.command()
def prepare(config: Path):
    from crystal_field.crystallography.io import prepare_reflections
    print(json.dumps(prepare_reflections(_cfg(config)), indent=2))


@app.command("make-rho0")
def make_rho0(config: Path):
    from crystal_field.crystallography.io import make_model_density
    print(json.dumps(make_model_density(_cfg(config)), indent=2))


@app.command("make-solvent-mask")
def make_solvent_mask(config: Path):
    from crystal_field.crystallography.io import make_solvent_mask
    print(json.dumps(make_solvent_mask(_cfg(config)), indent=2))


@app.command("fit-scaling")
def fit_scaling(config: Path):
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
    from crystal_field.hpc import estimate_memory
    print(json.dumps(estimate_memory(_cfg(config)), indent=2))


@app.command("baseline-metrics")
def baseline_metrics(config: Path):
    cfg = _cfg(config)
    _configure(cfg)
    import jax
    import jax.numpy as jnp

    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays
    arrays = load_problem_arrays(cfg)
    *_, metrics, _ = build_functions(arrays, cfg)
    z = jnp.zeros_like(arrays.rho0)
    result = {k: float(v) for k, v in jax.jit(metrics)(z).items()}
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


@app.command("expand-priors")
def expand_priors(base_config: Path, prior_grid: Path, output_dir: Path):
    from crystal_field.analysis.model_selection import expand_prior_grid
    print(json.dumps(expand_prior_grid(base_config, prior_grid, output_dir), indent=2))


@app.command("select-prior")
def select_prior(base_config: Path, manifest: Path, selected_config: Path):
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
    cfg = _cfg(config)
    from crystal_field.analysis.model_selection import (
        free_set_ledger_path,
        read_free_set_ledger,
        record_free_evaluation,
        verify_lock,
    )
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
    *_, metrics, free_metrics = build_functions(arrays, cfg)
    z = jnp.asarray(np.load(cfg.run.output_dir / "fit" / "z_map.npy"), dtype=arrays.rho0.dtype)
    z0 = jnp.zeros_like(z)
    free_fn = jax.jit(free_metrics)
    work_fn = jax.jit(metrics)
    field_free = {k: float(v) for k, v in free_fn(z).items()}
    base_free = {k: float(v) for k, v in free_fn(z0).items()}
    field_work = {k: float(v) for k, v in work_fn(z).items()}
    base_work = {k: float(v) for k, v in work_fn(z0).items()}
    result = {
        "baseline": {"r_work": base_work["r_work"], "r_free": base_free["r_free"]},
        "field": {"r_work": field_work["r_work"], "r_free": field_free["r_free"]},
        "delta": {
            "r_work": field_work["r_work"] - base_work["r_work"],
            "r_free": field_free["r_free"] - base_free["r_free"],
            "gap": (field_free["r_free"] - field_work["r_work"]) - (base_free["r_free"] - base_work["r_work"]),
        },
        "baseline_gap": base_free["r_free"] - base_work["r_work"],
        "field_gap": field_free["r_free"] - field_work["r_work"],
        "lock_config_sha256": lock["config_sha256"],
        "evaluation_index": evaluation_index,
        "one_shot": evaluation_index == 1,
    }
    out.write_text(json.dumps(result, indent=2))
    entry = record_free_evaluation(cfg, lock, out, result["delta"])
    print(json.dumps({**result, "ledger_entry": entry}, indent=2))


if __name__ == "__main__":
    app()
