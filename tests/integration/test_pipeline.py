"""End-to-end pipeline on a synthetic dataset.

This walks the same stages the SLURM DAG runs, on a grid small enough for CPU CI:
prepare, rho0, solvent mask, scaling, the FFT and derivative gates, a three-iteration
fit, the information spectrum, the freeze and the one-shot free evaluation.
"""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_prepare_writes_a_consistent_split_and_metadata(prepared_dataset):
    cfg = prepared_dataset["cfg"]
    metadata = prepared_dataset["metadata"]
    data = np.load(cfg.run.data_dir / "reflections.npz")

    assert metadata["n_train"] > metadata["n_tune"] > 0
    assert metadata["n_free"] > 0
    assert len(data["hkls"]) == metadata["n_total_kept"]
    assert metadata["d_min"] >= cfg.resolution.d_min_angstrom
    assert (cfg.run.data_dir / "splits.json").exists()
    assert json.loads((cfg.run.data_dir / "metadata.json").read_text())["grid_shape"] == list(metadata["grid_shape"])


def test_rho0_solvent_mask_and_scaling_share_one_grid(prepared_dataset):
    cfg = prepared_dataset["cfg"]
    declared = tuple(prepared_dataset["metadata"]["grid_shape"])
    assert tuple(np.load(cfg.run.data_dir / "rho0.npy").shape) == declared
    assert tuple(np.load(cfg.run.data_dir / "solvent_mask.npy").shape) == declared
    assert 0.0 < prepared_dataset["mask_stats"]["solvent_fraction"] < 1.0
    assert prepared_dataset["scaling"]["enabled"] is True
    assert (cfg.run.data_dir / "scaling_train.npz").exists()


def test_rho0_matches_the_crystallographic_model(prepared_dataset):
    """Section 14: the starting density must be verifiably the model's density."""
    from crystal_field.crystallography.io import check_model_density

    result = check_model_density(prepared_dataset["cfg"])
    assert result["pass"] is True
    assert result["electron_count_relative_error"] < 0.02
    assert result["amplitude_relative_l2_error"] < 0.05


def test_a_grid_mismatch_between_rho0_and_the_mask_is_caught(prepared_dataset):
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = prepared_dataset["cfg"]
    np.save(cfg.run.data_dir / "solvent_mask.npy", np.zeros((4, 4, 4), dtype=np.float32))
    with pytest.raises(RuntimeError, match="Grid mismatch"):
        load_problem_arrays(cfg)


def test_missing_scaling_for_the_fit_scope_is_reported(prepared_dataset):
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = prepared_dataset["cfg"]
    work_scope = cfg.model_copy(update={"optimizer": cfg.optimizer.model_copy(update={"fit_scope": "work"})})
    with pytest.raises(FileNotFoundError, match="fieldrefine fit-scaling"):
        load_problem_arrays(work_scope)


def test_fft_and_derivative_gates_pass(prepared_dataset):
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.validation.derivatives import run_derivative_check
    from crystal_field.validation.fft_check import run_fft_check

    cfg = prepared_dataset["cfg"]
    arrays = load_problem_arrays(cfg)
    fft = run_fft_check(cfg, arrays)
    assert fft["pass_expected_relation"] is True
    assert fft["ambiguous"] is False

    *_, residuals, objective, _, _ = build_functions(arrays, cfg)
    derivative = run_derivative_check(cfg, arrays, residuals, objective)
    assert derivative["pass"] is True
    assert derivative["adjoint_relative_error"] < derivative["adjoint_tolerance"]


def test_fit_improves_the_training_residual_and_writes_its_artifacts(prepared_dataset):
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.output import write_fit_map

    cfg = prepared_dataset["cfg"]
    arrays = load_problem_arrays(cfg)
    _, density, _, _, objective, metrics, _ = build_functions(arrays, cfg)

    import jax.numpy as jnp

    baseline = {k: float(v) for k, v in metrics(jnp.zeros_like(arrays.rho0)).items()}
    result = run_map_fit(cfg, arrays, objective, metrics, density)
    write_fit_map(cfg)

    assert result["chi2_train"] <= baseline["chi2_train"]
    assert result["fit_scope"] == "train"
    for name in ("z_map.npy", "rho_map_raw.npy", "rho_map.npy", "metrics.json", "history.csv"):
        assert (cfg.run.output_dir / "fit" / name).exists(), name
    assert np.load(cfg.run.output_dir / "fit" / "z_map.npy").shape == tuple(prepared_dataset["metadata"]["grid_shape"])


def test_information_spectrum_runs_matrix_free(prepared_dataset):
    from crystal_field.inference.information import run_information_spectrum
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = prepared_dataset["cfg"]
    arrays = load_problem_arrays(cfg)
    _, _, _, residuals, _, _, _ = build_functions(arrays, cfg)
    summary = run_information_spectrum(cfg, arrays, residuals)

    assert summary["n_modes"] == cfg.information.n_modes
    assert len(summary["eigenvalues"]) == cfg.information.n_modes
    assert all(value >= 0 for value in summary["eigenvalues"])
    assert summary["d_eff_lower_bound_from_top_modes"] <= cfg.information.n_modes
    saved = np.load(cfg.run.output_dir / "info_spectrum.npz")
    assert saved["eigenvectors_latent"].shape[1] == cfg.information.n_modes


def test_work_refit_freeze_and_one_shot_free_evaluation(prepared_dataset, tmp_path):
    """The final third of the DAG, including the guarantee that free is read once."""
    from crystal_field.analysis.model_selection import freeze_model, read_free_set_ledger, verify_lock
    from crystal_field.config import dump_config
    from crystal_field.crystallography.scaling import fit_baseline_scaling
    from crystal_field.inference.information import run_information_spectrum
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    base = prepared_dataset["cfg"]
    cfg = base.model_copy(
        update={
            "run": base.run.model_copy(update={"output_dir": tmp_path / "final"}),
            "optimizer": base.optimizer.model_copy(update={"fit_scope": "work"}),
        }
    )
    config_path = tmp_path / "selected.yaml"
    dump_config(cfg, config_path)

    fit_baseline_scaling(cfg)
    arrays = load_problem_arrays(cfg)
    _, density, _, residuals, objective, metrics, free_metrics = build_functions(arrays, cfg)
    run_map_fit(cfg, arrays, objective, metrics, density)
    run_information_spectrum(cfg, arrays, residuals)

    lock = freeze_model(config_path)
    assert json.loads(_read(lock["lock"]))["free_set_used_at_freeze"] is False
    assert verify_lock(config_path)["free_set_evaluations"] == []

    import jax.numpy as jnp

    z = jnp.asarray(np.load(cfg.run.output_dir / "fit" / "z_map.npy"), dtype=arrays.rho0.dtype)
    z0 = jnp.zeros_like(z)
    r_work_atomic = float(metrics(z0)["r_work"])
    r_work_field = float(metrics(z)["r_work"])
    r_free_atomic = float(free_metrics(z0)["r_free"])
    r_free_field = float(free_metrics(z)["r_free"])
    assert r_work_field <= r_work_atomic, "a work-scope fit must not worsen the work residual"
    assert all(np.isfinite([r_free_atomic, r_free_field]))

    from crystal_field.analysis.model_selection import record_free_evaluation

    out = cfg.run.output_dir / "FREE_EVALUATION.json"
    out.write_text(json.dumps({"r_free_field": r_free_field, "r_free_atomic": r_free_atomic}))
    record_free_evaluation(cfg, verify_lock(config_path), out, {"delta_r_free": r_free_field - r_free_atomic})

    assert len(read_free_set_ledger(cfg)) == 1
    with pytest.raises(RuntimeError, match="one-shot"):
        verify_lock(config_path)


def _read(path):
    from pathlib import Path

    return Path(path).read_text()
