"""Datasets with no held-out set at all (split.strategy = none).

Some entries cannot support a holdout. In that case the only target is R_work and every
free-set stage must degrade to a clearly labelled no-op rather than raising. The
scientific caveat is carried in the output, not dropped: with nothing held out, a lower
R_work is not evidence of recovered density.
"""

import json
import pathlib

import numpy as np
import pytest
import yaml
from conftest import config_payload, problem_arrays, problem_config, synthetic_reflections, write_mtz, write_tiny_model
from typer.testing import CliRunner

from crystal_field.cli import app
from crystal_field.config import AppConfig, check_config, resolve_payload_paths
from crystal_field.crystallography.io import prepare_reflections
from crystal_field.inference.problem import build_functions

runner = CliRunner()


@pytest.fixture
def no_free_dataset(tmp_path):
    model = write_tiny_model(tmp_path / "tiny.pdb")
    hkls, observed, sigma, free = synthetic_reflections(model)
    mtz = write_mtz(tmp_path / "tiny.mtz", hkls, observed, sigma, free)
    payload = config_payload(tmp_path, model, mtz, split={"strategy": "none"}, optimizer={"fit_scope": "work"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    cfg = AppConfig.model_validate(resolve_payload_paths(dict(payload), tmp_path))
    return {"cfg": cfg, "config_path": config_path, "payload": payload}


def test_config_declares_no_free_set(no_free_dataset):
    cfg = no_free_dataset["cfg"]
    assert cfg.split.strategy == "none"
    assert cfg.has_free_set is False


def test_none_strategy_needs_no_free_column_or_test_value(tmp_path):
    """The requirements that exist for deposited flags must not apply here."""
    cfg = AppConfig.model_validate(
        {
            "run": {"output_dir": str(tmp_path / "out")},
            "input": {
                "reflections": str(tmp_path / "a.mtz"),
                "model": str(tmp_path / "a.pdb"),
                "columns": {"observation": "FP", "sigma": "SIGFP"},
            },
            "split": {"strategy": "none"},
            "resolution": {"d_min_angstrom": 2.0},
        }
    )
    assert cfg.input.columns.free is None
    assert cfg.input.free_test_value is None
    assert cfg.has_free_set is False


def test_check_config_reports_the_reduced_target(no_free_dataset):
    report = check_config(no_free_dataset["config_path"])
    assert report["ok"] is True
    assert report["has_free_set"] is False
    assert report["target"] == "R_work only (no held-out set)"


def test_prepare_assigns_everything_to_work(no_free_dataset):
    metadata = prepare_reflections(no_free_dataset["cfg"])
    assert metadata["n_free"] == 0
    assert metadata["has_free_set"] is False
    assert metadata["n_train"] + metadata["n_tune"] == metadata["n_total_kept"]
    split = np.load(no_free_dataset["cfg"].run.data_dir / "reflections.npz")["split"]
    assert set(np.unique(split)) == {0, 1}


def test_tune_still_exists_for_prior_selection(no_free_dataset):
    """Prior selection and early stopping still need something to rank on."""
    metadata = prepare_reflections(no_free_dataset["cfg"])
    assert metadata["n_tune"] > 0
    assert metadata["n_train"] > metadata["n_tune"]


def test_work_objective_covers_every_reflection(tmp_path):
    import jax.numpy as jnp

    cfg = problem_config(tmp_path, split={"strategy": "none"}, optimizer={"fit_scope": "work"})
    arrays = problem_arrays(observation=(1.0, 2.0, 3.0, 1.5, 2.5), split=(0, 0, 1, 1, 0))
    metrics = build_functions(arrays, cfg)[5]
    result = {k: float(v) for k, v in metrics(jnp.zeros_like(arrays.rho0)).items()}
    assert result["n_work"] == 5
    assert np.isfinite(result["r_work"])


def test_free_metrics_are_finite_and_empty_rather_than_nan(tmp_path):
    """An empty mask must not produce NaN: the R-factor denominator is floored."""
    import jax.numpy as jnp

    cfg = problem_config(tmp_path, split={"strategy": "none"}, optimizer={"fit_scope": "work"})
    arrays = problem_arrays(observation=(1.0, 2.0, 3.0), split=(0, 1, 0))
    free_metrics = build_functions(arrays, cfg)[6]
    result = {k: float(v) for k, v in free_metrics(jnp.zeros_like(arrays.rho0)).items()}
    assert result["n_free"] == 0
    assert all(np.isfinite(value) for value in result.values())
    assert result["r_free"] == 0.0


def test_evaluate_free_reports_instead_of_failing(no_free_dataset):
    result = runner.invoke(app, ["evaluate-free", str(no_free_dataset["config_path"])])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output.replace("\n", ""))
    assert report["has_free_set"] is False
    assert report["primary_criterion_met"] is None
    assert report["target"] == "R_work only"
    # The caveat must survive into the artifact, not just the console.
    written = json.loads((no_free_dataset["cfg"].run.output_dir / "FREE_EVALUATION_SKIPPED.json").read_text())
    assert "not evidence" in written["note"]
    assert not (no_free_dataset["cfg"].run.output_dir / "FREE_EVALUATION.json").exists()


def test_a_holdout_strategy_that_empties_the_free_set_still_fails_loudly(tiny_dataset):
    """Only `none` is allowed to have no free set; a misconfigured holdout must not pass."""
    cfg = tiny_dataset["cfg"]
    broken = cfg.model_copy(
        update={
            "input": cfg.input.model_copy(update={"free_test_value": 999}),
            "split": cfg.split.model_copy(update={"strategy": "existing_free_then_hash"}),
        }
    )
    with pytest.raises(ValueError, match="free fraction"):
        prepare_reflections(broken)


def test_the_split_guard_suggests_the_none_strategy(tiny_dataset, monkeypatch):
    """A holdout strategy that yields no free reflections points at the way out."""
    from crystal_field.crystallography import io

    cfg = tiny_dataset["cfg"].model_copy(
        update={"split": tiny_dataset["cfg"].split.model_copy(update={"strategy": "hash"})}
    )
    monkeypatch.setattr(io, "_u01_for_key", lambda key, seed: 0.99)
    with pytest.raises(ValueError, match=r"split\.strategy=none"):
        prepare_reflections(cfg)


def test_freeze_records_the_reduced_target(no_free_dataset, tmp_path):
    from crystal_field.analysis.model_selection import freeze_model
    from crystal_field.config import dump_config
    from crystal_field.crystallography.io import make_model_density, make_solvent_mask
    from crystal_field.crystallography.scaling import fit_baseline_scaling
    from crystal_field.inference.information import run_information_spectrum
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions as build
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = no_free_dataset["cfg"]
    prepare_reflections(cfg)
    make_model_density(cfg)
    make_solvent_mask(cfg)
    fit_baseline_scaling(cfg)
    arrays = load_problem_arrays(cfg)
    _, density, _, residuals, objective, metrics, _ = build(arrays, cfg)
    run_map_fit(cfg, arrays, objective, metrics, density)
    run_information_spectrum(cfg, arrays, residuals)

    config_path = tmp_path / "selected.yaml"
    dump_config(cfg, config_path)
    result = freeze_model(config_path)
    lock = json.loads(pathlib.Path(result["lock"]).read_text())
    assert lock["has_free_set"] is False
    assert lock["target"] == "R_work only (no held-out set)"


def test_init_pdb_can_declare_no_free_set(tmp_path, monkeypatch):
    from crystal_field.dataset_init import build_config_payload

    entry = {"pdb_id": "9XYZ", "model": str(tmp_path / "m.cif"), "reflections": str(tmp_path / "r.cif")}
    refl = {"free": "FREE", "free_test_value": 1, "observation": "FP", "sigma": "SIGFP", "d_min": 2.0, "d_max": 30.0}

    with_holdout = build_config_payload(entry, refl, tmp_path / "runs")
    assert with_holdout["split"]["strategy"] == "existing_free_then_hash"

    without = build_config_payload(entry, refl, tmp_path / "runs", no_free_set=True)
    assert without["split"]["strategy"] == "none"
    assert without["input"]["free_test_value"] is None
