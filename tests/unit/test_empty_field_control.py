"""Refinement from an empty cell: the capacity control.

Not a phasing method. With amplitudes alone the problem is phase-degenerate, so this run
cannot recover a structure. It exists to calibrate the main experiment: it measures how
far the field class can drive R_work down using no structural information whatsoever. A
model-start result is only meaningful if it beats this on held-out reflections.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import config_payload, synthetic_reflections, write_mtz, write_tiny_model

from crystal_field.config import AppConfig, load_config, resolve_payload_paths
from crystal_field.crystallography.io import check_model_density, make_model_density, prepare_reflections

ROOT = Path(__file__).resolve().parents[2]

EMPTY = {
    "baseline": {"starting_density": "zero", "scaling": {"enabled": False}, "bulk_solvent": {"enabled": False}},
    "likelihood": {"global_scale": "profile"},
    "optimizer": {"init": "random", "init_scale": 1.0},
}


def _config(tmp_path, **overrides):
    model = write_tiny_model(tmp_path / "t.pdb")
    hkls, observed, sigma, free = synthetic_reflections(model)
    mtz = write_mtz(tmp_path / "t.mtz", hkls, observed, sigma, free)
    payload = config_payload(tmp_path, model, mtz, optimizer={"fit_scope": "train", "max_iterations": 4})
    for section, values in overrides.items():
        payload.setdefault(section, {}).update(values)
    return AppConfig.model_validate(resolve_payload_paths(dict(payload), tmp_path))


def test_the_committed_control_config_validates():
    cfg = load_config(ROOT / "configs/1ubq/empty_field_control.yaml")
    assert cfg.baseline.starting_density == "zero"
    assert cfg.optimizer.init == "random"
    assert cfg.baseline.scaling.enabled is False
    assert cfg.baseline.bulk_solvent.enabled is False
    assert cfg.likelihood.global_scale == "profile"
    # It must not clobber the main experiment's run directory.
    assert cfg.run.output_dir != load_config(ROOT / "configs/1ubq/default.yaml").run.output_dir


def test_the_control_differs_from_the_main_run_only_in_the_starting_density():
    """A control that changed the prior too would compare nothing."""
    main = load_config(ROOT / "configs/1ubq/default.yaml")
    control = load_config(ROOT / "configs/1ubq/empty_field_control.yaml")
    assert control.prior.model_dump() == main.prior.model_dump()
    assert control.resolution.model_dump() == main.resolution.model_dump()
    assert control.grid.model_dump() == main.grid.model_dump()
    assert control.split.model_dump() == main.split.model_dump()
    assert control.run.seed == main.run.seed


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"baseline": {"scaling": {"enabled": True}}}, "scaling.enabled must be false"),
        ({"baseline": {"bulk_solvent": {"enabled": True}}}, "bulk_solvent.enabled must be false"),
        ({"likelihood": {"global_scale": "fixed"}}, "global_scale must be 'profile'"),
        ({"optimizer": {"init": "zero"}}, "init must be 'random'"),
    ],
)
def test_an_inconsistent_control_configuration_is_rejected(tmp_path, override, message):
    payload = {}
    for section, values in EMPTY.items():
        payload[section] = {**values}
    for section, values in override.items():
        payload.setdefault(section, {}).update(values)
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **payload)


def test_the_starting_density_is_actually_empty(tmp_path):
    cfg = _config(tmp_path, **EMPTY)
    prepare_reflections(cfg)
    stats = make_model_density(cfg)
    assert stats["starting_density"] == "zero"
    rho = np.load(cfg.run.data_dir / "rho0.npy")
    assert rho.shape == tuple(stats["shape"])
    assert np.all(rho == 0.0)


def test_the_model_agreement_check_is_skipped_but_records_why(tmp_path):
    cfg = _config(tmp_path, **EMPTY)
    prepare_reflections(cfg)
    make_model_density(cfg)
    result = check_model_density(cfg)
    assert result["pass"] is True
    assert result["starting_density"] == "zero"
    assert "do not apply" in result["note"]
    assert json.loads((cfg.run.output_dir / "rho0_check.json").read_text())["starting_density"] == "zero"


def test_a_tampered_empty_density_is_caught(tmp_path):
    cfg = _config(tmp_path, **EMPTY)
    prepare_reflections(cfg)
    make_model_density(cfg)
    rho = np.load(cfg.run.data_dir / "rho0.npy")
    rho[0, 0, 0] = 1.0
    np.save(cfg.run.data_dir / "rho0.npy", rho)
    with pytest.raises(RuntimeError, match="not identically zero"):
        check_model_density(cfg)


def test_the_gradient_at_zero_is_undefined_for_an_empty_field(tmp_path):
    """Why optimizer.init='random' is mandatory rather than advisory."""
    from conftest import problem_arrays, problem_config

    from crystal_field.inference.problem import build_functions

    cfg = problem_config(tmp_path, optimizer={"fit_scope": "work"}, likelihood={"global_scale": "profile"})
    arrays = problem_arrays(observation=(10.0, 20.0, 30.0), split=(0, 1, 0))
    empty = arrays.__class__(**{**arrays.__dict__, "rho0": jnp.zeros_like(arrays.rho0)})
    objective = build_functions(empty, cfg)[4]

    z0 = jnp.zeros_like(empty.rho0)
    assert bool(jnp.isnan(jax.grad(objective)(z0)).any()), "|F| is not differentiable at F=0"

    perturbed = 0.5 * jax.random.normal(jax.random.PRNGKey(0), z0.shape, dtype=z0.dtype)
    assert not bool(jnp.isnan(jax.grad(objective)(perturbed)).any())


def test_random_initialisation_is_seeded_and_reproducible(tmp_path):
    from crystal_field.crystallography.io import make_solvent_mask
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    results = []
    for _ in range(2):
        cfg = _config(tmp_path, **EMPTY)
        prepare_reflections(cfg)
        make_model_density(cfg)
        make_solvent_mask(cfg)
        arrays = load_problem_arrays(cfg)
        _, density, _, _, objective, metrics, _ = build_functions(arrays, cfg)
        results.append(run_map_fit(cfg, arrays, objective, metrics, density))
    assert results[0]["r_train"] == pytest.approx(results[1]["r_train"])
    assert results[0]["starting_density"] == "zero"
    assert results[0]["initialisation"] == "random"


def test_an_empty_field_fits_training_data_but_does_not_generalise(tmp_path):
    """The result the control exists to produce.

    Fitting the training amplitudes from nothing is easy -- the field has far more
    degrees of freedom than there are reflections. Predicting held-out reflections from
    nothing is not. If this ever stopped holding, R_work would have lost its meaning.
    """
    from crystal_field.crystallography.io import make_solvent_mask
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    settings = {**EMPTY}
    settings["optimizer"] = {
        **EMPTY["optimizer"],
        "fit_scope": "train",
        "max_iterations": 60,
        "checkpoint_every": 10,
        "validation_patience": 1000,
    }
    cfg = _config(tmp_path, **settings)
    prepare_reflections(cfg)
    make_model_density(cfg)
    make_solvent_mask(cfg)
    arrays = load_problem_arrays(cfg)
    _, density, _, _, objective, metrics, _ = build_functions(arrays, cfg)
    result = run_map_fit(cfg, arrays, objective, metrics, density)

    assert result["r_train"] < 0.30, "an unconstrained field should fit the training amplitudes"
    assert result["r_tune"] > 2 * result["r_train"], "but it must not generalise to held-out reflections"
