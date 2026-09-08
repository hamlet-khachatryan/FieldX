"""Configuration schema and validation (plan section 20A)."""

from pathlib import Path

import pytest
import yaml

from crystal_field.config import AppConfig, check_config, load_config, resolve_payload_paths

ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CONFIGS = sorted(ROOT.glob("configs/**/*.yaml"))


def _base(tiny_dataset, **overrides):
    payload = {section: dict(values) for section, values in tiny_dataset["payload"].items()}
    for section, values in overrides.items():
        payload.setdefault(section, {}).update(values)
    return payload


@pytest.mark.parametrize("path", [p for p in COMMITTED_CONFIGS if p.name != "prior_grid.yaml"], ids=lambda p: p.name)
def test_every_committed_config_validates(path):
    cfg = load_config(path)
    assert cfg.grid.samples_per_dmin == 3.0
    assert cfg.input.reflections.is_absolute()
    assert cfg.run.output_dir.is_absolute()


def test_1ubq_pilot_is_the_small_v3_default():
    cfg = load_config(ROOT / "configs/1ubq/default.yaml")
    assert cfg.grid.samples_per_dmin == 3.0, "v3 sampling convention must not change for the first experiment"
    assert cfg.optimizer.fit_scope == "train"
    assert cfg.prior.latent_distribution == "gaussian"
    assert cfg.optimizer.max_iterations <= 100
    assert cfg.information.n_modes == 8
    # 1UBQ deposits no usable free flag, so the holdout must be the deterministic hash.
    assert cfg.split.strategy == "hash"


def test_relative_paths_resolve_against_the_config_directory(tmp_path):
    (tmp_path / "nested").mkdir()
    payload = {
        "run": {"output_dir": "../out"},
        "input": {
            "reflections": "data/x.mtz",
            "model": "data/x.pdb",
            "columns": {"observation": "FP", "sigma": "SIGFP"},
        },
        "split": {"strategy": "hash"},
        "resolution": {"d_min_angstrom": 2.0},
    }
    config = tmp_path / "nested" / "c.yaml"
    config.write_text(yaml.safe_dump(payload))
    cfg = load_config(config)
    assert cfg.input.reflections == tmp_path / "nested" / "data" / "x.mtz"
    assert cfg.run.output_dir == tmp_path / "out"


def test_absolute_paths_are_left_alone(tmp_path):
    payload = {"input": {"model": "/absolute/model.cif"}}
    assert resolve_payload_paths(payload, tmp_path)["input"]["model"] == "/absolute/model.cif"


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_check_config_reports_missing_inputs(tiny_dataset, tmp_path):
    payload = _base(tiny_dataset, input={"model": str(tmp_path / "no-such-model.cif")})
    config = tmp_path / "broken.yaml"
    config.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=r"input\.model does not exist"):
        check_config(config)


def test_check_config_accepts_a_complete_dataset(tiny_dataset):
    assert check_config(tiny_dataset["config_path"])["ok"] is True


def test_unknown_key_is_rejected(tiny_dataset):
    payload = _base(tiny_dataset, prior={"correlaton_length_angstrom": 1.0})
    with pytest.raises(ValueError, match="correlaton_length_angstrom"):
        AppConfig.model_validate(payload)


def test_existing_free_strategy_requires_a_free_column(tiny_dataset):
    payload = _base(tiny_dataset)
    payload["input"]["columns"] = {"observation": "FP", "sigma": "SIGFP"}
    with pytest.raises(ValueError, match=r"columns\.free is required"):
        AppConfig.model_validate(payload)


def test_existing_free_strategy_requires_a_test_value(tiny_dataset):
    payload = _base(tiny_dataset, input={"free_test_value": None})
    with pytest.raises(ValueError, match=r"free_test_value is required"):
        AppConfig.model_validate(payload)


def test_anomalous_is_rejected(tiny_dataset):
    with pytest.raises(ValueError, match="does not support anomalous"):
        AppConfig.model_validate(_base(tiny_dataset, split={"anomalous": True}))


def test_d_max_below_d_min_is_rejected(tiny_dataset):
    with pytest.raises(ValueError, match="d_max_angstrom must be"):
        AppConfig.model_validate(_base(tiny_dataset, resolution={"d_min_angstrom": 2.0, "d_max_angstrom": 1.0}))


def test_intensity_observations_conflict_with_gemmi_scaling(tiny_dataset):
    with pytest.raises(ValueError, match="requires amplitude observations"):
        AppConfig.model_validate(_base(tiny_dataset, input={"observation_kind": "intensity"}))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tau_density": 0.0}, "greater than 0"),
        ({"correlation_length_angstrom": -1.0}, "greater than 0"),
        ({"kernel": "matern", "alpha": 1.2}, "alpha must exceed 1.5"),
        ({"kernel": "multiscale_matern", "components": []}, "at least one component"),
        ({"kernel": "matern", "components": [{"correlation_length_angstrom": 1.0, "weight": 1.0}]}, "only meaningful"),
        ({"latent_distribution": "gamma"}, "latent_distribution"),
        ({"student_t_df": 1.5}, "greater than 2"),
    ],
)
def test_invalid_prior_parameters(tiny_dataset, override, message):
    with pytest.raises(ValueError, match=message):
        AppConfig.model_validate(_base(tiny_dataset, prior=override))


@pytest.mark.parametrize(
    ("shape", "message"),
    [((2, 16, 16), "at least 4"), ((15, 16, 16), "must be even"), ((0, 0, 0), "at least 4")],
)
def test_invalid_grid_shape(tiny_dataset, shape, message):
    with pytest.raises(ValueError, match=message):
        AppConfig.model_validate(_base(tiny_dataset, grid={"shape": shape}))


@pytest.mark.parametrize("samples", [1.5, 7.0])
def test_samples_per_dmin_is_bounded(tiny_dataset, samples):
    with pytest.raises(ValueError):
        AppConfig.model_validate(_base(tiny_dataset, grid={"samples_per_dmin": samples}))


def test_free_fraction_window_must_be_ordered(tiny_dataset):
    with pytest.raises(ValueError, match="expected_free_fraction_min"):
        AppConfig.model_validate(
            _base(tiny_dataset, input={"expected_free_fraction_min": 0.2, "expected_free_fraction_max": 0.1})
        )
