from pathlib import Path

import pytest

from crystal_field.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_6o2h_default_loads():
    cfg = load_config(ROOT / "configs/6o2h/default.yaml")
    assert cfg.resolution.d_min_angstrom == pytest.approx(1.21)
    assert cfg.input.observation_kind == "amplitude"
    assert cfg.grid.samples_per_dmin == 3.0
    assert cfg.optimizer.fit_scope == "train"
    assert cfg.run.data_dir.name == "shared"
    assert cfg.baseline.refmac_compatible_density_blur is False
    assert cfg.baseline.scaling.enabled is True
    assert cfg.likelihood.global_scale == "fixed"


def test_anomalous_rejected():
    payload = load_config(ROOT / "configs/6o2h/default.yaml").model_dump(mode="json")
    payload["split"]["anomalous"] = True
    with pytest.raises(ValueError, match="does not support anomalous"):
        AppConfig.model_validate(payload)


def test_multiscale_requires_components():
    payload = load_config(ROOT / "configs/6o2h/default.yaml").model_dump(mode="json")
    payload["prior"]["kernel"] = "multiscale_matern"
    payload["prior"]["components"] = []
    with pytest.raises(ValueError, match="requires at least one component"):
        AppConfig.model_validate(payload)
