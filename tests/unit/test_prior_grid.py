from pathlib import Path

import yaml

from crystal_field.config import AppConfig, load_config


def test_every_6o2h_prior_candidate_validates():
    base = load_config("configs/6o2h/default.yaml").model_dump(mode="json", exclude_none=True)
    grid = yaml.safe_load(Path("configs/6o2h/prior_grid.yaml").read_text())
    assert len(grid["candidates"]) >= 10
    for candidate in grid["candidates"]:
        payload = {**base, "prior": {**base["prior"], **candidate.get("prior", {})}}
        payload["likelihood"] = {**base["likelihood"], **candidate.get("likelihood", {})}
        payload["baseline"] = {**base["baseline"], **candidate.get("baseline", {})}
        AppConfig.model_validate(payload)
