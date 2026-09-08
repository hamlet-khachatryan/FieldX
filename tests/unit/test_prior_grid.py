"""Every committed prior grid must expand into valid configurations."""

from pathlib import Path

import pytest
import yaml

from crystal_field.analysis.model_selection import expand_prior_grid
from crystal_field.config import AppConfig, dump_config, load_config

ROOT = Path(__file__).resolve().parents[2]
GRIDS = sorted(ROOT.glob("configs/*/prior_grid.yaml"))


@pytest.mark.parametrize("grid_path", GRIDS, ids=lambda p: p.parent.name)
def test_every_candidate_validates(grid_path):
    base = load_config(grid_path.parent / "default.yaml").model_dump(mode="json", exclude_none=True)
    candidates = yaml.safe_load(grid_path.read_text())["candidates"]
    assert candidates
    names = [candidate["name"] for candidate in candidates]
    assert len(set(names)) == len(names)
    for candidate in candidates:
        prior = {**base["prior"], **candidate.get("prior", {})}
        if prior["kernel"] != "multiscale_matern":
            prior["components"] = []
        AppConfig.model_validate(
            {
                **base,
                "prior": prior,
                "likelihood": {**base["likelihood"], **candidate.get("likelihood", {})},
                "baseline": {**base["baseline"], **candidate.get("baseline", {})},
            }
        )


def test_the_1ubq_pilot_grid_is_the_small_gaussian_set():
    """Section 15: the first experiment uses four spatial candidates, Gaussian only."""
    candidates = yaml.safe_load((ROOT / "configs/1ubq/prior_grid.yaml").read_text())["candidates"]
    assert len(candidates) == 4
    assert {c["prior"]["kernel"] for c in candidates} == {"matern", "squared_exponential"}
    assert {c["prior"]["latent_distribution"] for c in candidates} == {"gaussian"}


@pytest.mark.parametrize("grid_path", GRIDS, ids=lambda p: p.parent.name)
def test_expansion_produces_one_train_scope_candidate_per_entry(grid_path, tiny_dataset, tmp_path):
    """Expansion is a login-node operation and must work for any dataset's grid."""
    cfg = tiny_dataset["cfg"]
    base = tmp_path / "base.yaml"
    dump_config(cfg, base)
    result = expand_prior_grid(base, grid_path, tmp_path / "expanded")
    candidates = yaml.safe_load(grid_path.read_text())["candidates"]
    assert result["n_candidates"] == len(candidates)

    import json

    manifest = json.loads(Path(result["manifest"]).read_text())
    for entry, candidate in zip(manifest, candidates, strict=True):
        expanded = load_config(entry["config"])
        assert expanded.optimizer.fit_scope == "train"
        # Every candidate shares the one prepared split, so no candidate can re-split.
        assert expanded.run.data_dir == cfg.run.data_dir
        assert expanded.run.output_dir == cfg.run.output_dir / "candidates" / candidate["name"]
        for key, value in candidate.get("prior", {}).items():
            if key != "components":
                assert getattr(expanded.prior, key) == value


def test_switching_away_from_multiscale_clears_inherited_components(tiny_dataset, tmp_path):
    cfg = tiny_dataset["cfg"]
    multiscale = cfg.model_copy(
        update={
            "prior": cfg.prior.model_copy(
                update={
                    "kernel": "multiscale_matern",
                    "components": [{"correlation_length_angstrom": 1.0, "alpha": 2.5, "weight": 1.0}],
                }
            )
        }
    )
    base = tmp_path / "base.yaml"
    dump_config(multiscale, base)
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"candidates": [{"name": "plain", "prior": {"kernel": "matern"}}]}))
    result = expand_prior_grid(base, grid, tmp_path / "out")

    import json

    entry = json.loads(Path(result["manifest"]).read_text())[0]
    assert load_config(entry["config"]).prior.components == []


def test_duplicate_candidate_names_are_rejected(tiny_dataset, tmp_path):
    base = tmp_path / "base.yaml"
    dump_config(tiny_dataset["cfg"], base)
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"candidates": [{"name": "a", "prior": {}}, {"name": "a", "prior": {}}]}))
    with pytest.raises(ValueError, match="duplicate candidate names"):
        expand_prior_grid(base, grid, tmp_path / "out")


def test_an_empty_grid_is_rejected(tiny_dataset, tmp_path):
    base = tmp_path / "base.yaml"
    dump_config(tiny_dataset["cfg"], base)
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"candidates": []}))
    with pytest.raises(ValueError, match="no candidates"):
        expand_prior_grid(base, grid, tmp_path / "out")


def test_a_missing_grid_is_reported(tiny_dataset, tmp_path):
    base = tmp_path / "base.yaml"
    dump_config(tiny_dataset["cfg"], base)
    with pytest.raises(FileNotFoundError, match="Prior grid not found"):
        expand_prior_grid(base, tmp_path / "absent.yaml", tmp_path / "out")
