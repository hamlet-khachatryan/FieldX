"""Prior selection ranks candidates on the tune subset only (plan section 17)."""

import json
from pathlib import Path

import pytest
import yaml

from crystal_field.analysis.model_selection import expand_prior_grid, select_prior
from crystal_field.config import dump_config, load_config


def _expand(tiny_dataset, tmp_path, candidates):
    base = tmp_path / "base.yaml"
    dump_config(tiny_dataset["cfg"], base)
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"candidates": candidates}))
    result = expand_prior_grid(base, grid, tmp_path / "expanded")
    return base, json.loads(Path(result["manifest"]).read_text())


def _write_metrics(entry, chi2_tune, n_tune=10, r_tune=0.2, r_train=0.1):
    fit = Path(entry["output_dir"]) / "fit"
    fit.mkdir(parents=True, exist_ok=True)
    (fit / "metrics.json").write_text(
        json.dumps({"chi2_tune": chi2_tune, "n_tune": n_tune, "r_tune": r_tune, "r_train": r_train})
    )


def test_the_best_tune_score_wins(tiny_dataset, tmp_path):
    base, manifest = _expand(
        tiny_dataset,
        tmp_path,
        [{"name": "a", "prior": {"tau_density": 0.02}}, {"name": "b", "prior": {"tau_density": 0.04}}],
    )
    _write_metrics(manifest[0], chi2_tune=20.0, r_tune=0.20)
    _write_metrics(manifest[1], chi2_tune=10.0, r_tune=0.10)

    selected = tmp_path / "selected.yaml"
    report = select_prior(base, tmp_path / "expanded" / "manifest.json", selected)
    assert report["winner"]["name"] == "b"
    assert [row["name"] for row in report["ranking"]] == ["b", "a"]

    cfg = load_config(selected)
    assert cfg.optimizer.fit_scope == "work", "the winner is refit on all work reflections"
    assert cfg.prior.tau_density == 0.04
    assert cfg.run.output_dir == tiny_dataset["cfg"].run.output_dir / "final"
    assert cfg.run.data_dir == tiny_dataset["cfg"].run.data_dir


def test_selection_ignores_train_and_free_quality(tiny_dataset, tmp_path):
    """Only the tune subset may decide. A candidate that fits train better must not win."""
    base, manifest = _expand(
        tiny_dataset,
        tmp_path,
        [{"name": "overfit", "prior": {"tau_density": 0.2}}, {"name": "honest", "prior": {"tau_density": 0.02}}],
    )
    _write_metrics(manifest[0], chi2_tune=50.0, r_train=0.001, r_tune=0.5)
    _write_metrics(manifest[1], chi2_tune=10.0, r_train=0.400, r_tune=0.1)
    report = select_prior(base, tmp_path / "expanded" / "manifest.json", tmp_path / "selected.yaml")
    assert report["winner"]["name"] == "honest"


def test_a_missing_candidate_result_is_an_error(tiny_dataset, tmp_path):
    base, manifest = _expand(tiny_dataset, tmp_path, [{"name": "a", "prior": {}}, {"name": "b", "prior": {}}])
    _write_metrics(manifest[0], chi2_tune=10.0)
    with pytest.raises(FileNotFoundError, match="Missing candidate metrics"):
        select_prior(base, tmp_path / "expanded" / "manifest.json", tmp_path / "selected.yaml")


def test_the_selection_report_is_written_next_to_the_config(tiny_dataset, tmp_path):
    base, manifest = _expand(tiny_dataset, tmp_path, [{"name": "a", "prior": {}}])
    _write_metrics(manifest[0], chi2_tune=10.0)
    selected = tmp_path / "selected.yaml"
    select_prior(base, tmp_path / "expanded" / "manifest.json", selected)
    report = json.loads(selected.with_suffix(".selection.json").read_text())
    assert report["winner"]["name"] == "a"
    assert len(report["ranking"]) == 1
