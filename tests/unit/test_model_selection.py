import json
from pathlib import Path

import yaml

from crystal_field.analysis.model_selection import expand_prior_grid, select_prior
from crystal_field.config import dump_config, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_expand_and_select_prior(tmp_path):
    base_cfg = load_config(ROOT / "configs/6o2h/default.yaml")
    base_cfg = base_cfg.model_copy(
        update={"run": base_cfg.run.model_copy(update={"output_dir": tmp_path / "run", "shared_data_dir": tmp_path / "shared"})}
    )
    base = tmp_path / "base.yaml"
    dump_config(base_cfg, base)
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"candidates": [
        {"name": "a", "prior": {"tau_density": 0.02}},
        {"name": "b", "prior": {"tau_density": 0.04}},
    ]}))
    result = expand_prior_grid(base, grid, tmp_path / "expanded")
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert len(manifest) == 2
    for item, score in zip(manifest, [2.0, 1.0], strict=True):
        p = Path(item["output_dir"]) / "fit"
        p.mkdir(parents=True, exist_ok=True)
        (p / "metrics.json").write_text(json.dumps({
            "chi2_tune": score * 10,
            "n_tune": 10,
            "r_tune": score / 10,
            "r_train": score / 20,
        }))
    selected = tmp_path / "selected.yaml"
    report = select_prior(base, Path(result["manifest"]), selected)
    assert report["winner"]["name"] == "b"
    payload = yaml.safe_load(selected.read_text())
    assert payload["optimizer"]["fit_scope"] == "work"
