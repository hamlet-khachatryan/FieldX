"""Generic PDB initialisation (plan section 20I).

The pipeline must work for any PDB entry with deposited Bragg structure factors and
must never silently substitute another dataset. RCSB is mocked throughout; nothing here
touches the network.
"""

import gzip
import json

import pytest
import yaml
from conftest import CELL, SPACEGROUP, synthetic_reflections, write_mmcif, write_mtz, write_sf_cif, write_tiny_model

from crystal_field.config import load_config
from crystal_field.crystallography import pdb
from crystal_field.crystallography.pdb import (
    MissingStructureFactors,
    discover_reflection_columns,
    download_pdb_entry,
    normalize_pdb_id,
)
from crystal_field.dataset_init import V3_PILOT_PRIOR_GRID, init_pdb_dataset


@pytest.mark.parametrize("raw", ["1UBQ", "1ubq", " 1Ubq ", "1uBQ\n"])
def test_pdb_id_normalisation(raw):
    assert normalize_pdb_id(raw) == "1ubq"


@pytest.mark.parametrize("raw", ["ubiquitin", "1ub", "1ubqq", "abcd", "1ub-", ""])
def test_invalid_pdb_ids_are_rejected(raw):
    with pytest.raises(ValueError, match="not a PDB identifier"):
        normalize_pdb_id(raw)


@pytest.fixture
def fake_rcsb(monkeypatch, tmp_path):
    """Serve a synthetic model and structure-factor file in place of RCSB."""
    source = tmp_path / "source"
    source.mkdir()
    pdb_model = write_tiny_model(source / "model.pdb")
    model = write_mmcif(pdb_model, source / "model.cif")
    hkls, observed, sigma, free = synthetic_reflections(pdb_model)
    mtz = write_mtz(source / "refl.mtz", hkls, observed, sigma, free)
    sf = write_sf_cif(mtz, source / "refl-sf.cif")
    state = {
        "pdb_model": pdb_model,
        "model": model,
        "reflections": sf,
        "structure_factors_available": True,
        "requested": [],
    }

    def fake_download(url, path):
        state["requested"].append(url)
        if url.endswith("-sf.cif.gz"):
            if not state["structure_factors_available"]:
                raise RuntimeError(f"RCSB download failed (404): {url}")
            path.write_bytes(gzip.compress(state["reflections"].read_bytes()))
        else:
            path.write_bytes(state["model"].read_bytes())

    monkeypatch.setattr(pdb, "_download", fake_download)
    return state


def test_download_writes_the_entry_record(fake_rcsb, tmp_path):
    entry = download_pdb_entry("9xyz", tmp_path / "data")
    assert entry["pdb_id"] == "9XYZ"
    assert (tmp_path / "data" / "9xyz" / "9xyz.cif").exists()
    assert (tmp_path / "data" / "9xyz" / "9xyz-sf.cif").exists()
    recorded = json.loads((tmp_path / "data" / "9xyz" / "entry.json").read_text())
    assert recorded == entry
    assert recorded["cell"] == pytest.approx(list(CELL))


def test_an_entry_without_structure_factors_is_an_error(fake_rcsb, tmp_path):
    """A coordinate file alone is not enough for Bragg refinement."""
    fake_rcsb["structure_factors_available"] = False
    with pytest.raises(MissingStructureFactors, match="no downloadable deposited structure-factor file"):
        download_pdb_entry("9xyz", tmp_path / "data")
    assert (tmp_path / "data" / "9xyz" / "9xyz.cif").exists()
    assert not (tmp_path / "data" / "9xyz" / "9xyz-sf.cif").exists()
    assert not (tmp_path / "data" / "9xyz" / "9xyz-sf.cif.gz").exists()


def test_a_failed_download_never_substitutes_another_dataset(fake_rcsb, tmp_path):
    fake_rcsb["structure_factors_available"] = False
    with pytest.raises(MissingStructureFactors):
        download_pdb_entry("9xyz", tmp_path / "data")
    assert not (tmp_path / "configs").exists()


def test_column_discovery_finds_amplitudes_sigmas_and_free_flags(fake_rcsb, tmp_path):
    entry = download_pdb_entry("9xyz", tmp_path / "data")
    refl = discover_reflection_columns(entry["reflections"])
    assert refl["observation"] == "FP"
    assert refl["sigma"] == "SIGFP"
    assert refl["free"] == "FreeR_flag"
    assert refl["free_test_value"] in (0, 1)
    assert 0.02 <= refl["free_fraction"] <= 0.15
    assert refl["spacegroup"] == SPACEGROUP
    assert refl["d_min"] > 0 and refl["d_max"] > refl["d_min"]
    assert refl["merged"] is True


def test_a_constant_free_column_yields_no_free_value(tmp_path):
    """1UBQ's situation: a deposited flag column that holds out nothing."""
    import numpy as np

    model = write_tiny_model(tmp_path / "m.pdb")
    hkls, observed, sigma, _ = synthetic_reflections(model)
    mtz = write_mtz(tmp_path / "constant.mtz", hkls, observed, sigma, np.ones(len(hkls), dtype=np.int32))
    refl = discover_reflection_columns(mtz)
    assert refl["free"] == "FREE"
    assert refl["free_test_value"] is None
    assert refl["free_distribution"] == {"1": 1.0}


def test_missing_amplitude_columns_are_reported(tmp_path):
    import numpy as np
    import reciprocalspaceship as rs

    model = write_tiny_model(tmp_path / "m.pdb")
    hkls, observed, _, _ = synthetic_reflections(model)
    ds = rs.DataSet(
        {
            "H": hkls[:, 0].astype(np.int32),
            "K": hkls[:, 1].astype(np.int32),
            "L": hkls[:, 2].astype(np.int32),
            "IMEAN": observed.astype(np.float32),
        },
        cell=CELL,
        spacegroup=SPACEGROUP,
    )
    ds = ds.set_index(["H", "K", "L"]).infer_mtz_dtypes()
    ds.merged = True
    ds.write_mtz(str(tmp_path / "no-amplitudes.mtz"))
    with pytest.raises(ValueError, match="Could not infer amplitude/sigma columns"):
        discover_reflection_columns(tmp_path / "no-amplitudes.mtz")


def test_init_writes_a_usable_config_and_prior_grid(fake_rcsb, tmp_path):
    result = init_pdb_dataset(
        "9xyz", data_root=tmp_path / "data", runs_root=tmp_path / "runs", config_out=tmp_path / "cfg" / "default.yaml"
    )
    cfg = load_config(result["config"])
    assert cfg.grid.samples_per_dmin == 3.0
    assert cfg.split.strategy == "existing_free_then_hash"
    assert cfg.input.free_test_value in (0, 1)
    assert cfg.run.output_dir == (tmp_path / "runs" / "9XYZ").resolve()
    assert cfg.input.reflections.exists() and cfg.input.model.exists()

    grid = yaml.safe_load((tmp_path / "cfg" / "prior_grid.yaml").read_text())
    assert grid == V3_PILOT_PRIOR_GRID
    assert [c["name"] for c in grid["candidates"]] == [
        "matern_fine",
        "matern_baseline",
        "matern_broad",
        "squared_exponential",
    ]
    assert all(c["prior"]["latent_distribution"] == "gaussian" for c in grid["candidates"])


def test_init_falls_back_to_a_hash_holdout_without_deposited_flags(fake_rcsb, tmp_path):
    import numpy as np

    hkls, observed, sigma, _ = synthetic_reflections(fake_rcsb["pdb_model"])
    mtz = write_mtz(tmp_path / "constant.mtz", hkls, observed, sigma, np.ones(len(hkls), dtype=np.int32))
    fake_rcsb["reflections"] = write_sf_cif(mtz, tmp_path / "constant-sf.cif")
    result = init_pdb_dataset(
        "9xyz", data_root=tmp_path / "data", runs_root=tmp_path / "runs", config_out=tmp_path / "cfg" / "default.yaml"
    )
    cfg = load_config(result["config"])
    assert cfg.split.strategy == "hash"
    assert cfg.input.free_test_value is None
    assert "NOT the deposited" in result["free_set"]


def test_init_does_not_clobber_an_existing_config(fake_rcsb, tmp_path):
    config = tmp_path / "cfg" / "default.yaml"
    first = init_pdb_dataset("9xyz", data_root=tmp_path / "data", runs_root=tmp_path / "runs", config_out=config)
    config.write_text(config.read_text().replace("tau_density: 0.04", "tau_density: 0.011"))
    second = init_pdb_dataset("9xyz", data_root=tmp_path / "data", runs_root=tmp_path / "runs", config_out=config)
    assert first["written"]
    assert second["written"] == []
    assert load_config(config).prior.tau_density == 0.011


def test_force_regenerates_the_config(fake_rcsb, tmp_path):
    config = tmp_path / "cfg" / "default.yaml"
    init_pdb_dataset("9xyz", data_root=tmp_path / "data", runs_root=tmp_path / "runs", config_out=config)
    config.write_text(config.read_text().replace("tau_density: 0.04", "tau_density: 0.011"))
    init_pdb_dataset("9xyz", data_root=tmp_path / "data", runs_root=tmp_path / "runs", config_out=config, force=True)
    assert load_config(config).prior.tau_density == 0.04


def test_dataset_directories_are_named_generically(fake_rcsb, tmp_path):
    for pdb_id in ("9xyz", "7abc"):
        result = init_pdb_dataset(
            pdb_id,
            data_root=tmp_path / "data",
            runs_root=tmp_path / "runs",
            config_out=tmp_path / "cfg" / pdb_id / "default.yaml",
        )
        assert result["data_dir"].endswith(f"/{pdb_id}")
        assert result["output_dir"].endswith(f"/{pdb_id.upper()}")


def test_roots_come_from_the_environment(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.setenv("FIELDX_DATA_ROOT", str(tmp_path / "workspace" / "data"))
    monkeypatch.setenv("FIELDX_RUNS_ROOT", str(tmp_path / "workspace" / "runs"))
    result = init_pdb_dataset("9xyz", config_out=tmp_path / "cfg" / "default.yaml")
    assert result["data_dir"] == str((tmp_path / "workspace" / "data" / "9xyz").resolve())
    assert result["output_dir"] == str((tmp_path / "workspace" / "runs" / "9XYZ").resolve())


def test_the_core_pipeline_has_no_dataset_special_cases():
    """Section 13: no per-entry branching outside example configuration files."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for path in root.glob("src/**/*.py"):
        text = path.read_text().lower()
        code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
        code = code.split('"""')
        code = "".join(code[::2])  # drop docstrings; prose may name the example datasets
        assert "6o2h" not in code, f"{path} branches on a specific entry"
        assert "1ubq" not in code, f"{path} branches on a specific entry"
