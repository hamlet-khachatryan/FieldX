"""Initialising several PDB entries in one command.

The guards matter more than the loop. Identifiers are validated and de-duplicated before
any network call, so a typo in the last argument does not cost downloads for the ones
before it; and one entry failing -- most often because it deposits no structure factors --
does not stop the others. Every failure is collected and reported, with a non-zero exit
code. --fail-fast stops at the first instead.
"""

import gzip
import json

import pytest
from conftest import synthetic_reflections, write_mmcif, write_mtz, write_sf_cif, write_tiny_model
from typer.testing import CliRunner

from crystal_field.cli import app
from crystal_field.config import load_config
from crystal_field.crystallography import pdb
from crystal_field.crystallography.pdb import MissingStructureFactors
from crystal_field.dataset_init import init_pdb_datasets

runner = CliRunner()


@pytest.fixture
def fake_rcsb(monkeypatch, tmp_path):
    """Serve synthetic entries; `broken` lists ids whose structure factors 404."""
    source = tmp_path / "source"
    source.mkdir()
    pdb_model = write_tiny_model(source / "model.pdb")
    model = write_mmcif(pdb_model, source / "model.cif")
    hkls, observed, sigma, free = synthetic_reflections(pdb_model)
    sf = write_sf_cif(write_mtz(source / "r.mtz", hkls, observed, sigma, free), source / "r-sf.cif")
    state = {"broken": set(), "requested": []}

    def fake_download(url, path):
        state["requested"].append(url)
        entry = url.rsplit("/", 1)[-1].split(".")[0].replace("-sf", "")
        if url.endswith("-sf.cif.gz"):
            if entry in state["broken"]:
                raise RuntimeError(f"RCSB download failed (404): {url}")
            path.write_bytes(gzip.compress(sf.read_bytes()))
        else:
            path.write_bytes(model.read_bytes())

    monkeypatch.setattr(pdb, "_download", fake_download)
    return state


def _roots(tmp_path):
    return {"data_root": tmp_path / "data", "runs_root": tmp_path / "runs"}


# --- guards that run before any download ------------------------------------------


def test_no_identifiers_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one PDB identifier"):
        init_pdb_datasets([], **_roots(tmp_path))


def test_every_identifier_is_validated_before_anything_is_downloaded(fake_rcsb, tmp_path):
    """A typo in the last argument must not cost the earlier downloads."""
    with pytest.raises(ValueError, match="nothing was downloaded"):
        init_pdb_datasets(["9xyz", "7abc", "not-a-pdb-id"], **_roots(tmp_path))
    assert fake_rcsb["requested"] == []
    assert not (tmp_path / "data").exists()


def test_the_error_names_every_bad_identifier(fake_rcsb, tmp_path):
    with pytest.raises(ValueError) as excinfo:
        init_pdb_datasets(["9xyz", "nope", "alsobad!"], **_roots(tmp_path))
    assert "nope" in str(excinfo.value)
    assert "alsobad!" in str(excinfo.value)


@pytest.mark.parametrize("ids", [["9xyz", "9XYZ"], ["9xyz", "7abc", "9xyz"], ["9xyz", " 9xyz "]])
def test_repeated_identifiers_are_rejected(fake_rcsb, tmp_path, ids):
    """They normalise to the same output paths, so the second would race the first."""
    with pytest.raises(ValueError, match="Repeated PDB identifier"):
        init_pdb_datasets(ids, **_roots(tmp_path))
    assert fake_rcsb["requested"] == []


# --- batch behaviour ----------------------------------------------------------------


def test_all_entries_are_initialised(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = init_pdb_datasets(["9xyz", "7abc", "8def"], **_roots(tmp_path))
    assert report["ok"] is True
    assert report["requested"] == ["9XYZ", "7ABC", "8DEF"]
    assert report["n_succeeded"] == 3
    assert report["n_failed"] == 0
    for entry in report["succeeded"]:
        cfg = load_config(entry["config"])
        assert cfg.input.reflections.exists()
        assert cfg.grid.samples_per_dmin == 3.0


def test_each_entry_gets_its_own_config_and_directories(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = init_pdb_datasets(["9xyz", "7abc"], **_roots(tmp_path))
    configs = {entry["pdb_id"]: entry["config"] for entry in report["succeeded"]}
    assert configs["9XYZ"].endswith("configs/9xyz/default.yaml")
    assert configs["7ABC"].endswith("configs/7abc/default.yaml")
    outputs = {entry["output_dir"] for entry in report["succeeded"]}
    assert len(outputs) == 2


def test_one_failure_does_not_stop_the_others(fake_rcsb, tmp_path, monkeypatch):
    """The default: a dud entry costs that entry and nothing else."""
    monkeypatch.chdir(tmp_path)
    fake_rcsb["broken"] = {"7abc"}
    report = init_pdb_datasets(["9xyz", "7abc", "8def"], **_roots(tmp_path))

    assert report["ok"] is False, "the batch still reports failure overall"
    assert report["n_succeeded"] == 2
    assert report["n_failed"] == 1
    assert report["skipped_after_failure"] == []
    assert {e["pdb_id"] for e in report["succeeded"]} == {"9XYZ", "8DEF"}
    assert report["failed"][0]["error_type"] == MissingStructureFactors.__name__
    # The entry after the failure really was initialised, not merely recorded.
    after = next(e for e in report["succeeded"] if e["pdb_id"] == "8DEF")
    assert load_config(after["config"]).input.reflections.exists()


def test_fail_fast_stops_the_batch(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_rcsb["broken"] = {"7abc"}
    report = init_pdb_datasets(["9xyz", "7abc", "8def"], **_roots(tmp_path), fail_fast=True)

    assert report["ok"] is False
    assert report["n_succeeded"] == 1
    assert [f["pdb_id"] for f in report["failed"]] == ["7ABC"]
    assert report["skipped_after_failure"] == ["8DEF"]
    # Whatever already succeeded is left intact.
    assert load_config(report["succeeded"][0]["config"]).input.model.exists()
    assert not (tmp_path / "configs" / "8def").exists()


def test_every_failure_is_reported_not_just_the_first(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_rcsb["broken"] = {"7abc", "8def"}
    report = init_pdb_datasets(["9xyz", "7abc", "8def"], **_roots(tmp_path))
    assert [f["pdb_id"] for f in report["failed"]] == ["7ABC", "8DEF"]
    assert report["n_succeeded"] == 1
    assert all("structure-factor" in f["error"] for f in report["failed"])


def test_a_failure_in_the_first_entry_still_runs_the_rest(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_rcsb["broken"] = {"9xyz"}
    report = init_pdb_datasets(["9xyz", "7abc"], **_roots(tmp_path))
    assert report["n_failed"] == 1
    assert [e["pdb_id"] for e in report["succeeded"]] == ["7ABC"]


def test_a_failed_entry_leaves_no_partial_structure_factor_file(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_rcsb["broken"] = {"7abc"}
    init_pdb_datasets(["7abc"], **_roots(tmp_path))
    data = tmp_path / "data" / "7abc"
    assert not (data / "7abc-sf.cif").exists()
    assert not (data / "7abc-sf.cif.gz").exists()


# --- command line -------------------------------------------------------------------


def test_cli_accepts_several_identifiers(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init-pdb", "9xyz", "7abc", "--data-root", "d", "--runs-root", "r"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output.replace("\n", ""))["n_succeeded"] == 2


def test_cli_exits_non_zero_when_an_entry_fails(fake_rcsb, tmp_path, monkeypatch):
    """Non-zero even though the other entries succeeded, so scripts notice."""
    monkeypatch.chdir(tmp_path)
    fake_rcsb["broken"] = {"7abc"}
    result = runner.invoke(app, ["init-pdb", "9xyz", "7abc", "8def", "--data-root", "d", "--runs-root", "r"])
    assert result.exit_code == 1
    report = json.loads(result.output.replace("\n", ""))
    assert report["n_succeeded"] == 2
    assert report["n_failed"] == 1


def test_cli_fail_fast_flag_stops_the_batch(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_rcsb["broken"] = {"7abc"}
    result = runner.invoke(
        app, ["init-pdb", "9xyz", "7abc", "8def", "--fail-fast", "--data-root", "d", "--runs-root", "r"]
    )
    assert result.exit_code == 1
    assert json.loads(result.output.replace("\n", ""))["skipped_after_failure"] == ["8DEF"]


def test_cli_rejects_config_out_with_several_identifiers(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init-pdb", "9xyz", "7abc", "--config-out", "one.yaml"])
    assert result.exit_code != 0
    assert fake_rcsb["requested"] == []


def test_cli_still_supports_a_single_identifier_with_config_out(fake_rcsb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init-pdb", "9xyz", "--config-out", "custom.yaml", "--data-root", "d"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "custom.yaml").exists()
