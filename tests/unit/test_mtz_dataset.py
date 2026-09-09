"""Explicit MTZ dataset selection (Workstream A).

reciprocalspaceship resolves a multi-dataset MTZ silently and inconsistently: columns
are assigned by label in file order so a duplicated label keeps the LAST occurrence,
while the recorded wavelength is the FIRST dataset's (normally HKL_base's 0.0). A
refinement run that way cannot say which wavelength it used. These tests pin the
explicit behaviour that replaces it.
"""

import warnings

import pytest
from conftest import CELL, SPACEGROUP, write_multi_dataset_mtz

from crystal_field.crystallography.mtz import describe_mtz, is_mtz, read_mtz_dataset, resolve_mtz_dataset

FS = {"FP": "F", "SIGFP": "Q"}
MAD = [("peak", 0.9792, FS), ("remote", 0.9184, FS)]
SINGLE = [("native", 0.9795, {"FP": "F", "SIGFP": "Q", "FREE": "I"})]


@pytest.fixture
def mad_mtz(tmp_path):
    return write_multi_dataset_mtz(tmp_path / "mad.mtz", MAD)


@pytest.fixture
def single_mtz(tmp_path):
    return write_multi_dataset_mtz(tmp_path / "native.mtz", SINGLE)


def test_is_mtz_recognises_the_suffix(tmp_path):
    assert is_mtz(tmp_path / "x.mtz")
    assert is_mtz(tmp_path / "x.mtz.gz")
    assert not is_mtz(tmp_path / "x-sf.cif")


def test_describe_enumerates_datasets_and_duplicate_labels(mad_mtz):
    path, _ = mad_mtz
    described = describe_mtz(path)
    assert described["spacegroup"] == SPACEGROUP
    assert described["cell"] == pytest.approx(list(CELL))
    # HKL_base plus the two data datasets.
    assert len(described["datasets"]) == 3
    assert len(described["data_datasets"]) == 2
    assert [d["wavelength"] for d in described["data_datasets"]] == pytest.approx([0.9792, 0.9184])
    assert described["duplicate_labels"] == ["FP", "SIGFP"]


def test_a_single_dataset_file_needs_no_selection(single_mtz):
    path, _ = single_mtz
    selection = resolve_mtz_dataset(path, ["FP", "SIGFP", "FREE"])
    assert selection["dataset_name"] == "native"
    assert selection["wavelength"] == pytest.approx(0.9795)
    assert selection["selected_explicitly"] is False
    assert selection["n_datasets_with_data"] == 1


def test_ambiguous_file_refuses_to_guess(mad_mtz):
    path, _ = mad_mtz
    with pytest.raises(ValueError, match=r"Set input\.mtz_dataset|occurs in"):
        resolve_mtz_dataset(path, ["FP", "SIGFP"])


def test_the_error_names_every_candidate(mad_mtz):
    path, _ = mad_mtz
    with pytest.raises(ValueError) as excinfo:
        resolve_mtz_dataset(path, ["FP", "SIGFP"])
    message = str(excinfo.value)
    for fragment in ["peak", "remote", "0.9792", "0.9184", "mtz_dataset"]:
        assert fragment in message, fragment


@pytest.mark.parametrize(("selector", "expected"), [(1, "peak"), (2, "remote"), ("peak", "peak"), ("remote", "remote")])
def test_explicit_selection_by_id_or_name(mad_mtz, selector, expected):
    path, _ = mad_mtz
    selection = resolve_mtz_dataset(path, ["FP", "SIGFP"], selector)
    assert selection["dataset_name"] == expected
    assert selection["selected_explicitly"] is True


def test_unknown_selector_is_reported(mad_mtz):
    path, _ = mad_mtz
    with pytest.raises(ValueError, match="matches no dataset"):
        resolve_mtz_dataset(path, ["FP", "SIGFP"], "inflection")


def test_a_dataset_missing_the_configured_columns_is_reported(tmp_path):
    path, _ = write_multi_dataset_mtz(
        tmp_path / "mixed.mtz",
        [("native", 0.98, {"FP": "F", "SIGFP": "Q"}), ("anom", 1.54, {"DANO": "D", "SIGDANO": "Q"})],
    )
    with pytest.raises(ValueError, match="does not contain"):
        resolve_mtz_dataset(path, ["FP", "SIGFP"], "anom")


def test_distinct_labels_across_datasets_resolve_without_a_selector(tmp_path):
    """Two datasets are fine when the configured labels single one out."""
    path, _ = write_multi_dataset_mtz(
        tmp_path / "mixed.mtz",
        [("native", 0.98, {"FP": "F", "SIGFP": "Q"}), ("anom", 1.54, {"DANO": "D", "SIGDANO": "Q"})],
    )
    selection = resolve_mtz_dataset(path, ["FP", "SIGFP"])
    assert selection["dataset_name"] == "native"
    assert selection["wavelength"] == pytest.approx(0.98)


def test_columns_split_across_datasets_are_refused(tmp_path):
    path, _ = write_multi_dataset_mtz(
        tmp_path / "split.mtz",
        [("a", 0.98, {"FP": "F"}), ("b", 1.54, {"SIGFP": "Q"})],
    )
    with pytest.raises(ValueError, match="different datasets"):
        resolve_mtz_dataset(path, ["FP", "SIGFP"])


@pytest.mark.parametrize(("selector", "name"), [(1, "peak"), (2, "remote")])
def test_read_returns_the_selected_dataset_values_and_wavelength(mad_mtz, selector, name):
    path, truth = mad_mtz
    with warnings.catch_warnings():
        # Any silent reciprocalspaceship fallback must fail the test, not pass quietly.
        warnings.simplefilter("error")
        ds, selection = read_mtz_dataset(path, ["FP", "SIGFP"], selector)
    assert float(ds["FP"].iloc[0]) == pytest.approx(truth[(name, "FP")])
    assert ds.wavelength == pytest.approx(selection["wavelength"])
    assert selection["dataset_name"] == name
    assert sorted(ds.columns) == ["FP", "SIGFP"]
    assert ds.spacegroup.xhm() == SPACEGROUP


def test_selection_differs_from_the_implicit_reciprocalspaceship_result(mad_mtz):
    """The bug this replaces: rs keeps the LAST duplicate and records the FIRST wavelength."""
    import gemmi
    import reciprocalspaceship as rs

    path, truth = mad_mtz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        implicit = rs.io.from_gemmi(gemmi.read_mtz_file(str(path)))
    assert float(implicit["FP"].iloc[0]) == pytest.approx(truth[("remote", "FP")])
    assert implicit.wavelength == pytest.approx(0.0), "rs records HKL_base's wavelength"

    peak, selection = read_mtz_dataset(path, ["FP", "SIGFP"], "peak")
    assert float(peak["FP"].iloc[0]) == pytest.approx(truth[("peak", "FP")])
    assert float(peak["FP"].iloc[0]) != pytest.approx(float(implicit["FP"].iloc[0]))
    assert selection["wavelength"] == pytest.approx(0.9792)


# --- wiring: config -> loader -> prepared metadata --------------------------------

FS_FREE = {"FP": "F", "SIGFP": "Q", "FREE": "I"}


def _mad_config(tmp_path, mtz_dataset):
    from crystal_field.config import AppConfig

    path, truth = write_multi_dataset_mtz(
        tmp_path / "mad.mtz", [("peak", 0.9792, FS_FREE), ("remote", 0.9184, FS_FREE)]
    )
    payload = {
        "run": {"output_dir": str(tmp_path / "run"), "shared_data_dir": str(tmp_path / "shared"), "seed": 3},
        "input": {
            "reflections": str(path),
            "model": str(tmp_path / "unused.pdb"),
            "columns": {"observation": "FP", "sigma": "SIGFP", "free": "FREE"},
            "mtz_dataset": mtz_dataset,
            "free_test_value": 1,
            "expected_free_fraction_min": 0.02,
            "expected_free_fraction_max": 0.30,
        },
        "split": {"strategy": "existing_free_then_hash", "tune_fraction_of_work": 0.2},
        "resolution": {"d_min_angstrom": 2.6},
        "baseline": {"scaling": {"enabled": False}},
    }
    return AppConfig.model_validate(payload), truth


def test_loader_refuses_an_ambiguous_file(tmp_path):
    from crystal_field.crystallography.io import load_reflections

    cfg, _ = _mad_config(tmp_path, None)
    with pytest.raises(ValueError, match=r"Set input\.mtz_dataset|occurs in"):
        load_reflections(cfg)


@pytest.mark.parametrize(("selector", "name", "wavelength"), [("peak", "peak", 0.9792), (2, "remote", 0.9184)])
def test_loader_honours_the_configured_dataset(tmp_path, selector, name, wavelength):
    from crystal_field.crystallography.io import load_reflections

    cfg, truth = _mad_config(tmp_path, selector)
    ds, selection = load_reflections(cfg, with_selection=True)
    assert selection["dataset_name"] == name
    assert selection["wavelength"] == pytest.approx(wavelength)
    assert float(ds["FP"].iloc[0]) == pytest.approx(truth[(name, "FP")])


def test_prepared_metadata_records_the_dataset_and_wavelength(tmp_path):
    """Provenance must survive into the artifact that MODEL_LOCK.json hashes."""
    from crystal_field.crystallography.io import prepare_reflections

    cfg, _ = _mad_config(tmp_path, "peak")
    metadata = prepare_reflections(cfg)
    assert metadata["mtz_dataset_name"] == "peak"
    assert metadata["mtz_dataset_id"] == 1
    assert metadata["wavelength"] == pytest.approx(0.9792)


def test_reflection_cif_has_no_dataset_concept(tmp_path, tiny_dataset):
    """A structure-factor CIF carries no datasets, so selection must not apply at all."""
    from conftest import write_sf_cif

    from crystal_field.crystallography.io import load_reflections

    cif = write_sf_cif(tiny_dataset["reflections"], tmp_path / "refl-sf.cif")
    cfg = tiny_dataset["cfg"]
    cfg = cfg.model_copy(update={"input": cfg.input.model_copy(update={"reflections": cif})})
    ds, selection = load_reflections(cfg, with_selection=True)
    assert selection is None, "a CIF has no dataset to select"
    assert len(ds) > 0


def test_config_check_reports_the_configured_dataset(tmp_path):
    from crystal_field.config import check_config, dump_config

    cfg, _ = _mad_config(tmp_path, "peak")
    cfg.input.model.write_text("REMARK placeholder\nEND\n")
    config_path = tmp_path / "c.yaml"
    dump_config(cfg, config_path)
    assert check_config(config_path)["mtz_dataset"] == "peak"
