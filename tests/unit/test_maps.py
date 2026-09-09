"""CCP4 map export.

Calculated maps (rho0, the correction, their sum) show what the model is. Coefficient
maps (2Fo-Fc, Fo-Fc) show what the data say and are the ones comparable to a conventional
2Fo-Fc map. All of them are built from work reflections only -- model interpretation is
one of the things the free set must stay out of.
"""

import json

import gemmi
import jax.numpy as jnp
import numpy as np
import pytest

from crystal_field.maps import export_maps

EXPECTED = [
    "rho0.ccp4",
    "field.ccp4",
    "refined.ccp4",
    "2FoFc_field.ccp4",
    "2FoFc_atomic.ccp4",
    "FoFc_field.ccp4",
    "FoFc_atomic.ccp4",
    "2mFoDFc_field.ccp4",
    "2mFoDFc_atomic.ccp4",
    "mFoDFc_field.ccp4",
    "mFoDFc_atomic.ccp4",
    "maps.json",
]


@pytest.fixture
def fitted(prepared_dataset):
    """A prepared dataset with a short work-scope fit, so z_map.npy exists."""
    from crystal_field.crystallography.scaling import fit_baseline_scaling
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = prepared_dataset["cfg"]
    cfg = cfg.model_copy(
        update={"optimizer": cfg.optimizer.model_copy(update={"fit_scope": "work", "max_iterations": 15})}
    )
    fit_baseline_scaling(cfg)
    arrays = load_problem_arrays(cfg)
    _, density, _, _, objective, metrics, _ = build_functions(arrays, cfg)
    run_map_fit(cfg, arrays, objective, metrics, density)
    prepared_dataset["cfg"] = cfg
    return prepared_dataset


def test_every_map_is_written(fitted):
    manifest = export_maps(fitted["cfg"])
    directory = fitted["cfg"].run.output_dir / "maps"
    for name in EXPECTED:
        assert (directory / name).is_file(), name
    assert set(manifest["maps"]) == {n for n in EXPECTED if n.endswith(".ccp4")}


def test_the_maps_are_readable_with_the_right_cell_and_grid(fitted):
    export_maps(fitted["cfg"])
    metadata = fitted["metadata"]
    for name in EXPECTED:
        if not name.endswith(".ccp4"):
            continue
        ccp4 = gemmi.read_ccp4_map(str(fitted["cfg"].run.output_dir / "maps" / name))
        assert ccp4.grid.shape == tuple(metadata["grid_shape"]), name
        assert ccp4.grid.unit_cell.a == pytest.approx(metadata["cell"][0], rel=1e-4)


def test_the_field_map_is_exactly_the_difference_of_the_other_two(fitted):
    """`field.ccp4` must be the inferred correction, not a re-derivation of it."""
    export_maps(fitted["cfg"])
    directory = fitted["cfg"].run.output_dir / "maps"

    def read(name):
        return np.array(gemmi.read_ccp4_map(str(directory / name)).grid, copy=True)

    np.testing.assert_allclose(read("field.ccp4"), read("refined.ccp4") - read("rho0.ccp4"), atol=1e-5)


def test_the_correction_is_not_trivially_zero(fitted):
    manifest = export_maps(fitted["cfg"])
    assert manifest["statistics"]["field"]["rms"] > 0.0


def test_free_reflections_are_excluded_from_map_calculation(fitted):
    manifest = export_maps(fitted["cfg"])
    metadata = fitted["metadata"]
    assert manifest["n_reflections_used"] == metadata["n_train"] + metadata["n_tune"]
    assert manifest["n_reflections_used"] < metadata["n_total_kept"]
    assert "free set is excluded" in manifest["reflections_used"]


def test_the_manifest_records_provenance(fitted):
    manifest = export_maps(fitted["cfg"])
    written = json.loads((fitted["cfg"].run.output_dir / "maps" / "maps.json").read_text())
    assert written == manifest
    assert written["spacegroup"] == fitted["metadata"]["spacegroup"]
    assert written["grid_shape"] == list(fitted["metadata"]["grid_shape"])
    assert "sigma-A" in written["weighting"], "the absence of weighting must be stated"


def test_related_paths_resolve_against_the_manifest_directory(fitted):
    """Every key in the manifest is a path relative to maps.json's own directory.

    `related` points at another stage's output, and `decomposition/` is a SIBLING of
    `maps/`, not a child: a key written without the `../` resolves to
    `final/maps/decomposition/...`, a path that never exists on any run. The check is
    performed the way a reader would -- join the key to the manifest's directory and open
    the file -- with a real decomposition on disk to resolve against.
    """
    from crystal_field.analysis.decomposition import run_decomposition

    cfg = fitted["cfg"]
    cfg = cfg.model_copy(update={"decomposition": cfg.decomposition.model_copy(update={"n_capacity_trials": 1})})
    run_decomposition(cfg)
    manifest = export_maps(cfg)

    directory = cfg.run.output_dir / "maps"
    assert manifest["related"], "the cross-reference must not be empty"
    for key in manifest["related"]:
        resolved = (directory / key).resolve()
        assert resolved.is_file(), f"{key} resolves to {resolved}, which does not exist"
    for key in manifest["maps"]:
        assert (directory / key).resolve().is_file(), key


def test_refinement_flattens_the_difference_map(fitted):
    """Fo-Fc with the fitted phases should be flatter than with the atomic ones."""
    manifest = export_maps(fitted["cfg"])
    assert manifest["statistics"]["FoFc_field"]["rms"] < manifest["statistics"]["FoFc_atomic"]["rms"]


def test_an_fc_map_reproduces_the_density_it_came_from(fitted):
    """The end-to-end check on map construction.

    Rebuilding a map from the model's own structure factors must return the model's
    density. It catches both ways this can go wrong: the forward model uses
    exp(-2 pi i h.r) while Gemmi's map transform expects exp(+2 pi i h.r), so failing to
    conjugate negates every phase; and ComplexAsuData expands by symmetry, so feeding it
    a full sphere double-counts. Either mistake drops the correlation from ~0.87 to ~0.12.
    """
    import jax

    from crystal_field.inference.problem import build_complex_fcalc
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.maps import _map_from_coefficients

    cfg = fitted["cfg"]
    metadata = fitted["metadata"]
    arrays = load_problem_arrays(cfg)
    complex_fcalc, _ = build_complex_fcalc(arrays, cfg)

    cell = gemmi.UnitCell(*metadata["cell"])
    spacegroup = gemmi.SpaceGroup(metadata["spacegroup"])
    hkls = np.asarray(jax.device_get(arrays.hkls))
    rho0 = np.asarray(jax.device_get(arrays.rho0))

    fcalc = np.conj(np.asarray(jax.device_get(complex_fcalc(jnp.zeros_like(arrays.rho0))), dtype=np.complex128))
    grid = _map_from_coefficients(hkls, fcalc, cell, spacegroup, metadata["grid_shape"])
    correlation = float(np.corrcoef(grid.ravel(), rho0.ravel())[0, 1])
    assert correlation > 0.8, f"Fc map should reproduce rho0; got r={correlation:.4f}"

    # Without the conjugation the same calculation must be visibly wrong.
    wrong = _map_from_coefficients(hkls, np.conj(fcalc), cell, spacegroup, metadata["grid_shape"])
    assert float(np.corrcoef(wrong.ravel(), rho0.ravel())[0, 1]) < 0.5


def test_coefficient_maps_differ_from_calculated_maps(fitted):
    """The point of producing both: they are not the same object."""
    export_maps(fitted["cfg"])
    directory = fitted["cfg"].run.output_dir / "maps"

    def read(name):
        return np.array(gemmi.read_ccp4_map(str(directory / name)).grid, copy=True)

    experimental, calculated = read("2FoFc_field.ccp4"), read("refined.ccp4")
    correlation = np.corrcoef(experimental.ravel(), calculated.ravel())[0, 1]
    assert 0.5 < correlation < 0.999, f"expected related but distinct maps, got r={correlation:.4f}"


def test_a_missing_fit_is_reported(prepared_dataset):
    with pytest.raises(FileNotFoundError, match="fieldrefine fit-map"):
        export_maps(prepared_dataset["cfg"])


def test_complex_structure_factors_match_the_amplitudes_the_likelihood_uses(fitted):
    """Pins build_complex_fcalc to build_functions so the two cannot drift apart."""
    from crystal_field.inference.problem import build_complex_fcalc, build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = fitted["cfg"]
    arrays = load_problem_arrays(cfg)
    _, _, predict_all, _, _, _, _ = build_functions(arrays, cfg)
    complex_fcalc, correction = build_complex_fcalc(arrays, cfg)

    z = jnp.asarray(np.load(cfg.run.output_dir / "fit" / "z_map.npy"), dtype=arrays.rho0.dtype)
    for candidate in (jnp.zeros_like(z), z):
        np.testing.assert_allclose(
            np.abs(np.asarray(complex_fcalc(candidate))),
            np.asarray(predict_all(candidate)),
            rtol=1e-5,
            atol=1e-6,
        )
    np.testing.assert_allclose(np.asarray(correction(jnp.zeros_like(z))), 0.0, atol=1e-7)


def test_export_maps_is_wired_into_the_final_fit_job():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "slurm/60_final_fit.sbatch").read_text()
    assert "run_cfi export-maps" in text


# --- sigma-A weighted maps -----------------------------------------------------------


def test_sigma_a_provenance_is_recorded(fitted):
    manifest = export_maps(fitted["cfg"])
    sigma_a = manifest["sigma_a"]
    assert sigma_a["source"] == "work"
    assert "biased" in sigma_a["estimated_on"], "the bias must be stated, not hidden"
    assert sigma_a["n_reflections"] == manifest["n_reflections_used"]
    assert 0.0 <= sigma_a["mean_figure_of_merit"] <= 1.0
    assert len(sigma_a["shells"]) == sigma_a["n_bins"]


def test_sigma_a_shells_span_the_resolution_range(fitted):
    manifest = export_maps(fitted["cfg"])
    shells = manifest["sigma_a"]["shells"]
    assert shells[0]["d_max"] >= shells[-1]["d_min"]
    assert sum(shell["n"] for shell in shells) == manifest["sigma_a"]["n_reflections"]


def test_the_weighted_difference_map_is_flatter_after_refinement(fitted):
    manifest = export_maps(fitted["cfg"])
    statistics = manifest["statistics"]
    assert statistics["mFoDFc_field"]["rms"] < statistics["mFoDFc_atomic"]["rms"]


def test_the_weighted_map_resembles_the_unweighted_one_but_is_not_identical(fitted):
    """2mFo-DFc down-weights poorly explained reflections; 2Fo-Fc does not."""
    export_maps(fitted["cfg"])
    directory = fitted["cfg"].run.output_dir / "maps"

    def read(name):
        return np.array(gemmi.read_ccp4_map(str(directory / name)).grid, copy=True)

    weighted, plain = read("2mFoDFc_field.ccp4"), read("2FoFc_field.ccp4")
    correlation = float(np.corrcoef(weighted.ravel(), plain.ravel())[0, 1])
    assert correlation > 0.9, f"the two should broadly agree, got r={correlation:.4f}"


def test_estimating_sigma_a_on_the_free_set_is_refused_before_the_lock(fitted):
    """Weighting from free reflections before the one-shot read would spend them."""
    with pytest.raises(RuntimeError, match="one-shot free-set evaluation"):
        export_maps(fitted["cfg"], sigma_a_from="free")


def test_estimating_sigma_a_on_the_free_set_is_allowed_once_it_is_spent(fitted):
    """After the free set has been read it is no longer held out, so weighting is free."""
    from crystal_field.analysis.model_selection import append_free_set_ledger

    cfg = fitted["cfg"]
    append_free_set_ledger(cfg, {"free_set_used": True, "evaluation_index": 1})
    manifest = export_maps(cfg, sigma_a_from="free")
    assert manifest["sigma_a"]["source"] == "free"
    assert "unbiased" in manifest["sigma_a"]["estimated_on"]
    assert manifest["sigma_a"]["n_reflections"] == fitted["metadata"]["n_free"]


def test_an_unknown_sigma_a_source_is_rejected(fitted):
    with pytest.raises(ValueError, match="must be 'work' or 'free'"):
        export_maps(fitted["cfg"], sigma_a_from="tune")
