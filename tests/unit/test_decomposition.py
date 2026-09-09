"""Decomposition of the inferred correction onto the atomic tangent space."""

import json

import gemmi
import numpy as np
import pytest
from conftest import write_tiny_model

from crystal_field.analysis.decomposition import (
    capacity_control,
    correction_from_fit,
    decompose_target,
    run_decomposition,
    solve_normal_equations,
    split_symmetric,
    symmetrize_grid,
)
from crystal_field.analysis.tangent import build_tangent_basis, selected_atoms, tangent_column
from crystal_field.crystallography.density import model_density_on_grid


def _problem(n_rows=200, n_columns=12, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((n_rows, n_columns))
    truth = rng.standard_normal(n_columns)
    target = basis @ truth
    return basis, truth, target


def test_exact_recovery_when_the_target_lies_in_the_span():
    basis, truth, target = _problem()
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    np.testing.assert_allclose(result["amplitudes"], truth, rtol=1e-8, atol=1e-8)
    assert result["explained_fraction"] == pytest.approx(1.0, abs=1e-10)
    assert result["rank"] == basis.shape[1]


def test_a_target_orthogonal_to_the_basis_explains_nothing():
    basis, _, _ = _problem()
    rng = np.random.default_rng(7)
    orthogonal = rng.standard_normal(basis.shape[0])
    orthogonal -= basis @ np.linalg.lstsq(basis, orthogonal, rcond=None)[0]
    result = solve_normal_equations(basis.T @ basis, basis.T @ orthogonal, float(orthogonal @ orthogonal))
    assert result["explained_fraction"] == pytest.approx(0.0, abs=1e-8)


def test_explained_fraction_is_bounded():
    basis, _, _ = _problem()
    rng = np.random.default_rng(3)
    for _ in range(5):
        target = rng.standard_normal(basis.shape[0])
        result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
        assert -1e-9 <= result["explained_fraction"] <= 1.0 + 1e-9


def test_ridge_shrinks_the_amplitudes():
    basis, _, target = _problem()
    plain = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    ridged = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target), ridge=10.0)
    assert np.linalg.norm(ridged["amplitudes"]) < np.linalg.norm(plain["amplitudes"])
    assert ridged["explained_fraction"] < plain["explained_fraction"]


def test_rank_deficiency_is_reported_not_hidden():
    """A duplicated column makes the basis rank deficient; that is a fact, not an error."""
    basis, _, _ = _problem(n_columns=6)
    basis = np.hstack([basis, basis[:, :1]])
    target = basis @ np.ones(basis.shape[1])
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    assert result["rank"] < basis.shape[1]
    assert np.isfinite(result["condition_number"])
    assert result["explained_fraction"] == pytest.approx(1.0, abs=1e-8)


def test_a_zero_target_does_not_divide_by_zero():
    basis, _, _ = _problem()
    result = solve_normal_equations(basis.T @ basis, np.zeros(basis.shape[1]), 0.0)
    assert np.isfinite(result["explained_fraction"])
    assert result["explained_fraction"] == 0.0


def test_ridge_does_not_leak_into_the_reported_residual():
    """The reported residual and explained_fraction must track the DATA misfit, not the
    ridge-regularized objective. A ridge penalty inflates ||target - Phi a||^2 by
    ridge * ||a||^2 if the ridged gram is reused for the residual computation instead of
    the original one -- that bug is invisible to test_ridge_shrinks_the_amplitudes because
    it only checks direction, and inflating the residual pushes the fraction in the same
    direction that ridge is expected to push it anyway.
    """
    basis, _, target = _problem(n_rows=50, n_columns=8, seed=1)
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target), ridge=1.0)

    data_residual = float(np.sum((target - basis @ result["amplitudes"]) ** 2))
    np.testing.assert_allclose(result["residual_norm_squared"], data_residual, rtol=1e-8, atol=1e-10)

    expected_fraction = 1.0 - data_residual / float(target @ target)
    np.testing.assert_allclose(result["explained_fraction"], expected_fraction, rtol=1e-8, atol=1e-10)


def test_symmetrizing_an_already_symmetric_grid_changes_nothing(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    from crystal_field.crystallography.density import model_density_on_grid

    density, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, (24, 30, 36), 2.6, 1e-6)
    again = symmetrize_grid(density.astype(np.float64), structure.cell, spacegroup)
    np.testing.assert_allclose(again, density, rtol=1e-5, atol=1e-7)


def test_symmetrizing_is_idempotent(tmp_path):
    rng = np.random.default_rng(0)
    cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    field = rng.standard_normal((24, 30, 36))
    once = symmetrize_grid(field, cell, spacegroup)
    twice = symmetrize_grid(once, cell, spacegroup)
    np.testing.assert_allclose(twice, once, rtol=1e-5, atol=1e-7)


def test_the_antisymmetric_fraction_is_reported():
    """A random field in a symmetric group is mostly antisymmetric; that must be visible."""
    rng = np.random.default_rng(1)
    cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    field = rng.standard_normal((24, 30, 36))
    symmetric, antisymmetric_fraction = split_symmetric(field, cell, spacegroup)
    assert 0.0 < antisymmetric_fraction < 1.0
    assert np.linalg.norm(symmetric) < np.linalg.norm(field)


def test_a_symmetric_field_has_no_antisymmetric_part(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    from crystal_field.crystallography.density import model_density_on_grid

    density, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, (24, 30, 36), 2.6, 1e-6)
    _, fraction = split_symmetric(density.astype(np.float64), structure.cell, spacegroup)
    assert fraction < 1e-3


def test_p1_symmetrization_is_the_identity():
    rng = np.random.default_rng(2)
    field = rng.standard_normal((16, 16, 16))
    result = symmetrize_grid(field, gemmi.UnitCell(20, 20, 20, 90, 90, 90), gemmi.SpaceGroup("P 1"))
    np.testing.assert_allclose(result, field, rtol=1e-6, atol=1e-8)


SHAPE = (24, 30, 36)


@pytest.fixture
def tiny_basis(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 1")
    basis = build_tangent_basis(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6)
    return structure, spacegroup, basis


def test_ground_truth_a_known_displacement_is_recovered(tiny_basis):
    """The decisive test: move an atom by a known amount, get that amount back."""
    structure, spacegroup, basis = tiny_basis
    atoms = selected_atoms(structure[0])
    index, atom = atoms[1]
    delta = 0.05

    before, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6)
    original = atom.pos.x
    atom.pos.x = original + delta
    after, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6)
    atom.pos.x = original

    target = after.astype(np.float64) - before.astype(np.float64)
    result = decompose_target(basis, target)

    assert result["explained_fraction"] > 0.99, "an atomic displacement must be atom-explainable"
    column = basis.labels.index((index, "x"))
    assert result["amplitudes"][column] == pytest.approx(delta, rel=0.1)
    others = np.delete(result["amplitudes"], column)
    assert np.abs(others).max() < 0.2 * abs(result["amplitudes"][column])


def test_explained_and_unexplained_sum_to_the_target(tiny_basis):
    _, _, basis = tiny_basis
    rng = np.random.default_rng(4)
    target = rng.standard_normal(SHAPE)
    result = decompose_target(basis, target)
    np.testing.assert_allclose(result["explained"] + result["unexplained"], target, atol=1e-8)


def test_a_random_field_is_less_explainable_than_a_real_displacement(tiny_basis):
    structure, spacegroup, basis = tiny_basis
    atom = selected_atoms(structure[0])[1][1]
    real = 0.05 * tangent_column(atom, "x", structure.cell, spacegroup, SHAPE, 2.6, 1e-6)

    rng = np.random.default_rng(11)
    noise = rng.standard_normal(SHAPE)
    noise *= np.linalg.norm(real) / np.linalg.norm(noise)

    assert decompose_target(basis, real)["explained_fraction"] > decompose_target(basis, noise)["explained_fraction"]


def test_the_control_detects_over_parameterisation(tmp_path):
    """The guard must be a measurement, not a decoration.

    A larger basis fits noise better. If moving from `coordinates` to `full` did not
    raise the control's explained fraction, the control would not be measuring capacity.
    """
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 1")
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(SHAPE)

    small = build_tangent_basis(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6, basis="coordinates")
    large = build_tangent_basis(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6, basis="full")
    assert large.n_columns > small.n_columns
    assert decompose_target(large, noise)["explained_fraction"] > decompose_target(small, noise)["explained_fraction"]


def test_capacity_trials_are_seeded_and_reproducible(tiny_basis):
    _, spacegroup, basis = tiny_basis
    kwargs = dict(
        basis=basis,
        reference_norm=1.0,
        transfer=None,
        unit_cell_volume=13440.0,
        cell=gemmi.UnitCell(20, 24, 28, 90, 90, 90),
        spacegroup=spacegroup,
        seed=17,
        n_trials=3,
    )
    first = capacity_control(**kwargs)
    second = capacity_control(**kwargs)
    assert first["fractions"] == second["fractions"]
    assert first["n_trials"] == 3
    assert 0.0 <= first["mean"] <= 1.0

    # Reproducible is only half of the requirement. Spec section 6.1 asks for n_trials
    # DISTINCT draws: a control that draws the same field every time is perfectly
    # reproducible, reports sd == 0, and measures nothing. Dropping the trial index from
    # the seed produces exactly that, so the distinctness is asserted directly rather than
    # through `sd >= 0`, which no implementation can fail.
    assert len(set(first["fractions"])) == 3, "each trial must be a distinct draw"
    assert first["sd"] > 0.0


@pytest.fixture
def fitted(prepared_dataset):
    from crystal_field.crystallography.scaling import fit_baseline_scaling
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = prepared_dataset["cfg"]
    cfg = cfg.model_copy(
        update={
            "optimizer": cfg.optimizer.model_copy(update={"fit_scope": "work", "max_iterations": 10}),
            "decomposition": cfg.decomposition.model_copy(update={"n_capacity_trials": 2}),
        }
    )
    fit_baseline_scaling(cfg)
    arrays = load_problem_arrays(cfg)
    _, density, _, _, objective, metrics, _ = build_functions(arrays, cfg)
    run_map_fit(cfg, arrays, objective, metrics, density)
    prepared_dataset["cfg"] = cfg
    return prepared_dataset


def test_all_artifacts_are_written(fitted):
    report = run_decomposition(fitted["cfg"])
    directory = fitted["cfg"].run.output_dir / "decomposition"
    for name in ("decomposition.json", "explained.ccp4", "unexplained.ccp4"):
        assert (directory / name).is_file(), name
    assert json.loads((directory / "decomposition.json").read_text()) == report


def test_the_report_carries_both_targets_and_the_control(fitted):
    """Spec section 11: no explained fraction may be quoted without its own control.

    The bound `0 <= explained_fraction <= 1` is deliberately NOT asserted here:
    `solve_normal_equations` ends with `np.clip(explained, 0.0, 1.0)`, so the code under
    test enforces it and the assertion could not fail whatever the decomposition did.
    """
    report = run_decomposition(fitted["cfg"])
    assert set(report["targets"]) == {"full_correction", "data_supported"}
    # Every target carries its own control, computed on that target -- the full-correction
    # control does not stand in for the data-supported one.
    for name, target in report["targets"].items():
        control = target["control"]
        assert control["n_trials"] == 2, name
        assert 0.0 <= control["mean"] <= 1.0, name
        assert target["explained_above_control"] == pytest.approx(target["explained_fraction"] - control["mean"]), name
    assert report["basis"]["name"] == "coordinates"
    assert report["basis"]["rank"] > 0
    assert "antisymmetric_norm_fraction" in report
    # The two controls score different targets, so they must not be the same numbers.
    assert (
        report["targets"]["data_supported"]["control"]["fractions"]
        != report["targets"]["full_correction"]["control"]["fractions"]
    )


def test_no_truncation_means_no_norm_fraction_is_reported(fitted):
    """`box_radius_angstrom` defaults to null, and then `min_norm_fraction` is vacuous.

    `_truncate` returns the column unchanged when no radius is given, so the fraction would
    compare a column with itself and be 1.0 for every run ever made. A reader told to read
    this report must not find a number there that cannot vary.
    """
    cfg = fitted["cfg"]
    assert cfg.decomposition.box_radius_angstrom is None
    assert "min_norm_fraction" not in run_decomposition(cfg)["basis"]

    truncated = cfg.model_copy(
        update={"decomposition": cfg.decomposition.model_copy(update={"box_radius_angstrom": 2.5})}
    )
    report = run_decomposition(truncated)
    assert 0.999 <= report["basis"]["min_norm_fraction"] < 1.0


def test_explained_and_unexplained_reconstruct_the_symmetrized_correction(fitted):
    """Per spec section 3, the target is symmetrize(u), not u."""
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = fitted["cfg"]
    report = run_decomposition(cfg)
    directory = cfg.run.output_dir / "decomposition"
    explained = np.array(gemmi.read_ccp4_map(str(directory / "explained.ccp4")).grid, copy=True)
    unexplained = np.array(gemmi.read_ccp4_map(str(directory / "unexplained.ccp4")).grid, copy=True)
    assert explained.shape == unexplained.shape
    assert np.all(np.isfinite(explained)) and np.all(np.isfinite(unexplained))
    assert report["targets"]["full_correction"]["target_norm"] > 0

    # The name's promise: explained + unexplained reconstructs symmetrize(u) exactly, not
    # just "some finite grid of the right shape". Rebuild the expected target the same way
    # run_decomposition does, following the loading pattern the rest of this module and
    # test_maps.py use (fitted["metadata"] for cell/spacegroup, load_problem_arrays for arrays).
    metadata = fitted["metadata"]
    cell = gemmi.UnitCell(*metadata["cell"])
    spacegroup = gemmi.SpaceGroup(metadata["spacegroup"])
    arrays = load_problem_arrays(cfg)
    expected = symmetrize_grid(correction_from_fit(cfg, arrays), cell, spacegroup)

    # explained.ccp4 and unexplained.ccp4 each round-trip through float32 independently
    # (crystal_field.maps._write casts before writing), so their sum carries float32
    # rounding on the order of ~1.2e-7 relative to the largest value in the grid. 1e-5 of
    # the peak magnitude is ~80x that noise floor -- generous enough to absorb it and any
    # incidental FFT/symmetrization rounding, but 10,000x tighter than would be needed to
    # let a swapped-map or transposed-axis bug pass.
    tolerance = 1e-5 * float(np.abs(expected).max())
    np.testing.assert_allclose(explained + unexplained, expected, atol=tolerance, rtol=0.0)

    # The reconstruction above is, on its own, an algebraic identity: production sets
    # `unexplained = target - explained`, so `explained + unexplained == target` holds for
    # ANY `explained` whatsoever -- scale `explained` by two and it still passes. These two
    # maps are the stage's primary scientific deliverable, so they are also tied to the
    # scalar the report quotes: the unexplained map must carry exactly the residual power
    # that `explained_fraction` claims is left over. That pins the CONTENT of `explained`,
    # not just its arithmetic relationship to `unexplained`.
    residual_fraction = float(np.sum(unexplained**2) / np.sum(expected**2))
    assert 1.0 - residual_fraction == pytest.approx(
        report["targets"]["full_correction"]["explained_fraction"], abs=1e-6
    )


def test_run_decomposition_symmetrizes_in_a_space_group_where_that_matters(symmetric_prepared_dataset):
    """Spec section 3: the symmetrization is "not optional". This is where that is proved.

    Every other `run_decomposition` test derives from `tiny_dataset`, whose space group is
    `P 1` -- there `symmetrize_avg` is the identity, so deleting the symmetrization from the
    production path changes nothing any of them can see. This one runs the real entry point
    in `P 21 21 21`.

    The latent field is written directly rather than fitted, and deliberately so: the
    likelihood reaches the grid only through `symmetry_projected_fcalc`, so its gradient is
    symmetric and a fit starting from z = 0 produces a `u` whose antisymmetric fraction is
    ~1e-6. A fitted z would leave this test as blind as `P 1` does. `z` is unconstrained in
    the model, which is exactly why spec section 3 requires the projection at all, so an
    arbitrary z is the case that has to be covered.
    """
    from crystal_field.inference.runtime import load_problem_arrays

    dataset = symmetric_prepared_dataset
    cfg = dataset["cfg"]
    cfg = cfg.model_copy(update={"decomposition": cfg.decomposition.model_copy(update={"n_capacity_trials": 2})})

    metadata = dataset["metadata"]
    cell = gemmi.UnitCell(*metadata["cell"])
    spacegroup = gemmi.SpaceGroup(metadata["spacegroup"])
    assert len(spacegroup.operations()) > 1, "the fixture must have real symmetry to test"

    fit_directory = cfg.run.output_dir / "fit"
    fit_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(19)
    np.save(fit_directory / "z_map.npy", rng.standard_normal(tuple(metadata["grid_shape"])).astype(np.float32))

    arrays = load_problem_arrays(cfg)
    correction = correction_from_fit(cfg, arrays)
    symmetric = symmetrize_grid(correction, cell, spacegroup)
    assert np.linalg.norm(symmetric) < 0.9 * np.linalg.norm(correction), (
        "the fixture must have a symmetric part to lose"
    )

    report = run_decomposition(cfg)

    # The decisive assertion: the target the stage decomposed is symmetrize(u), not u.
    # Passing the raw correction through instead makes target_norm equal ||u||.
    assert report["targets"]["full_correction"]["target_norm"] == pytest.approx(
        float(np.linalg.norm(symmetric)), rel=1e-6
    )
    assert report["targets"]["full_correction"]["target_norm"] < 0.9 * float(np.linalg.norm(correction))
    assert report["antisymmetric_norm_fraction"] > 0.1

    # And the maps written from that target reconstruct symmetrize(u), not u.
    directory = cfg.run.output_dir / "decomposition"
    explained = np.array(gemmi.read_ccp4_map(str(directory / "explained.ccp4")).grid, copy=True)
    unexplained = np.array(gemmi.read_ccp4_map(str(directory / "unexplained.ccp4")).grid, copy=True)
    tolerance = 1e-5 * float(np.abs(symmetric).max())
    np.testing.assert_allclose(explained + unexplained, symmetric, atol=tolerance, rtol=0.0)


def test_a_missing_fit_is_reported(prepared_dataset):
    with pytest.raises(FileNotFoundError, match="fieldrefine fit-map"):
        run_decomposition(prepared_dataset["cfg"])


def test_the_free_set_does_not_influence_any_reported_number(fitted):
    """`_data_supported_target` reads exactly one field to decide what counts as free:
    `split`, partitioned as `work = split != 2` (decomposition.py:169-170). It never reads
    `observation` or `uncertainty`. A test that mutates `observation` on the free rows
    passes identically against a broken implementation that includes free reflections
    (e.g. `work = np.ones_like(split, dtype=bool)`), because nothing in this feature ever
    looks at `observation` -- so this test mutates `split` instead, the field the code
    actually consumes, with a positive control proving the mutation is observable.
    """
    cfg = fitted["cfg"]
    baseline = run_decomposition(cfg)

    path = cfg.run.data_dir / "reflections.npz"
    original = dict(np.load(path))
    baseline_split = original["split"]
    baseline_n_reflections = baseline["targets"]["data_supported"]["n_reflections"]
    # Pin the exclusion directly: the reported reflection count is exactly the work count.
    assert baseline_n_reflections == int((baseline_split != 2).sum())

    # --- Negative half: relabeling WORK reflections between the two WORK codes (train=0,
    # tune=1) must leave every reported number unchanged, because the feature partitions
    # only on split != 2 -- neither code is free, so this is invisible to `work`. ---
    stored = {key: value.copy() for key, value in original.items()}
    split = stored["split"]
    work_indices = np.flatnonzero(split != 2)
    assert work_indices.size > 1
    relabel = work_indices[: work_indices.size // 2]
    split[relabel] = np.where(split[relabel] == 0, 1, 0)
    np.savez_compressed(path, **stored)
    try:
        relabeled = run_decomposition(cfg)
    finally:
        np.savez_compressed(path, **original)

    for name in ("full_correction", "data_supported"):
        assert relabeled["targets"][name]["explained_fraction"] == pytest.approx(
            baseline["targets"][name]["explained_fraction"]
        )
        assert relabeled["targets"][name]["target_norm"] == pytest.approx(baseline["targets"][name]["target_norm"])
    assert relabeled["targets"]["data_supported"]["n_reflections"] == baseline_n_reflections

    # --- Positive control: moving WORK reflections to FREE (2) MUST change data_supported,
    # and must leave full_correction untouched -- it reads no reflections at all. This is
    # the evidence that the shape of test above can actually fail. ---
    stored = {key: value.copy() for key, value in original.items()}
    split = stored["split"]
    work_indices = np.flatnonzero(split != 2)
    to_free = work_indices[: work_indices.size // 3]
    assert to_free.size > 0
    split[to_free] = 2
    np.savez_compressed(path, **stored)
    try:
        shrunk = run_decomposition(cfg)
    finally:
        np.savez_compressed(path, **original)

    shrunk_n_reflections = shrunk["targets"]["data_supported"]["n_reflections"]
    assert shrunk_n_reflections < baseline_n_reflections
    assert shrunk["targets"]["data_supported"]["explained_fraction"] != pytest.approx(
        baseline["targets"]["data_supported"]["explained_fraction"]
    )
    assert shrunk["targets"]["full_correction"]["explained_fraction"] == pytest.approx(
        baseline["targets"]["full_correction"]["explained_fraction"]
    )
    assert shrunk["targets"]["full_correction"]["target_norm"] == pytest.approx(
        baseline["targets"]["full_correction"]["target_norm"]
    )


def test_the_basis_can_be_overridden(fitted):
    report = run_decomposition(fitted["cfg"], basis="full")
    assert report["basis"]["name"] == "full"
    assert report["basis"]["n_columns"] > 0


def test_the_span_property_v3_is_the_case_phi_equals_zero(fitted):
    """Spec section 9.4, and the invariant increment 2 must preserve.

    docs/V4_DESIGN.md section 2 states that v3 is the special case Phi = 0. With the
    basis disabled nothing may be claimed as explained. Trivial here, but it is the
    property joint refinement has to keep, so it is pinned from the start.
    """
    cfg = fitted["cfg"]
    disabled = cfg.model_copy(update={"decomposition": cfg.decomposition.model_copy(update={"enabled": False})})
    report = run_decomposition(disabled)
    assert report["enabled"] is False
    assert report["targets"]["full_correction"]["explained_fraction"] == 0.0
    assert report["basis"]["n_columns"] == 0


def test_a_disabled_decomposition_writes_nothing(fitted):
    cfg = fitted["cfg"]
    disabled = cfg.model_copy(update={"decomposition": cfg.decomposition.model_copy(update={"enabled": False})})
    run_decomposition(disabled)
    assert not (disabled.run.output_dir / "decomposition" / "decomposition.json").exists()
    assert not (disabled.run.output_dir / "decomposition").exists()
