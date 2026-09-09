"""Declarative prior sweeps.

A grid may declare axes instead of enumerating candidates. The product is taken over the
declared axes, and combinations that differ only in a parameter the kernel or latent
distribution ignores are collapsed: alpha does nothing to a squared-exponential kernel,
correlation length does nothing to band-limited white noise. "Compute every mode" must
not mean "fit the same operator repeatedly".
"""

import json
from pathlib import Path

import pytest
import yaml

from crystal_field.analysis.model_selection import DEFAULT_MAX_CANDIDATES, expand_prior_grid, expand_sweep
from crystal_field.config import PriorConfig, dump_config, load_config
from crystal_field.model.prior import effective_parameters

ROOT = Path(__file__).resolve().parents[2]
BASE = PriorConfig().model_dump(mode="json")


def _names(candidates):
    return [candidate["name"] for candidate in candidates]


def test_a_single_axis_expands_to_one_candidate_each():
    candidates = expand_sweep({"tau_density": [0.02, 0.04, 0.06]}, BASE)
    assert len(candidates) == 3
    assert [c["prior"]["tau_density"] for c in candidates] == [0.02, 0.04, 0.06]


def test_axes_multiply():
    candidates = expand_sweep({"tau_density": [0.02, 0.04], "correlation_length_angstrom": [0.5, 1.0, 1.5]}, BASE)
    assert len(candidates) == 6


def test_a_scalar_axis_is_accepted():
    assert len(expand_sweep({"tau_density": 0.03, "alpha": [2.0, 2.5]}, BASE)) == 2


def test_alpha_is_inert_for_squared_exponential():
    """Sweeping an ignored parameter must not multiply the job count."""
    candidates = expand_sweep({"kernel": ["squared_exponential"], "alpha": [2.0, 2.5, 3.0]}, BASE)
    assert len(candidates) == 1
    assert "alpha" not in candidates[0]["prior"]


def test_correlation_length_and_alpha_are_inert_for_white_noise():
    candidates = expand_sweep(
        {"kernel": ["bandlimited_white"], "correlation_length_angstrom": [0.5, 1.0], "alpha": [2.0, 3.0]}, BASE
    )
    assert len(candidates) == 1
    assert set(candidates[0]["prior"]) == {"kernel", "tau_density", "remove_mean", "latent_distribution"}


def test_latent_parameters_are_inert_under_a_gaussian_latent():
    candidates = expand_sweep({"latent_distribution": ["gaussian"], "student_t_df": [3.0, 4.0, 8.0]}, BASE)
    assert len(candidates) == 1


def test_latent_parameters_matter_for_their_own_distribution():
    candidates = expand_sweep({"latent_distribution": ["student_t"], "student_t_df": [3.0, 8.0]}, BASE)
    assert len(candidates) == 2
    assert sorted(c["prior"]["student_t_df"] for c in candidates) == [3.0, 8.0]


def test_the_full_kernel_product_collapses_correctly():
    """3 kernels x 2 tau x 3 ell x 2 alpha = 36 declared, 20 distinct operators."""
    candidates = expand_sweep(
        {
            "kernel": ["matern", "squared_exponential", "bandlimited_white"],
            "tau_density": [0.025, 0.040],
            "correlation_length_angstrom": [0.5, 0.75, 1.5],
            "alpha": [2.0, 2.5],
        },
        BASE,
    )
    assert len(candidates) == 2 + 6 + 12
    assert len(set(_names(candidates))) == len(candidates), "names must be unique"


def test_every_candidate_is_a_distinct_operator():
    candidates = expand_sweep(
        {"kernel": ["matern", "squared_exponential"], "tau_density": [0.02, 0.05], "alpha": [2.0, 2.5]}, BASE
    )
    signatures = {json.dumps(effective_parameters({**BASE, **c["prior"]}), sort_keys=True) for c in candidates}
    assert len(signatures) == len(candidates)


def test_names_encode_the_swept_values():
    candidates = expand_sweep(
        {"kernel": ["matern"], "tau_density": [0.025], "correlation_length_angstrom": [0.5]}, BASE
    )
    assert candidates[0]["name"] == "matern_t0p025_l0p5_a2p5"


@pytest.mark.parametrize("axis", ["kernel_name", "components", "tau", "d_min"])
def test_an_unsweepable_axis_is_rejected(axis):
    with pytest.raises(ValueError, match="Cannot sweep"):
        expand_sweep({axis: [1, 2]}, BASE)


def test_an_empty_axis_is_rejected():
    with pytest.raises(ValueError, match="is empty"):
        expand_sweep({"tau_density": []}, BASE)


def test_multiscale_cannot_be_swept():
    with pytest.raises(ValueError, match="multiscale_matern cannot be swept"):
        expand_sweep({"kernel": ["multiscale_matern"]}, BASE)


# --- integration with the grid file ------------------------------------------------


def _base_config(tiny_dataset, tmp_path):
    path = tmp_path / "base.yaml"
    dump_config(tiny_dataset["cfg"], path)
    return path


def _grid(tmp_path, spec):
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_a_grid_may_declare_a_sweep_instead_of_candidates(tiny_dataset, tmp_path):
    base = _base_config(tiny_dataset, tmp_path)
    grid = _grid(tmp_path, {"sweep": {"tau_density": [0.02, 0.04, 0.06]}})
    result = expand_prior_grid(base, grid, tmp_path / "out")
    assert result["n_candidates"] == 3


def test_a_sweep_and_explicit_candidates_combine(tiny_dataset, tmp_path):
    base = _base_config(tiny_dataset, tmp_path)
    grid = _grid(
        tmp_path,
        {
            "sweep": {"tau_density": [0.02, 0.04]},
            "candidates": [{"name": "hand_written", "prior": {"kernel": "bandlimited_white"}}],
        },
    )
    result = expand_prior_grid(base, grid, tmp_path / "out")
    assert result["n_candidates"] == 3
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert "hand_written" in [entry["name"] for entry in manifest]


def test_the_manifest_records_the_effective_prior(tiny_dataset, tmp_path):
    base = _base_config(tiny_dataset, tmp_path)
    grid = _grid(tmp_path, {"sweep": {"kernel": ["matern", "bandlimited_white"], "tau_density": [0.03]}})
    result = expand_prior_grid(base, grid, tmp_path / "out")
    manifest = json.loads(Path(result["manifest"]).read_text())
    by_kernel = {entry["prior"]["kernel"]: entry["prior"] for entry in manifest}
    assert by_kernel["matern"]["alpha"] is not None
    assert "alpha" not in by_kernel["bandlimited_white"]
    for entry in manifest:
        assert load_config(entry["config"]).prior.kernel == entry["prior"]["kernel"]


def test_an_oversized_sweep_is_refused(tiny_dataset, tmp_path):
    base = _base_config(tiny_dataset, tmp_path)
    grid = _grid(tmp_path, {"sweep": {"tau_density": [0.01 * i for i in range(1, DEFAULT_MAX_CANDIDATES + 5)]}})
    with pytest.raises(ValueError, match="above max_candidates"):
        expand_prior_grid(base, grid, tmp_path / "out")


def test_the_cap_can_be_raised_deliberately(tiny_dataset, tmp_path):
    base = _base_config(tiny_dataset, tmp_path)
    n = DEFAULT_MAX_CANDIDATES + 4
    grid = _grid(
        tmp_path,
        {"sweep": {"tau_density": [0.001 * i for i in range(1, n + 1)]}, "max_candidates": n},
    )
    assert expand_prior_grid(base, grid, tmp_path / "out")["n_candidates"] == n


def test_an_empty_grid_is_still_refused(tiny_dataset, tmp_path):
    base = _base_config(tiny_dataset, tmp_path)
    with pytest.raises(ValueError, match="no candidates and no sweep"):
        expand_prior_grid(base, _grid(tmp_path, {}), tmp_path / "out")


def test_the_committed_sweep_expands_within_its_declared_cap(tiny_dataset, tmp_path):
    base = _base_config(tiny_dataset, tmp_path)
    result = expand_prior_grid(base, ROOT / "configs/1ubq/prior_sweep.yaml", tmp_path / "out")
    manifest = json.loads(Path(result["manifest"]).read_text())
    kernels = {entry["prior"]["kernel"] for entry in manifest}
    assert kernels == {"matern", "squared_exponential", "bandlimited_white", "multiscale_matern"}
    assert (
        result["n_candidates"] == yaml.safe_load((ROOT / "configs/1ubq/prior_sweep.yaml").read_text())["max_candidates"]
    )
