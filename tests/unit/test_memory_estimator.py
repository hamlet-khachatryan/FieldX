"""GPU memory planning (plan section 21)."""

import pytest
from conftest import problem_config

from crystal_field.hpc import check_information_budget, estimate_memory, grid_shape_for


def test_grid_comes_from_prepared_metadata_when_it_exists(prepared_dataset):
    cfg = prepared_dataset["cfg"]
    shape, source = grid_shape_for(cfg)
    assert shape == tuple(prepared_dataset["metadata"]["grid_shape"])
    assert source == "prepared metadata"


def test_grid_is_derived_from_the_model_before_preparation(tiny_dataset):
    """The estimator must work on a login node before any job has run."""
    shape, source = grid_shape_for(tiny_dataset["cfg"])
    assert all(n > 0 for n in shape)
    assert source == "derived from input.model cell and d_min"


def test_explicit_grid_shape_is_used(tiny_dataset):
    cfg = tiny_dataset["cfg"]
    cfg = cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"shape": (32, 32, 32)})})
    assert grid_shape_for(cfg) == ((32, 32, 32), "grid.shape")


def test_estimates_scale_with_the_grid_and_the_mode_count(tiny_dataset):
    cfg = tiny_dataset["cfg"]
    small = estimate_memory(cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"shape": (16, 16, 16)})}))
    large = estimate_memory(cfg.model_copy(update={"grid": cfg.grid.model_copy(update={"shape": (32, 32, 32)})}))
    assert large["n_voxels"] == 8 * small["n_voxels"]
    assert large["estimated_gpu_GiB_lbfgs"] > small["estimated_gpu_GiB_lbfgs"]

    more_modes = estimate_memory(
        cfg.model_copy(
            update={
                "grid": cfg.grid.model_copy(update={"shape": (16, 16, 16)}),
                "information": cfg.information.model_copy(update={"n_modes": 64}),
            }
        )
    )
    assert more_modes["estimated_gpu_GiB_information"] > small["estimated_gpu_GiB_information"]


def test_double_precision_doubles_the_field_cost(tiny_dataset):
    cfg = tiny_dataset["cfg"].model_copy(
        update={"grid": tiny_dataset["cfg"].grid.model_copy(update={"shape": (16, 16, 16)})}
    )
    single = estimate_memory(cfg)
    double = estimate_memory(cfg.model_copy(update={"run": cfg.run.model_copy(update={"enable_x64": True})}))
    assert double["single_real_field_GiB"] == pytest.approx(2 * single["single_real_field_GiB"])
    assert double["precision"] == "float64"


def test_an_oversized_information_job_is_refused_before_it_allocates(tiny_dataset):
    cfg = tiny_dataset["cfg"]
    oversized = cfg.model_copy(
        update={
            "grid": cfg.grid.model_copy(update={"shape": (512, 512, 512)}),
            "information": cfg.information.model_copy(update={"n_modes": 256, "memory_budget_gib": 40.0}),
        }
    )
    assert estimate_memory(oversized)["information_within_budget"] is False
    with pytest.raises(MemoryError, match=r"information\.memory_budget_gib"):
        check_information_budget(oversized)


def test_lobpcg_shape_constraint_is_checked(tiny_dataset):
    cfg = tiny_dataset["cfg"]
    impossible = cfg.model_copy(
        update={
            "grid": cfg.grid.model_copy(update={"shape": (4, 4, 4)}),
            "information": cfg.information.model_copy(update={"n_modes": 40}),
        }
    )
    with pytest.raises(ValueError, match=r"0 < 5\*n_modes < n_voxels"):
        check_information_budget(impossible)


def test_a_reasonable_job_passes_the_budget(prepared_dataset):
    assert check_information_budget(prepared_dataset["cfg"])["information_within_budget"] is True


def test_estimator_reports_a_clear_error_without_metadata_or_a_model(tmp_path):
    cfg = problem_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="Cannot estimate memory"):
        estimate_memory(cfg)
