import jax.numpy as jnp
import pytest

from crystal_field.config import load_config
from crystal_field.inference.problem import ProblemArrays, build_functions


def _arrays(observation):
    return ProblemArrays(
        rho0=jnp.zeros((4, 4, 4), dtype=jnp.float32),
        solvent_mask=jnp.zeros((4, 4, 4), dtype=jnp.float32),
        hkls=jnp.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=jnp.int32),
        observation=jnp.asarray(observation, dtype=jnp.float32),
        sigma=jnp.ones(3, dtype=jnp.float32),
        split=jnp.asarray([0, 1, 2], dtype=jnp.int8),
        reciprocal_metric=jnp.eye(3, dtype=jnp.float32) / 100.0,
        symmetry_rotations=jnp.eye(3, dtype=jnp.int32)[None, ...],
        symmetry_translations=jnp.zeros((1, 3), dtype=jnp.float32),
        overall_scale=jnp.ones(3, dtype=jnp.float32),
        solvent_scale=jnp.zeros(3, dtype=jnp.float32),
        unit_cell_volume=1000.0,
        d_min_angstrom=2.0,
        observation_kind="amplitude",
    )


def _cfg(scope):
    cfg = load_config("configs/6o2h/default.yaml")
    return cfg.model_copy(
        update={
            "optimizer": cfg.optimizer.model_copy(update={"fit_scope": scope}),
            "baseline": cfg.baseline.model_copy(
                update={"bulk_solvent": cfg.baseline.bulk_solvent.model_copy(update={"enabled": False})}
            ),
        }
    )


def test_train_objective_ignores_tune_and_free_observations():
    cfg = _cfg("train")
    objective_a = build_functions(_arrays([1.0, 2.0, 3.0]), cfg)[4]
    objective_b = build_functions(_arrays([1.0, 200.0, 300.0]), cfg)[4]
    z = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    assert float(objective_a(z)) == pytest.approx(float(objective_b(z)))


def test_work_objective_ignores_only_free_observations():
    cfg = _cfg("work")
    objective_a = build_functions(_arrays([1.0, 2.0, 3.0]), cfg)[4]
    objective_b = build_functions(_arrays([1.0, 2.0, 300.0]), cfg)[4]
    objective_c = build_functions(_arrays([1.0, 200.0, 3.0]), cfg)[4]
    z = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    assert float(objective_a(z)) == pytest.approx(float(objective_b(z)))
    assert float(objective_a(z)) != pytest.approx(float(objective_c(z)))
