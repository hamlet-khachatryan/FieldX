"""Train / tune / free separation (plan section 20G).

D_work = D_train u D_tune. D_free is never visible to the optimizer, prior selection,
hyperparameter selection, early stopping, grid selection, nuisance calibration or model
interpretation. These tests mutate the free observations and require every quantity that
may inform a decision to be bit-for-bit unchanged.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import problem_arrays, problem_config

from crystal_field.crystallography.io import _canonical_hash_key, _u01_for_key
from crystal_field.inference.problem import build_functions

TRAIN, TUNE, FREE = 0, 1, 2
SPLIT = (TRAIN, TRAIN, TUNE, TUNE, FREE)


def _functions(tmp_path, observation, scope):
    cfg = problem_config(tmp_path, optimizer={"fit_scope": scope})
    return build_functions(problem_arrays(observation=observation, split=SPLIT), cfg), cfg


def test_train_scope_objective_ignores_tune_and_free(tmp_path):
    z = jnp.zeros((6, 6, 6), dtype=jnp.float32)
    base = _functions(tmp_path, (1.0, 2.0, 3.0, 4.0, 5.0), "train")[0][4]
    mutated = _functions(tmp_path, (1.0, 2.0, 300.0, 400.0, 500.0), "train")[0][4]
    assert float(base(z)) == pytest.approx(float(mutated(z)))


def test_work_scope_objective_ignores_free_but_not_tune(tmp_path):
    z = jnp.zeros((6, 6, 6), dtype=jnp.float32)
    base = _functions(tmp_path, (1.0, 2.0, 3.0, 4.0, 5.0), "work")[0][4]
    free_changed = _functions(tmp_path, (1.0, 2.0, 3.0, 4.0, 500.0), "work")[0][4]
    tune_changed = _functions(tmp_path, (1.0, 2.0, 3.0, 400.0, 5.0), "work")[0][4]
    assert float(base(z)) == pytest.approx(float(free_changed(z)))
    assert float(base(z)) != pytest.approx(float(tune_changed(z)))


@pytest.mark.parametrize("scope", ["train", "work"])
def test_every_selection_metric_is_invariant_to_the_free_set(tmp_path, scope):
    """Nothing that can steer a decision may move when a free observation changes."""
    z = 0.05 * jnp.ones((6, 6, 6), dtype=jnp.float32)
    base_metrics = _functions(tmp_path, (1.0, 2.0, 3.0, 4.0, 5.0), scope)[0][5]
    mutated_metrics = _functions(tmp_path, (1.0, 2.0, 3.0, 4.0, 999.0), scope)[0][5]
    base = {k: float(v) for k, v in base_metrics(z).items()}
    mutated = {k: float(v) for k, v in mutated_metrics(z).items()}
    assert set(base) == set(mutated)
    assert not any(key.endswith("_free") for key in base), "work metrics must not expose free quantities"
    for key in base:
        assert base[key] == pytest.approx(mutated[key], rel=1e-6, abs=1e-9), key


def test_free_metrics_do_see_the_free_set(tmp_path):
    """The mirror image: the one-shot report must actually read the free reflections."""
    z = jnp.zeros((6, 6, 6), dtype=jnp.float32)
    base = _functions(tmp_path, (1.0, 2.0, 3.0, 4.0, 5.0), "work")[0][6]
    mutated = _functions(tmp_path, (1.0, 2.0, 3.0, 4.0, 999.0), "work")[0][6]
    assert float(base(z)["r_free"]) != pytest.approx(float(mutated(z)["r_free"]))


def test_nuisance_scale_is_fitted_without_the_free_set(tmp_path):
    """The profiled global scale is calibrated on the fit scope only."""
    z = jnp.zeros((6, 6, 6), dtype=jnp.float32)
    cfg = problem_config(tmp_path, optimizer={"fit_scope": "work"}, likelihood={"global_scale": "profile"})
    base = build_functions(problem_arrays(observation=(1.0, 2.0, 3.0, 4.0, 5.0), split=SPLIT), cfg)[6]
    mutated = build_functions(problem_arrays(observation=(1.0, 2.0, 3.0, 4.0, 999.0), split=SPLIT), cfg)[6]
    assert float(base(z)["scale_from_work"]) == pytest.approx(float(mutated(z)["scale_from_work"]))


def test_hash_split_is_deterministic_and_seed_dependent():
    key = _canonical_hash_key((3, 1, 4), True)
    assert _u01_for_key(key, 10) == _u01_for_key(key, 10)
    assert _u01_for_key(key, 10) != _u01_for_key(key, 11)


def test_friedel_mates_share_a_split():
    assert _canonical_hash_key((1, 2, -3), True) == _canonical_hash_key((-1, -2, 3), True)
    assert _canonical_hash_key((1, 2, -3), False) != _canonical_hash_key((-1, -2, 3), False)


def test_prepare_produces_the_same_split_every_time(tiny_dataset):
    from crystal_field.crystallography.io import prepare_reflections

    cfg = tiny_dataset["cfg"]
    first = prepare_reflections(cfg)
    split_a = np.load(cfg.run.data_dir / "reflections.npz")["split"].copy()
    second = prepare_reflections(cfg)
    split_b = np.load(cfg.run.data_dir / "reflections.npz")["split"]
    assert first == second
    np.testing.assert_array_equal(split_a, split_b)


def test_changing_the_seed_changes_the_split(tiny_dataset):
    from crystal_field.crystallography.io import prepare_reflections

    cfg = tiny_dataset["cfg"]
    prepare_reflections(cfg)
    baseline = np.load(cfg.run.data_dir / "reflections.npz")["split"].copy()
    reseeded = cfg.model_copy(update={"run": cfg.run.model_copy(update={"seed": cfg.run.seed + 1})})
    prepare_reflections(reseeded)
    assert not np.array_equal(baseline, np.load(cfg.run.data_dir / "reflections.npz")["split"])


def test_split_covers_every_reflection_exactly_once(prepared_dataset):
    metadata = prepared_dataset["metadata"]
    split = np.load(prepared_dataset["cfg"].run.data_dir / "reflections.npz")["split"]
    assert set(np.unique(split)) == {TRAIN, TUNE, FREE}
    assert metadata["n_train"] + metadata["n_tune"] + metadata["n_free"] == metadata["n_total_kept"] == len(split)


def test_empty_split_is_rejected(tiny_dataset):
    """A tune fraction that empties a set must fail loudly, not train on nothing."""
    from crystal_field.crystallography.io import prepare_reflections

    cfg = tiny_dataset["cfg"]
    impossible = cfg.model_copy(
        update={
            "input": cfg.input.model_copy(update={"free_test_value": 12345}),
            "split": cfg.split.model_copy(update={"strategy": "existing_free_then_hash"}),
        }
    )
    with pytest.raises(ValueError, match="free fraction"):
        prepare_reflections(impossible)
