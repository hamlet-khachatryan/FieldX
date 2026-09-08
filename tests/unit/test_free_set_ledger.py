"""The free-set evaluation must be physically one-shot, not one-shot by operator discipline.

MODEL_LOCK.json is verified but not consumed, so on its own it does not stop an operator
from evaluating on free, tweaking a hyperparameter, re-freezing and evaluating again. The
ledger closes that path, and it lives beside the prepared reflections so that re-freezing
into a fresh output_dir does not escape it.
"""

import numpy as np
import pytest

from crystal_field.analysis.model_selection import (
    free_set_ledger_path,
    freeze_model,
    read_free_set_ledger,
    record_free_evaluation,
    verify_lock,
)
from crystal_field.config import dump_config, load_config


def _make_frozen_run(tmp_path, shared, name):
    """Materialize the artifacts freeze_model requires and return its config path."""
    cfg = load_config("configs/6o2h/default.yaml")
    cfg = cfg.model_copy(
        update={
            "run": cfg.run.model_copy(update={"output_dir": tmp_path / name, "shared_data_dir": shared}),
            "optimizer": cfg.optimizer.model_copy(update={"fit_scope": "work"}),
        }
    )
    config_path = tmp_path / f"{name}.yaml"
    dump_config(cfg, config_path)

    cfg.run.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.run.output_dir.joinpath("fit").mkdir(parents=True, exist_ok=True)
    cfg.run.data_dir.joinpath("metadata.json").write_text("{}")
    np.savez_compressed(cfg.run.data_dir / "reflections.npz", x=np.asarray([1]))
    np.save(cfg.run.data_dir / "rho0.npy", np.zeros((2, 2, 2), dtype=np.float32))
    np.save(cfg.run.data_dir / "solvent_mask.npy", np.zeros((2, 2, 2), dtype=np.float32))
    np.savez_compressed(
        cfg.run.data_dir / "scaling_work.npz",
        overall_scale=np.ones(1, dtype=np.float32),
        solvent_scale=np.zeros(1, dtype=np.float32),
    )
    np.save(cfg.run.output_dir / "fit" / "z_map.npy", np.zeros((2, 2, 2), dtype=np.float32))
    cfg.run.output_dir.joinpath("fit", "metrics.json").write_text("{}")
    return cfg, config_path


def _evaluate(cfg, lock, name="FREE_EVALUATION.json"):
    result = cfg.run.output_dir / name
    result.write_text('{"delta": {"r_free": -0.004}}')
    return record_free_evaluation(cfg, lock, result, {"r_free": -0.004})


def test_free_set_ledger_blocks_a_second_evaluation(tmp_path):
    shared = tmp_path / "shared"
    cfg, config_path = _make_frozen_run(tmp_path, shared, "final")

    freeze_model(config_path)
    lock = verify_lock(config_path)
    assert lock["free_set_evaluations"] == []
    assert lock["free_set_used_at_freeze"] is False
    assert not free_set_ledger_path(cfg).exists()

    entry = _evaluate(cfg, lock)
    assert entry["free_set_used"] is True
    assert entry["evaluation_index"] == 1
    assert entry["lock_config_sha256"] == lock["config_sha256"]
    assert entry["timestamp_utc"].endswith("+00:00")

    with pytest.raises(RuntimeError, match="one-shot"):
        verify_lock(config_path)

    # An explicit override still works, and still reports the prior read.
    reopened = verify_lock(config_path, allow_consumed=True)
    assert len(reopened["free_set_evaluations"]) == 1


def test_freeze_model_refuses_to_refreeze_a_revised_model_after_free_was_read(tmp_path):
    shared = tmp_path / "shared"
    cfg, config_path = _make_frozen_run(tmp_path, shared, "final")
    freeze_model(config_path)
    _evaluate(cfg, verify_lock(config_path))

    # Same prepared split, a revised model, a fresh output_dir: the loophole the ledger closes.
    revised_cfg, revised_path = _make_frozen_run(tmp_path, shared, "final_revised")
    with pytest.raises(RuntimeError, match="one-shot"):
        freeze_model(revised_path)

    lock = freeze_model(revised_path, allow_after_free_evaluation=True)
    assert lock["config_sha256"]
    assert verify_lock(revised_path, allow_consumed=True)["free_set_used_at_freeze"] is True
    assert revised_cfg.run.output_dir.joinpath("MODEL_LOCK.json").exists()


def test_ledger_is_append_only_and_counts_up(tmp_path):
    shared = tmp_path / "shared"
    cfg, config_path = _make_frozen_run(tmp_path, shared, "final")
    freeze_model(config_path)
    lock = verify_lock(config_path)

    first = _evaluate(cfg, lock)
    second = _evaluate(cfg, verify_lock(config_path, allow_consumed=True), "FREE_EVALUATION_repeat_2.json")

    entries = read_free_set_ledger(cfg)
    assert [e["evaluation_index"] for e in entries] == [1, 2]
    assert entries[0] == first and entries[1] == second
    assert entries[0]["result_file"] != entries[1]["result_file"]
    assert entries[0]["result_sha256"] == entries[1]["result_sha256"]  # same content, distinct files
