"""Model locking and the one-shot free-set ledger (plan sections 17, 18, 20G).

MODEL_LOCK.json is verified but not consumed, so on its own it does not stop an operator
from evaluating on free, tweaking a hyperparameter, re-freezing and evaluating again. The
append-only ledger closes that path, and it lives beside the prepared reflections so that
re-freezing into a fresh output_dir does not escape it.
"""

import json

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


def _frozen_run(tmp_path, shared, name, base_cfg):
    """Materialise every artifact freeze_model requires; return the config and its path."""
    cfg = base_cfg.model_copy(
        update={
            "run": base_cfg.run.model_copy(update={"output_dir": tmp_path / name, "shared_data_dir": shared}),
            "optimizer": base_cfg.optimizer.model_copy(update={"fit_scope": "work"}),
        }
    )
    config_path = tmp_path / f"{name}.yaml"
    dump_config(cfg, config_path)

    cfg.run.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run.output_dir / "fit").mkdir(parents=True, exist_ok=True)
    (cfg.run.data_dir / "metadata.json").write_text("{}")
    np.savez_compressed(cfg.run.data_dir / "reflections.npz", x=np.asarray([1]))
    np.save(cfg.run.data_dir / "rho0.npy", np.zeros((2, 2, 2), dtype=np.float32))
    np.save(cfg.run.data_dir / "solvent_mask.npy", np.zeros((2, 2, 2), dtype=np.float32))
    np.savez_compressed(
        cfg.run.data_dir / "scaling_work.npz",
        overall_scale=np.ones(1, dtype=np.float32),
        solvent_scale=np.zeros(1, dtype=np.float32),
    )
    np.save(cfg.run.output_dir / "fit" / "z_map.npy", np.zeros((2, 2, 2), dtype=np.float32))
    (cfg.run.output_dir / "fit" / "metrics.json").write_text("{}")
    (cfg.run.output_dir / "info_spectrum.json").write_text('{"n_modes": 2}')
    return cfg, config_path


@pytest.fixture
def frozen(tiny_dataset, tmp_path):
    shared = tmp_path / "lockshared"
    return _frozen_run(tmp_path, shared, "final", tiny_dataset["cfg"]), shared, tiny_dataset["cfg"]


def _evaluate(cfg, lock, name="FREE_EVALUATION.json"):
    result = cfg.run.output_dir / name
    result.write_text('{"delta_r_free": -0.004}')
    return record_free_evaluation(cfg, lock, result, {"delta_r_free": -0.004})


def test_only_a_work_scope_config_can_be_frozen(tiny_dataset, tmp_path):
    train_cfg = tiny_dataset["cfg"]
    path = tmp_path / "train.yaml"
    dump_config(train_cfg, path)
    with pytest.raises(ValueError, match="work-refit configuration"):
        freeze_model(path)


def test_freeze_requires_every_artifact(frozen):
    (cfg, config_path), _, _ = frozen
    (cfg.run.output_dir / "info_spectrum.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"info_spectrum\.json"):
        freeze_model(config_path)


def test_lock_rejects_a_changed_config(frozen):
    (_, config_path), _, _ = frozen
    freeze_model(config_path)
    verify_lock(config_path)
    config_path.write_text(config_path.read_text() + "\n# tampered\n")
    with pytest.raises(RuntimeError, match="Configuration changed"):
        verify_lock(config_path)


def test_lock_rejects_a_changed_artifact(frozen):
    (cfg, config_path), _, _ = frozen
    freeze_model(config_path)
    np.save(cfg.run.output_dir / "fit" / "z_map.npy", np.ones((2, 2, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="Frozen artifact changed"):
        verify_lock(config_path)


def test_evaluation_requires_a_lock(frozen):
    (_, config_path), _, _ = frozen
    with pytest.raises(RuntimeError, match=r"MODEL_LOCK\.json is required"):
        verify_lock(config_path)


def test_lock_records_the_artifacts_it_hashed(frozen):
    (cfg, config_path), _, _ = frozen
    freeze_model(config_path)
    lock = json.loads((cfg.run.output_dir / "MODEL_LOCK.json").read_text())
    hashed = {p.rsplit("/", 1)[-1] for p in lock["artifacts"]}
    assert {
        "reflections.npz",
        "rho0.npy",
        "solvent_mask.npy",
        "scaling_work.npz",
        "z_map.npy",
        "info_spectrum.json",
    } <= hashed
    assert lock["free_set_used_at_freeze"] is False


def test_ledger_blocks_a_second_evaluation(frozen):
    (cfg, config_path), _, _ = frozen
    freeze_model(config_path)
    lock = verify_lock(config_path)
    assert lock["free_set_evaluations"] == []
    assert not free_set_ledger_path(cfg).exists()

    entry = _evaluate(cfg, lock)
    assert entry["free_set_used"] is True
    assert entry["evaluation_index"] == 1
    assert entry["lock_config_sha256"] == lock["config_sha256"]
    assert entry["timestamp_utc"].endswith("+00:00")

    with pytest.raises(RuntimeError, match="one-shot"):
        verify_lock(config_path)
    assert len(verify_lock(config_path, allow_consumed=True)["free_set_evaluations"]) == 1


def test_refreezing_a_revised_model_after_a_free_read_is_blocked(frozen, tmp_path):
    (cfg, config_path), shared, base_cfg = frozen
    freeze_model(config_path)
    _evaluate(cfg, verify_lock(config_path))

    # Same prepared split, a revised model, a fresh output_dir: the loophole the ledger closes.
    _, revised_path = _frozen_run(tmp_path, shared, "final_revised", base_cfg)
    with pytest.raises(RuntimeError, match="one-shot"):
        freeze_model(revised_path)

    assert freeze_model(revised_path, allow_after_free_evaluation=True)["config_sha256"]
    assert verify_lock(revised_path, allow_consumed=True)["free_set_used_at_freeze"] is True


def test_ledger_is_append_only_and_counts_up(frozen):
    (cfg, config_path), _, _ = frozen
    freeze_model(config_path)
    first = _evaluate(cfg, verify_lock(config_path))
    second = _evaluate(cfg, verify_lock(config_path, allow_consumed=True), "FREE_EVALUATION_repeat_2.json")

    entries = read_free_set_ledger(cfg)
    assert [entry["evaluation_index"] for entry in entries] == [1, 2]
    assert entries[0] == first and entries[1] == second
    assert entries[0]["result_file"] != entries[1]["result_file"]


def test_a_corrupt_ledger_is_reported(frozen):
    (cfg, _), _, _ = frozen
    free_set_ledger_path(cfg).write_text('{"not": "a list"}')
    with pytest.raises(RuntimeError, match="corrupt"):
        read_free_set_ledger(cfg)


def test_the_ledger_follows_the_prepared_split_not_the_run(frozen):
    (cfg, _), shared, _ = frozen
    assert free_set_ledger_path(cfg).parent == shared
    assert load_config  # keeps the import meaningful for readers of this module
