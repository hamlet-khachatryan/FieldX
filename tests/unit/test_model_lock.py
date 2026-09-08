
import numpy as np
import pytest

from crystal_field.analysis.model_selection import freeze_model, verify_lock
from crystal_field.config import dump_config, load_config


def test_model_lock_rejects_changed_config(tmp_path):
    cfg = load_config("configs/6o2h/default.yaml")
    cfg = cfg.model_copy(
        update={
            "run": cfg.run.model_copy(update={"output_dir": tmp_path / "final", "shared_data_dir": tmp_path / "shared"}),
            "optimizer": cfg.optimizer.model_copy(update={"fit_scope": "work"}),
        }
    )
    config_path = tmp_path / "selected.yaml"
    dump_config(cfg, config_path)

    cfg.run.data_dir.mkdir(parents=True)
    cfg.run.output_dir.joinpath("fit").mkdir(parents=True)
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

    freeze_model(config_path)
    verify_lock(config_path)

    config_path.write_text(config_path.read_text() + "\n# changed\n")
    with pytest.raises(RuntimeError, match="Configuration changed"):
        verify_lock(config_path)
