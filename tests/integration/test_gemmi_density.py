from pathlib import Path

import gemmi
import numpy as np
import pytest

from crystal_field.config import AppConfig
from crystal_field.crystallography.io import make_model_density, make_solvent_mask

pytestmark = pytest.mark.integration


def _write_pdb(path: Path):
    path.write_text(
        "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  CA  ALA A   1       5.000   5.000   5.000  1.00 10.00           C\n"
        "ATOM      2  O   ALA A   1       6.200   5.000   5.000  1.00 12.00           O\n"
        "END\n"
    )


def test_gemmi_atomic_density_and_solvent_mask(tmp_path):
    pytest.importorskip("reciprocalspaceship")
    model = tmp_path / "tiny.pdb"
    _write_pdb(model)
    data_dir = tmp_path / "shared"
    data_dir.mkdir()
    metadata = {
        "spacegroup": "P 1",
        "cell": [20, 20, 20, 90, 90, 90],
        "grid_shape": [32, 32, 32],
        "unit_cell_volume": 8000.0,
    }
    import json
    (data_dir / "metadata.json").write_text(json.dumps(metadata))
    cfg = AppConfig.model_validate({
        "run": {"output_dir": str(tmp_path / "run"), "shared_data_dir": str(data_dir)},
        "input": {
            "reflections": str(tmp_path / "dummy.mtz"), "model": str(model),
            "columns": {"observation": "FP", "sigma": "SIGFP"},
        },
        "split": {"strategy": "hash"},
        "resolution": {"d_min_angstrom": 2.0},
    })
    make_model_density(cfg)
    mask_stats = make_solvent_mask(cfg)
    rho = np.load(data_dir / "rho0.npy")
    mask = np.load(data_dir / "solvent_mask.npy")
    assert rho.shape == (32, 32, 32)
    assert float(rho.max()) > 0
    assert mask.shape == rho.shape
    assert 0 < mask_stats["solvent_fraction"] < 1
    assert gemmi.__version__
