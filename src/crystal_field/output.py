import json

import gemmi
import numpy as np


def write_fit_map(cfg):
    try:
        import reciprocalspaceship as rs
    except ImportError as exc:
        raise RuntimeError("reciprocalspaceship is required to write CCP4 output") from exc
    m = json.loads((cfg.run.data_dir / "metadata.json").read_text())
    raw = np.load(cfg.run.output_dir / "fit" / "rho_map_raw.npy")
    cell = gemmi.UnitCell(*m["cell"])
    sg = gemmi.SpaceGroup(m["spacegroup"])
    grid = gemmi.FloatGrid(raw.astype(np.float32, copy=False))
    grid.set_unit_cell(cell)
    grid.spacegroup = sg
    grid.symmetrize_avg()
    rho = np.array(grid.array, dtype=np.float32, copy=True, order="C")
    np.save(cfg.run.output_dir / "fit" / "rho_map.npy", rho)
    rs.io.write_ccp4_map(rho, str(cfg.run.output_dir / "fit" / "rho_map.ccp4"), cell, sg)
