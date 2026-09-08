from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from pathlib import Path

import gemmi
import numpy as np


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as fh:
            fh.write(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"RCSB download failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"RCSB download failed: {url}: {exc.reason}") from exc


def _decompress_gzip(path: Path, output: Path) -> Path:
    with gzip.open(path, "rb") as src, output.open("wb") as dst:
        dst.write(src.read())
    path.unlink()
    return output


def download_pdb_entry(pdb_id: str, root: str | Path) -> dict:
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError("PDB ID must be a four-character alphanumeric identifier")

    root = Path(root)
    data_dir = root / pdb_id
    data_dir.mkdir(parents=True, exist_ok=True)
    model = data_dir / f"{pdb_id}.cif"
    sf = data_dir / f"{pdb_id}-sf.cif"

    if not model.exists():
        _download(f"https://files.rcsb.org/download/{pdb_id}.cif", model)

    if not sf.exists():
        tmp = data_dir / f"{pdb_id}-sf.cif.gz"
        try:
            _download(f"https://files.rcsb.org/download/{pdb_id}-sf.cif.gz", tmp)
            _decompress_gzip(tmp, sf)
        except RuntimeError as exc:
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(
                f"PDB {pdb_id.upper()} has no downloadable deposited structure-factor file. "
                "Provide reflections=... in a custom config or choose another entry."
            ) from exc

    st = gemmi.read_structure(str(model))
    cell = st.cell
    result = {
        "pdb_id": pdb_id.upper(),
        "model": str(model),
        "reflections": str(sf),
        "spacegroup": st.spacegroup_hm or None,
        "cell": [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
    }
    (data_dir / "entry.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def discover_reflection_columns(reflection_path: str | Path) -> dict:
    import reciprocalspaceship as rs

    ds = rs.read_cif(str(reflection_path))
    columns = {str(column).upper(): str(column) for column in ds.columns}

    observation = next((columns[name] for name in ("FP", "F", "FOBS", "FMEAN") if name in columns), None)
    sigma = next((columns[name] for name in ("SIGFP", "SIGF", "SIGFOBS", "SIGFMEAN") if name in columns), None)
    free = next((columns[name] for name in ("FREE", "FREER_FLAG", "FREERFLAG", "RFREE", "R-FREE-FLAGS") if name in columns), None)
    if observation is None or sigma is None:
        raise ValueError(
            f"Could not infer amplitude/sigma columns from {list(ds.columns)}. "
            "Create a config manually with input.columns.observation/sigma."
        )

    ds.compute_dHKL(inplace=True)
    free_value = None
    if free is not None:
        values, counts = np.unique(np.asarray(ds[free]), return_counts=True)
        fractions = counts / counts.sum()
        candidates = [(abs(float(frac) - 0.05), int(value), float(frac)) for value, frac in zip(values, fractions, strict=False)]
        candidates = [item for item in candidates if 0.02 <= item[2] <= 0.15]
        if candidates:
            candidates.sort()
            free_value = candidates[0][1]

    return {
        "observation": observation,
        "sigma": sigma,
        "free": free,
        "free_test_value": free_value,
        "d_min": float(np.nanmin(np.asarray(ds["dHKL"]))),
        "d_max": float(np.nanmax(np.asarray(ds["dHKL"]))),
        "spacegroup": ds.spacegroup.xhm(),
        "cell": [ds.cell.a, ds.cell.b, ds.cell.c, ds.cell.alpha, ds.cell.beta, ds.cell.gamma],
        "n_reflections": len(ds),
    }
