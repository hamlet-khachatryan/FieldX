"""Generic PDB-entry acquisition.

Any four-character PDB entry with deposited experimental Bragg structure factors is
a valid FieldX target. Nothing here is specialised to a particular entry, and an
entry without deposited structure factors is reported as an error rather than
silently substituted by another dataset.
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from pathlib import Path

import gemmi
import numpy as np

RCSB_MODEL_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_SF_URL = "https://files.rcsb.org/download/{pdb_id}-sf.cif.gz"

AMPLITUDE_NAMES = ("FP", "F", "FOBS", "F-OBS", "FMEAN", "F_MEAS_AU", "F_MEAS")
SIGMA_NAMES = ("SIGFP", "SIGF", "SIGFOBS", "SIGF-OBS", "SIGFMEAN", "F_MEAS_SIGMA_AU", "F_MEAS_SIGMA")
FREE_NAMES = ("FREE", "FREER_FLAG", "FREERFLAG", "RFREE", "R-FREE-FLAGS", "STATUS")


class MissingStructureFactors(RuntimeError):
    """The PDB entry has no deposited experimental structure factors."""


def normalize_pdb_id(pdb_id: str) -> str:
    """Return the canonical lowercase four-character identifier."""
    normalized = str(pdb_id).strip().lower()
    if len(normalized) != 4 or not normalized.isalnum() or not normalized[0].isdigit():
        raise ValueError(
            f"{pdb_id!r} is not a PDB identifier: expected four alphanumeric characters starting with a digit"
        )
    return normalized


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as fh:
            fh.write(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"RCSB download failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"RCSB download failed: {url}: {exc.reason}") from exc


def download_pdb_entry(pdb_id: str, data_root: str | Path) -> dict:
    """Fetch coordinates and deposited structure factors into `data_root/<pdbid>/`."""
    pdb_id = normalize_pdb_id(pdb_id)
    data_dir = Path(data_root) / pdb_id
    data_dir.mkdir(parents=True, exist_ok=True)
    model = data_dir / f"{pdb_id}.cif"
    sf = data_dir / f"{pdb_id}-sf.cif"

    if not model.exists():
        _download(RCSB_MODEL_URL.format(pdb_id=pdb_id), model)

    if not sf.exists():
        tmp = data_dir / f"{pdb_id}-sf.cif.gz"
        try:
            _download(RCSB_SF_URL.format(pdb_id=pdb_id), tmp)
            with gzip.open(tmp, "rb") as src, sf.open("wb") as dst:
                dst.write(src.read())
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if sf.exists():
                sf.unlink()
            raise MissingStructureFactors(
                f"PDB {pdb_id.upper()} has no downloadable deposited structure-factor file. "
                "A coordinate file alone is not sufficient for Bragg refinement. Supply your own "
                "MTZ/CIF reflection file and write input.reflections by hand, or choose another entry."
            ) from exc
        tmp.unlink()

    st = gemmi.read_structure(str(model))
    cell = st.cell
    entry = {
        "pdb_id": pdb_id.upper(),
        "model": str(model.resolve()),
        "reflections": str(sf.resolve()),
        "spacegroup": st.spacegroup_hm or None,
        "cell": [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
    }
    (data_dir / "entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return entry


def _pick(columns: dict, names) -> str | None:
    return next((columns[name] for name in names if name in columns), None)


def discover_reflection_columns(reflection_path: str | Path, mtz_dataset=None) -> dict:
    """Detect observation/sigma/free columns, resolution range and free-flag convention.

    For a multi-dataset MTZ the dataset must be settled before columns mean anything: the
    same label can name different wavelengths. Discovery therefore refuses to guess and
    reports the available datasets instead.
    """
    import reciprocalspaceship as rs

    from crystal_field.crystallography.mtz import describe_mtz, is_mtz, read_mtz_dataset

    path = Path(reflection_path)
    selection = None
    if is_mtz(path):
        described = describe_mtz(path)
        candidates = described["data_datasets"]
        if mtz_dataset is None and len(candidates) > 1:
            summary = "; ".join(
                f"id={d['id']} {d['dataset_name']!r} wavelength={d['wavelength']:.5g}" for d in candidates
            )
            raise ValueError(
                f"{path} contains {len(candidates)} data-bearing datasets: {summary}. "
                "Pass mtz_dataset (an id or name) to say which one to configure."
            )
        # No labels are known yet, so resolution rests on the dataset alone.
        ds, selection = read_mtz_dataset(path, [], mtz_dataset)
    else:
        ds = rs.read_cif(str(path))
    columns = {str(column).upper(): str(column) for column in ds.columns}

    observation = _pick(columns, AMPLITUDE_NAMES)
    sigma = _pick(columns, SIGMA_NAMES)
    free = _pick(columns, FREE_NAMES)
    if observation is None or sigma is None:
        raise ValueError(
            f"Could not infer amplitude/sigma columns from {list(map(str, ds.columns))}. "
            "Write input.columns.observation and input.columns.sigma by hand."
        )

    ds.compute_dHKL(inplace=True)
    free_value, free_fraction, free_distribution = None, None, None
    if free is not None:
        values = np.asarray(ds[free])
        if np.issubdtype(values.dtype, np.number):
            unique, counts = np.unique(values, return_counts=True)
            fractions = counts / counts.sum()
            free_distribution = {
                str(int(v)): float(f) for v, f in zip(unique.tolist(), fractions.tolist(), strict=True)
            }
            # A deposited free flag holds out a small minority of reflections. Pick the
            # value whose share is closest to 5% among those in a plausible band; if
            # nothing qualifies (e.g. a constant column) there is no usable free set.
            plausible = [
                (abs(float(f) - 0.05), int(v), float(f))
                for v, f in zip(unique.tolist(), fractions.tolist(), strict=True)
                if 0.02 <= float(f) <= 0.15
            ]
            if plausible:
                plausible.sort()
                _, free_value, free_fraction = plausible[0]

    return {
        "observation": observation,
        "sigma": sigma,
        "free": free,
        "free_test_value": free_value,
        "free_fraction": free_fraction,
        "free_distribution": free_distribution,
        "d_min": float(np.nanmin(np.asarray(ds["dHKL"]))),
        "d_max": float(np.nanmax(np.asarray(ds["dHKL"]))),
        "spacegroup": ds.spacegroup.xhm(),
        "cell": [ds.cell.a, ds.cell.b, ds.cell.c, ds.cell.alpha, ds.cell.beta, ds.cell.gamma],
        "n_reflections": len(ds),
        "merged": bool(ds.merged),
        "mtz_selection": selection,
    }
