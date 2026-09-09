"""Explicit MTZ dataset selection.

An MTZ may carry several datasets -- MAD/SAD wavelengths, a native plus a derivative,
reprocessed versions of the same crystal -- and they frequently reuse column labels.
reciprocalspaceship resolves that silently and inconsistently: it assigns columns by
label in file order, so a duplicated label keeps the LAST occurrence, while the
wavelength it records is the FIRST dataset's (usually HKL_base's 0.0). The result is a
refinement against an unidentified wavelength.

FieldX therefore resolves the dataset itself and refuses to guess, in the same spirit as
never substituting another PDB entry for a missing one. The chosen dataset and its
wavelength are recorded in the prepared metadata, so a published result names the data
it used.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np

INDEX_COLUMN_TYPE = "H"


def is_mtz(path: str | Path) -> bool:
    return "".join(Path(path).suffixes).lower().endswith((".mtz", ".mtz.gz"))


def _index_columns(mtz):
    return [column for column in mtz.columns if column.type == INDEX_COLUMN_TYPE][:3]


def describe_mtz(path: str | Path) -> dict:
    """Enumerate the datasets in an MTZ and the columns each one owns."""
    mtz = gemmi.read_mtz_file(str(path))
    index_ids = {column.idx for column in _index_columns(mtz)}
    datasets = []
    for dataset in mtz.datasets:
        columns = [
            {"label": column.label, "type": column.type}
            for column in mtz.columns
            if column.dataset_id == dataset.id and column.idx not in index_ids
        ]
        datasets.append(
            {
                "id": int(dataset.id),
                "project_name": dataset.project_name,
                "crystal_name": dataset.crystal_name,
                "dataset_name": dataset.dataset_name,
                "wavelength": float(dataset.wavelength),
                "columns": columns,
                "carries_data": bool(columns),
            }
        )
    labels = [column.label for column in mtz.columns if column.idx not in index_ids]
    return {
        "path": str(path),
        "spacegroup": mtz.spacegroup.xhm() if mtz.spacegroup else None,
        "cell": [mtz.cell.a, mtz.cell.b, mtz.cell.c, mtz.cell.alpha, mtz.cell.beta, mtz.cell.gamma],
        "n_reflections": int(mtz.nreflections),
        "datasets": datasets,
        "data_datasets": [d for d in datasets if d["carries_data"]],
        "duplicate_labels": sorted({label for label in labels if labels.count(label) > 1}),
    }


def _format_datasets(datasets) -> str:
    lines = []
    for dataset in datasets:
        labels = ", ".join(column["label"] for column in dataset["columns"]) or "(none)"
        lines.append(
            f"  id={dataset['id']}  dataset={dataset['dataset_name']!r}  "
            f"crystal={dataset['crystal_name']!r}  wavelength={dataset['wavelength']:.5g}\n"
            f"      columns: {labels}"
        )
    return "\n".join(lines)


def resolve_mtz_dataset(path: str | Path, requested_labels, mtz_dataset=None) -> dict:
    """Decide which MTZ dataset supplies the configured columns.

    `mtz_dataset` may be a dataset id or a dataset name. When it is unset the choice is
    made only if it is forced: either a single data-bearing dataset exists, or every
    requested label occurs exactly once and in the same dataset. Anything else raises.
    """
    described = describe_mtz(path)
    candidates = described["data_datasets"]
    requested = [label for label in requested_labels if label]

    if not candidates:
        raise ValueError(f"{path} contains no data columns outside the reflection indices")

    if mtz_dataset is not None:
        if isinstance(mtz_dataset, bool):
            raise TypeError("input.mtz_dataset must be a dataset id or name, not a bool")
        matches = [
            dataset
            for dataset in described["datasets"]
            if (dataset["id"] == mtz_dataset)
            or (isinstance(mtz_dataset, str) and dataset["dataset_name"] == mtz_dataset)
        ]
        if not matches:
            raise ValueError(
                f"input.mtz_dataset={mtz_dataset!r} matches no dataset in {path}. Available:\n"
                + _format_datasets(described["datasets"])
            )
        if len(matches) > 1:
            raise ValueError(
                f"input.mtz_dataset={mtz_dataset!r} is ambiguous in {path}; use the numeric id. Matching:\n"
                + _format_datasets(matches)
            )
        chosen = matches[0]
    elif len(candidates) == 1:
        chosen = candidates[0]
    else:
        # Several data-bearing datasets. Only proceed if the configured labels themselves
        # single one out; otherwise the choice would be a guess about which wavelength.
        owners = {}
        for label in requested:
            holding = [d for d in candidates if any(c["label"] == label for c in d["columns"])]
            if len(holding) != 1:
                raise ValueError(
                    f"{path} has {len(candidates)} datasets and column {label!r} occurs in "
                    f"{len(holding)}. Set input.mtz_dataset to the id or name of the one to use.\n"
                    + _format_datasets(candidates)
                )
            owners[label] = holding[0]["id"]
        distinct = set(owners.values())
        if len(distinct) != 1:
            raise ValueError(
                f"{path}: the configured columns come from different datasets {sorted(distinct)}. "
                "Set input.mtz_dataset, or use columns from a single dataset.\n" + _format_datasets(candidates)
            )
        chosen = next(d for d in candidates if d["id"] == distinct.pop())

    available = {column["label"] for column in chosen["columns"]}
    missing = [label for label in requested if label not in available]
    if missing:
        raise ValueError(
            f"{path}: dataset id={chosen['id']} ({chosen['dataset_name']!r}) does not contain "
            f"{missing}. It provides: {sorted(available)}"
        )
    return {
        "dataset_id": chosen["id"],
        "dataset_name": chosen["dataset_name"],
        "crystal_name": chosen["crystal_name"],
        "wavelength": chosen["wavelength"],
        "n_datasets_with_data": len(candidates),
        "selected_explicitly": mtz_dataset is not None,
        "available_datasets": [{k: d[k] for k in ("id", "dataset_name", "wavelength")} for d in described["datasets"]],
    }


def read_mtz_dataset(path: str | Path, requested_labels, mtz_dataset=None):
    """Read exactly one MTZ dataset into a reciprocalspaceship DataSet."""
    import reciprocalspaceship as rs

    selection = resolve_mtz_dataset(path, requested_labels, mtz_dataset)
    source = gemmi.read_mtz_file(str(path))
    origin = next(d for d in source.datasets if d.id == selection["dataset_id"])
    index_columns = _index_columns(source)
    data_columns = [
        column
        for column in source.columns
        if column.dataset_id == selection["dataset_id"] and column.idx not in {c.idx for c in index_columns}
    ]

    single = gemmi.Mtz(with_base=True)
    single.spacegroup = source.spacegroup
    single.title = source.title
    single.set_cell_for_all(source.cell)
    # reciprocalspaceship compares the wavelengths of ALL datasets including HKL_base and
    # keeps the first; leaving base at 0.0 makes it record 0.0 for real data.
    single.datasets[0].wavelength = origin.wavelength
    target = single.add_dataset(origin.dataset_name)
    target.crystal_name = origin.crystal_name
    target.project_name = origin.project_name
    target.wavelength = origin.wavelength
    for column in data_columns:
        single.add_column(column.label, column.type, dataset_id=target.id)
    single.set_data(
        np.column_stack([column.array for column in index_columns] + [column.array for column in data_columns])
    )
    return rs.io.from_gemmi(single), selection
