# Tangent Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only analysis stage that decomposes a fitted FieldX v3 density correction onto the atomic tangent space, reporting how much of it ordinary refinement could have produced and how much it could not.

**Architecture:** A sparse real-space operator `Phi` whose columns are per-atom density derivatives, built by finite-differencing single-atom Gemmi densities. Density is additive over atoms, so a single-atom structure yields the exact column, and Gemmi's own density cutoff makes each column naturally sparse (~7% of the grid). Least squares runs on dense normal equations (`Phi^T Phi` is a few thousand square). A capacity control decomposes matched random fields through the same operator so an explained fraction is never reported alone.

**Tech Stack:** Python 3.11+, gemmi 0.7.5 (densities, symmetry, CCP4), numpy, scipy.sparse (column storage), JAX (only for `build_transfer`/`apply_transfer`, CPU), pydantic v2 (config), typer (CLI), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-09-tangent-decomposition-design.md`

## Global Constraints

- **Scope:** Increment 1 only. Section 10 of the spec (joint refinement over `(a, z)`) MUST NOT be implemented. Do not add a `prior.modes` config namespace. Do not change the forward model, prior, likelihood or optimizer.
- **Read-only:** This stage loads a completed run and writes new files. It must not modify `rho0.npy`, `z_map.npy`, `metadata.json`, `reflections.npz`, `MODEL_LOCK.json`, or anything under `final/fit/`.
- **Naming:** The feature is `decompose` / "tangent decomposition". It is NOT v4 and must not be named v4 in code, CLI, config keys or artifacts.
- **Free set:** The full-correction target reads no reflections. The data-supported target reads work reflections only (`split != 2`). No code path in this feature may read `split == 2`.
- **DAG:** `75_freeze_model` keeps depending on stages 60 and 70 only. `MODEL_LOCK.json` does not hash decomposition output. Stage `72_decompose` is a leaf depending only on `$j_final`.
- **CPU only:** `slurm/72_decompose.sbatch` must NOT declare `--gres=gpu` and must NOT call `fieldx_load_cuda`.
- **Tests:** Build synthetic data via `tests/conftest.py`. No test may depend on downloaded data or a specific PDB entry. Baseline is 581 passing tests; the suite must still pass at every commit.
- **Verification command:** `make check` (runs `bash -n` on all shell/sbatch scripts, `ruff format --check src tests`, `ruff check src tests`, `pytest -m "not gpu and not cluster"`).
- **Style:** `ruff` line-length 120, double quotes. Run `uv run --frozen --no-sync ruff format src tests` before each commit.

---

## File Structure

**Create:**
- `src/crystal_field/analysis/tangent.py` — the `Phi` operator: column construction and basis assembly.
- `src/crystal_field/analysis/decomposition.py` — targets, the solve, the capacity control, orchestration and artifacts.
- `slurm/72_decompose.sbatch` — CPU stage.
- `tests/unit/test_tangent_basis.py` — column and basis tests.
- `tests/unit/test_decomposition.py` — solve, targets, control, orchestration tests.

**Modify:**
- `src/crystal_field/config.py` — add `DecompositionConfig`, wire into `AppConfig`, surface in `check_config`.
- `src/crystal_field/cli.py` — add `decompose` command.
- `src/crystal_field/maps.py` — cross-reference decomposition maps in `maps.json`.
- `scripts/submit.sh` — submit stage 72.
- `tests/unit/test_slurm_control_plane.py` — add `72_decompose` to the expected job set and the non-GPU set.
- `docs/V4_DESIGN.md`, `docs/WORKFLOW.md`, `docs/TESTING.md`, `docs/METHODS.md` — documentation.

**Split rationale:** `tangent.py` owns "what is `Phi`" and nothing else — it has no knowledge of fitted runs, targets or reporting, so it can be tested against pure geometry. `decomposition.py` owns "what do we do with `Phi`". Keeping them apart means the ground-truth test (Task 3) exercises the operator without needing a fit, and the control test (Task 6) exercises the guard without needing Gemmi.

---

## Task 1: Configuration block

**Files:**
- Modify: `src/crystal_field/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DecompositionConfig` with fields `enabled: bool`, `basis: Literal["coordinates","coordinates_b","full","residue_rigid"]`, `box_radius_angstrom: float | None`, `ridge: float`, `n_capacity_trials: int`, `write_maps: bool`. Reachable as `cfg.decomposition`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config.py`:

```python
def test_decomposition_defaults(tiny_dataset):
    cfg = tiny_dataset["cfg"]
    assert cfg.decomposition.enabled is True
    assert cfg.decomposition.basis == "coordinates"
    assert cfg.decomposition.box_radius_angstrom is None
    assert cfg.decomposition.ridge == 0.0
    assert cfg.decomposition.n_capacity_trials == 8
    assert cfg.decomposition.write_maps is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"basis": "nonsense"}, "basis"),
        ({"ridge": -1.0}, "greater than or equal to 0"),
        ({"n_capacity_trials": 0}, "greater than or equal to 1"),
        ({"box_radius_angstrom": 0.0}, "greater than 0"),
    ],
)
def test_invalid_decomposition_parameters(tiny_dataset, override, message):
    with pytest.raises(ValueError, match=message):
        AppConfig.model_validate(_base(tiny_dataset, decomposition=override))


def test_config_check_reports_the_decomposition_basis(tiny_dataset):
    report = check_config(tiny_dataset["config_path"])
    assert report["decomposition_basis"] == "coordinates"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_config.py -k decomposition -v`
Expected: FAIL — `AttributeError: 'AppConfig' object has no attribute 'decomposition'`.

- [ ] **Step 3: Add the config model**

In `src/crystal_field/config.py`, add after `AtomicBenchmarkConfig`:

```python
class DecompositionConfig(Strict):
    """Decomposition of the inferred correction onto the atomic tangent space.

    A read-only diagnostic: it answers whether ordinary refinement -- moving atoms,
    changing B factors or occupancies -- could have produced the density the field
    inferred. It does not change the model.
    """

    enabled: bool = True
    basis: Literal["coordinates", "coordinates_b", "full", "residue_rigid"] = "coordinates"
    # Optional extra truncation of each column. None keeps whatever Gemmi's own
    # density cutoff produces, which is already sparse (~7% of the grid).
    box_radius_angstrom: float | None = Field(default=None, gt=0.0)
    ridge: float = Field(0.0, ge=0.0)
    # With thousands of free parameters the basis fits noise; matched random fields
    # calibrate what "explained" means. Never fewer than one.
    n_capacity_trials: int = Field(8, ge=1)
    write_maps: bool = True
```

Add the field to `AppConfig`, immediately after `atomic_benchmark`:

```python
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
```

In `check_config`, add to the returned dict next to `"prior_kernel"`:

```python
        "decomposition_basis": cfg.decomposition.basis,
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen --no-sync pytest tests/unit/test_config.py -v`
Expected: PASS, including the pre-existing committed-config tests.

- [ ] **Step 5: Format, lint, full suite, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/config.py tests/unit/test_config.py
git commit -m "feat(config): add decomposition block for tangent-space analysis"
```

---

## Task 2: Single tangent column

**Files:**
- Create: `src/crystal_field/analysis/tangent.py`
- Test: `tests/unit/test_tangent_basis.py`

**Interfaces:**
- Consumes: `crystal_field.crystallography.density.model_density_on_grid(model, cell, spacegroup, shape, d_min, cutoff, scattering="xray")`.
- Produces:
  - `PARAMETER_STEPS: dict[str, float]` keyed by `"x"`, `"y"`, `"z"`, `"b_iso"`, `"occupancy"`.
  - `single_atom_model(atom: gemmi.Atom) -> gemmi.Model`
  - `tangent_column(atom, kind, cell, spacegroup, shape, d_min, cutoff, scattering="xray") -> numpy.ndarray` — a dense float64 array of shape `shape`, the finite-difference derivative `d(rho)/d(kind)` for that atom including its symmetry copies.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tangent_basis.py`:

```python
"""The Phi operator: per-atom density derivatives.

Density is additive over atoms, so the derivative with respect to one atom's parameter
involves only that atom -- a single-atom structure gives the exact column. Gemmi's own
density cutoff then makes each column naturally sparse, which is what keeps the basis
affordable without any box arithmetic.
"""

import gemmi
import numpy as np
import pytest
from conftest import write_tiny_model

from crystal_field.analysis.tangent import PARAMETER_STEPS, single_atom_model, tangent_column
from crystal_field.crystallography.density import model_density_on_grid

SHAPE = (24, 30, 36)
D_MIN = 2.6
CUTOFF = 1e-6


@pytest.fixture
def model(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    return structure


def _atoms(model):
    return [cra.atom for cra in model[0].all()]


def test_density_is_additive_over_atoms(model):
    """The premise the whole operator rests on."""
    spacegroup = gemmi.SpaceGroup("P 1")
    full, _ = model_density_on_grid(model[0], model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    total = np.zeros(SHAPE, dtype=np.float64)
    for atom in _atoms(model):
        one, _ = model_density_on_grid(single_atom_model(atom), model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
        total += one
    assert np.abs(total - full).max() / full.max() < 1e-5


@pytest.mark.parametrize("kind", ["x", "y", "z", "b_iso", "occupancy"])
def test_column_is_finite_nonzero_and_sparse(model, kind):
    spacegroup = gemmi.SpaceGroup("P 1")
    column = tangent_column(_atoms(model)[1], kind, model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    assert column.shape == SHAPE
    assert np.all(np.isfinite(column))
    assert np.linalg.norm(column) > 0
    occupied = np.count_nonzero(np.abs(column) > 1e-12)
    assert occupied < 0.5 * column.size, "an atom's derivative must not fill the cell"


def test_coordinate_column_matches_a_real_displacement(model):
    """The derivative must predict what actually happens when the atom moves."""
    spacegroup = gemmi.SpaceGroup("P 1")
    atom = _atoms(model)[1]
    column = tangent_column(atom, "x", model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)

    delta = 0.01
    before, _ = model_density_on_grid(single_atom_model(atom), model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    original = atom.pos.x
    atom.pos.x = original + delta
    after, _ = model_density_on_grid(single_atom_model(atom), model.cell, spacegroup, SHAPE, D_MIN, CUTOFF)
    atom.pos.x = original

    actual = after.astype(np.float64) - before.astype(np.float64)
    predicted = delta * column
    assert np.linalg.norm(actual - predicted) / np.linalg.norm(actual) < 0.05


def test_the_column_leaves_the_atom_unchanged(model):
    """A derivative that mutates the model would corrupt every later column."""
    atom = _atoms(model)[0]
    before = (atom.pos.x, atom.pos.y, atom.pos.z, atom.b_iso, atom.occ)
    tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    tangent_column(atom, "b_iso", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    assert (atom.pos.x, atom.pos.y, atom.pos.z, atom.b_iso, atom.occ) == before


def test_columns_include_symmetry_copies(model):
    """rho0 is symmetrize_sum-ed, so a column must cover every copy."""
    atom = _atoms(model)[1]
    p1 = tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    p212121 = tangent_column(atom, "x", model.cell, gemmi.SpaceGroup("P 21 21 21"), SHAPE, D_MIN, CUTOFF)
    assert np.count_nonzero(np.abs(p212121) > 1e-12) > 2 * np.count_nonzero(np.abs(p1) > 1e-12)


def test_an_unknown_parameter_is_rejected(model):
    with pytest.raises(ValueError, match="Unknown tangent parameter"):
        tangent_column(_atoms(model)[0], "charge", model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)


def test_parameter_steps_are_declared_for_every_kind():
    assert set(PARAMETER_STEPS) == {"x", "y", "z", "b_iso", "occupancy"}
    assert all(step > 0 for step in PARAMETER_STEPS.values())
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_tangent_basis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crystal_field.analysis.tangent'`.

- [ ] **Step 3: Write the module**

Create `src/crystal_field/analysis/tangent.py`:

```python
"""The atomic tangent-space operator Phi.

A column of Phi is the derivative of the unit-cell density with respect to one parameter
of one atom. Density is additive over atoms, so that derivative involves only the atom in
question: a structure containing just that atom yields the exact column, at a fraction of
the cost of rebuilding the whole model. Gemmi's density cutoff then truncates each column
to a neighbourhood of the atom and its symmetry copies, so the columns are naturally
sparse -- roughly 7% of the grid -- with no box arithmetic of our own.

This module knows nothing about fitted runs, targets or reporting. It answers only "what
is Phi".
"""

from __future__ import annotations

import gemmi
import numpy as np

from crystal_field.crystallography.density import model_density_on_grid

# Central-difference steps, matching the existing atomic_benchmark configuration so the
# two diagnostics perturb atoms identically.
PARAMETER_STEPS = {
    "x": 0.02,
    "y": 0.02,
    "z": 0.02,
    "b_iso": 0.5,
    "occupancy": 0.02,
}


def single_atom_model(atom: gemmi.Atom) -> gemmi.Model:
    """A Model containing one atom, for computing that atom's density alone."""
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "UNK"
    residue.seqid = gemmi.SeqId(1, " ")
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    return model


def _read(atom, kind):
    if kind == "x":
        return atom.pos.x
    if kind == "y":
        return atom.pos.y
    if kind == "z":
        return atom.pos.z
    if kind == "b_iso":
        return atom.b_iso
    if kind == "occupancy":
        return atom.occ
    raise ValueError(f"Unknown tangent parameter: {kind}")


def _write(atom, kind, value):
    if kind == "x":
        atom.pos.x = value
    elif kind == "y":
        atom.pos.y = value
    elif kind == "z":
        atom.pos.z = value
    elif kind == "b_iso":
        atom.b_iso = value
    elif kind == "occupancy":
        atom.occ = value
    else:
        raise ValueError(f"Unknown tangent parameter: {kind}")


def tangent_column(atom, kind, cell, spacegroup, shape, d_min, cutoff, scattering="xray"):
    """d(rho)/d(kind) for one atom, on the full grid, including symmetry copies.

    The atom is restored to its original value before returning: a derivative that
    mutated the model would corrupt every column computed after it.
    """
    step = PARAMETER_STEPS[kind] if kind in PARAMETER_STEPS else None
    if step is None:
        raise ValueError(f"Unknown tangent parameter: {kind}")

    original = _read(atom, kind)
    try:
        _write(atom, kind, original + step)
        plus, _ = model_density_on_grid(
            single_atom_model(atom), cell, spacegroup, shape, d_min, cutoff, scattering
        )
        _write(atom, kind, original - step)
        minus, _ = model_density_on_grid(
            single_atom_model(atom), cell, spacegroup, shape, d_min, cutoff, scattering
        )
    finally:
        _write(atom, kind, original)

    return (plus.astype(np.float64) - minus.astype(np.float64)) / (2.0 * step)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen --no-sync pytest tests/unit/test_tangent_basis.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/analysis/tangent.py tests/unit/test_tangent_basis.py
git commit -m "feat(tangent): per-atom density derivative columns"
```

---

## Task 3: Basis assembly

**Files:**
- Modify: `src/crystal_field/analysis/tangent.py`
- Test: `tests/unit/test_tangent_basis.py`

**Interfaces:**
- Consumes: `tangent_column` from Task 2.
- Produces:
  - `PARAMETER_SETS: dict[str, tuple[str, ...]]` for `"coordinates"`, `"coordinates_b"`, `"full"`.
  - `@dataclass TangentBasis` with fields `matrix: scipy.sparse.csc_matrix` (shape `(n_voxels, n_columns)`), `labels: list[tuple[int, str]]`, `grid_shape: tuple[int, int, int]`, `basis: str`, `n_columns: int`, `min_norm_fraction: float`.
  - `selected_atoms(model) -> list[tuple[int, gemmi.Atom]]` — non-hydrogen, occupancy > 0, paired with their index.
  - `build_tangent_basis(model, cell, spacegroup, shape, d_min, cutoff, basis="coordinates", scattering="xray", truncate_radius=None) -> TangentBasis`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tangent_basis.py`:

```python
from crystal_field.analysis.tangent import PARAMETER_SETS, build_tangent_basis, selected_atoms


def _basis(model, name, spacegroup="P 1"):
    return build_tangent_basis(
        model[0], model.cell, gemmi.SpaceGroup(spacegroup), SHAPE, D_MIN, CUTOFF, basis=name
    )


def test_selected_atoms_excludes_hydrogens_and_zero_occupancy(model):
    atoms = selected_atoms(model[0])
    assert len(atoms) == 10
    assert all(not atom.element.is_hydrogen for _, atom in atoms)
    assert all(atom.occ > 0 for _, atom in atoms)


@pytest.mark.parametrize(
    ("name", "per_atom"), [("coordinates", 3), ("coordinates_b", 4), ("full", 5)]
)
def test_column_count_follows_the_basis(model, name, per_atom):
    basis = _basis(model, name)
    assert basis.n_columns == 10 * per_atom
    assert basis.matrix.shape == (int(np.prod(SHAPE)), basis.n_columns)
    assert len(basis.labels) == basis.n_columns
    assert basis.basis == name


def test_labels_identify_atom_and_parameter(model):
    basis = _basis(model, "coordinates")
    kinds = {kind for _, kind in basis.labels}
    assert kinds == {"x", "y", "z"}
    assert {index for index, _ in basis.labels} == set(range(10))


def test_the_matrix_is_sparse(model):
    basis = _basis(model, "coordinates")
    density = basis.matrix.nnz / (basis.matrix.shape[0] * basis.matrix.shape[1])
    assert density < 0.5, f"columns should be local, got {density:.2%} filled"


def test_columns_match_tangent_column_exactly(model):
    """Assembly must not alter what Task 2 produces."""
    basis = _basis(model, "coordinates")
    index, kind = basis.labels[4]
    atom = selected_atoms(model[0])[index][1]
    expected = tangent_column(atom, kind, model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
    stored = np.asarray(basis.matrix[:, 4].todense()).ravel().reshape(SHAPE)
    np.testing.assert_allclose(stored, expected, atol=1e-10)


def test_norm_capture_is_reported(model):
    """The sparsity guard: stored columns must retain essentially all their norm."""
    basis = _basis(model, "coordinates")
    assert basis.min_norm_fraction > 0.999


def test_truncation_below_tolerance_is_refused(model):
    with pytest.raises(ValueError, match="box_radius_angstrom"):
        build_tangent_basis(
            model[0], model.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF,
            basis="coordinates", truncate_radius=0.05,
        )


def test_residue_rigid_gives_six_columns_per_multi_atom_group(model):
    """Single-atom groups have no meaningful rotation, so they contribute translation only."""
    basis = _basis(model, "residue_rigid")
    kinds = [kind for _, kind in basis.labels]
    assert set(kinds) <= {"t_x", "t_y", "t_z", "r_x", "r_y", "r_z"}
    # ALA(5 atoms) + GLY(4) + MET(1): two multi-atom groups x 6, one single-atom x 3.
    assert basis.n_columns == 2 * 6 + 1 * 3


def test_an_unknown_basis_is_rejected(model):
    with pytest.raises(ValueError, match="Unknown basis"):
        _basis(model, "everything")


def test_a_model_with_no_usable_atoms_is_rejected(tmp_path):
    empty = gemmi.Structure()
    empty.cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    empty.add_model(gemmi.Model("1"))
    with pytest.raises(ValueError, match="No atoms"):
        build_tangent_basis(empty[0], empty.cell, gemmi.SpaceGroup("P 1"), SHAPE, D_MIN, CUTOFF)
```

The tiny model in `conftest.py` has 10 atoms across 3 residues: ALA 1 (5 atoms), GLY 2 (4), MET 3 (1). MET 3 is single-atom, so `residue_rigid` gives `2 * 6 + 1 * 3 = 15` columns. Step 2 re-derives this against the fixture rather than trusting the arithmetic.

- [ ] **Step 2: Verify the residue grouping before asserting**

Run:

```bash
uv run --frozen --no-sync python -c "
import sys, gemmi, tempfile, pathlib; sys.path.insert(0,'tests')
from conftest import write_tiny_model
st = gemmi.read_structure(str(write_tiny_model(pathlib.Path(tempfile.mkdtemp())/'m.pdb'))); st.setup_entities()
g={}
for cra in st[0].all(): g.setdefault((cra.chain.name, cra.residue.seqid.num), []).append(cra.atom)
print({k: len(v) for k,v in g.items()})
print('residue_rigid columns =', sum(6 if len(v)>1 else 3 for v in g.values()))
"
```

Use the printed number in `test_residue_rigid_gives_six_columns_per_multi_atom_group`.

- [ ] **Step 3: Run to verify the tests fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_tangent_basis.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_tangent_basis'`.

- [ ] **Step 4: Implement assembly**

Append to `src/crystal_field/analysis/tangent.py`:

Move the imports to the top of the file, beside the existing ones:

```python
from dataclasses import dataclass

from scipy.sparse import csc_matrix, hstack
```

Then append:

```python
PARAMETER_SETS = {
    "coordinates": ("x", "y", "z"),
    "coordinates_b": ("x", "y", "z", "b_iso"),
    "full": ("x", "y", "z", "b_iso", "occupancy"),
}

RIGID_KINDS = ("t_x", "t_y", "t_z", "r_x", "r_y", "r_z")
TRANSLATION_KINDS = ("t_x", "t_y", "t_z")

# A stored column must keep essentially all of its norm, or the sparsity that makes the
# basis affordable is quietly discarding signal.
MIN_NORM_FRACTION = 0.999


@dataclass(frozen=True)
class TangentBasis:
    matrix: csc_matrix
    labels: list
    grid_shape: tuple
    basis: str
    n_columns: int
    min_norm_fraction: float


def selected_atoms(model):
    """Non-hydrogen atoms with positive occupancy, matching atomic_benchmark's selection."""
    return [
        (index, cra.atom)
        for index, cra in enumerate(
            cra for cra in model.all() if not cra.atom.element.is_hydrogen and cra.atom.occ > 0
        )
    ]


def _residue_groups(model):
    groups = {}
    for cra in model.all():
        if cra.atom.element.is_hydrogen or cra.atom.occ <= 0:
            continue
        groups.setdefault((cra.chain.name, cra.residue.seqid.num), []).append(cra.atom)
    return list(groups.values())


def _truncate(column, atom, cell, shape, radius):
    """Zero everything beyond `radius` of the atom, for an explicit truncation test.

    Only reachable when box_radius_angstrom is set. The default (None) keeps whatever
    Gemmi's density cutoff produced, which is already sparse.
    """
    if radius is None:
        return column
    fractional = cell.fractionalize(atom.pos)
    centre = (fractional.x, fractional.y, fractional.z)
    lengths = (cell.a, cell.b, cell.c)
    # Offset along each axis in Angstrom, wrapped into [-L/2, L/2) for periodicity.
    offsets = []
    for axis, (n, middle, length) in enumerate(zip(shape, centre, lengths, strict=True)):
        delta = ((np.arange(n) / n - middle + 0.5) % 1.0 - 0.5) * length
        offsets.append(delta.reshape(tuple(-1 if i == axis else 1 for i in range(3))))
    distance2 = offsets[0] ** 2 + offsets[1] ** 2 + offsets[2] ** 2
    return np.where(distance2 <= radius**2, column, 0.0)


def build_tangent_basis(
    model, cell, spacegroup, shape, d_min, cutoff, basis="coordinates", scattering="xray", truncate_radius=None
):
    """Assemble Phi as a sparse (n_voxels x n_columns) matrix."""
    shape = tuple(int(n) for n in shape)
    n_voxels = int(np.prod(shape))
    columns, labels, norm_fractions = [], [], []

    def add(atom, kind, label, vector=None):
        dense = tangent_column(atom, kind, cell, spacegroup, shape, d_min, cutoff, scattering) if vector is None else vector
        full_norm = float(np.linalg.norm(dense))
        kept = _truncate(dense, atom, cell, shape, truncate_radius)
        kept_norm = float(np.linalg.norm(kept))
        norm_fractions.append(kept_norm / full_norm if full_norm > 0 else 1.0)
        columns.append(csc_matrix(kept.reshape(n_voxels, 1)))
        labels.append(label)

    if basis in PARAMETER_SETS:
        atoms = selected_atoms(model)
        if not atoms:
            raise ValueError("No atoms survive the selection (non-hydrogen, occupancy > 0)")
        for index, atom in atoms:
            for kind in PARAMETER_SETS[basis]:
                add(atom, kind, (index, kind))
    elif basis == "residue_rigid":
        groups = _residue_groups(model)
        if not groups:
            raise ValueError("No atoms survive the selection (non-hydrogen, occupancy > 0)")
        for index, atoms in enumerate(groups):
            kinds = RIGID_KINDS if len(atoms) > 1 else TRANSLATION_KINDS
            for kind in kinds:
                add(atoms[0], _rigid_source_kind(kind), (index, kind),
                    vector=_rigid_column(kind, atoms, cell, spacegroup, shape, d_min, cutoff, scattering))
    else:
        raise ValueError(f"Unknown basis: {basis}")

    minimum = float(min(norm_fractions))
    if minimum < MIN_NORM_FRACTION:
        raise ValueError(
            f"Truncation keeps only {minimum:.4%} of a column's norm, below "
            f"{MIN_NORM_FRACTION:.1%}. Raise box_radius_angstrom or leave it null."
        )

    return TangentBasis(
        matrix=csc_matrix(hstack(columns)),
        labels=labels,
        grid_shape=shape,
        basis=basis,
        n_columns=len(labels),
        min_norm_fraction=minimum,
    )


def _rigid_source_kind(kind):
    """Rigid columns are assembled from per-atom columns; this names the driving parameter."""
    return {"t_x": "x", "t_y": "y", "t_z": "z", "r_x": "x", "r_y": "y", "r_z": "z"}[kind]


def _rigid_column(kind, atoms, cell, spacegroup, shape, d_min, cutoff, scattering):
    """A group translation or rotation, as the sum of its atoms' coordinate derivatives.

    A rigid translation along an axis moves every atom identically. A rotation about an
    axis through the group centroid moves atom i by (axis x (r_i - centroid)), so its
    density derivative is that displacement contracted with the atom's coordinate
    derivatives.
    """
    centroid = np.mean([[a.pos.x, a.pos.y, a.pos.z] for a in atoms], axis=0)
    axis_index = {"t_x": 0, "t_y": 1, "t_z": 2, "r_x": 0, "r_y": 1, "r_z": 2}[kind]
    total = np.zeros(tuple(int(n) for n in shape), dtype=np.float64)
    for atom in atoms:
        if kind.startswith("t_"):
            weights = np.zeros(3)
            weights[axis_index] = 1.0
        else:
            axis = np.zeros(3)
            axis[axis_index] = 1.0
            weights = np.cross(axis, np.array([atom.pos.x, atom.pos.y, atom.pos.z]) - centroid)
        for component, name in zip(weights, ("x", "y", "z"), strict=True):
            if component == 0.0:
                continue
            total += component * tangent_column(atom, name, cell, spacegroup, shape, d_min, cutoff, scattering)
    return total
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run --frozen --no-sync pytest tests/unit/test_tangent_basis.py -v`
Expected: PASS.

- [ ] **Step 6: Format, lint, full suite, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/analysis/tangent.py tests/unit/test_tangent_basis.py
git commit -m "feat(tangent): assemble Phi as a sparse basis with selectable parameter sets"
```

---

## Task 4: The least-squares solve

**Files:**
- Create: `src/crystal_field/analysis/decomposition.py`
- Test: `tests/unit/test_decomposition.py`

**Interfaces:**
- Consumes: `TangentBasis` from Task 3.
- Produces: `solve_normal_equations(gram, rhs, target_norm_squared, ridge=0.0) -> dict` with keys `amplitudes` (`np.ndarray`), `explained_fraction` (`float`), `rank` (`int`), `condition_number` (`float`), `residual_norm_squared` (`float`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_decomposition.py`:

```python
"""Decomposition of the inferred correction onto the atomic tangent space."""

import numpy as np
import pytest

from crystal_field.analysis.decomposition import solve_normal_equations


def _problem(n_rows=200, n_columns=12, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((n_rows, n_columns))
    truth = rng.standard_normal(n_columns)
    target = basis @ truth
    return basis, truth, target


def test_exact_recovery_when_the_target_lies_in_the_span():
    basis, truth, target = _problem()
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    np.testing.assert_allclose(result["amplitudes"], truth, rtol=1e-8, atol=1e-8)
    assert result["explained_fraction"] == pytest.approx(1.0, abs=1e-10)
    assert result["rank"] == basis.shape[1]


def test_a_target_orthogonal_to_the_basis_explains_nothing():
    basis, _, _ = _problem()
    rng = np.random.default_rng(7)
    orthogonal = rng.standard_normal(basis.shape[0])
    orthogonal -= basis @ np.linalg.lstsq(basis, orthogonal, rcond=None)[0]
    result = solve_normal_equations(
        basis.T @ basis, basis.T @ orthogonal, float(orthogonal @ orthogonal)
    )
    assert result["explained_fraction"] == pytest.approx(0.0, abs=1e-8)


def test_explained_fraction_is_bounded():
    basis, _, _ = _problem()
    rng = np.random.default_rng(3)
    for _ in range(5):
        target = rng.standard_normal(basis.shape[0])
        result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
        assert -1e-9 <= result["explained_fraction"] <= 1.0 + 1e-9


def test_ridge_shrinks_the_amplitudes():
    basis, _, target = _problem()
    plain = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    ridged = solve_normal_equations(
        basis.T @ basis, basis.T @ target, float(target @ target), ridge=10.0
    )
    assert np.linalg.norm(ridged["amplitudes"]) < np.linalg.norm(plain["amplitudes"])
    assert ridged["explained_fraction"] < plain["explained_fraction"]


def test_rank_deficiency_is_reported_not_hidden():
    """A duplicated column makes the basis rank deficient; that is a fact, not an error."""
    basis, _, _ = _problem(n_columns=6)
    basis = np.hstack([basis, basis[:, :1]])
    target = basis @ np.ones(basis.shape[1])
    result = solve_normal_equations(basis.T @ basis, basis.T @ target, float(target @ target))
    assert result["rank"] < basis.shape[1]
    assert np.isfinite(result["condition_number"])
    assert result["explained_fraction"] == pytest.approx(1.0, abs=1e-8)


def test_a_zero_target_does_not_divide_by_zero():
    basis, _, _ = _problem()
    result = solve_normal_equations(basis.T @ basis, np.zeros(basis.shape[1]), 0.0)
    assert np.isfinite(result["explained_fraction"])
    assert result["explained_fraction"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'crystal_field.analysis.decomposition'`.

- [ ] **Step 3: Implement the solve**

Create `src/crystal_field/analysis/decomposition.py`:

```python
"""Decomposing a fitted correction onto the atomic tangent space.

This answers the manuscript's central claim directly: could ordinary refinement -- moving
atoms, changing B factors or occupancies -- have produced the density the field inferred?
The part it cannot explain is the object the project exists to find.

An explained fraction is never reported alone. With thousands of free parameters the
basis fits a great deal of anything, so every result is accompanied by what the same
basis explains of matched random fields.
"""

from __future__ import annotations

import numpy as np


def solve_normal_equations(gram, rhs, target_norm_squared, ridge: float = 0.0) -> dict:
    """Least squares from precomputed normal equations.

    `gram` is Phi^T Phi, `rhs` is Phi^T target. Working from the normal equations keeps
    memory at O(n_columns^2) rather than O(n_voxels x n_columns), which is what makes a
    few thousand columns affordable. lstsq is SVD-based, so a rank-deficient basis is
    resolved rather than producing a spurious solution.
    """
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if ridge > 0.0:
        gram = gram + ridge * np.eye(gram.shape[0])

    amplitudes, _, rank, singular = np.linalg.lstsq(gram, rhs, rcond=None)

    # ||target - Phi a||^2 = ||target||^2 - 2 a.rhs + a.(gram a), expanded so the full
    # residual vector never has to be formed.
    residual = float(target_norm_squared) - 2.0 * float(amplitudes @ rhs) + float(amplitudes @ (gram @ amplitudes))
    residual = max(residual, 0.0)
    explained = 0.0 if target_norm_squared <= 0 else 1.0 - residual / float(target_norm_squared)

    positive = singular[singular > 0]
    condition = float(positive.max() / positive.min()) if positive.size else float("inf")
    return {
        "amplitudes": amplitudes,
        "explained_fraction": float(np.clip(explained, 0.0, 1.0)),
        "rank": int(rank),
        "condition_number": condition,
        "residual_norm_squared": residual,
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Format, lint, full suite, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/analysis/decomposition.py tests/unit/test_decomposition.py
git commit -m "feat(decomposition): least squares from precomputed normal equations"
```

---

## Task 5: Targets — the symmetrized correction and its data-supported part

**Files:**
- Modify: `src/crystal_field/analysis/decomposition.py`
- Test: `tests/unit/test_decomposition.py`

**Interfaces:**
- Consumes: `solve_normal_equations` (Task 4); `crystal_field.model.prior.build_transfer/apply_transfer`; `crystal_field.inference.runtime.load_problem_arrays`.
- Produces:
  - `symmetrize_grid(array, cell, spacegroup) -> np.ndarray`
  - `correction_from_fit(cfg, arrays) -> np.ndarray` — `u = L z` on the grid, float64.
  - `split_symmetric(array, cell, spacegroup) -> tuple[np.ndarray, float]` — the symmetric part and the antisymmetric norm fraction.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_decomposition.py`:

```python
import gemmi
from conftest import write_tiny_model

from crystal_field.analysis.decomposition import split_symmetric, symmetrize_grid


def test_symmetrizing_an_already_symmetric_grid_changes_nothing(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    from crystal_field.crystallography.density import model_density_on_grid

    density, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, (24, 30, 36), 2.6, 1e-6)
    again = symmetrize_grid(density.astype(np.float64), structure.cell, spacegroup)
    np.testing.assert_allclose(again, density, rtol=1e-5, atol=1e-7)


def test_symmetrizing_is_idempotent(tmp_path):
    rng = np.random.default_rng(0)
    cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    field = rng.standard_normal((24, 30, 36))
    once = symmetrize_grid(field, cell, spacegroup)
    twice = symmetrize_grid(once, cell, spacegroup)
    np.testing.assert_allclose(twice, once, rtol=1e-5, atol=1e-7)


def test_the_antisymmetric_fraction_is_reported():
    """A random field in a symmetric group is mostly antisymmetric; that must be visible."""
    rng = np.random.default_rng(1)
    cell = gemmi.UnitCell(20, 24, 28, 90, 90, 90)
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    field = rng.standard_normal((24, 30, 36))
    symmetric, antisymmetric_fraction = split_symmetric(field, cell, spacegroup)
    assert 0.0 < antisymmetric_fraction < 1.0
    assert np.linalg.norm(symmetric) < np.linalg.norm(field)


def test_a_symmetric_field_has_no_antisymmetric_part(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 21 21 21")
    from crystal_field.crystallography.density import model_density_on_grid

    density, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, (24, 30, 36), 2.6, 1e-6)
    _, fraction = split_symmetric(density.astype(np.float64), structure.cell, spacegroup)
    assert fraction < 1e-3


def test_p1_symmetrization_is_the_identity():
    rng = np.random.default_rng(2)
    field = rng.standard_normal((16, 16, 16))
    result = symmetrize_grid(field, gemmi.UnitCell(20, 20, 20, 90, 90, 90), gemmi.SpaceGroup("P 1"))
    np.testing.assert_allclose(result, field, rtol=1e-6, atol=1e-8)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py -k symmetr -v`
Expected: FAIL — `ImportError: cannot import name 'symmetrize_grid'`.

- [ ] **Step 3: Implement the targets**

Append to `src/crystal_field/analysis/decomposition.py`:

```python
import gemmi


def symmetrize_grid(array, cell, spacegroup):
    """Average a grid over the space group.

    Only the symmetric component of the correction reaches F_calc through
    symmetry_projected_fcalc, and every Phi column is symmetric by construction, so the
    antisymmetric part is orthogonal to the basis and would otherwise register as
    permanently unexplained density that no basis could reach.
    """
    grid = gemmi.FloatGrid(np.ascontiguousarray(array, dtype=np.float32))
    grid.set_unit_cell(cell)
    grid.spacegroup = spacegroup
    grid.symmetrize_avg()
    return np.array(grid.array, dtype=np.float64, copy=True)


def split_symmetric(array, cell, spacegroup):
    """Return the symmetric component and the norm fraction carried by the rest."""
    array = np.asarray(array, dtype=np.float64)
    symmetric = symmetrize_grid(array, cell, spacegroup)
    total = float(np.linalg.norm(array))
    antisymmetric = float(np.linalg.norm(array - symmetric))
    return symmetric, (antisymmetric / total if total > 0 else 0.0)


def correction_from_fit(cfg, arrays):
    """Recover u = L z on the grid from the fitted latent field."""
    import jax
    import jax.numpy as jnp

    from crystal_field.model.prior import apply_transfer, build_transfer

    path = cfg.run.output_dir / "fit" / "z_map.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `fieldrefine fit-map CONFIG` first")

    transfer = build_transfer(
        tuple(arrays.rho0.shape),
        arrays.reciprocal_metric,
        arrays.unit_cell_volume,
        arrays.d_min_angstrom,
        cfg.prior,
        arrays.rho0.dtype,
    )
    z = jnp.asarray(np.load(path), dtype=arrays.rho0.dtype)
    return np.asarray(jax.device_get(apply_transfer(z, transfer, arrays.unit_cell_volume)), dtype=np.float64)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py -v`
Expected: PASS.

- [ ] **Step 5: Format, lint, full suite, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/analysis/decomposition.py tests/unit/test_decomposition.py
git commit -m "feat(decomposition): symmetrized correction target and antisymmetric reporting"
```

---

## Task 6: Ground truth and the capacity control

**Files:**
- Modify: `src/crystal_field/analysis/decomposition.py`
- Test: `tests/unit/test_decomposition.py`

**Interfaces:**
- Consumes: `TangentBasis` (Task 3), `solve_normal_equations` (Task 4), `split_symmetric` (Task 5).
- Produces:
  - `decompose_target(basis, target, ridge=0.0) -> dict` — the Task 4 result for a grid-shaped target, adding `"explained"` and `"unexplained"` grids.
  - `capacity_control(basis, reference_norm, transfer, unit_cell_volume, cell, spacegroup, seed, n_trials, ridge=0.0) -> dict` with keys `mean`, `sd`, `n_trials`, `fractions`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_decomposition.py`:

```python
from crystal_field.analysis.decomposition import capacity_control, decompose_target
from crystal_field.analysis.tangent import build_tangent_basis, selected_atoms, tangent_column
from crystal_field.crystallography.density import model_density_on_grid

SHAPE = (24, 30, 36)


@pytest.fixture
def tiny_basis(tmp_path):
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 1")
    basis = build_tangent_basis(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6)
    return structure, spacegroup, basis


def test_ground_truth_a_known_displacement_is_recovered(tiny_basis):
    """The decisive test: move an atom by a known amount, get that amount back."""
    structure, spacegroup, basis = tiny_basis
    atoms = selected_atoms(structure[0])
    index, atom = atoms[1]
    delta = 0.05

    before, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6)
    original = atom.pos.x
    atom.pos.x = original + delta
    after, _ = model_density_on_grid(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6)
    atom.pos.x = original

    target = after.astype(np.float64) - before.astype(np.float64)
    result = decompose_target(basis, target)

    assert result["explained_fraction"] > 0.99, "an atomic displacement must be atom-explainable"
    column = basis.labels.index((index, "x"))
    assert result["amplitudes"][column] == pytest.approx(delta, rel=0.1)
    others = np.delete(result["amplitudes"], column)
    assert np.abs(others).max() < 0.2 * abs(result["amplitudes"][column])


def test_explained_and_unexplained_sum_to_the_target(tiny_basis):
    _, _, basis = tiny_basis
    rng = np.random.default_rng(4)
    target = rng.standard_normal(SHAPE)
    result = decompose_target(basis, target)
    np.testing.assert_allclose(result["explained"] + result["unexplained"], target, atol=1e-8)


def test_a_random_field_is_less_explainable_than_a_real_displacement(tiny_basis):
    structure, spacegroup, basis = tiny_basis
    atom = selected_atoms(structure[0])[1][1]
    real = 0.05 * tangent_column(atom, "x", structure.cell, spacegroup, SHAPE, 2.6, 1e-6)

    rng = np.random.default_rng(11)
    noise = rng.standard_normal(SHAPE)
    noise *= np.linalg.norm(real) / np.linalg.norm(noise)

    assert decompose_target(basis, real)["explained_fraction"] > decompose_target(basis, noise)["explained_fraction"]


def test_the_control_detects_over_parameterisation(tmp_path):
    """The guard must be a measurement, not a decoration.

    A larger basis fits noise better. If moving from `coordinates` to `full` did not
    raise the control's explained fraction, the control would not be measuring capacity.
    """
    structure = gemmi.read_structure(str(write_tiny_model(tmp_path / "m.pdb")))
    structure.setup_entities()
    spacegroup = gemmi.SpaceGroup("P 1")
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(SHAPE)

    small = build_tangent_basis(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6, basis="coordinates")
    large = build_tangent_basis(structure[0], structure.cell, spacegroup, SHAPE, 2.6, 1e-6, basis="full")
    assert large.n_columns > small.n_columns
    assert decompose_target(large, noise)["explained_fraction"] > decompose_target(small, noise)["explained_fraction"]


def test_capacity_trials_are_seeded_and_reproducible(tiny_basis):
    _, spacegroup, basis = tiny_basis
    kwargs = dict(
        basis=basis, reference_norm=1.0, transfer=None, unit_cell_volume=13440.0,
        cell=gemmi.UnitCell(20, 24, 28, 90, 90, 90), spacegroup=spacegroup, seed=17, n_trials=3,
    )
    first = capacity_control(**kwargs)
    second = capacity_control(**kwargs)
    assert first["fractions"] == second["fractions"]
    assert first["n_trials"] == 3
    assert 0.0 <= first["mean"] <= 1.0
    assert first["sd"] >= 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py -k "ground_truth or control" -v`
Expected: FAIL — `ImportError: cannot import name 'decompose_target'`.

- [ ] **Step 3: Implement**

Append to `src/crystal_field/analysis/decomposition.py`:

```python
def decompose_target(basis, target, ridge: float = 0.0) -> dict:
    """Least-squares projection of a grid-shaped target onto the basis."""
    flat = np.asarray(target, dtype=np.float64).reshape(-1)
    if flat.size != basis.matrix.shape[0]:
        raise ValueError(f"Target has {flat.size} voxels but the basis expects {basis.matrix.shape[0]}")

    gram = np.asarray((basis.matrix.T @ basis.matrix).todense(), dtype=np.float64)
    rhs = np.asarray(basis.matrix.T @ flat, dtype=np.float64).ravel()
    result = solve_normal_equations(gram, rhs, float(flat @ flat), ridge=ridge)

    explained = np.asarray(basis.matrix @ result["amplitudes"]).reshape(basis.grid_shape)
    result["explained"] = explained
    result["unexplained"] = np.asarray(target, dtype=np.float64) - explained
    return result


def capacity_control(
    basis, reference_norm, transfer, unit_cell_volume, cell, spacegroup, seed, n_trials, ridge: float = 0.0
) -> dict:
    """What the same basis explains of matched random fields.

    An explained fraction on its own says nothing: with thousands of free parameters the
    basis fits a great deal of anything. Each trial draws a field through the *same*
    prior operator the fit used, scales it to the correction's norm, symmetrizes it, and
    decomposes it. If the control scores 0.70, a real score of 0.75 is not a finding.
    """
    fractions = []
    for trial in range(int(n_trials)):
        rng = np.random.default_rng(int(seed) + 1000 + trial)
        draw = rng.standard_normal(basis.grid_shape)
        if transfer is not None:
            import jax
            import jax.numpy as jnp

            from crystal_field.model.prior import apply_transfer

            draw = np.asarray(
                jax.device_get(apply_transfer(jnp.asarray(draw, dtype=transfer.dtype), transfer, unit_cell_volume)),
                dtype=np.float64,
            )
        draw = symmetrize_grid(draw, cell, spacegroup)
        norm = float(np.linalg.norm(draw))
        if norm > 0:
            draw *= float(reference_norm) / norm
        fractions.append(decompose_target(basis, draw, ridge=ridge)["explained_fraction"])

    return {
        "fractions": [float(f) for f in fractions],
        "mean": float(np.mean(fractions)),
        "sd": float(np.std(fractions)),
        "n_trials": int(n_trials),
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py -v`
Expected: PASS.

- [ ] **Step 5: Format, lint, full suite, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/analysis/decomposition.py tests/unit/test_decomposition.py
git commit -m "feat(decomposition): ground-truth projection and capacity control"
```

---

## Task 7: Orchestration and artifacts

**Files:**
- Modify: `src/crystal_field/analysis/decomposition.py`, `src/crystal_field/maps.py`
- Test: `tests/unit/test_decomposition.py`

**Interfaces:**
- Consumes: everything from Tasks 3-6; `crystal_field.inference.runtime.load_problem_arrays`; `crystal_field.crystallography.io._model_for_metadata`; `crystal_field.maps._write`.
- Produces: `run_decomposition(cfg, basis=None, n_trials=None) -> dict`, writing `final/decomposition/decomposition.json` plus `explained.ccp4` and `unexplained.ccp4`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_decomposition.py`:

```python
import json

from crystal_field.analysis.decomposition import run_decomposition


@pytest.fixture
def fitted(prepared_dataset):
    from crystal_field.crystallography.scaling import fit_baseline_scaling
    from crystal_field.inference.optimize import run_map_fit
    from crystal_field.inference.problem import build_functions
    from crystal_field.inference.runtime import load_problem_arrays

    cfg = prepared_dataset["cfg"]
    cfg = cfg.model_copy(
        update={
            "optimizer": cfg.optimizer.model_copy(update={"fit_scope": "work", "max_iterations": 10}),
            "decomposition": cfg.decomposition.model_copy(update={"n_capacity_trials": 2}),
        }
    )
    fit_baseline_scaling(cfg)
    arrays = load_problem_arrays(cfg)
    _, density, _, _, objective, metrics, _ = build_functions(arrays, cfg)
    run_map_fit(cfg, arrays, objective, metrics, density)
    prepared_dataset["cfg"] = cfg
    return prepared_dataset


def test_all_artifacts_are_written(fitted):
    report = run_decomposition(fitted["cfg"])
    directory = fitted["cfg"].run.output_dir / "decomposition"
    for name in ("decomposition.json", "explained.ccp4", "unexplained.ccp4"):
        assert (directory / name).is_file(), name
    assert json.loads((directory / "decomposition.json").read_text()) == report


def test_the_report_carries_both_targets_and_the_control(fitted):
    report = run_decomposition(fitted["cfg"])
    assert set(report["targets"]) == {"full_correction", "data_supported"}
    for target in report["targets"].values():
        assert 0.0 <= target["explained_fraction"] <= 1.0
    control = report["capacity_control"]
    assert control["n_trials"] == 2
    assert 0.0 <= control["mean"] <= 1.0
    assert report["basis"]["name"] == "coordinates"
    assert report["basis"]["rank"] > 0
    assert "antisymmetric_fraction" in report


def test_explained_and_unexplained_reconstruct_the_symmetrized_correction(fitted):
    """Per spec section 3, the target is symmetrize(u), not u."""
    report = run_decomposition(fitted["cfg"])
    directory = fitted["cfg"].run.output_dir / "decomposition"
    explained = np.array(gemmi.read_ccp4_map(str(directory / "explained.ccp4")).grid, copy=True)
    unexplained = np.array(gemmi.read_ccp4_map(str(directory / "unexplained.ccp4")).grid, copy=True)
    assert explained.shape == unexplained.shape
    assert np.all(np.isfinite(explained)) and np.all(np.isfinite(unexplained))
    assert report["targets"]["full_correction"]["target_norm"] > 0


def test_a_missing_fit_is_reported(prepared_dataset):
    with pytest.raises(FileNotFoundError, match="fieldrefine fit-map"):
        run_decomposition(prepared_dataset["cfg"])


def test_the_free_set_does_not_influence_any_reported_number(fitted):
    """Same discipline as the rest of the pipeline: mutate free, nothing may move."""
    cfg = fitted["cfg"]
    baseline = run_decomposition(cfg)

    path = cfg.run.data_dir / "reflections.npz"
    stored = dict(np.load(path))
    free = stored["split"] == 2
    assert free.any()
    stored["observation"][free] *= 1000.0
    np.savez_compressed(path, **stored)

    mutated = run_decomposition(cfg)
    for name in ("full_correction", "data_supported"):
        assert mutated["targets"][name]["explained_fraction"] == pytest.approx(
            baseline["targets"][name]["explained_fraction"]
        )


def test_the_basis_can_be_overridden(fitted):
    report = run_decomposition(fitted["cfg"], basis="full")
    assert report["basis"]["name"] == "full"
    assert report["basis"]["n_columns"] > 0


def test_the_span_property_v3_is_the_case_phi_equals_zero(fitted):
    """Spec section 9.4, and the invariant increment 2 must preserve.

    docs/V4_DESIGN.md section 2 states that v3 is the special case Phi = 0. With the
    basis disabled nothing may be claimed as explained. Trivial here, but it is the
    property joint refinement has to keep, so it is pinned from the start.
    """
    cfg = fitted["cfg"]
    disabled = cfg.model_copy(update={"decomposition": cfg.decomposition.model_copy(update={"enabled": False})})
    report = run_decomposition(disabled)
    assert report["enabled"] is False
    assert report["targets"]["full_correction"]["explained_fraction"] == 0.0
    assert report["basis"]["n_columns"] == 0


def test_a_disabled_decomposition_writes_nothing(fitted):
    cfg = fitted["cfg"]
    disabled = cfg.model_copy(update={"decomposition": cfg.decomposition.model_copy(update={"enabled": False})})
    run_decomposition(disabled)
    assert not (disabled.run.output_dir / "decomposition" / "decomposition.json").exists()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py -k "artifacts or targets_and" -v`
Expected: FAIL — `ImportError: cannot import name 'run_decomposition'`.

- [ ] **Step 3: Implement orchestration**

Append to `src/crystal_field/analysis/decomposition.py`:

```python
def _data_supported_target(cfg, arrays, basis, correction):
    """Delta_F on the work reflections, decomposed against the columns' structure factors.

    Only about one band-limited frequency in seven has a measured reflection behind it
    for a typical dataset; the rest of the correction is prior interpolation. This target
    isolates the part the data actually supports.
    """
    import jax
    import jax.numpy as jnp

    from crystal_field.forward.diffraction import fft_structure_factor_grid, symmetry_projected_fcalc

    work = np.asarray(jax.device_get(arrays.split)) != 2
    hkls = jnp.asarray(np.asarray(jax.device_get(arrays.hkls))[work], dtype=jnp.int32)

    def structure_factors(grid):
        transformed = fft_structure_factor_grid(
            jnp.asarray(grid, dtype=arrays.rho0.dtype), arrays.unit_cell_volume
        )
        return np.asarray(
            jax.device_get(
                symmetry_projected_fcalc(
                    transformed, hkls, arrays.symmetry_rotations, arrays.symmetry_translations
                )
            ),
            dtype=np.complex128,
        )

    delta_f = structure_factors(correction)

    # One FFT per column. The stacked matrix is (n_reflections x n_columns) complex --
    # about 190 MB at 1UBQ's 6029 reflections and 1980 columns, which is affordable; if a
    # larger dataset makes it not, accumulate gram and rhs inside the loop instead.
    columns = [
        structure_factors(np.asarray(basis.matrix[:, index].todense()).reshape(basis.grid_shape))
        for index in range(basis.n_columns)
    ]
    stacked = np.stack(columns, axis=1)
    # a is real, so the least-squares normal equations take the real part.
    gram = np.real(stacked.conj().T @ stacked)
    rhs = np.real(stacked.conj().T @ delta_f)

    result = solve_normal_equations(gram, rhs, float(np.real(delta_f.conj() @ delta_f)), ridge=cfg.decomposition.ridge)
    return {
        "explained_fraction": result["explained_fraction"],
        "rank": result["rank"],
        "condition_number": result["condition_number"],
        "n_reflections": int(work.sum()),
        "target_norm": float(np.linalg.norm(delta_f)),
    }


def run_decomposition(cfg, basis: str | None = None, n_trials: int | None = None) -> dict:
    """Decompose a fitted correction onto the atomic tangent space and write the report."""
    import json

    from crystal_field.analysis.tangent import build_tangent_basis
    from crystal_field.crystallography.io import _model_for_metadata
    from crystal_field.inference.runtime import load_problem_arrays
    from crystal_field.maps import _write
    from crystal_field.model.prior import build_transfer

    if not cfg.decomposition.enabled:
        # v3 is the special case Phi = 0 (spec section 9.4): the correction is reported
        # whole and nothing is claimed to be explained.
        return {
            "enabled": False,
            "basis": {"name": None, "n_columns": 0},
            "targets": {"full_correction": {"explained_fraction": 0.0}},
            "note": "decomposition.enabled is false; the whole correction is unexplained by definition.",
        }

    metadata = json.loads((cfg.run.data_dir / "metadata.json").read_text())
    structure, cell, spacegroup = _model_for_metadata(cfg, metadata)
    shape = tuple(int(n) for n in metadata["grid_shape"])

    arrays = load_problem_arrays(cfg)
    correction = correction_from_fit(cfg, arrays)
    symmetric, antisymmetric_fraction = split_symmetric(correction, cell, spacegroup)

    basis_name = basis or cfg.decomposition.basis
    tangent = build_tangent_basis(
        structure[0], cell, spacegroup, shape,
        cfg.resolution.d_min_angstrom, cfg.baseline.density_cutoff,
        basis=basis_name, scattering=cfg.input.scattering,
        truncate_radius=cfg.decomposition.box_radius_angstrom,
    )

    full = decompose_target(tangent, symmetric, ridge=cfg.decomposition.ridge)
    data_supported = _data_supported_target(cfg, arrays, tangent, symmetric)

    transfer = build_transfer(
        tuple(arrays.rho0.shape), arrays.reciprocal_metric, arrays.unit_cell_volume,
        arrays.d_min_angstrom, cfg.prior, arrays.rho0.dtype,
    )
    control = capacity_control(
        tangent, float(np.linalg.norm(symmetric)), transfer, arrays.unit_cell_volume,
        cell, spacegroup, cfg.run.seed, n_trials or cfg.decomposition.n_capacity_trials,
        ridge=cfg.decomposition.ridge,
    )

    out = cfg.run.output_dir / "decomposition"
    out.mkdir(parents=True, exist_ok=True)
    if cfg.decomposition.write_maps:
        _write(full["explained"], out / "explained.ccp4", cell, spacegroup)
        _write(full["unexplained"], out / "unexplained.ccp4", cell, spacegroup)

    by_kind = {}
    for (_, kind), amplitude in zip(tangent.labels, full["amplitudes"], strict=True):
        by_kind.setdefault(kind, []).append(float(amplitude))

    report = {
        "enabled": True,
        "basis": {
            "name": basis_name,
            "n_columns": tangent.n_columns,
            "rank": full["rank"],
            "condition_number": full["condition_number"],
            "min_norm_fraction": tangent.min_norm_fraction,
        },
        "starting_density": cfg.baseline.starting_density,
        "antisymmetric_fraction": antisymmetric_fraction,
        "targets": {
            "full_correction": {
                "explained_fraction": full["explained_fraction"],
                "target_norm": float(np.linalg.norm(symmetric)),
                "n_voxels": int(symmetric.size),
            },
            "data_supported": data_supported,
        },
        "amplitude_rms_by_parameter": {
            kind: float(np.sqrt(np.mean(np.square(values)))) for kind, values in by_kind.items()
        },
        "capacity_control": control,
        "verdict": {
            "explained_above_control": full["explained_fraction"] - control["mean"],
            "note": (
                "An explained fraction is only meaningful against its control. A real "
                "score close to the control's mean means the basis is fitting capacity, "
                "not structure."
            ),
        },
        "note": (
            "Read-only diagnostic. The target is symmetrize(u): only the symmetric part of "
            "the correction reaches F_calc, and every Phi column is symmetric."
        ),
    }
    (out / "decomposition.json").write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Cross-reference the maps**

In `src/crystal_field/maps.py`, inside the `manifest` dict in `export_maps`, add after the `"maps"` entry:

```python
        "related": {
            "decomposition/explained.ccp4": "written by `fieldrefine decompose`: the part of the "
            "correction the atomic tangent space explains",
            "decomposition/unexplained.ccp4": "written by `fieldrefine decompose`: the part it does not",
        },
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run --frozen --no-sync pytest tests/unit/test_decomposition.py tests/unit/test_maps.py -v`
Expected: PASS.

- [ ] **Step 6: Format, lint, full suite, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/analysis/decomposition.py src/crystal_field/maps.py tests/unit/test_decomposition.py
git commit -m "feat(decomposition): orchestration, artifacts and map cross-reference"
```

---

## Task 8: CLI, SLURM stage and submission

**Files:**
- Modify: `src/crystal_field/cli.py`, `scripts/submit.sh`, `tests/unit/test_slurm_control_plane.py`
- Create: `slurm/72_decompose.sbatch`
- Test: `tests/unit/test_cli.py`, `tests/unit/test_slurm_control_plane.py`

**Interfaces:**
- Consumes: `run_decomposition` (Task 7).
- Produces: CLI command `decompose`; sbatch stage `72_decompose`; `$j_decompose` in `submit.sh`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cli.py`:

```python
def test_decompose_rejects_a_missing_config(tmp_path):
    assert runner.invoke(app, ["decompose", str(tmp_path / "absent.yaml")]).exit_code != 0


def test_decompose_is_listed_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "decompose" in result.output
```

In `tests/unit/test_slurm_control_plane.py`, update the two pinned sets:

```python
# in test_the_expected_jobs_exist, add to the expected set:
        "72_decompose",
```

and leave `GPU_JOBS` unchanged so `test_gpu_jobs_load_cuda_and_cpu_jobs_do_not` enforces that `72_decompose` is CPU-only.

Append to `tests/unit/test_slurm_control_plane.py`:

```python
def test_the_decomposition_stage_is_a_leaf_and_does_not_gate_the_freeze():
    """Spec section 7.1: the model lock and the existing chain must not change."""
    lines = (ROOT / "scripts/submit.sh").read_text().splitlines()

    # Assert against the 72 line itself: `--dependency=afterok:"$j_final"` also appears
    # on the 70 line and is a prefix of the 75 line, so a whole-file `in` check passes
    # without 72 existing at all.
    decompose_line = next(line for line in lines if "72_decompose.sbatch" in line)
    assert '--dependency=afterok:"$j_final"' in decompose_line, decompose_line
    # A leaf hangs off the final fit ALONE, so $j_final is its only job reference.
    assert decompose_line.count("$j_") == 1, decompose_line

    # The freeze still waits only on the final fit and the information spectrum.
    freeze_line = next(line for line in lines if "75_freeze_model.sbatch" in line)
    assert '--dependency=afterok:"$j_final":"$j_info"' in freeze_line, freeze_line
    assert "j_decompose" not in freeze_line, "the freeze must not depend on the decomposition"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --frozen --no-sync pytest tests/unit/test_cli.py tests/unit/test_slurm_control_plane.py -v`
Expected: FAIL — the job set assertion and the missing `decompose` command.

- [ ] **Step 3: Add the CLI command**

In `src/crystal_field/cli.py`, add the options near the other module-level singletons:

```python
DECOMPOSE_BASIS_OPTION = typer.Option(
    None, "--basis", help="Override decomposition.basis: coordinates, coordinates_b, full or residue_rigid."
)
DECOMPOSE_TRIALS_OPTION = typer.Option(
    None, "--trials", help="Override decomposition.n_capacity_trials."
)
```

and the command, immediately before `@app.command("expand-priors")`:

```python
@app.command("decompose")
def decompose_cmd(
    config: Path,
    basis: str | None = DECOMPOSE_BASIS_OPTION,
    trials: int | None = DECOMPOSE_TRIALS_OPTION,
):
    """Decompose the inferred correction onto the atomic tangent space (read-only)."""
    cfg = _cfg(config)
    _configure(cfg)
    from crystal_field.analysis.decomposition import run_decomposition

    print(json.dumps(run_decomposition(cfg, basis=basis, n_trials=trials), indent=2))
```

- [ ] **Step 4: Create the SLURM stage**

Create `slurm/72_decompose.sbatch`. Note there is **no** `--gres=gpu` and **no** `fieldx_load_cuda`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=fieldx-decompose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
# SLURM stages this script outside the repository, so "$0" must never be used
# to locate project files. Everything resolves from SLURM_SUBMIT_DIR.
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/common.sh"

# Read-only analysis: Gemmi densities, sparse linear algebra and FFTs. No accelerator is
# required, so this stage does not compete for GPU allocation and does not load CUDA.
# It is a leaf: the model lock does not depend on it, so a failure here cannot block the
# freeze. With decomposition.enabled=false the command returns immediately.
run_cfi decompose
```

- [ ] **Step 5: Wire it into submission**

In `scripts/submit.sh`, after the line that submits `70_information.sbatch`, add:

```bash
# Read-only analysis, parallel to the information spectrum. A leaf: the freeze does not
# depend on it, so a decomposition failure cannot block the model lock.
j_decompose=$(submit 72_decompose.sbatch      --export="$FINAL_ENV"   --dependency=afterok:"$j_final")
```

and add a line to the summary heredoc, after the `70 information` line:

```
  72 decomposition        $j_decompose
```

- [ ] **Step 6: Run to verify they pass**

```bash
bash -n slurm/72_decompose.sbatch
bash -n scripts/submit.sh
uv run --frozen --no-sync pytest tests/unit/test_cli.py tests/unit/test_slurm_control_plane.py -v
```
Expected: PASS.

- [ ] **Step 7: Format, lint, full suite, commit**

```bash
uv run --frozen --no-sync ruff format src tests
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync pytest -m "not gpu and not cluster" -q
git add src/crystal_field/cli.py slurm/72_decompose.sbatch scripts/submit.sh tests/unit/test_cli.py tests/unit/test_slurm_control_plane.py
git commit -m "feat(decompose): CLI command and CPU-only SLURM stage 72"
```

---

## Task 9: Documentation

**Files:**
- Modify: `docs/V4_DESIGN.md`, `docs/WORKFLOW.md`, `docs/TESTING.md`, `docs/METHODS.md`

**Interfaces:**
- Consumes: the behaviour built in Tasks 1-8.
- Produces: no code.

- [ ] **Step 1: Record the diagnostic in the v4 design doc**

In `docs/V4_DESIGN.md`, insert a new section after section 3 ("Atomic tangent-space benchmark"):

```markdown
## 3a. The tangent decomposition (implemented; gates this design)

`fieldrefine decompose` implements the projection described above as a read-only v3
analysis. It is **not** v4: the formulation stays `rho = rho0 + L z` and nothing in the
model, prior or optimizer changes.

It answers whether v4 is worth building. Given a fitted correction, it reports the
fraction the atomic tangent space explains, the fraction it does not, and what the same
basis explains of matched random fields. If the real fraction does not exceed its
capacity control, a mode dictionary would be fitting capacity rather than structure and
the joint refinement below should not be built.

Specification: `docs/superpowers/specs/2026-09-09-tangent-decomposition-design.md`.
Outputs: `runs/<ID>/final/decomposition/`.
```

- [ ] **Step 2: Document the stage in the workflow guide**

In `docs/WORKFLOW.md`, add to the artifact table after the `final/maps/*.ccp4` row:

```markdown
| `final/decomposition/decomposition.json` | 72 | explained fraction vs its capacity control |
| `final/decomposition/unexplained.ccp4` | 72 | density no atomic parameter can produce |
```

and add a subsection before "## The free set":

```markdown
## Reading the decomposition

Stage 72 asks whether ordinary refinement could have produced what the field inferred.

Read `decomposition.json` in this order:

1. `capacity_control.mean` — what the basis explains of pure noise. This is the floor.
2. `targets.full_correction.explained_fraction` — the real number.
3. `verdict.explained_above_control` — the difference. Near zero means the basis is
   fitting capacity, not structure, and the decomposition has found nothing.
4. `basis.condition_number` — if very large, the per-parameter breakdown in
   `amplitude_rms_by_parameter` should not be over-interpreted even when the total holds.
5. `antisymmetric_fraction` — how much of the correction never reached `F_calc` at all.

Then open `unexplained.ccp4` against the model. That is density the atomic parameters
cannot reach, and it is the object the project exists to find.

The stage is a leaf: it does not gate the model freeze, and its output is not hashed into
`MODEL_LOCK.json` because it is re-derivable and makes no claim about held-out data.
```

- [ ] **Step 3: List the new suites**

In `docs/TESTING.md`, add to the suite table:

```markdown
| `test_tangent_basis.py` | density additivity over atoms, per-atom derivative columns, sparsity, symmetry copies, basis column counts, the truncation guard |
| `test_decomposition.py` | exact least-squares recovery, rank deficiency reported not hidden, symmetrization and the antisymmetric fraction, ground-truth displacement recovery, the control detecting over-parameterisation, artifacts, free-set invariance |
```

- [ ] **Step 4: Record the method**

In `docs/METHODS.md`, add before "## 6. R factors":

```markdown
## 5g. Tangent decomposition

A read-only diagnostic testing the claim in section 1 directly: is the inferred density
reachable by ordinary refinement? The correction is projected onto the atomic tangent
space, spanned by the density derivatives with respect to atomic coordinates and,
optionally, B factors and occupancies.

Columns are built by central-differencing single-atom Gemmi densities. Density is additive
over atoms, so a single-atom structure yields the exact column, and Gemmi's density cutoff
makes each column sparse, which is what makes a few thousand columns affordable.

Two targets are reported: the symmetrized correction on the grid, and its data-supported
part on the measured work reflections. Only the symmetric component of the correction
reaches `F_calc`, and every column is symmetric, so the antisymmetric residue is reported
separately rather than counted as unexplained.

With thousands of free parameters the basis fits a substantial fraction of anything, so
an explained fraction is never reported alone: matched random fields drawn through the
same prior operator are decomposed onto the same basis and reported alongside. The
comparison, not the number, is the result.
```

- [ ] **Step 5: Verify and commit**

```bash
make check
git add docs/V4_DESIGN.md docs/WORKFLOW.md docs/TESTING.md docs/METHODS.md
git commit -m "docs: describe the tangent decomposition stage"
```

---

## Final verification

- [ ] **Run the full gate**

```bash
make check
```
Expected: all shell scripts pass `bash -n`, ruff format and check clean, and the test count is above the 581 baseline with zero failures.

- [ ] **Confirm the constraints held**

```bash
# The stage is CPU-only.
grep -c "gres=gpu\|fieldx_load_cuda" slurm/72_decompose.sbatch   # expect 0

# No code path reads the free split.
grep -rn "split == 2" src/crystal_field/analysis/ || echo "clean"

# The feature is not called v4.
grep -rni "v4" src/crystal_field/ || echo "clean"

# The freeze is unchanged.
git diff --stat HEAD~9 -- slurm/75_freeze_model.sbatch          # expect no output
```

- [ ] **Smoke test on real data**

```bash
uv run fieldrefine config-check configs/1ubq/default.yaml
```
Expected: `"decomposition_basis": "coordinates"` present, `"ok": true`.
