# FieldX

Direct refinement of experimental crystallographic density, starting from an already
refined atomistic model.

## The v3 question

Conventional refinement represents density through a compact atomistic parameterization
— coordinates, occupancies, displacement parameters. That is efficient, and it is also a
restriction: refinement can only reach densities in the image of that parameterization.

FieldX v3 asks what experimentally supported density remains outside it. The inferred
field is

```
rho(r) = rho0(r) + L_theta z,        z ~ N(0, I)
```

where `rho0` is generated from the refined atomic model with Gemmi, and `L_theta` is a
spatially correlated, band-limited density-field operator.

The correlated prior exists so that

```
coherent multi-voxel density changes   are plausible,
isolated single-voxel excursions       are strongly disfavored.
```

It is deliberately *generic*. v3 does not force density change into atoms, coordinates,
B factors, occupancies or predefined physical modes: a meaningful density feature is
allowed to be unfamiliar and non-atomistic. The refined atomic model is the starting
point, not the hypothesis space.

### Primary criterion

```
R_work(FieldX) < R_work(atomic baseline)
        AND
R_free(FieldX) < R_free(atomic baseline)
```

with `R_free` excluded from optimization, prior selection, hyperparameter selection,
stopping criteria and model selection. `G = R_free - R_work` is tracked as a diagnostic,
never as an optimization target.

v3 uses **Bragg observations only**. Diffuse scattering is a separate, later formulation
and is not part of this likelihood. The v4 design — which keeps a generic correlated
residual field while allowing optional interpretable coherent modes — is in
[`docs/V4_DESIGN.md`](docs/V4_DESIGN.md).

## Official workflow

```
GitHub -> clone -> uv sync -> configure PDB -> validate config
       -> sbatch -> SLURM GPU jobs -> freeze -> manual R_free
```

Nothing else is supported. There is no pip installation path, no generated environment
file, and no environment variable that must be set to find the Python environment.

### 1. Clone and create the environment

```bash
git clone git@github.com:YOUR_ORG/FieldX.git
cd FieldX
uv sync --locked --extra cpu --group dev      # laptop, login node, CI
```

On a GPU cluster use the matching accelerator extra instead of `cpu`:

```bash
uv sync --locked --extra cuda13 --group dev
```

### 2. Check the checkout

```bash
uv run pytest
uv run ruff check .
scripts/cluster_preflight.sh
```

The preflight is a login-node check. It needs no GPU: reporting `JAX devices = CPU` on a
login node is correct, not a failure.

### 3. Configure a dataset

```bash
uv run fieldrefine init-pdb 1UBQ
uv run fieldrefine inspect      configs/1ubq/default.yaml
uv run fieldrefine config-check configs/1ubq/default.yaml
```

Several entries can be initialised at once:

```bash
uv run fieldrefine init-pdb 1UBQ 6O2H 3K0N
```

Identifiers are validated and de-duplicated **before** anything is downloaded, so a typo
in the last argument costs nothing. One entry failing — most often because it deposits no
structure factors — does not stop the others: the batch finishes and every failure is
reported at the end. The command exits non-zero if any entry failed, so it still composes
in a script. `--fail-fast` stops at the first failure instead.

`init-pdb` works for any four-character PDB entry with deposited experimental Bragg
structure factors. It downloads coordinates and structure factors from RCSB, detects the
crystallographic metadata, reflection columns and free-flag convention, and writes both
`configs/<pdbid>/default.yaml` and its dataset-specific `prior_grid.yaml`. An entry with
coordinates but no deposited structure factors is reported as an error — a coordinate
file alone is not sufficient for Bragg refinement, and no other dataset is ever
substituted.

Relative paths inside a configuration resolve against the directory containing it, so a
committed config works unchanged on a laptop and on a cluster.

### 4. Submit

```bash
scripts/submit.sh configs/1ubq/default.yaml
```

Optionally with a different grid:

```bash
scripts/submit.sh configs/1ubq/default.yaml configs/1ubq/prior_grid.yaml
```

Site options, without editing any job file:

```bash
export CFI_SBATCH_ARGS="--account=MYACCOUNT --partition=MYGPU"
export CFI_ARRAY_LIMIT=4
```

Monitor:

```bash
squeue -u "$USER"
sacct  -u "$USER" --starttime today
```

### 5. Evaluate the free set, once, by hand

The pipeline stops at the model freeze. `scripts/submit.sh` prints the exact command for
the free-set evaluation; it is never submitted automatically.

## Diamond Light Source

The repository lives at

```
/dls/data2temp01/hamlet/workspace/FieldX
```

with all data and results under `/dls/data2temp01/hamlet/workspace/`:

```
/dls/data2temp01/hamlet/workspace/
    FieldX/          this repository, including its .venv
    data/            <PDBID>/ coordinate and structure-factor files
    runs/            <PDBID>/ run outputs, logs and prepared reflections
    uv-cache/        uv cache, kept off the small home quota (UV_CACHE_DIR)
    uv-python/       uv-managed interpreters (UV_PYTHON_INSTALL_DIR)
    tools/
```

On a login node:

```bash
cd /dls/data2temp01/hamlet/workspace/FieldX

# Keep uv off the home quota BEFORE syncing: the CUDA wheels are several GB and a
# home-quota cache fails partway through extraction with "Disk quota exceeded".
# These are uv's own variables, so they also apply to bare `uv` commands.
export UV_CACHE_DIR=/dls/data2temp01/hamlet/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/dls/data2temp01/hamlet/workspace/uv-python

export FIELDX_DATA_ROOT=/dls/data2temp01/hamlet/workspace/data
export FIELDX_RUNS_ROOT=/dls/data2temp01/hamlet/workspace/runs

uv cache dir     # confirm it is NOT under $HOME
uv sync --locked --extra cuda13 --group dev

uv run pytest
uv run ruff check .

uv run fieldrefine init-pdb 1UBQ --force
uv run fieldrefine inspect      configs/1ubq/default.yaml
uv run fieldrefine config-check configs/1ubq/default.yaml

scripts/submit.sh configs/1ubq/default.yaml
```

`--force` regenerates the committed example config against the workspace roots.

The login node has no GPU and correctly reports `[CpuDevice(id=0)]`. Every GPU SLURM job
loads CUDA itself with

```bash
module load cuda
```

which on DLS provides `/dls_sw/apps/cuda/13.3.1` and CUPTI at
`extras/CUPTI/lib64/libcupti.so`. That is the only site-specific logic in the repository
and it lives in [`slurm/dls/cuda.sh`](slurm/dls/cuda.sh); point `FIELDX_CUDA_INIT` at
your own equivalent on another cluster.

An interactive GPU node is never required for normal operation.

## SLURM pipeline

```
00_inspect
  -> 10_prepare              reflections, train/tune/free split
  -> 20_make_rho0            Gemmi density, solvent mask, rho0 sanity check
  -> 25_scaling_train        Gemmi scaling on train only
  -> 30_numerics             device report, memory estimate, FFT and derivative gates
  -> 40_gpu_tests            accelerator pytest gate
  -> 45_atomic_benchmark     xyz / B / occupancy / ADP representability
  -> 50_prior_candidate      SLURM array, one GPU job per prior candidate
  -> 52_select_prior         ranked on the tune subset only
  -> 55_scaling_work         Gemmi scaling refit on all work reflections
  -> 60_final_fit            field refit on all work reflections
  -> 70_information          matrix-free local information spectrum
  -> 75_freeze_model         MODEL_LOCK.json
STOP.
  -- manual --
     80_evaluate_free        one-shot free-set evaluation
```

Every job resolves the repository through `$SLURM_SUBMIT_DIR`, never through `$0`: SLURM
may stage the submitted script under `/var/spool/slurm/...`, where `$0` does not point
into the checkout. This is enforced by `tests/unit/test_slurm_control_plane.py`.

No production FFT optimization, information-spectrum solve or accelerator test is ever
intended for a login node.

## Maps

Stage 60 writes `final/maps/`: the starting density, the inferred correction on its own,
the refined total, and 2Fo-Fc / Fo-Fc maps with both FieldX and atomic-model phases.
`maps.json` describes each one. Compare `2FoFc_field` against `2FoFc_atomic` — not
against `refined.ccp4`, which is a calculated density and smooth by construction. See
[`docs/WORKFLOW.md`](docs/WORKFLOW.md#looking-at-the-maps).

## Statistical barrier

Reflections have three roles:

```
D_work = D_train  u  D_tune          D_free untouched
```

1. `train` — field fitting and nuisance-scale calibration during prior comparison;
2. `tune` — prior selection, hyperparameter selection and early stopping;
3. `free` — held out from everything until a model lock exists.

After a prior is selected it is frozen, the nuisance scaling and the field are refit on
all of `D_work`, and the configuration, prepared split, starting density, work scaling,
fitted latent field and information spectrum are hashed into `MODEL_LOCK.json`.

The free-set evaluation is enforced as one-shot, not merely documented as one-shot. Each
read appends to an append-only `FREE_SET_LEDGER.json` beside the prepared reflections;
afterwards both `evaluate-free` and `freeze-model` refuse to run without an explicit,
recorded override. The ledger lives with the prepared split, so re-freezing a revised
model into a fresh output directory does not escape it.

The one-shot report contains

```
R_work_atomic  R_work_field   delta_R_work
R_free_atomic  R_free_field   delta_R_free
gap_atomic     gap_field      delta_gap
```

When an entry deposits no usable free flags — as for 1UBQ, whose `FreeR_flag` column is
constant — the split falls back to a deterministic Friedel-paired hash holdout. That set
is genuinely untouched and a valid cross-validation statistic, but it is **not** the
deposited crystallographic R-free, and the report says so in its `free_set_kind` field.

### When no holdout is possible at all

`split.strategy: none` declares that no held-out set exists. The whole dataset becomes
work, the sole target is `R_work`, and every free-set stage degrades to a clearly
labelled no-op instead of failing: `prepare` assigns nothing to free, the R-factor
denominators stay finite rather than producing NaN, `freeze-model` records
`target: R_work only`, and `evaluate-free` writes `FREE_EVALUATION_SKIPPED.json` and
exits 0. Generate such a config with `fieldrefine init-pdb <PDBID> --no-free-set`.

Use it only when a holdout is genuinely impossible. With nothing held out the primary
criterion cannot be evaluated at all, and a lower `R_work` is not evidence of recovered
density — a correlated field with this much capacity can always reduce it. Every
artifact produced in this mode says so; do not quote `R_work` from it as a result.

## Repository layout

```
configs/       dataset configurations and prior grids (1UBQ, 6O2H, template)
src/           scientific code, free of site- and dataset-specific paths
slurm/         SLURM jobs, common runtime, and slurm/dls/ site initialisation
scripts/       submission, preflight and transfer helpers
tests/         unit, integration and GPU suites
docs/          methods, HPC guide, testing notes, v4 design
paper/         manuscript source
containers/    optional Apptainer definition
```

## Datasets

| Entry | Role | Notes |
|---|---|---|
| 1UBQ | first v3 experiment | ubiquitin, `P 21 21 21`, 1.80 Å, hash holdout ([details](configs/1ubq/README.md)) |
| 6O2H | next major benchmark | ambient-temperature triclinic lysozyme, `P1`, 1.21 Å; has diffuse-scattering data that v3 does not use |

## Documentation

- [`docs/METHODS.md`](docs/METHODS.md) — the model, the priors and the forward calculation
- [`docs/HPC.md`](docs/HPC.md) — SLURM, CUDA, memory planning and site portability
- [`docs/TESTING.md`](docs/TESTING.md) — what each suite covers and where it runs
- [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — day-to-day operation and troubleshooting
- [`docs/V4_DESIGN.md`](docs/V4_DESIGN.md) — the next formulation, and how it relates to v3
- [`docs/RESULTS_TEMPLATE.md`](docs/RESULTS_TEMPLATE.md) — how to report a run honestly

## References

- Brünger AT. Free R value. *Nature* 355, 472–475 (1992). DOI: 10.1038/355472a0
- Holton JM *et al.* The R-factor gap in macromolecular crystallography. *FEBS J* 281, 4046–4060 (2014). DOI: 10.1111/febs.12922
- Meisburger SP, Case DA, Ando N. Diffuse X-ray scattering from correlated motions in a protein crystal. *Nat Commun* 11, 1271 (2020). DOI: 10.1038/s41467-020-14933-6
- Wojdyr M. GEMMI: a library for structural biology. *JOSS* 7, 4200 (2022). DOI: 10.21105/joss.04200
- Greisman JB, Dalton KM, Hekstra DR. reciprocalspaceship. *J Appl Cryst* 54, 1521–1529 (2021). DOI: 10.1107/S160057672100755X
