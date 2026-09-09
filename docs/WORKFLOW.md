# Day-to-day workflow

## Setup

```bash
git clone git@github.com:YOUR_ORG/FieldX.git
cd FieldX
uv sync --locked --extra cpu --group dev     # or --extra cuda13 on a GPU cluster
make check
```

In PyCharm: *Get from VCS*, then select `.venv` as the project interpreter. Do not
install CUDA packages in a laptop environment; GPU work happens in the cluster
environment.

## A new dataset

```bash
uv run fieldrefine init-pdb 6O2H
# or several at once; a dud entry costs that entry only, and the batch reports it:
uv run fieldrefine init-pdb 6O2H 3K0N 5E6Y
uv run fieldrefine inspect      configs/6o2h/default.yaml
uv run fieldrefine config-check configs/6o2h/default.yaml
```

`inspect` is the command to read before anything heavy. Check:

- `spacegroup` and `cell` agree between the model and the reflections (`model.cell_consistent`);
- `column_suggestions` matches what the config selected;
- `free_counts` shows a plausible held-out minority for the chosen `free_test_value`;
- `d_min_configured` is not finer than `d_min_in_file`;
- `suggested_grid_shape` is what you expect for `samples_per_dmin = 3.0`.

If `free_counts` shows a single value, the entry deposits no usable free set. `init-pdb`
will already have selected `split.strategy: hash`; the resulting statistic must be
reported as a held-out test R, not as the deposited R-free.

## Submitting

```bash
export CFI_SBATCH_ARGS="--account=MYACCOUNT --partition=MYGPU"
export CFI_ARRAY_LIMIT=4
scripts/submit.sh configs/6o2h/default.yaml
```

The script validates the config, expands the prior grid on the submission host, and
submits the whole chain with `afterok` dependencies. It prints every job id and the
exact free-evaluation command, then stops.

```bash
squeue -u "$USER"
sacct  -u "$USER" --starttime today --format=JobID,JobName%24,State,Elapsed,MaxRSS
```

Logs land in `<run_root>/logs/`.

## Reading the artifacts

| File | Stage | What to check |
|---|---|---|
| `shared/metadata.json` | 10 | grid shape, split sizes, free fraction |
| `shared/rho0_stats.json` | 20 | density range, declared grid |
| `rho0_check.json` | 20 | `pass`, electron-count and amplitude error |
| `scaling_train.json` | 25 | `r_factor`, `k_overall`, `b_sol` |
| `fft_check.json` | 30 | `pass_expected_relation`, `ambiguous` false |
| `derivative_check.json` | 30 | `pass`, both relative errors |
| `baseline_metrics.json` | 30 | the atomic baseline `r_work`, before any fitting |
| `atomic_benchmark.json` | 45 | `per_kind_max_latent_rms` |
| `candidates/*/fit/metrics.json` | 50 | per-candidate `chi2_tune`, `r_tune`, and `prior` (the effective parameters) |
| `selected.selection.json` | 52 | the ranking and the winner |
| `final/fit/metrics.json` | 60 | work-set fit; `history.csv` for convergence |
| `final/maps/*.ccp4` | 60 | the map set; read `maps/maps.json` first |
| `final/decomposition/decomposition.json` | 72 | explained fraction vs its capacity control |
| `final/decomposition/explained.ccp4` | 72 | the part of the correction the atomic tangent space explains |
| `final/decomposition/unexplained.ccp4` | 72 | density no atomic parameter can produce |
| `final/info_spectrum.json` | 70 | eigenvalues, `d_eff` lower bound |
| `final/MODEL_LOCK.json` | 75 | hashes of everything frozen |
| `final/FREE_EVALUATION.json` | manual | the one-shot result |

## Looking at the maps

Stage 60 writes `final/maps/`, described by `maps.json`:

| file | what it is |
|---|---|
| `rho0.ccp4` | the starting atomistic density (calculated) |
| `field.ccp4` | **the inferred correction on its own** — what FieldX added |
| `refined.ccp4` | `rho0 + L z` (calculated) |
| `2FoFc_field.ccp4` | experimental amplitudes, FieldX phases |
| `2FoFc_atomic.ccp4` | experimental amplitudes, atomic-model phases |
| `FoFc_field.ccp4` | difference map, FieldX phases |
| `FoFc_atomic.ccp4` | difference map, atomic-model phases |
| `2mFoDFc_field.ccp4` | **sigma-A weighted**, FieldX phases — the map to rebuild against |
| `2mFoDFc_atomic.ccp4` | sigma-A weighted, atomic-model phases |
| `mFoDFc_field.ccp4` | sigma-A weighted difference map, FieldX phases |
| `mFoDFc_atomic.ccp4` | sigma-A weighted difference map, atomic-model phases |

Open them against the coordinate model in Coot or ChimeraX.

**Compare like with like.** `refined.ccp4` is a *calculated* density and is smooth by
construction — ideal form factors, band-limited correction. A conventional 2Fo-Fc map is
built from measured amplitudes and carries noise and series termination. The comparable
pair is `2FoFc_field` against `2FoFc_atomic`; comparing `refined.ccp4` to a deposited
2Fo-Fc map is not like-for-like and will always look "too clean".

`field.ccp4` is the object worth most of your attention: it is exactly what the
refinement inferred. Ask whether it is spatially coherent and near chemically plausible
locations, or scattered.

`FoFc_field` should be flatter than `FoFc_atomic` if the field explained real residual
density; `maps.json` reports the RMS of each.

Maps are built from work reflections only — the free set stays out of interpretation too.

**Sigma-A weighting.** `2mFoDFc` and `mFoDFc` follow Read (1986): per resolution shell the
model gets a scale `D` and a residual variance, from which each reflection gets a figure
of merit `m`. Centric reflections use `m·Fo` rather than `2m·Fo − D·Fc`, since a
restricted phase would otherwise count the observation twice.

Where `m` and `D` come from is recorded in `maps.json` under `sigma_a`:

- `--sigma-a-from work` (default) estimates on the work reflections. Safe at any time, and
  biased towards `m` too large because the model was fitted to them. The manifest says so.
- `--sigma-a-from free` estimates on the free set — the conventional, unbiased choice —
  and is **refused** until the one-shot evaluation has already happened. After that the
  set is spent, so using it for weighting costs nothing further.

Check `sigma_a.mean_figure_of_merit` and the per-shell `D`: `D` far from 1 means the
scaling is off, and `m` near 1 everywhere means either an excellent model or (more often)
that the weights were estimated on data the model had already fitted.

## Running and reading the decomposition

Stage 72 asks whether ordinary refinement could have produced what the field inferred.
It runs as part of `scripts/submit.sh`, needs no GPU, and can be re-run by hand against
a finished run at any time:

```bash
fieldrefine decompose configs/<name>.yaml
fieldrefine decompose configs/<name>.yaml --basis full --n-trials 16
```

Both flags override the config for that invocation only. The `decomposition:` block
holds the defaults, and every key has one, so an absent block behaves as shown:

```yaml
decomposition:
  enabled: true              # false skips the stage entirely and writes nothing at all
  basis: coordinates         # coordinates | coordinates_b | full | residue_rigid
  box_radius_angstrom: null  # extra per-column truncation; null keeps Gemmi's own cutoff
  ridge: 0.0                 # Tikhonov damping, for an ill-conditioned basis
  n_capacity_trials: 8       # matched random fields used to calibrate the control
  write_maps: true           # write explained.ccp4 and unexplained.ccp4
```

A larger basis explains more of anything at all, which is precisely what the capacity
control is there to measure: widening `basis` raises the control floor along with the
reported fraction. Compare `explained_above_control` across bases, never
`explained_fraction` on its own.

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

## The free set

After the freeze job succeeds:

```bash
cd /path/to/FieldX
sbatch $CFI_SBATCH_ARGS --export=ALL,CFI_CONFIG=<run_root>/selected.yaml,CFI_PROJECT_ROOT=$PWD \
  slurm/80_evaluate_free.sbatch
```

Run this once. It verifies `MODEL_LOCK.json`, reads the free reflections, reports the
atomic baseline and the field model side by side, and appends to the ledger. Afterwards
both `evaluate-free` and `freeze-model` refuse to run again without an override that is
itself recorded.

## Troubleshooting

**`Disk quota exceeded` during `uv sync`** — uv is caching into your home quota. Run
`uv cache clean`, then `export UV_CACHE_DIR=<shared path>` and
`export UV_PYTHON_INSTALL_DIR=<shared path>` and retry. `FIELDX_UV_CACHE` does not cover
a bare `uv sync`; it is only read once `slurm/common.sh` is sourced.

**MTZ dataset errors** — `... contains N data-bearing datasets` or `column 'FP' occurs
in 2`. The file holds more than one dataset (MAD/SAD wavelengths, native + derivative)
and reuses column labels. Run `fieldrefine inspect` and read the `mtz` block, then set
`input.mtz_dataset` to the id or name you want. FieldX will not choose for you:
reciprocalspaceship's implicit behaviour keeps the *last* duplicate column while
recording the *first* dataset's wavelength, so a run made that way cannot say which data
it used. Reflection CIFs have no datasets and ignore this setting.

**`grid.shape ... is incompatible with space group`** — tetragonal, trigonal, hexagonal
and cubic groups require equal sampling along symmetry-related axes. `config-check`
catches this on the login node now; leave `grid.shape: null` to have it derived from the
cell, `d_min` and `samples_per_dmin`.

**`unit cell ... is not metrically compatible with space group`** — the model's cell
cannot host that symmetry (e.g. `a != b` in a tetragonal group). Check the coordinate
file before anything else.

**`CUDA_ERROR_OUT_OF_MEMORY` / `Failed to load in-memory CUBIN`** — usually preallocation,
not data size; it can fire while copying one scalar. Check `estimate-memory` first: if it
reports well under the card's capacity, the arena is starving the driver. The default is
now `XLA_PYTHON_CLIENT_PREALLOCATE=false`; if you set it true, keep
`XLA_PYTHON_CLIENT_MEM_FRACTION` at 0.75–0.8. Read `device-report`'s `memory` block to
confirm. If jobs have run on different GPU models, also clear `.jax-cache`.

**`sbatch not found`** — you are not on a submission host.

**`'<dir>' is not a FieldX repository`** — `sbatch` was invoked from outside the
checkout, so `SLURM_SUBMIT_DIR` points elsewhere. Run `scripts/submit.sh` from anywhere;
it changes to the repository root before submitting.

**`no usable uv executable`** — uv is neither on `PATH` nor in `.venv/bin`. Set
`FIELDX_UV` to an absolute path.

**`Prior grid not found`** — every dataset needs its own grid. There is no fallback, by
design. Regenerate with `uv run fieldrefine init-pdb <PDBID> --force`.

**`FFT grid ... is too small for retained HKLs`** — raise `grid.samples_per_dmin` or set
an explicit even `grid.shape`. Do not raise it for the 1UBQ pilot; that run is testing
the existing v3 sampling convention.

**`free_test_value=... gives free fraction ...`** — the free-flag convention is wrong for
this file. Read `free_counts` from `inspect` before changing anything.

**`Every required split must contain at least one reflection`** — the chosen holdout
leaves train, tune or free empty. Adjust `tune_fraction_of_work` /
`final_fraction_if_hash` first. If the dataset genuinely cannot support a holdout, set
`split.strategy: none` (or regenerate with `init-pdb --no-free-set`): the run then
targets `R_work` only, nothing errors, and `evaluate-free` writes
`FREE_EVALUATION_SKIPPED.json` instead of a result. Note that no cross-validated claim
can be made from such a run.

**Derivative check fails** — the gate reports the finite-difference step it chose along
with both relative errors. A failure here means the forward model and its adjoint
disagree; do not proceed to fitting.

**Information spectrum refused with `MemoryError`** — the estimate exceeded
`information.memory_budget_gib`. Reduce `information.n_modes`, raise the budget for a
larger GPU, or coarsen the grid.
