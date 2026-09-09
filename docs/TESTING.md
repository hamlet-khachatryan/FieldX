# Testing

## Local and CI

```bash
make check
```

which is

```bash
bash -n every shell and sbatch script
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest -m "not gpu and not cluster"
```

CI (`.github/workflows/ci.yml`) runs exactly this on Python 3.11, 3.12 and 3.13, with
CPU JAX only. **CI never requires CUDA.**

Every test builds its own tiny synthetic dataset — a ten-atom `P 1` model, structure
factors by direct summation, an MTZ written through reciprocalspaceship — so no test
depends on a downloaded reflection file or on a particular PDB entry.

## What each suite covers

| Suite | Covers |
|---|---|
| `test_config.py` | schema validation, path resolution, invalid priors and grids, empty splits, every committed config |
| `test_priors.py` | each kernel, tau normalisation, **grid independence in physical units**, band limit, mean removal, latent distributions, multiscale mixing, spike-vs-blob cost |
| `test_crystallography.py` | metric tensors, resolution agreement with Gemmi, **22 space groups across all 7 crystal systems and all centrings (P/A/C/I/F/R, both rhombohedral settings)**, the reduced-projection equivalence proof and its precondition, systematic absences, declared-grid density in symmetric groups, electron count, Nyquist and grid-symmetry validation |
| `test_forward_model.py` | constant and point-like densities, analytic Gaussian transform, FFT index wrapping, P1 symmetry projection, observables |
| `test_fft_convention.py` | the Gemmi/JAX sign convention and its coupling to the symmetry phase |
| `test_derivatives.py` | finite differences vs autodiff at and away from zero, `J v`, `J^T w`, the adjoint identity, and that no dense Jacobian is formed |
| `test_statistical_separation.py` | train/tune/free leakage, invariance of every selection metric to free-set mutation, deterministic and seed-dependent splits |
| `test_model_lock.py` | required artifacts, config and artifact tampering, the append-only one-shot ledger |
| `test_model_selection.py` | tune-only ranking, work-scope refit config, missing candidate results |
| `test_prior_sweep.py` | declarative sweeps: axis products, collapsing of inert parameters, name generation, the candidate cap, and combination with hand-written candidates |
| `test_prior_grid.py` | every committed grid expands into valid train-scope candidates |
| `test_slurm_control_plane.py` | `bash -n`, no `$0` path resolution, `SLURM_SUBMIT_DIR` usage, GPU jobs load CUDA, no retired environment variable, no dataset fallback, dependency chain, fake-SLURM path resolution |
| `test_multi_pdb_init.py` | batch initialisation: identifiers validated and de-duplicated before any download, one failure not stopping the rest, `--fail-fast`, per-entry isolation, non-zero exit |
| `test_pdb_generic.py` | ID normalisation, mocked RCSB, missing structure factors, column and free-flag discovery, generic dataset naming |
| `test_mtz_dataset.py` | multi-dataset MTZ enumeration, explicit selection by id/name, refusal to guess, provenance into `metadata.json`, and a regression pinning how this differs from reciprocalspaceship's implicit result |
| `test_empty_field_control.py` | `starting_density: zero` -- the empty cell really is empty, the mandatory settings are enforced, the NaN gradient at z=0 is pinned, and the control fits training data without generalising |
| `test_no_free_set.py` | `split.strategy: none` — everything becomes work, free metrics stay finite rather than NaN, `evaluate-free` reports instead of raising, and a *misconfigured* holdout still fails loudly |
| `test_sigmaa.py` | figure of merit bounded, monotonic and overflow-safe; centric vs acentric forms; scale recovery; `m` falling for an unrelated model; shells applied across reflection sets |
| `test_maps.py` | the CCP4 map set: every file written and readable, `field` exactly `refined - rho0`, free reflections excluded, an Fc map reproducing the density it came from (pinning both the phase conjugation and the ASU reduction), and the sigma-A source gate |
| `test_memory_estimator.py` | grid sources, scaling with grid and mode count, the information budget guard |
| `test_tangent_basis.py` | density additivity over atoms, per-atom derivative columns, sparsity, symmetry copies, basis column counts, the truncation guard |
| `test_decomposition.py` | exact least-squares recovery, rank deficiency reported not hidden, symmetrization and the antisymmetric fraction, ground-truth displacement recovery, the control detecting over-parameterisation, artifacts, free-set invariance |
| `test_cli.py` | login-node commands run without an accelerator |
| `integration/test_pipeline.py` | prepare → rho0 → mask → scaling → gates → fit → information → freeze → one-shot free evaluation |
| `integration/test_atomic_representability.py` | coordinate, occupancy, B and anisotropic ADP perturbations; the model is left untouched |

## GPU gate

`slurm/40_gpu_tests.sbatch` runs `pytest -m gpu tests/gpu`. Those tests assert that an
accelerator is actually present, that the field operator is differentiable on it, that
FFTs round-trip, and that jitted executables are reused. They perform no production
fitting.

## Production numerical gates

Before any prior sweep the SLURM DAG requires:

- the starting density to match the crystallographic model (electron count and
  direct-summation structure factors);
- Gemmi and JAX FFT structure factors to agree under the expected conjugation;
- finite-difference and autodiff directional derivatives to agree;
- `<J v, w> = <v, J^T w>` within tolerance;
- the accelerator test suite to pass;
- the atomic representability benchmark to complete.

Each gate raises and blocks the `afterok` chain rather than warning.

## Cluster-only tests

The `cluster` marker is reserved for expensive end-to-end runs on real data. Nothing in
CI or in the default `pytest` invocation runs them.
