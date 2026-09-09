# Contributing

1. Create a feature branch.
2. **Keep the statistical barrier intact.** No code path used for fitting, early stopping,
   hyperparameter selection, prior selection, grid selection, nuisance calibration or
   diagnostics may read free reflections. `tests/unit/test_statistical_separation.py`
   mutates the free observations and requires every decision-bearing quantity to be
   unchanged; extend it whenever you add one.
3. **Keep v3 generic.** The correlated prior exists so coherent multi-voxel density is
   plausible and isolated spikes are not. Do not make physical or atomic modes a
   requirement of v3; that is a v4 extension, and `docs/V4_DESIGN.md` explains why it
   must remain optional there too.
4. **Keep the SLURM contract.** No sbatch script may locate repository files through
   `$0`; use `$SLURM_SUBMIT_DIR`. Every GPU job loads CUDA itself. No pipeline code may
   reference a specific PDB entry or a site-specific path outside `slurm/dls/`.
   `tests/unit/test_slurm_control_plane.py` enforces all of this.
5. Run `make check` before opening a pull request. It runs `bash -n` on every shell and
   sbatch script, `ruff format --check`, `ruff check` and the CPU test suite.
6. Mark accelerator-heavy tests `@pytest.mark.gpu`; expensive end-to-end runs
   `@pytest.mark.cluster`. Neither runs in CI.
7. Never commit crystallographic data, HDF5 files or run products. `data/` and `runs/`
   are gitignored; datasets are fetched with `fieldrefine init-pdb`.
8. Use `uv` exclusively. No `pip install --user`, no `python -m venv`, no generated
   environment file.

Scientific changes that alter the prior, likelihood, split logic, grid convention or
R-factor calculation need a focused test and a note in `docs/METHODS.md`.

Tests build their own synthetic dataset (see `tests/conftest.py`); do not add a test that
depends on a downloaded reflection file or on a particular PDB entry.
